import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

import torch.optim as optim

import gym
import os

from KUKA_as_gym import KUKA_environment
from Networks import PPO_UNet, value_network

from gym.wrappers import flatten_observation

import pickle

# Call from U-Net forward-fn
'''
**************************************************************************

Class contains all PPO functions for the algorithm to work. It connects to 
KUKAs environment previously implemented as a gym-type, 
as mentioned in: https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html

**************************************************************************
'''


# Register the environment
gym.register(
    id="KUKA-v0",                                        # Unique identifier for this environment
    entry_point="KUKA_as_gym:KUKA_environment",          # Reference to the module and class
    max_episode_steps=200,                               # Optional: set max steps per episode
    kwargs={"goal_point": 'chest', "time_steps": 100},   # Default arguments
)

class PPO():
        
    def __init__(self, goal_point, frames_per_batch = 1000, total_frames = 50_000, sub_batch_size = 64, num_epochs = 10, 
                 clip_epsilon = 0.2, gamma = 0.99, lmbda = 0.95, entropy_eps = 1e-4, alpha = 0.1):        

        # Define hyperparameters
        is_fork = multiprocessing.get_start_method() == "fork"
        self.device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
            else torch.device("cpu")
        )

        # Data collection parameters
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        
        # PPO parameters
        self.sub_batch_size = sub_batch_size  
        self.num_epochs = num_epochs  
        self.clip_epsilon = (
            clip_epsilon  
        )
        self.gamma = gamma 
        self.lmbda = lmbda 
        self.entropy_eps = entropy_eps 
        self.alpha = alpha
        
        # Value network parameters
        self.num_cells = 256  # number of cells in each layer i.e. output dim.
        self.lr = 3e-4
        self.max_grad_norm = 1.0

        # Define environment
        self.base_env = GymEnv(
            "KUKA-v0", 
            device=self.device, 
            kwargs={"goal_point": goal_point, "time_steps": 100}
        )

        # Normalization of the environment
        self.env = TransformedEnv(
            self. base_env,
            # self.wrapped_env,
            Compose(
                # normalize observations
                ObservationNorm(in_keys=[
                    'Image', 'Joints positions', 'Distance end effector'
                ]),
                DoubleToFloat(),
                StepCounter(),
            ),
        )
        
        for key in ['Image', 'Joints positions', 'Distance end effector']:
            self.env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0, key = key) 
            # print("normalization constant shape:", self.env.transform[0].loc.shape)
        
        # UNet 
        inp = 3 * 256 * 256 # joints + image (resolution)
        out = 7 # joint positions
        self.actor = PPO_UNet(inp, out, self.alpha)

        # Policy 
        # “talk” with the environment through the tensordict data carrier
        self.policy_module = TensorDictModule(self.actor, in_keys=['observation'], out_keys=['loc', 'scale'])
        # distribution out of the location and scale of our normal distribution
        self.policy_module = ProbabilisticActor(
            module=self.policy_module,
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": self.env.action_spec.space.low,
                "max": self.env.action_spec.space.high,
            },
            return_log_prob=True,
            # we'll need the log-prob for the numerator of the importance weights
        )

        # Value Network
        self.value_network = value_network(self.num_cells, self.device)
        print("Running policy:", self.policy_module(self.env.reset()))
        print("Running value:", self.value_network.value_module(self.env.reset()))

        # Collector reset an environment, compute an action given the latest observation,
        # execute a step in the environment, and repeat the last two steps until the 
        # environment signals a stop
        self.collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=self.device,
        )

        # Refilled every time a batch of data is collected
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.frames_per_batch), 
            sampler=SamplerWithoutReplacement(),
            )
        
        # Loss function
        # How better is the current policy compared to previous
        # Loss function already implemented by PPO

        self.advantage_module = GAE(
            gamma = self.gamma, 
            lmbda = self.lmbda, 
            value_network = self.value_network.value_module,
            average_gae = True
        )

        self.loss_module = ClipPPOLoss(
            actor_network = self.policy_module,
            critic_network = self.value_network.value_module,
            clip_epsilon = self.clip_epsilon,
            entropy_bonus = bool(self.entropy_eps),
            entropy_coef = 1.0,
            loss_critic_type = "smooth_11",
        )

        # Plot parameters
        self.logs = defaultdict(list)

    def save_model(self):
        self.actor.save_checkpoint()

    def load_model(self):
        self.actor.load_checkpoint()

    def set_state(self, observation):
        self.state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

    def train(self):

        '''
        Main trainning algorithm.
        '''

        pbar = tqdm(total=self.total_frames)
        eval_str = ""

        # Iterate over the collector until it reaches total number of frames designed to 
        # collect
        for i, tensordict_data in enumerate(self.collector):

            # For this batch
            self.advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            self.replay_buffer.extend(data_view.cpu())

            for _ in range(self.frames_per_batch // self.sub_batch_size):
                
                # Need advantage to make network work. Compute atvantage each epoch
                # which depends on value_network
                subdata = self.replay_buffer.sample(self.sub_batch_size)
                loss_vals = self.loss_module(subdata.to(self.device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
                optim.step()
                optim.zero_grad()
        
        self.logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={self.logs['reward'][-1]: 4.4f} (init={self.logs['reward'][0]: 4.4f})"
        )
        self.logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {self.logs['step_count'][-1]}"
        self.logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {self.logs['lr'][-1]: 4.4f}"

        if i % 10 == 0:

            # Evaluation of the policy every 10 batches of new data. Performed an execution
            # of policy without exploration for a number of steps.
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                
                # execute a rollout with the trained policy
                eval_rollout = self.env.rollout(1000, self.policy_module)
                self.logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                self.logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                self.logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {self.logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {self.logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {self.logs['eval step_count'][-1]}"
                )
                del eval_rollout
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

            # scheduler.step() Nor necessary

            # Save logs into file
            with open("ppo_logs.pkl", "wb") as f:
                pickle.dump(self.logs, f)

            self.load_model()
    
    def results(self):

        # Plot the results of the training
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self.logs["reward"])
        plt.title("training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(self.logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(self.logs["eval reward (sum)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(self.logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.show()

    def results_from_file(self):
        
        # Open logs file
        with open("ppo_logs.pkl", "rb") as f:
            logs = pickle.load(f)

        # Plot the results of the training
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(logs["reward"])
        plt.title("training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(logs["eval reward (sum)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.show()