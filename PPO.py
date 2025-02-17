import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
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

import gym
import os

from KUKA_as_gym import KUKA_environment
from Networks import PPO_UNet, value_network

from gym.wrappers import flatten_observation

import pickle

import torchvision.models as models

class PPO():

    '''
    **************************************************************************

        Class contains all PPO functions for the algorithm to work. It connects to 
        KUKAs environment previously implemented as a gym-type, 
        as mentioned in: https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html

    **************************************************************************
    '''
        
    def __init__(self, goal_point, frames_per_batch = 1000, total_frames = 50_000, sub_batch_size = 64, num_epochs = 10, 
                 clip_epsilon = 0.2, gamma = 0.99, lmbda = 0.95, entropy_eps = 1e-4, alpha = 0.1):        
        
        '''
            Parameters
        '''

        # Define hyperparameters
        is_fork = multiprocessing.get_start_method() == "fork"
        self.device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
            else torch.device("cpu")
        )
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        self.lr = 3e-4
        self.max_grad_norm = 1.0
        
        '''
            Environment
        '''

        # Register the environment
        gym.register(
            id="KUKA-v0",                                        # Unique identifier for this environment
            entry_point="KUKA_as_gym:KUKA_environment",          # Reference to the module and class
            max_episode_steps=200,                               # Optional: set max steps per episode
            kwargs={"goal_point": 'chest', "time_steps": 100},   # Default arguments
        )

        # Define environment
        self.base_env = GymEnv(
            "KUKA-v0", 
            device=self.device, 
            kwargs={"goal_point": goal_point, "time_steps": 100}
        )

        # Data to loosely match a unit Gaussian distribution (Normalization layer). 
        # What key to read (in_keys) and what key to write (out_keys).
        self.env = TransformedEnv(
            self. base_env,
            Compose(
                # normalize observations
                ObservationNorm(in_keys=[
                    'Image'
                ]),
                DoubleToFloat(),    # Transform converts double entries to single-precision numbers, ready to be read by the policy
                StepCounter(),      # To count the steps before the environment is terminated
            ),
        )
        
        # # Set normalization layer its normalization parameters. Gather the summary statistics
        self.env.transform[0].init_stats(num_iter=3, reduce_dim=0, cat_dim=0) 
        # self.env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
       
        # self.print_env_specs() # Print environment specs

        '''
            Actor - UNet 
        '''

        # inp = (3, 256, 256) # joints + image (resolution)
        inp = 3 # Image size
        out = 7 # joint positions
        self.actor = PPO_UNet(inp, out)

        '''
            Policy 
        '''

        # “Talk” with the environment through the tensordict data carrier. Reads and writes the tensorDict.
        self.policy_module = TensorDictModule(
            self.actor,                 # Calls the UNet
            in_keys=['Image'],          # -> Keys to be read,, which will be sent to the UNet
            out_keys=["loc", "scale"]   # <- Keys to be written to the input tensordict
            )
        self.policy_module.to(self.device)

        # ProbabilisticActor class to build a TanhNormal out of the location and scale parameters.
        self.policy_module = ProbabilisticActor(
            module=self.policy_module,
            spec=self.env.action_spec,          # Keyword-only argument containing the specs of the output tensor.
            in_keys=["loc", "scale"],           # key(s) that will be read from module (UNet output)
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": self.env.action_spec.space.low,
                "max": self.env.action_spec.space.high,
            },
            return_log_prob=True,
            )

        '''
            Value Network
        '''

        # Read the observations and return an estimation of the discounted return 
        # for the following trajectory. 
        self.value_network = value_network(3,1,self.device) # The key is ok
        self.value_module = ValueOperator(
            module=self.value_network,
            in_keys=["Image"],
        )
        
        # print("Running policy:", self.policy_module(self.env.reset()))  
        # print("Running value:", self.value_network.value_module(self.env.reset()))

        '''
            Collector
        '''

        # Collector reset an environment, compute an action given the latest observation,
        # execute a step in the environment, and repeat the last two steps until the 
        # environment signals a stop.
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
        
        '''
            Loss function
        '''

        # How better is the current policy compared to previous
        # Loss function already implemented by PPO

        self.advantage_module = GAE(
            gamma = self.gamma, 
            lmbda = self.lmbda, 
            value_network = self.value_module,
            average_gae = True
        )

        self.loss_module = ClipPPOLoss(
            actor_network = self.policy_module,
            critic_network = self.value_module,
            clip_epsilon = self.clip_epsilon,
            entropy_bonus = bool(self.entropy_eps),
            entropy_coef = 1.0,
            loss_critic_type = "smooth_11",
        )

        self.optim = torch.optim.Adam(self.loss_module.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, total_frames // frames_per_batch, 0.0
        )

        '''
            Plot parameters
        '''

        self.logs = defaultdict(list)

    def print_env_specs(self):

        '''
            Prints Observation, Reward, Input & Action specs
        '''

        print("observation_spec:", self.env.observation_spec)
        print("reward_spec:", self.env.reward_spec)
        print("input_spec:", self.env.input_spec)
        print("action_spec (as defined by input_spec):", self.env.action_spec)

    def train(self):

        '''
            Main trainning algorithm:

            Collect data

                Compute advantage

                    Loop over the collected to compute loss values

                    Back propagate

                    Optimize

                    Repeat

                Repeat

            Repeat
        '''

        self.actor.utils.save_checkpoint(self.actor.checkpoint)
        
        pbar = tqdm(total=self.total_frames)
        eval_str = ""

        # Iterate over the collector until it reaches total number of frames designed to 
        # collect
        
        self.actor.utils.load_checkpoint(self.actor.checkpoint)

        for i, tensordict_data in enumerate(self.collector):
            
            # For this batch
            self.advantage_module(tensordict_data) #here
            data_view = tensordict_data.reshape(-1)
            self.replay_buffer.extend(data_view.cpu())

            for j in range(self.frames_per_batch // self.sub_batch_size):
                
                print('batch: ', i, ' frame: ',j)
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
                self.optim.step()
                self.optim.zero_grad()
        
        self.logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={self.logs['reward'][-1]: 4.4f} (init={self.logs['reward'][0]: 4.4f})"
        )
        self.logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {self.logs['step_count'][-1]}"
        self.logs["lr"].append(self.optim.param_groups[0]["lr"])
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

            self.scheduler.step() # Not necessary

            # Save logs into file
            with open("ppo_logs.pkl", "wb") as f:
                pickle.dump(self.logs, f)
    
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