import gym
import torch
import time
from gym import spaces
from gym.spaces import Box
import torchvision.transforms as transforms
import numpy as np
from tensordict.tensordict import TensorDict
from gym.wrappers import flatten_observation
import torch.nn.functional as F

from KUKA import KUKA

from multiprocessing import Process
from threading import Thread


class KUKA_environment(gym.Env):

    '''
    ******************************************************************

        Class contains KUKAs simulation as a gym environment. The
        neccessary functions are implemented,
        as mentioned in: https://gymnasium.farama.org/introduction/create_custom_env/

    ******************************************************************
    '''
    
    # Goal options
    goal_opitons = ['chest', 'left arm', 'right arm', 'left leg', 'right leg', 'waist']
    # Observations (joints positions and image size)
    N_observations = 7 + (256 * 256 * 3) # image contains 196608 elements (196615)

    def __init__(self, goal_point = None, time_steps = None, **kwargs):
        super().__init__()

        # Gym initialization

        '''
        Box: gym data-type describing numpy arrays with elements taking values continuously in
        a range.
        '''

        self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                                # Box: e.g. array between [-1,1] of 7 elements, and type
                # 'Joint positions': Box(low = -np.inf, high = np.inf, shape = (7,), dtype = np.float32), # [theta1, ..., theta7]
                'Image': Box(low = -np.inf, high = np.inf, shape = (3, 256, 256), dtype = np.float32),           #RGB
            }
        )
        self.time_steps = time_steps if time_steps is not None else 100
        self.goal_point = goal_point if goal_point is not None else "chest"
        self.kuka = KUKA()

    def get_observations(self):

        # Observations composed by joints thetas and image from sensor
        img = self.kuka.torch_vs_data() # torch.Size([256, 256, 3])
        joints_pos = self.kuka.joints_positions()

        # return {'Joint positions': joints_pos, "Image": img}
        return {"Image": img}

    def info(self):

        # Information of the environment current distance to goal
        return {'Distance end effector': self.kuka.end_effector_pos()}

    def reset(self, *, seed = None, options = None):

        if seed is not None:
            self.seed(seed)  # Set the seed using the seed method

        # Restarts simulation and place robot in original position
        self.kuka.sim.stopSimulation()
        self.kuka.sim.startSimulation()

        # Move robot to home position
        self.kuka.move_joints(self.kuka.home_pos)

        # Reset goal point
        self.goal_point = np.random.choice(self.goal_opitons, 1)
        self.kuka.goal_point_handle_selection(self.goal_point)

        # Returns observations
        return self.get_observations()

    def step(self, action):

        # Logic of the environment.
        """
        # Takes an action and computes the state of the environment after applying the action.
        # Returns next observation and resulting reward
        """

        joined_distances_links = 0
        terminated = False
        truncated = False
        timer = time.time()

        # Map the action & move mannequin simultaneously
        f1 = Thread(target=self.kuka.move_joints(action))
        f2 = Thread(target=self.kuka.move_mannequin("one"))
        f1.start()
        f2.start()

        # Reward parameters
        a = 1 # scaling factors sensitivity to distances
        b = 1
        w1 = 0.7  # weight factors
        w2 = 0.5
        w3 = 2

        # Distances
        distance_EF_goal = -1 * self.kuka.endE_euclidean_goal()
        # distance_EF_goal = 0 if distance_EF_goal == [] else distance_EF_goal[0] # [] -> int

        distance_links_mannequin = self.kuka.links_distances()
        # distance_links_mannequin = [item for sublist in distance_links_mannequin for item in sublist] # [[]] -> []

        collision = self.kuka.collisions()

        for i in range(7):
            joined_distances_links += distance_links_mannequin[i]

        # Reward function
        """
        It gives a bigger reward if the position of the end effector is closer to the
        goal point and the links of the robot are more distant to the mannequin.
        Also, collisions are penalized
        """
        t_1 = (w1 * np.exp(a * distance_EF_goal))
        t_2 = (w2 * np.exp(b * joined_distances_links))
        t_3 = (w3 * collision)
        reward = t_1 + t_2 - t_3

        # f = open('rewards', 'a')
        # to_write = str(reward) + '\n'
        # f.write(to_write)
        # f.close()

        """
        Terminated: episode ending after reaching a terminal state that is defined as
        part of the environment definition. Examples are - task success, task failure,
        robot falling down etc.

        Truncated: episode ending after an externally defined condition (that is outside
        the scope of the Markov Decision Process).
        """

        # Environment terminated if end_effector close enough to goal point
        # terminated = True if all(dist <= 0.01 for dist in distance_EF_goal) else False
        terminated = True if distance_EF_goal <= 0.01 else False

        # Truncation if episode lasts more than 4 seconds
        truncated = True if timer == 4 else False

        # Sends whether is done or not
        done = terminated or truncated

        # Dictionary to return
        info = self.info()
        # send = {"Reward": reward, "Done": done, "Info": info}
        # ob = self.get_observations()
        # send.update(ob)
        # send.update(info)
        # print(send)

        return self.get_observations(), reward, done, info
        # return send