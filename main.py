from PPO import PPO

'''
**************************************************************************

Performs the simulation.

**************************************************************************
'''

'''
    to do:
    PPO error -> Can only get one observation (doesn't make sense joint positions)
    np.inline(action)to discretize and give reward for each step of trajectory (like mannequin)
    Fix IK, give reward if closer to it
    
'''

goal_point = "chest" # choose between "chest", "left arm", "left leg", 
                     # "right arm", "right leg" and "waist"

PPO = PPO(
    goal_point, 
    frames_per_batch = 2,#00,
    total_frames = 5000,  # For a complete training, bring the number of frames up to 1M
    sub_batch_size = 3,#2,    # Cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 3,         # Optimization steps per batch of data collected
    clip_epsilon = 0.2,     # Clip value for PPO loss
    gamma = 0.99,           # Discount factor
    lmbda = 0.95,           # Learning rate
    entropy_eps = 1e-4,     # Clipping parameter
    alpha = 0.1
)  




# PPO = PPO(
#     goal_point, 
#     frames_per_batch = 1000, 
#     total_frames = 50_000,  # For a complete training, bring the number of frames up to 1M
#     sub_batch_size = 64,    # Cardinality of the sub-samples gathered from the current data in the inner loop
#     num_epochs = 10,        # Optimization steps per batch of data collected
#     clip_epsilon = 0.2,     # Clip value for PPO loss
#     gamma = 0.99,           # Discount factor
#     lmbda = 0.95,           # Learning rate
#     entropy_eps = 1e-4,     # Clipping parameter
#     alpha = 0.1
# )  

PPO.train()
PPO.results()

