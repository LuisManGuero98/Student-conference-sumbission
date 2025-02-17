from coppeliasim_zmqremoteapi_client import RemoteAPIClient # python3 -m pip install coppeliasim-zmqremoteapi-client
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import os
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F




class KUKA():

    '''
    **************************************************************************

    Class contains functions to establish connection with the scene simulation 
    and perform all neccessary actions for a gym-type simulation.

    **************************************************************************
    '''

    end_effector_name = '/tableKUKA/LBRiiwa7R800/link8_resp/link/Ultrasound_sensor'

    # Robot
    collection_kuka = []
    joints_count = 7
    links_count = 7
    joints_handles = [0] * joints_count
    links_handles = [0] * links_count

    # Mannequin
    leg_handles_count = 4
    mannequin_left_leg_handles = [0] * leg_handles_count
    mannequin_right_leg_handles = [0] * leg_handles_count

    arm_handles_count = 4
    mannequin_left_arm_handles = [0] * arm_handles_count
    mannequin_right_arm_handles = [0] * arm_handles_count

    mann_ob_count = 4
    mannequin_object_targets = [0] * mann_ob_count

    home_pos = [0,0,0,-np.deg2rad(100),0,np.deg2rad(50),0]

    Juan = '/tableJuan/Juan'

    # Juan's Table
    table_Juan_handles = [0] * 5

    # Goal handles
    chest_pos = '/tableJuan/Juan/Joint[2]/Group'
    left_arm_pos = '/tableJuan/Juan/Joint[2]/Joint[0]/Group'
    right_arm_pos = '/tableJuan/Juan/Joint[2]/Joint[1]/Group'
    left_leg_pos = '/tableJuan/Juan/Joint/Joint[0]/Group'
    right_leg_pos = '/tableJuan/Juan/Joint/Joint[1]/Group'
    waist_pos = '/tableJuan/Juan/Group[0]'  

    def __init__(self):

        # Connect to coppeliasim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # Inverse Kinematics 
        self.simIK = self.client.require('simIK')

        # Ultrasound handle 
        self.end_effector_handle = self.sim.getObject(self.end_effector_name)

        # Get vision sensor handle
        self.vision_sensor_handle = self.sim.getObject('/tableKUKA/LBRiiwa7R800/visionSensor')
       
        '''
        Get KUKAs joints handles
        '''

        try: 
            for num in range(self.joints_count):

                if num + 1 == 1:
                    
                    string_objectPath = '/tableKUKA/LBRiiwa7R800/joint'

                else:
                    
                    string_objectPath = '/tableKUKA/LBRiiwa7R800/link' + str(num+1) + '_resp/joint'

                self.joints_handles[num] = self.sim.getObject(string_objectPath) 
                # print(string_objectPath, self.joints_handles[num])
                    
        except:
            print('Could not get joints handles') 

        # Get KUKAs links handles
        try: 
            for num in range(self.links_count):

                if num + 1 == 1:
                    
                    string_objectPath = '/tableKUKA/LBRiiwa7R800/link'

                else:
                    
                    string_objectPath = '/tableKUKA/LBRiiwa7R800/link' + str(num+2) + '_resp/link'

                self.links_handles[num] = self.sim.getObject(string_objectPath) 
                # print(string_objectPath, self.joints_handles[num])
                    
        except:
            print('Could not get links handles') 

        # Kuka collection handles
        self.collection_kuka = self.sim.createCollection()
        self.sim.addItemToCollection(self.collection_kuka, self.sim.handle_tree, self.sim.getObject('/tableKUKA/LBRiiwa7R800'), 0)

        '''
         Get mannequin handles
        '''


        # Left leg
        self.mannequin_left_leg_handles[0] = self.sim.getObject('/tableJuan/Juan/Joint[0]') # Spherical
        self.mannequin_left_leg_handles[1] = self.sim.getObject('/tableJuan/Juan/Joint[0]/Joint') 
        self.mannequin_left_leg_handles[2] = self.sim.getObject('/tableJuan/Juan/Joint[0]/Joint/Group/Joint')
        self.mannequin_left_leg_handles[3] = self.sim.getObject('/tableJuan/Juan/Joint[0]/Joint/Group/Joint/Joint')

        # Right leg
        self.mannequin_right_leg_handles[0] = self.sim.getObject('/tableJuan/Juan/Joint[1]') # Spherical
        self.mannequin_right_leg_handles[1] = self.sim.getObject('/tableJuan/Juan/Joint[1]/Joint')
        self.mannequin_right_leg_handles[2] = self.sim.getObject('/tableJuan/Juan/Joint[1]/Joint/Group/Joint')
        self.mannequin_right_leg_handles[3] = self.sim.getObject('/tableJuan/Juan/Joint[1]/Joint/Group/Joint/Joint')

        # Left arm
        self.mannequin_left_arm_handles[0] = self.sim.getObject('/tableJuan/Juan/Joint[2]/Joint[0]') # Spherical
        self.mannequin_left_arm_handles[1] = self.sim.getObject('/tableJuan/Juan/Joint[2]/Joint[0]/Joint')
        self.mannequin_left_arm_handles[2] = self.sim.getObject('/tableJuan/Juan/Joint[2]/Joint[0]/Joint/Group/Joint')
        self.mannequin_left_arm_handles[3] = self.sim.getObject('/tableJuan/Juan/Joint[2]/Joint[0]/Joint/Group/Joint/Joint')

        # Right arm
        self.mannequin_right_arm_handles[0] = self.sim.getObject('/tableJuan/Juan/Joint[2]/Joint[1]') # Spherical
        self.mannequin_right_arm_handles[1] = self.sim.getObject('/tableJuan/Juan/Joint[2]/Joint[1]/Joint')
        self.mannequin_right_arm_handles[2] = self.sim.getObject('/tableJuan/Juan/Joint[2]/Joint[1]/Joint/Group/Joint')
        self.mannequin_right_arm_handles[3] = self.sim.getObject('/tableJuan/Juan/Joint[2]/Joint[1]/Joint/Group/Joint/Joint')

        # All mannequin handles
        self.mannequin_all_handles = self.mannequin_left_arm_handles + self.mannequin_right_arm_handles + self.mannequin_left_leg_handles + self.mannequin_right_leg_handles

        # Mannequin Object 
        self.mannequin_object_targets[0] = self.sim.getObject('/tableJuan/Juan/mannequinRightFootTarget') # Right foot
        self.mannequin_object_targets[1] = self.sim.getObject('/tableJuan/Juan/mannequinRightHandTarget') # Right hand
        self.mannequin_object_targets[2] = self.sim.getObject('/tableJuan/Juan/mannequinLeftFootTarget') # Left foot
        self.mannequin_object_targets[3] = self.sim.getObject('/tableJuan/Juan/mannequinLeftHandTarget') # Left hand

        # Juan handle
        self.Juan_handle = self.sim.getObjectHandle(self.Juan)

        # Mannequin collection
        self.collection_man_handle = self.sim.createCollection()
        self.sim.addItemToCollection(self.collection_man_handle, self.sim.handle_tree, self.sim.getObject('/tableJuan/Juan'), 0)

        '''
        Get table collection
        '''

        self.collection_table_Juan = self.sim.createCollection()
        self.sim.addItemToCollection(self.collection_table_Juan, self.sim.handle_tree, self.sim.getObject('/tableJuan/table'), 0)
        self.sim.addItemToCollection(self.collection_table_Juan, self.sim.handle_single, self.sim.getObject('/tableJuan/top'), 0)


        # KUKA home position
        self.move_joints(self.home_pos)

        # Goal point
        # self.goal_point_handle_selection("chest")

    def start(self):

        '''
        Start simulation
        '''

        self.sim.startSimulation()
        time.sleep(1)

    def stop(self):

        '''
        Stop simulation
        '''

        self.sim.stopSimulation()
        time.sleep(1)

    def goal_point_handle_selection(self, goal_point):

        '''
        Given the selected goal point, obtains its handle, and saves in the class
        the goal position coordinates.
        '''

        self.goal_point = goal_point if goal_point is not None else 'chest'
        if self.goal_point == 'chest':
            self.goal_handle = self.sim.getObjectHandle(self.chest_pos)
            self.goal_point_pos = self.sim.getObjectPosition(self.goal_handle)
        elif self.goal_point == 'left arm':
            self.goal_handle = self.sim.getObjectHandle(self.left_arm_pos)
            self.goal_point_pos = self.sim.getObjectPosition(self.goal_handle)
        elif self.goal_point == 'left leg':
            self.goal_handle = self.sim.getObjectHandle(self.left_leg_pos)
            self.goal_point_pos = self.sim.getObjectPosition(self.goal_handle)
        elif self.goal_point == 'right arm':
            self.goal_handle = self.sim.getObjectHandle(self.right_arm_pos)
            self.goal_point_pos = self.sim.getObjectPosition(self.goal_handle)
        elif self.goal_point == 'right leg':
            self.goal_handle = self.sim.getObjectHandle(self.right_leg_pos)
            self.goal_point_pos = self.sim.getObjectPosition(self.goal_handle)
        elif self.goal_point == 'waist':
            self.goal_handle = self.sim.getObjectHandle(self.waist_pos)
            self.goal_point_pos = self.sim.getObjectPosition(self.goal_handle)
        else:
            print('Wrong goal point selection')

    def show_m_handles(self):
        
        '''
        For programming purposes
        '''

        # Show mannequin handles
        print('Left hand handles: ')
        print(self.mannequin_left_arm_handles)
        print('Right hand handles: ')
        print(self.mannequin_right_arm_handles)
        print('Left leg handles: ')
        print(self.mannequin_left_leg_handles)
        print('Right leg handles: ')
        print(self.mannequin_right_leg_handles)

    def show_kuka_handles(self):
        
        '''
        For programming purposes
        '''

        # Shows the handles from the joints and links
        print('KUKA joint handles: ' + str(self.joints_handles))
        print('KUKA links handles: ' + str(self.links_handles))

    def vision_sensor_video(self):
        
        '''
        For programming purposes
        '''

        # Shows real time video image from the sensor

        self.move_joints(self.home_pos)
        # Set up the plot for live updating
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots()
        self.img_plot = self.ax.imshow(np.zeros((256, 256, 3), dtype=np.uint8))  # Initial dummy image
        plt.axis('off')  # Hide axes for better display

        # Create a video given the stream of images obtained
        try:
            
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                
                # Obtain image from vision sensor
                # image, resolution = self.sim.getVisionSensorImg(self.vision_sensor_handle)
                image, resolution = self.vs_data()

                # If image is not empty
                if image is not None and resolution is not None:
                    
                    # Obtain an array of integer values
                    image_int = self.sim.unpackUInt8Table(image)

                    # Convert the image data to a numpy array and reshape it
                    image_array = np.array(image_int, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))

                    # Convert from BGR to RGB
                    image_array = image_array[:, :, [2, 1, 0]]
                    
                    # Feature detection
                    # self.transformation_matrix = visual_servoing(image_array, reference_im)

                    # Update the plot with the new image
                    self.img_plot.set_data(image_array)
                    plt.draw()
                    plt.pause(0.01)  # Small pause to update the plot, adjust for smoothness

                    # pose_yolo(image_array)
                
                else:
                    print("No image data received. Retrying...")
                    time.sleep(0.1)  # Small delay before retrying

        except KeyboardInterrupt:

            print('Video stopped')

        finally:

            # Turn off interactive mode and show the last frame
            plt.ioff()
            plt.show() 

    def move_joints(self, thetas):
        
        '''
        Moves KUKA given set of joints thetas
        '''
        thetas_list = []
        if isinstance(thetas, torch.Tensor):
            # Converts tensor to list
            thetas_copy = thetas.clone().detach() 
            thetas_numpy = thetas_copy.numpy().flatten()
            thetas_list = np.ndarray.tolist(thetas_numpy)

        elif isinstance(thetas, np.ndarray):
            # Transform, if ndarray to python list, for Coppsm
            thetas_list = np.ndarray.tolist(thetas)
        else:
            thetas_list = thetas
        
        # Moves robot with joint values given
        for num in range(self.joints_count):
            self.sim.setJointTargetPosition(self.joints_handles[num], thetas_list[num])

    def move_mannequin(self, type_m = None):
        
        '''
        Simulates random mannequin parts movements. Could be only one body part, or
        all body parts.
        '''

        type_m = type_m if type_m is not None else 'one'
        
        steps = 30
        delay = 0.04

        if type_m == 'all':
            # Randomly move all parts
            for i in range(self.mann_ob_count):

                original_pos = self.sim.getObjectPosition(self.mannequin_object_targets[i])
                current_pos = original_pos.copy()
                
                if i == 1 or i == 3: # If any of the arms (Left arm position: 0.269984, 0.569823, 0.790344)
                    new_pos = current_pos.copy()
                    new_z = random.uniform(0.6, 0.9)
                    new_pos[2] = new_z
                else: # Legs
                    new_pos = np.random.uniform(0.3,0.7,3)

                for j in np.linspace(0, 1, steps):

                    # Move to new position
                    move_to = [current_pos[k] + j * (new_pos[k] - current_pos[k]) for k in range(3)]
                    self.sim.setObjectPosition(
                        self.mannequin_object_targets[i], 
                        move_to
                    )    

                    # Check if collision with table
                    collision, _ = self.sim.checkCollision(self.collection_man_handle, self.collection_table_Juan)
                    if collision > 0:
                        # stop current movement and return to original
                        for k in range(3):
                            new_pos[k] = move_to[k]
                        break

                    time.sleep(delay)

                time.sleep(0.5)

                for j in np.linspace(0, 1, steps):
                    # Return to original position
                    self.sim.setObjectPosition(
                        self.mannequin_object_targets[i],
                        [new_pos[k] + j * (original_pos[k] - new_pos[k]) for k in range(3)]
                    )
                    time.sleep(delay)

        elif type_m == 'one':
            
            # Randomly move only one part
            rand_object = np.random.randint(0,3)
            original_pos = self.sim.getObjectPosition(self.mannequin_object_targets[rand_object])
            current_pos = original_pos.copy()
                
            if rand_object == 1 or rand_object == 3:
                # Left arm position: 0.269984, 0.569823, 0.790344
                new_pos = current_pos.copy()
                new_z = random.uniform(0.6, 0.9)
                new_pos[2] = new_z
            else:
                new_pos = np.random.uniform(0.3,0.7,3)

            # print('new pose: ', new_pos)
            for j in np.linspace(0, 1, steps):

                # Move to new position
                move_to = [current_pos[k] + j * (new_pos[k] - current_pos[k]) for k in range(3)]
                # print('moving to: ', move_to)
                self.sim.setObjectPosition(
                    self.mannequin_object_targets[rand_object], 
                    move_to
                )

                # Check if collision with table
                collision, _ = self.sim.checkCollision(self.collection_man_handle, self.collection_table_Juan)
                if collision > 0:
                    # stop current movement and return to original
                    # print('Previous new pos: ', new_pos)
                    for k in range(3):
                        new_pos[k] = move_to[k]
                    # print('Collision, new pos: ', new_pos)
                    break
                time.sleep(delay)
                
            time.sleep(0.1)

            for j in np.linspace(0, 1, steps):

                # Return to original position
                # print('back to: ',[new_pos[k] + j * (original_pos[k] - new_pos[k]) for k in range(3)])
                self.sim.setObjectPosition(
                    self.mannequin_object_targets[rand_object],
                    [new_pos[k] + j * (original_pos[k] - new_pos[k]) for k in range(3)]
                )
                time.sleep(delay)
            
    def vs_data(self):
        
        '''
        Obtains information from the vision sensor
        '''
        
        image, resolution = self.sim.getVisionSensorImg(self.vision_sensor_handle)
        return image, resolution
    
    def torch_vs_data(self):

        '''
        Transforms image into a torch tensor
        Used for observation of the environment.
        '''
        
        image, resolution = self.vs_data() # Coppsm -> byte img [-128, 127]
        image_int = self.sim.unpackUInt8Table(image) # byte img -> uint8 [0, 255]
        image_array = np.array(image_int, dtype=np.float32).reshape((resolution[1], resolution[0], 3)) # reshape
        image_array = image_array[:, :, [2, 1, 0]] # Convert from BGR to RGB
        image = torch.from_numpy(image_array)
        image = image.permute(2, 0, 1).unsqueeze(0)
        
        return image
    
    def joints_positions(self):
        
        '''
        Returns joints current angular positions (thetas in radians), in a tensor.
        Used for observation of the environment.
        '''

        positions = []
        for i in range(self.joints_count):
            positions.append(self.sim.getJointPosition(self.joints_handles[i]))
        
        positions = torch.tensor(positions, dtype=float) 

        return positions
    
    def pos(self, object_handle):

        '''
        unused?
        Returns [x y z] from given handle, relative to the goal point
        '''

        position = self.sim.getObjectPosition(object_handle, self.goal_point) # Cambiar goal point o handle

        return torch.tensor(position, dtype=float)

    # def links_positions(self):
        
    #     '''
    #     Returns links current positions wrt the mannequin
    #     Used for reward function.
    #     '''

    #     positions = []
    #     for i in range(self.links_count):
    #         positions.append(self.sim.getObjectPosition(self.links_handles[i], self.Juan_handle))

    #     return positions # [[xyz], ..., [xyz]]
    
    def end_effector_pos(self):

        '''
        Returns end effector coordinates wrt world reference frame. 
        Used for info function.
        '''
        
        position = self.sim.getObjectPosition(self.end_effector_handle, self.sim.handle_world) 
        
        return position # [x,y,z]

    def endE_euclidean_goal(self):

        '''
        Returns Euclidean distance between given position and goal point.
        Used for reward function.
        '''
        position = self.sim.getObjectPosition(self.end_effector_handle, self.sim.handle_world) 
        
        return np.linalg.norm(np.array(position) - np.array(self.goal_point_pos))

    def links_distances(self):

        '''
        Returns Euclidean distance between links and mannequin.
        Used for reward function.
        '''

        # Links current positions wrt the mannequin
        positions = []
        for i in range(self.links_count):
            positions.append(self.sim.getObjectPosition(self.links_handles[i], self.Juan_handle))
        
        # Euclidean distances
        distances = []
        for i in range(7):
            distances.append(np.linalg.norm(np.array(positions[i]) -  np.array(self.Juan_handle)))

        return distances

    def collisions(self):
        
        '''
        Detects if robot collides with mannequin or the mannequins table
        Used for reward function.
        '''
        # Collision between KUKA and mannequin
        kuka_man, _ = self.sim.checkCollision(
            self.collection_kuka, self.collection_man_handle
        )

        kuka_table, _ = self.sim.checkCollision(
            self.collection_kuka, self.collection_table_Juan
        )

        collision = 1 if kuka_man > 0 or kuka_table > 0 else 0

        return collision
    
    def InvKin(self):

        '''
        Performs the inverse kinematics given the objective goal coordinates
        '''

        kuka_base = self.links_handles[0]
        # kuka_tip = self.end_effector_handle
        kuka_tip = self.sim.getObject('/tableKUKA/LBRiiwa7R800/kukaTip')
        target = self.goal_handle
        ikEnv = self.simIK.createEnvironment()
        # print(f"kuka_base: {kuka_base}, kuka_tip: {kuka_tip}, target: {target}")

        # Undamped method
        ikGroup_undamped = self.simIK.createGroup(ikEnv)
        self.simIK.setGroupCalculation(ikEnv, ikGroup_undamped, self.simIK.method_pseudo_inverse, 0, 6)
        self.simIK.addElementFromScene(ikEnv, ikGroup_undamped, kuka_base, kuka_tip, target, self.simIK.constraint_pose)

        # Damped Least Squares method
        ikGroup_damped = self.simIK.createGroup(ikEnv)
        self.simIK.setGroupCalculation(ikEnv, ikGroup_damped, self.simIK.method_damped_least_squares, 1, 99)
        self.simIK.addElementFromScene(ikEnv, ikGroup_damped, kuka_base, kuka_tip, target, self.simIK.constraint_pose)

        res, *_ = self.simIK.handleGroup(ikEnv, ikGroup_undamped, {'syncWorlds': True})
        if res != self.simIK.result_success:
            self.simIK.handleGroup(ikEnv, self.ikGroup_damped, {'syncWorlds': True})
            self.sim.addLog(self.sim.verbosity_scriptwarnings, "IK solver failed.")

        