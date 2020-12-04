#!/usr/bin/python
"""
Created on 19/05/14

@author: Sammy Pfeiffer
@email: sammypfeiffer@gmail.com

This file contains training related classes.
"""

import rospy
import subprocess, yaml
import math
import numpy as np
import rosbag
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
from dmp.srv import GetDMPPlan, GetDMPPlanRequest,LearnDMPFromDemo, LearnDMPFromDemoRequest, SetActiveDMP, SetActiveDMPRequest, TrainDMPWeights
from dmp.msg import DMPTraj, DMPData, DMPPoint
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest, GetPositionFKResponse, GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from scipy.interpolate import interp1d
import time

DEFAULT_IK_SERVICE = "/compute_ik"
DEFAULT_JOINT_STATES = "/joint_states"
PLAY_MOTION_AS = "/play_motion"


class gestureGeneration():
    def __init__(self):
        rospy.loginfo("Initializing gestureGeneration")
        self.head = ['head_1_joint', 'head_2_joint']
        self.torso = ['torso_lift_joint']
        self.arm = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint',
                           'arm_4_joint', 'arm_5_joint', 'arm_6_joint',
                           'arm_7_joint']

        self.all_joints = self.torso + self.head + self.arm
        self.arm_torso = self.torso + self.arm
        self.gesture_difference = []
        # Use for the learning part
        self.weights = []
        self.length = 0
        self.firstPolicy = True
        rospy.loginfo("End initialization gestureGeneration")


    def getNamesAndMsgList(self, joints, joint_state_msg):
        """ Get the joints for the specified group and return this name list and a list of it's values in joint_states
        Note: the names and values are correlated in position """

        list_to_iterate = joints
        curr_j_s = joint_state_msg
        ids_list = []
        msg_list = []
        rospy.logdebug("Current message: " + str(curr_j_s))
        for joint in list_to_iterate:
            idx_in_message = curr_j_s.name.index(joint)
            ids_list.append(idx_in_message)
            msg_list.append(curr_j_s.position[idx_in_message])
        rospy.logdebug("Current position of joints in message: " + str(ids_list))
        rospy.logdebug("Current msg:" + str(msg_list))

        return list_to_iterate, msg_list

    def printNamesAndValues(self, group='arm_torso'):
        """Given a group, print in screen in a pretty way joint names and it's values"""
        names, values = self.getNamesAndMsgList(group=group)
        print "Name =  Joint Value"
        print "================="
        for nam, val in zip(names, values):
            print nam + " = " + str(val)

    def loadGestureFromBagJointStates(self, bagname, joints, isReset = True):
        """Load gesture from the bag name given """
        # get bag info
        self.info_bag = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bagname],
                                                    stdout=subprocess.PIPE).communicate()[0])
        bases_rel_to_time = math.ceil(self.info_bag['duration'] * 20) # empirically for every second 20 bases it's ok
        
        
        # Create a DMP from a X-number of joints trajectory
        dims = len(joints) # number of dims as number of joints
        dt = 0.02#1.0 0.02
        K = 100 #100
        D = 2.0 * np.sqrt(K) #2.0 * np.sqrt(K)
        num_bases = bases_rel_to_time
    
        # Fill up traj with real trajectory points  
        traj = []  
        bag = rosbag.Bag(bagname)
        first_point = True
        for topic, msg, t in bag.read_messages(topics=[DEFAULT_JOINT_STATES]):
            # Get the joint and it's values...
            js = msg # JointState()
            # Process interesting joints here
            names, positions = self.getNamesAndMsgList(joints, msg)
            # Append interesting joints here
            traj.append(positions)
            if first_point:
                # Store first point
                self.gesture_x0 = positions
                first_point = False
        bag.close()
        # Store last point
        self.gesture_goal = positions
        
        # Compute the difference between initial and final point
        for val1, val2 in zip(self.gesture_x0, self.gesture_goal):                     
            self.gesture_difference.append(val2-val1)
        
        print str(len(traj)) + " points in example traj. Using " + str(num_bases) + " num_bases"
        #self.length = len(traj)

        resp = self.makeLFDRequest(dims, traj, dt, K, D, num_bases, isReset)
        #Set it as the active DMP
        self.makeSetActiveRequest(resp.dmp_list)
        self.resp_from_makeLFDRequest = resp

        
        #rospy.loginfo("Response of makeLDFRequest is:\n" + str(self.resp_from_makeLFDRequest) )
        #************ PRINTS OF INFO HERE ****************

        #rospy.loginfo("Joints:" + str(joints))
        #rospy.loginfo("Initial pose:" + str(self.gesture_x0))
        #rospy.loginfo("Final pose: " + str(self.gesture_goal))
        time = self.info_bag['duration']
        rospy.loginfo("Time: " + str(time))
        #rospy.loginfo("DMP result: " + str(self.resp_from_makeLFDRequest))
        gesture_dict = self.saveGestureYAML(bagname + ".yaml", bagname, joints, self.gesture_x0, self.gesture_goal, self.resp_from_makeLFDRequest, time)
        return gesture_dict

    def loadGestureFromBagJointStatesAndRemoveJerkiness(self, bagname, joints, frequency_to_downsample=15):
        """Load gesture from the bag name given and remove jerkiness by downsampling and
        interpolating with a cubic spline """
        # get bag info
        self.info_bag = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bagname],
                                                    stdout=subprocess.PIPE).communicate()[0])
        bases_rel_to_time = math.ceil(self.info_bag['duration'] * 20) # empirically for every second 20 bases it's ok
        
        
        # Create a DMP from a X-number of joints trajectory
        dims = len(joints) # number of dims as number of joints
        dt = 0.02#1.0 0.02
        K = 100 # 100
        D = 2 * np.sqrt(K) # 2.0 * np.sqrt(K)
        num_bases = bases_rel_to_time
    
        # Fill up traj with real trajectory points  
        traj = []  
        # Also fill up downsampled traj
        downsampled_traj = []
        bag = rosbag.Bag(bagname)
        first_point = True
        num_msgs = 0
        num_downsampled_data_points = 0
        for topic, msg, t in bag.read_messages(topics=[DEFAULT_JOINT_STATES]):
            num_msgs += 1
            # Get the joint and it's values...
            js = msg # JointState()
            # Process interesting joints here
            names, positions = self.getNamesAndMsgList(joints, msg)
            if num_msgs % frequency_to_downsample == 0:
                num_downsampled_data_points += 1
                downsampled_traj.append(positions)
            # Append interesting joints here
            traj.append(positions)
            if first_point:
                # Store first point
                self.gesture_x0 = positions
                first_point = False
        bag.close()
        # Store last point
        self.gesture_goal = positions
        
        # Compute the difference between initial and final point
        for val1, val2 in zip(self.gesture_x0, self.gesture_goal):
            self.gesture_difference.append(val2-val1)
        
        print str(len(traj)) + " points in example traj. Using " + str(num_bases) + " num_bases"
        print "Downsampled traj has: " + str(len(downsampled_traj)) + " points"
        
        trajs_by_joint = self.getTrajectoriesByJoint(downsampled_traj) # Checked, does it's job
        splined_trajs = []
        for traj in trajs_by_joint:
            # Now we do a cubic spline between the points to filter jerkiness
            ticks = range(0, num_downsampled_data_points)
            cubic_spline_func = interp1d(ticks, traj, kind='cubic')
            ticks_as_array = np.array(ticks)
            # We will end with the same number of points than the initial trajectory
            # TODO: Check if we can just get rid of some of them
            new_ticks = np.linspace(ticks_as_array.min(), ticks_as_array.max(), num_msgs)
            filtered_trajectory = cubic_spline_func(new_ticks)
            splined_trajs.append(filtered_trajectory.tolist())
            # Here the trajectories have the original size
            
        dmp_friendly_filtered_trajs = self.getTrajectoriesForDMP(splined_trajs)
        resp = self.makeLFDRequest(dims, dmp_friendly_filtered_trajs, dt, K, D, num_bases)
        #Set it as the active DMP
        self.makeSetActiveRequest(resp.dmp_list)
        self.resp_from_makeLFDRequest = resp
        
        #rospy.loginfo("Response of makeLDFRequest is:\n" + str(self.resp_from_makeLFDRequest) )
        
        rospy.loginfo("Joints:" + str(joints))
        rospy.loginfo("Initial pose:" + str(self.gesture_x0))
        rospy.loginfo("Final pose: " + str(self.gesture_goal))
        time = self.info_bag['duration']
        rospy.loginfo("Time: " + str(time))
        #rospy.loginfo("DMP result: " + str(self.resp_from_makeLFDRequest))
        gesture_dict = self.saveGestureYAML(bagname + ".yaml", bagname, joints, self.gesture_x0, self.gesture_goal, self.resp_from_makeLFDRequest, time)
        return gesture_dict

    def loadGestureFromBagJointStatesAndDownsample(self, bagname, joints, frequency_to_downsample=15):
        """This does not give good results. Load gesture from the bag name given and remove jerkiness by downsampling and
        interpolating with a cubic spline """
        # get bag info
        self.info_bag = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bagname],
                                                    stdout=subprocess.PIPE).communicate()[0])
        bases_rel_to_time = math.ceil(self.info_bag['duration'] * 20) # empirically for every second 20 bases it's ok
        
        
        # Create a DMP from a X-number of joints trajectory
        dims = len(joints) # number of dims as number of joints
        dt = 0.02#1.0
        K = 100
        D = 2.0 * np.sqrt(K)
        num_bases = bases_rel_to_time
    
        # Fill up traj with real trajectory points  
        traj = []  
        # Also fill up downsampled traj
        downsampled_traj = []
        bag = rosbag.Bag(bagname)
        first_point = True
        num_msgs = 0
        num_downsampled_data_points = 0
        for topic, msg, t in bag.read_messages(topics=[DEFAULT_JOINT_STATES]):
            num_msgs += 1
            # Get the joint and it's values...
            js = msg # JointState()
            # Process interesting joints here
            names, positions = self.getNamesAndMsgList(joints, msg)
            if num_msgs % frequency_to_downsample == 0:
                num_downsampled_data_points += 1
                downsampled_traj.append(positions)
            # Append interesting joints here
            traj.append(positions)
            if first_point:
                # Store first point
                self.gesture_x0 = positions
                first_point = False
        bag.close()
        # Store last point
        self.gesture_goal = positions
        
        # Compute the difference between initial and final point
        for val1, val2 in zip(self.gesture_x0, self.gesture_goal):
            self.gesture_difference.append(val2-val1)
        

#         trajs_by_joint = self.getTrajectoriesByJoint(downsampled_traj) # Checked, does it's job
#         splined_trajs = []
#         for traj in trajs_by_joint:
#             # Now we do a cubic spline between the points to filter jerkiness
#             ticks = range(0, num_downsampled_data_points)
#             cubic_spline_func = interp1d(ticks, traj, kind='cubic')
#             ticks_as_array = np.array(ticks)
#             # We will end with the same number of points than the initial trajectory
#             # TODO: Check if we can just get rid of some of them
#             new_ticks = np.linspace(ticks_as_array.min(), ticks_as_array.max(), num_msgs)
#             filtered_trajectory = cubic_spline_func(new_ticks)
#             splined_trajs.append(filtered_trajectory.tolist())
#             # Here the trajectories have the original size
#             
#         dmp_friendly_filtered_trajs = self.getTrajectoriesForDMP(splined_trajs)
        num_bases_rel_to_traj_size = len(downsampled_traj) * 2
        num_bases = num_bases_rel_to_traj_size
        print str(len(traj)) + " points in example traj. Using " + str(num_bases) + " num_bases"
        print "Downsampled traj has: " + str(len(downsampled_traj)) + " points"
        resp = self.makeLFDRequest(dims, downsampled_traj, dt, K, D, num_bases)
        #Set it as the active DMP
        self.makeSetActiveRequest(resp.dmp_list)
        self.resp_from_makeLFDRequest = resp
        
        #rospy.loginfo("Response of makeLDFRequest is:\n" + str(self.resp_from_makeLFDRequest) )
        
        rospy.loginfo("Joints:" + str(joints))
        rospy.loginfo("Initial pose:" + str(self.gesture_x0))
        rospy.loginfo("Final pose: " + str(self.gesture_goal))
        time = self.info_bag['duration']
        rospy.loginfo("Time: " + str(time))
        #rospy.loginfo("DMP result: " + str(self.resp_from_makeLFDRequest))
        gesture_dict = self.saveGestureYAML(bagname + ".yaml", bagname, joints, self.gesture_x0, self.gesture_goal, self.resp_from_makeLFDRequest, time)
        return gesture_dict


    def getTrajectoriesByJoint(self, points_joint_state):
        """Given the trajectories in a list of lists where every element
        in each list is the value for one joint, return a list of lists where every list is
        the values for only one joint"""
#         print "points_joint_state has len: " + str(len(points_joint_state))
#         print "and each sublist has len: " + str(len(points_joint_state[0]))
        number_of_joints = len(points_joint_state[0])
        trajectories_by_joint = []
        # Prepare number of lists
        for joint_num in range(number_of_joints):
            trajectories_by_joint.append([])
#         print "Trajectories_by_joint preparing has len:" + str(len(trajectories_by_joint))
        num_point = 0
        for point in points_joint_state:
            num_point += 1
#             print "  Point #" + str(num_point)
            for joint_num in range(number_of_joints):
#                 print "    joint #" + str(joint_num)
#                 print "       Appending: " + str(point[joint_num])
                trajectories_by_joint[joint_num].append(point[joint_num])
#         for joint_traj in trajectories_by_joint:
#             print "One joint traj ends being: " + str(joint_traj)
        return trajectories_by_joint

    def getTrajectoriesForDMP(self, trajectories_by_joint):
        """Given a trajectories by joint give the expected input for DMP messaging"""
        print "trajectories_by_joint has len: " + str(len(trajectories_by_joint))
        print "and each sublist has len: " + str(len(trajectories_by_joint[0]))
        points_joint_state = []
#         print "From trajectories_by_joint: "
#         print trajectories_by_joint
#         print "\n\nWe went to array2d:"
        array2d = np.array(trajectories_by_joint)
#         print array2d
        for column in range(len(trajectories_by_joint[0])):
            points_joint_state.append( array2d[:,column].tolist())
        return points_joint_state

    def saveGestureYAML(self, yamlname, name, joints, initial_pose, final_pose, computed_dmp, time):
        """Given the info of the gesture computed with the DMP server save it into a yaml file.
        @yamlname string name of the yaml file
        @name string name of the gesture
        @joints list of strings joints that are included in this gesture
        @initial_pose list of double initial pose for the gesture as it was recorded in the training
        @final_pose list of double final pose of the gesture as it was recorded in the training
        @computed_dmp python/object/new:dmp.srv._LearnDMPFromDemo.LearnDMPFromDemoResponse with the response of the DMP server
        @time double how long the gesture took"""
        gesture_dict = {"name": name,
                        "joints" : joints,
                        "initial_pose" : initial_pose,
                        "final_pose" : final_pose,
                        "computed_dmp" : computed_dmp,
                        "duration" : time}
        #rospy.loginfo("gesture_dict:\n" + str(gesture_dict))
        stream = file(yamlname, "w")
        yaml.dump(gesture_dict, stream)
        self.resp_from_makeLFDRequest = gesture_dict["computed_dmp"]
        return gesture_dict

    def saveYAML(self, yamlname, name, joints, initial_pose, final_pose, computed_dmp, time):
        """Given the info of the gesture computed with the DMP server save it into a yaml file.
        @yamlname string name of the yaml file
        @name string name of the gesture
        @joints list of strings joints that are included in this gesture
        @initial_pose list of double initial pose for the gesture as it was recorded in the training
        @final_pose list of double final pose of the gesture as it was recorded in the training
        @computed_dmp python/object/new:dmp.srv._LearnDMPFromDemo.LearnDMPFromDemoResponse with the response of the DMP server
        @time double how long the gesture took"""
        gesture_dict = {"name": name,
                        "joints" : joints,
                        "initial_pose" : initial_pose,
                        "final_pose" : final_pose,
                        "computed_dmp" : computed_dmp,
                        "duration" : time}
        #rospy.loginfo("gesture_dict:\n" + str(gesture_dict))
        stream = file(yamlname, "w")
        yaml.dump(gesture_dict, stream)
        self.resp_from_makeLFDRequest = gesture_dict["computed_dmp"]
        return gesture_dict

    def loadGestureYAML(self, yamlname):
        """Given a yamlname which has a gesture saved load it and set it as active in the DMP server"""
        try:
            stream = file(yamlname, "r")
        except:
            print "couldnt find file: " + yamlname
            return None
        gesture_dict = yaml.load(stream)
        #Set it as the active DMP
        #print gesture_dict
        self.makeSetActiveRequest(gesture_dict['computed_dmp'].dmp_list)
        self.resp_from_makeLFDRequest = gesture_dict['computed_dmp']
        return gesture_dict

    def makeLFDRequest(self, dims, traj, dt, K_gain,
                       D_gain, num_bases, isReset, initialPosition = False):
        """Learn a DMP from demonstration data """
        demotraj = DMPTraj()
    
        for i in range(len(traj)):
            pt = DMPPoint()
            pt.positions = traj[i]
            demotraj.points.append(pt)
            demotraj.times.append(dt*i)
    
        k_gains = [K_gain]*dims
        d_gains = [D_gain]*dims
    
        print "Starting ..."
        init_time = time.time()
        if (isReset or self.firstPolicy):
            rospy.loginfo("-------Learn from first policy or is reset----------")
            rospy.wait_for_service('learn_dmp_from_demo')
            try:
                lfd = rospy.ServiceProxy('learn_dmp_from_demo', LearnDMPFromDemo)
                resp = lfd(demotraj, k_gains, d_gains, num_bases)
                if (initialPosition == False and isReset):
                    print("***********FIRST POLICY************")
                    self.weights = resp.weights
                    self.length = resp.length
                    self.firstPolicy = False
            except rospy.ServiceException, e:
                print "Service call failed: %s"%e
        else:
            rospy.loginfo("-------------Training weights-------------")
            rospy.wait_for_service('train_dmp_weights')
            try:
                #self.length = len(demotraj.points)
                #self.weights = self.weights #Compute these values with a function calling A2C taking into account length
                tw = rospy.ServiceProxy('train_dmp_weights', TrainDMPWeights)
                resp = tw(demotraj, self.weights, num_bases)
            except rospy.ServiceException, e:
                print "Service call failed: %s"%e
        fin_time = time.time()
        print "LfD done, took: " + str(fin_time - init_time)
    
        return resp

    
    def makeSetActiveRequest(self, dmp_list):
        """Set a DMP as active for planning """
        try:
            sad = rospy.ServiceProxy('set_active_dmp', SetActiveDMP)
            sad(dmp_list)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    
    
    def makePlanRequest(self, x_0, x_dot_0, t_0, goal, goal_thresh,
                        seg_length, tau, dt, integrate_iter):
        """Generate a plan from a DMP """
        print "Starting DMP planning..."
        init_time = time.time()
        rospy.wait_for_service('get_dmp_plan')
        try:
            gdp = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)
            resp = gdp(x_0, x_dot_0, t_0, goal, goal_thresh,
                       seg_length, tau, dt, integrate_iter)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        fin_time = time.time()
        print "DMP planning done, took: " + str(fin_time - init_time)
    
        return resp


    def getPlan(self, initial_pose, goal_pose, seg_length=-1, initial_velocities=[], t_0 = None, tau=None, dt=None, integrate_iter=None,goal_thresh=[]):
        """Generate a plan...
        @initial_pose list of double initial pose for the gesture
        @goal_pose list of double final pose of the gesture
        @initial_velocities TODO list of double : initial velocities
        @t_0 TODO double initial time
        @goal_thresh TODO list of double : threshold for every joint
        @seg_length TODO integer... with -1 it's plan until convergence of goal, see docs of DMP
        @tau TODO
        @dt TODO
        @integrate_iter TODO"""
        x_0 = initial_pose
        #x_0 = [0.137,-0.264,1.211,0.0395796940422, 0.0202532964694, 0.165785921829]
        x_dot_0 = [0.0] * len(initial_pose)
        t_0 = 0
        
        goal = goal_pose
        
        #goal = [0.259,-0.252,1.289, 0.0212535586323, -0.00664429330438, 0.117483470173]
        if len(goal_thresh) > 0:
            this_goal_thresh = goal_thresh
        else:
            this_goal_thresh = [0.01] * len(initial_pose)
        seg_length = seg_length          #Plan until convergence to goal is -1
        #tau = 2 * self.resp_from_makeLFDRequest.tau       #Desired plan should take twice as long as demo
        if tau != None:
            print "input tau != None"
            print "its: " + str(tau)
            this_tau = tau
        else:
            print "input tau == None"
            this_tau = self.resp_from_makeLFDRequest.tau -1 # HEY WE NEED TO PUT -1 SEC HERE, WHY?? BUG?
        rospy.logwarn("tau is: " + str(this_tau))
        if dt != None:
            this_dt = dt
        else:
            this_dt = 0.05
        if integrate_iter != None:
            this_integrate_iter = integrate_iter
        else:
            this_integrate_iter = 1 #5       #dt is rather large, so this is > 1
        plan_resp = self.makePlanRequest(x_0, x_dot_0, t_0, goal, this_goal_thresh,
                               seg_length, this_tau, this_dt, this_integrate_iter)
        return plan_resp

class dmpPlanTrajectoryPlotter():
    def __init__(self):
        print "dmpPlanTrajectoryPlotter initialized."
        
    def planToPlot(self, plan, joint_names):
        """Given a GetDMPPlanResponse make a plot of the trajectory"""
        fig, plots = plt.subplots(nrows=len(joint_names), ncols=1)
        fig_vel, plots_vel = plt.subplots(nrows=len(joint_names), ncols=1) # Velocity plot
        # Set labels
        for plot, plot_vel, joint_name in zip(plots, plots_vel, joint_names):
            plot.set_xlabel('time')
            plot.set_ylabel('position')
            plot.set_title(joint_name)
            plot_vel.set_xlabel('time')
            plot_vel.set_ylabel('velocity')
            plot_vel.set_title(joint_name)
        # Read the plan and store the arrays for each plot
        trajectories = []
        velocities = []
        # prepare somewhere to accumulate the data
        for joint in joint_names:
            trajectories.append([])
            velocities.append([])
        num_points = 0
        for point in plan.points:
            num_points += 1
            joint_vals = point.positions
            vel_vals = point.velocities
#             print joint_names
#             print joint_vals
            counter = 0
            for joint_val, vel_val in  zip(joint_vals, vel_vals):
                trajectories[counter].append(joint_val)
                velocities[counter].append(vel_val)
                counter += 1
        ticks = range(0, num_points)
        for plot, vel_plot, trajectory, velocity in zip(plots, plots_vel, trajectories, velocities):
            plot.plot(ticks, trajectory, 'b-')
            vel_plot.plot(ticks, velocity, 'b-')
        return plt 



if __name__ == '__main__':
    rospy.init_node("test_generation_classes")
    rospy.loginfo("Initializing dmp_generation test.")
    joint_names = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint',
                           'arm_4_joint', 'arm_5_joint', 'arm_6_joint',
                           'arm_7_joint']
    gg = gestureGeneration()
    gg.loadGestureFromBagJointStates("test.bag", joint_names)
    print(gg.resp_from_makeLFDRequest)


