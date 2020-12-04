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
from dmp.srv import GetDMPPlan, GetDMPPlanRequest,LearnDMPFromDemo, LearnDMPFromDemoRequest, SetActiveDMP, SetActiveDMPRequest
from dmp.msg import DMPTraj, DMPData, DMPPoint
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest, GetPositionFKResponse, GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from scipy.interpolate import interp1d
import time
from functions.dmp_generation import gestureGeneration
from functions.dmp_generation import dmpPlanTrajectoryPlotter
from functions.dmp_execution import createExecuteKnownTrajectoryRequest
from functions.dmp_execution import gestureExecution
import ast

# Brings in the SimpleActionClient
from  actionlib import SimpleActionClient
# For text-to-speech
from pal_interaction_msgs.msg import TtsAction, TtsGoal
# For executing movements
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
# For looking at points
from control_msgs.msg import PointHeadAction, PointHeadGoal

class getJointStates:
    joint_position = ""

    def callback(self,data):
        self.joint_position = data.position[0:7]
    
    def __init__(self):
        rospy.Subscriber('joint_states', JointState, self.callback)
        rospy.sleep(1)
        
# To send goals to play_motion action server
class MotionPlayer(object):
    def __init__(self):
        rospy.loginfo("Initializing MotionPlayer...")
        self.ac = SimpleActionClient('/play_motion', PlayMotionAction)
        rospy.loginfo("Connecting to /play_motion...")
        # If more than 5s pass by, we crash
        if not self.ac.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Could not connect to /tts action server!")
            exit(0)
        rospy.loginfo("Connected!")

    def play_motion(self, motion_name, block=True):
        # To check motions:
        # rosparam list | grep play_motion | grep joints
        # for example:
        # open_hand, close_hand, offer_hand, wave
        # shake_hands, thumb_up_hand, pointing_hand, pinch_hand
        # head_tour, do_weights, pick_from_floor
        g = PlayMotionGoal()
        g.motion_name = motion_name

        if block:
            self.ac.send_goal_and_wait(g)
        else:
            self.ac.send_goal(g)


if __name__ == '__main__':
    rospy.init_node("test_generation_classes")
    rospy.loginfo("Initializing dmp_generation test.")
    joint_names = ['torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'];
    controllers = ['arm_controller', 'head_controller', 'torso_controller', 'gripper_controller']
    gg = gestureGeneration()
    gg.loadGestureFromBagJointStates("movement_new.bag", joint_names)
    initial_pose = [0.33298205586157925, 0.01849843995247369, -0.02382713489780741, 0.006523303899903077, 0.0018120146754379007, 0.007003304627368756, 0.0019429459867295051, 0.007481019894183483];
    goal_pose = [0, 0.16471575617593182, 0.8884589872474873, -1.6132110770646424, 1.4834595266009387, -1.2186207169225822, 1.3679727677743694, -1.3353656179158602];
    plan_resp = gg.getPlan(initial_pose, goal_pose)
    rospy.loginfo("Initializing dmp_execution test.")


    diction = []

    for i in range(0, len(plan_resp.plan.points)):
        diction.append({'positions': plan_resp.plan.points[i].positions, 'time_from_start':plan_resp.plan.times[i]})
    

    name = 'movement'
    #d = {'play_motion':{'controllers': controllers,'motions':{name:{'joints': joint_names , 'points': diction}}}}
    d = {'play_motion':{'controllers': controllers,'motions':{name:{'joints': joint_names , 'points': diction}}}}

    # Create params
    rospy.set_param('/play_motion/motions/movement/points', diction)
    rospy.set_param('/play_motion/motions/movement/joints', joint_names)
    
    #raw_input('Enter to perform action')
    
    # Create yaml
    #with open('result.yaml', 'w') as yaml_file:
    #    yaml.dump(d, yaml_file, default_flow_style=False)

    #print d
    
    rospy.loginfo("Parameters created")
    
    """
    obj = MotionPlayer()
    obj.play_motion('initial_position', block=False)
    #obj.play_motion('wave', block=False)
    """

    client = SimpleActionClient("play_motion", PlayMotionAction)
    client.wait_for_server()
    rospy.loginfo("...connected.")

    rospy.wait_for_message("joint_states", JointState)
    rospy.sleep(3.0)

    rospy.loginfo("Initial position...")
    goal = PlayMotionGoal()
    goal.motion_name = 'movement'
    goal.skip_planning = True

    client.send_goal(goal)
    client.wait_for_result(rospy.Duration(10.0))
    rospy.loginfo("Initial position reached.")
    rospy.loginfo("Action done")
