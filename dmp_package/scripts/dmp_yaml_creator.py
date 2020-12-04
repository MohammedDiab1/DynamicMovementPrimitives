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
    joint_names = ['torso_lift_joint', 'head_1_joint', 'head_2_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'];
    controllers = ['arm_controller', 'head_controller', 'torso_controller', 'gripper_controller']
    gg = gestureGeneration()
    gg.loadGestureFromBagJointStates("movement.bag", joint_names)
    initial_pose = [0, -0.0005489635547037963, -0.019133316703270786, 0.00012744042352608176, -0.010618968208951252, -0.00010177600870164838, 0.00400816240991464, 0.0001901484358279859, -0.000982966902768112, 3.3421034090430624e-05];
    goal_pose = [0.34408729391130444, -2.175129525916475e-05, -1.0205164314296917, 0.00012744042352608176, -0.010618968208951252, -0.00010177600870164838, 0.00400816240991464, 0.0001901484358279859, -0.000982966902768112, 3.3421034090430624e-05];
    plan_resp = gg.getPlan(initial_pose, goal_pose)
    rospy.loginfo("Initializing dmp_execution test.")
    #print(list(plan_resp.plan.points))

    diction = []

    for i in range(0, len(plan_resp.plan.points)):
        diction.append({'positions': plan_resp.plan.points[i].positions, 'time_from_start':plan_resp.plan.times[i]})
    

    name = 'movement'
    #d = {'play_motion':{'controllers': controllers,'motions':{name:{'joints': joint_names , 'points': diction}}}}
    d = {'play_motion':{'controllers': controllers,'motions':{name:{'joints': joint_names , 'points': diction}}}}

    # Create params
    rospy.set_param('/play_motion/motions/movement/points', diction)
    
    # Create yaml
    with open('movement.yaml', 'w') as yaml_file:
        yaml.dump(d, yaml_file, default_flow_style=False)

    #print d
    
    rospy.loginfo("Parameters created")
