#!/usr/bin/env python

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
from dmp_training import *
import ast

# Brings in the SimpleActionClient
from  actionlib import SimpleActionClient
# For text-to-speech
from pal_interaction_msgs.msg import TtsAction, TtsGoal
# For executing movements
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
# For looking at points
from control_msgs.msg import PointHeadAction, PointHeadGoal

from dmp_package.srv import *

class getJointStates:
    joint_position = None

    def callback(self,data):
        self.joint_position = []
        self.joint_position.append(data.position[11])
        for i in range(8):
            self.joint_position.append(data.position[i])
        self.joint_position = tuple(self.joint_position)
    
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
            
# Service

def moveArm(req):
    rospy.loginfo("Initializing dmp_training test.")

    joint_names = ['torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'];

    rospy.loginfo("Initializing dmp_generation test.")
    
    rospy.loginfo("Get goal joint states")
    #raw_input('Press Enter to save goal state')
    
    #state = getJointStates()
    goal_pose = req.joints[0:9]
    rospy.loginfo("I heard %s", goal_pose)
    
    state = getJointStates()
    initial_pose = state.joint_position
    rospy.loginfo("I heard initial %s", initial_pose)
    rospy.loginfo("I heard goal %s", goal_pose)
    
    name = req.name
    
    print("motion name = " + name)
    
    controllers = ['arm_controller', 'head_controller', 'torso_controller', 'gripper_controller']
    gg = gestureGeneration()
    gg.loadGestureFromBagJointStates(name + ".bag", joint_names)
    #initial_pose = initial.joint_position;
    #goal_pose = goal.joint_position;
    plan_resp = gg.getPlan(initial_pose, goal_pose)
    rospy.loginfo("Initializing dmp_execution test.")
    #print(list(plan_resp.plan.points))

    
    diction = []

    for i in range(0, len(plan_resp.plan.points)):
        diction.append({'positions': plan_resp.plan.points[i].positions, 'time_from_start':plan_resp.plan.times[i]})
    

    #d = {'play_motion':{'controllers': controllers,'motions':{name:{'joints': joint_names , 'points': diction}}}}
    d = {'play_motion':{'controllers': controllers,'motions':{name:{'joints': joint_names , 'points': diction}}}}

    # Create params
    rospy.set_param('/play_motion/motions/' + name + '/points', diction)
    rospy.set_param('/play_motion/motions/' + name + '/joints', joint_names)
    
    #raw_input('Enter to perform action')
    
    # Create yaml
    #with open('result.yaml', 'w') as yaml_file:
    #    yaml.dump(d, yaml_file, default_flow_style=False)

    #print d
    
    rospy.loginfo("Parameters created")
    
    obj = MotionPlayer()
    obj.play_motion(name, block=False)
    rospy.loginfo("Action done")

if __name__ == '__main__':
    rospy.init_node("all_test")
    move_arm_service = rospy.Service('moveArm', dmpService, moveArm)
    rospy.spin()
    
    
    
