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
    rospy.init_node("all_test")

    rospy.loginfo("Initializing dmp_training test.")

    joint_names = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'];
    """
    js_learn = LearnFromJointState()

    raw_input('Press Enter to start\n')

    js_learn.start_learn(joint_names, 'test');

    #r = rospy.Rate(1)
    #while not rospy.is_shutdown():
    #print 'Press enter to stop'
    raw_input('Press Enter to stop')
    #r.sleep()
        #rospy.loginfo(rospy.get_caller_id() + "I heard %s", joint_names)

    rospy.loginfo("Out")
    motion_data = js_learn.stop_learn();
    print(motion_data);

    plotRecord = recordedTrajectoryPlotter();
    plt = plotRecord.rosbagToPlot('test.bag', joint_names)
    plt.show();


    """
    rospy.loginfo("Initializing dmp_generation test.")
    
    rospy.loginfo("Get goal joint states")
    #raw_input('Press Enter to save goal state')
    
    #state = getJointStates()
    #goal_pose = state.joint_position
    goal_pose = [0.9255377625059442, 0.11754621638620011, 2.9038488088481187, 0.3778005583764132, -1.5510446684563268, 1.3709161964243322, -0.0049033035665370095]
    rospy.loginfo("I heard %s", goal_pose)
    
    rospy.loginfo("Get initial joint states. Move the robot until initial position")
    #raw_input('Press Enter to save initial state')
    
    #state = getJointStates()
    #initial_pose = state.joint_position
    initial_pose = [2.4402871834205158, 0.2886487030020402, 2.6416266548037246, 0.4475822700148798, -1.5510355375372085, 1.3709161964243322, -0.0049033035665370095]
    rospy.loginfo("I heard initial %s", initial_pose)
    rospy.loginfo("I heard goal %s", goal_pose)


    controllers = ['arm_controller', 'head_controller', 'torso_controller', 'gripper_controller']
    gg = gestureGeneration()
    gg.loadGestureFromBagJointStates("test.bag", joint_names)
    #initial_pose = initial.joint_position;
    #goal_pose = goal.joint_position;
    plan_resp = gg.getPlan(initial_pose, goal_pose)
    rospy.loginfo("Initializing dmp_execution test.")
    #print(list(plan_resp.plan.points))

    
    diction = []

    for i in range(0, len(plan_resp.plan.points)):
        diction.append({'positions': plan_resp.plan.points[i].positions, 'time_from_start':plan_resp.plan.times[i]})
    

    name = 'gesture_test'
    #d = {'play_motion':{'controllers': controllers,'motions':{name:{'joints': joint_names , 'points': diction}}}}
    d = {'play_motion':{'controllers': controllers,'motions':{name:{'joints': joint_names , 'points': diction}}}}

    # Create params
    rospy.set_param('/play_motion/motions/gesture_test/points', diction)
    rospy.set_param('/play_motion/motions/gesture_test/joints', joint_names)
    
    raw_input('Enter to perform action')
    
    # Create yaml
    #with open('result.yaml', 'w') as yaml_file:
    #    yaml.dump(d, yaml_file, default_flow_style=False)

    #print d
    
    rospy.loginfo("Parameters created")
    
    obj = MotionPlayer()
    obj.play_motion('gesture_test', block=False)
    rospy.loginfo("Action done")
    
    
