#!/usr/bin/env python

# ROS defines
import rospy
import subprocess, yaml
import actionlib
from  actionlib import SimpleActionClient
import rosbag
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion

# messages definition
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest, GetPositionFKResponse, GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from control_msgs.msg import PointHeadAction, PointHeadGoal

# gym defines
import gym
from gym import spaces
from gym.utils import seeding, EzPickle

# gym types
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

# other libraries
import numpy as np
import sys, math
from scipy.interpolate import interp1d
import time
import ast
import random

#dmp defines
from dmp.srv import GetDMPPlan, GetDMPPlanRequest,LearnDMPFromDemo, LearnDMPFromDemoRequest, SetActiveDMP, SetActiveDMPRequest
from dmp.msg import DMPTraj, DMPData, DMPPoint, DMPWeights
from functions.dmp_generation import gestureGeneration
from functions.dmp_generation import dmpPlanTrajectoryPlotter
from functions.dmp_execution import createExecuteKnownTrajectoryRequest
from functions.dmp_execution import gestureExecution

class getJointStates:
    joint_position = ""

    def callback(self,data):
        self.joint_position = (data.position[11],) + data.position[9:11] + data.position[0:7]
        #self.joint_position = tuple(data.position[12]) + data.position[10:12] + data.position[0:7]
    
    def __init__(self):
        rospy.Subscriber('joint_states', JointState, self.callback)
        rospy.sleep(1)

class LearningModel(gym.Env):
    """
    Una plantilla personalizada para crear entornos compatibles con OpenAI Gym
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.__version__ = "0.1"

        #self.joint_names = ['torso_lift_joint', 'head_1_joint', 'head_2_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'];
        self.joint_names = ['torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'];
        self.controllers = ['arm_controller', 'head_controller', 'torso_controller', 'gripper_controller']

        self.pub = rospy.Publisher('dmp_weights', DMPWeights, queue_size=10)
        self.learning_weights = []
        self.learning_old_weights = []
        self.learning_length = 0
        self.first_policy = True

        self.gg = gestureGeneration()

        
        # Modify observation space with minimum and maximum values we need taking into account the environment

        #self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        
        # Modify the action space according to the environment needs
        #self.action_space = spaces.Box(4)

    def step(self):
        """
        def step(self, action):
        Execute a determined action each step to guide the agent in the environment.
        The reset method will also execute the end of each episode
        : param action: The action that is going to be executed in the environment
        : return : (observation, reward, done, info)
            observation(object):
                Observation of the environment in the action execution instant
            reward(float):
                Environment reward according to the executed action
            done(bool):
                boolean flag to indicate if the epise is finished or not
            info(dict):
                A dictionary with additional information about the executed action
        """

        print("Before apply action")
        print(len(self.learning_weights))
        self.applyAction();

        if (self.first_policy):
        	self.learning_weights = self.gg.weights
        	self.learning_length = self.gg.length
        	self.first_policy = False
        self.learning_old_weights = self.learning_weights
        #self.learnWeights()
        self.gg.weights = self.learning_weights

        print("Inside step")
        print(len(self.learning_old_weights))
        print(len(self.learning_weights))

        rospy.loginfo("Publishing weights...")

        # Publishing weights
        weights = DMPWeights()
        weights.value = self.learning_weights
        weights.length = self.learning_length
        weights.joints = len(self.joint_names)
        self.pub.publish(weights)


        # Implement the following steps of step method here:
        #   - Compute the reward according to the action
        #   - Compute the next observation
        #   - Configure done to True if the episode has finished and to False otherwise
        #   - Optionally, define the values inside info dictionary
        # return(observation, reward, done, info)
        
    def reset(self):
        """
        Reset all the environment variables and return an initial observation
        : return : observation
            observation(object): 
                Initial observation after configuring a new episode
        """

        print("Before apply reset")
        print(len(self.learning_weights))
        self.applyReset()
        print("Inside reset")
        print(len(self.learning_weights))

        # Implement reset  method here
        # return observation
        
    def render(self, mode = 'human', close = False):
        """
        : param mode:
        : return :
        """
        return
    def applyAction(self):
        self.gg.loadGestureFromBagJointStates("movement.bag", self.joint_names, False)
        state = getJointStates()
        #initial_pose = state.joint_position
        initial_pose = [0.33298205586157925, -4.7589775664214073e-05, -0.021233773215461937, 0.00041857553360458155, 0.007812366709138985, -0.00022604317547436636, 0.018577739002537008, 0.00013233464147077711]
        goal_pose = [0, 0.16471575617593182, 0.8884589872474873, -1.6132110770646424, 1.4834595266009387, -1.2186207169225822, 1.3679727677743694, -1.3353656179158602]
        plan_resp = self.gg.getPlan(initial_pose, goal_pose)
        rospy.loginfo("Initializing dmp_execution test.")
        diction = []
        for i in range(0, len(plan_resp.plan.points)):
            diction.append({'positions': plan_resp.plan.points[i].positions, 'time_from_start':plan_resp.plan.times[i]})
        
        name = 'movement'

        d = {'play_motion':{'controllers': self.controllers,'motions':{name:{'joints': self.joint_names , 'points': diction}}}}

        # Create params
        rospy.set_param('/play_motion/motions/movement/points', diction)
        rospy.set_param('/play_motion/motions/movement/joints', self.joint_names)
        #rospy.loginfo("Parameters created")

        client = SimpleActionClient("play_motion", PlayMotionAction)

        client.wait_for_server()
        rospy.loginfo("...connected.")

        #rospy.wait_for_message("joint_states", JointState)
        #rospy.sleep(3.0)

if __name__ == "__main__":
  rospy.init_node("learning_model")
  rospy.loginfo("Initialization of learning_model...")
  model = LearningModel()
  states = getJointStates()
  #print("***************JOINTS*****************")
  #print("torso_lift_joint, head_1_joint, head_2_joint, arm_1_joint, arm_2_joint, arm_3_joint, arm_4_joint, arm_5_joint, arm_6_joint, arm_7_joint")
  #print(states.joint_position)
  cycle = 1
  while not rospy.is_shutdown():
	print("+++++++++++++++++ cycle = " + str(cycle) + " +++++++++++++++++")
	print("***************** STEP *******************")
	model.step()
	rospy.loginfo("Step launched")
	rospy.sleep(3.0)
	cycle += 1

  
