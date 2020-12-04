#!/usr/bin/env python

import rospy
import rosbag
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIKRequest, GetPositionIKResponse, GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory
from actionlib import SimpleActionClient, GoalStatus
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal, PlayMotionResult
import matplotlib.pyplot as plt
import time
import tf

DEFAULT_IK_SERVICE = "/compute_ik"
DEFAULT_JOINT_STATES = "/joint_states"
PLAY_MOTION_AS = "/play_motion"


class recordedTrajectoryPlotter():
    def __init__(self):
        print "recordedTrajectoryPlotter initialized."
        
    def rosbagToPlot(self, rosbag_name, joint_names):
        """Given a rosbag with joint states topic make a plot of the trajectory"""
        fig, plots = plt.subplots(nrows=len(joint_names), ncols=1)
        # Set labels
        for plot, joint_name in zip(plots, joint_names):
            plot.set_xlabel('time')
            plot.set_ylabel('position')
            plot.set_title(joint_name)
            
        # Fill up traj with real trajectory points  
        # Read the bag and store the arrays for each plot
        trajectories = []
        # prepare somewhere to accumulate the data
        for joint in joint_names:
            trajectories.append([])
        bag = rosbag.Bag(rosbag_name)
        num_points = 0
        for topic, msg, t in bag.read_messages(topics=[DEFAULT_JOINT_STATES]):
            num_points += 1
            # Get the joint and it's values...
            js = msg # JointState()
            # Process interesting joints here
            names, positions = self.getNamesAndMsgList(joint_names, msg)
            # Append interesting joints here
            counter = 0
            for joint_val in  positions:
                trajectories[counter].append(joint_val)
                counter += 1
        bag.close()

        ticks = range(0, num_points)
        for plot, trajectory in zip(plots, trajectories):
            plot.plot(ticks, trajectory, 'b-')
        return plt 

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

class LearnFromJointState():
    """Manage the learning from joint positions"""
    def __init__(self):
        """Initialize class.
        @arg joint_names list of strings with the name of the
        joints to subscribe on joint_states."""
        rospy.loginfo("Init LearnFromJointState()")
        #TODO: make joint states topic a param to change in yaml file
        self.joint_states_topic = DEFAULT_JOINT_STATES
        # Creating a subscriber to joint states
        self.start_recording = False
        self.joint_states_subs = rospy.Subscriber(self.joint_states_topic, JointState, self.joint_states_cb)
        rospy.loginfo("Connected.")
        self.current_rosbag_name = "uninitialized_rosbag_name"
        self.last_joint_states_data = None
        self.joint_states_accumulator = []
        self.motion_name = "no_motion_name"
        self.joints_to_record = []
        rospy.loginfo("End init.")

        
    def joint_states_cb(self, data):
        """joint_states topic cb"""
        #rospy.loginfo("In callback")
        #rospy.loginfo("Received joint_states:\n " + str(data))
        if self.start_recording:
            self.joint_states_accumulator.append(data)

    def start_learn(self, joints=[], bag_name="no_bag_name_set"):
        """Start the learning writting in the accumulator of msgs"""
        self.current_rosbag_name = bag_name
        self.start_recording = True
        if len(joints) > 0:
            self.joints_to_record = joints
        else:
            rospy.logerr("No joints provided to record, aborting")
            return

    def stop_learn(self):
        """Stop the learning writting the bag into disk and returning the info of the motion"""
        self.start_recording = False
        self.joint_states_subs.unregister()
        rospy.loginfo("Recording in bag!")
        self.current_rosbag = rosbag.Bag(self.current_rosbag_name + '.bag', 'w')
        for js_msg in self.joint_states_accumulator:
            self.current_rosbag.write(DEFAULT_JOINT_STATES, js_msg, t=js_msg.header.stamp)
        self.current_rosbag.close()
        rospy.loginfo("Motion finished and closed bag.")
        motion_data = {'motion_name' : self.motion_name,
                       'joints' : self.joints_to_record,
                       'rosbag_name': self.current_rosbag_name + '.bag'}
        return motion_data


if __name__ == '__main__':
    rospy.init_node("test_training_classes")
    rospy.loginfo("Initializing dmp_training test.")

    joints = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'];
    js_learn = LearnFromJointState()

    raw_input('Press Enter to start\n')

    js_learn.start_learn(joints, 'test1');

    #r = rospy.Rate(1)
    #while not rospy.is_shutdown():
    #    print 'ok...'
    #    r.sleep()
        #rospy.loginfo(rospy.get_caller_id() + "I heard %s", joints)
    print 'ok...'
    raw_input('Press Enter to stop\n')
    rospy.loginfo("Out")
    motion_data = js_learn.stop_learn();
    print(motion_data);

    plotRecord = recordedTrajectoryPlotter();
    plt = plotRecord.rosbagToPlot('test1.bag', joints)
    plt.show();
    


