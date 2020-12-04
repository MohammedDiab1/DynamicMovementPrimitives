#!/usr/bin/env python

import rospy
from dmp_package.msg import joints
from dmp_package.srv import dmpTrajectoryMotion
import subprocess, yaml

DEFAULT_IK_SERVICE = "/compute_ik"
DEFAULT_JOINT_STATES = "/joint_states"
PLAY_MOTION_AS = "/play_motion"
STAMP_INCREASE = 0.03


def trajectoryMotion(req):
    #print(list(str(req.trajectory[0]).replace("joint: ","").replace("[","").replace("]","").split(', ')))
    rospy.loginfo("Initializing dmp_trajectory_learning")
    joint_names = ['torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'];
    controllers = ['arm_controller', 'head_controller', 'torso_controller', 'gripper_controller']

    print(len(req.trajectory))
    # js_learn = LearnFromTrajectory()
    diction = []
    time_stamp = 0.0
    for i in range(len(req.trajectory)):
        list_strings = list(str(req.trajectory[i]).replace("joint: ","").replace("[","").replace("]","").split(', '))
        current_joints = [float(i) for i in list_strings]

        diction.append({'positions': current_joints, 'time_from_start':time_stamp})
        time_stamp += STAMP_INCREASE

        print(current_joints)
    name = 'movement_new'
    #d = {'play_motion':{'controllers': controllers,'motions':{name:{'joints': joint_names , 'points': diction}}}}
    d = {'play_motion':{'controllers': controllers,'motions':{name:{'joints': joint_names , 'points': diction}}}}

    # Create params
    rospy.set_param('/play_motion/motions/movement/points', diction)
    
    # Create yaml
    with open('movement_new.yaml', 'w') as yaml_file:
        yaml.dump(d, yaml_file, default_flow_style=False)

    #print d
    
    rospy.loginfo("Parameters created")

    return []

if __name__ == '__main__':
    rospy.init_node("fom_trajectory_to_motion")
    move_arm_service = rospy.Service('trajectory_motion', dmpTrajectoryMotion, trajectoryMotion)
    print("Service on")
    rospy.spin()

