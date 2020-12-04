#!/usr/bin/python
import rosbag
import rospy
from sensor_msgs.msg import JointState

bag = rosbag.Bag('test.bag', 'w')

def callback(joint_states):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", joint_states.position)


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("joint_states", JointState, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()

    

    try:
        str = String()
        str.data = 'foo'

        i = Int32()
        i.data = 42

        bag.write('chatter', str)
        bag.write('numbers', i)
    finally:
        bag.close()