#include <stdlib.h>
#include <ros/topic.h>

int main(int argc, char** argv) 
{
    // Init the ROS node
    ros::init(argc, argv, "dmp");


    ROS_INFO("Starting dmp node ...");
    ros::NodeHandle nh;

    while (nh.ok()){

    }

    return 0;

}