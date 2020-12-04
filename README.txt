# DynamicMovementPrimitives
Learning From Demonstrations

Author: Mohammed Diab


Imitation:

To train a movement:
Define the type of movement in movements.yaml
File:

hackaton_arm/config/movements.yaml

2. Save the defined trajectory in a rosparam

roslaunch hackaton_arm playmotion_movements.launch

*if you want to check it, you can run: rosparam list /play_motion/motions/
3. In order to execute the trajectory run:

rosrun play_motion run_motion openDraw     (or serving_finished)

4. Train the movement  running:

rosrun dmp_package dmp_training.py

Executing the movement with different position

Launch different service nodes

roslaunch dmp dmp.launch
rosrun dmp_package dmp_service.py (change the rute of .bag file (line 102))
Name of the service: ‘moveArm’

Example of how to use it in:
    
tiago_task_manager/src/tests/tiago_test.cpp

