cmake_minimum_required(VERSION 3.0.2)
project(emb_template_ros)

find_package(catkin REQUIRED COMPONENTS
    actionlib_msgs
    message_generation
    geometry_msgs
    rospy
    std_msgs
)

add_action_files(
    FILES
    getPose.action
)

generate_messages(
    DEPENDENCIES
    actionlib_msgs
    geometry_msgs
    std_msgs
)

catkin_package()