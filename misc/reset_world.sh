#!/bin/bash

echo "Starting continuous reset of Gazebo world.."
echo "Press [Ctrl+C] to stop"

while true
do
    ros2 service call /reset_world std_srvs/srv/Empty "{}"
    sleep 0.5 # wait 0.5 seconds between resets
done