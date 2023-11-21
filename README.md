# Faur hand (placeholder) control

## Prerequisites
Ubuntu 20.04 and ROS noetic

## Installation and build
1. Set up virtual environment in root directory
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Build ROS package from root directory
```
catkin_make
source devel/setup.bash
```
3. Set permissions for node files
```
cd src/faive_control/faive_control
chmod +x gripper_controller_node.py
chmod +x mano_faive_retargeter.py
chmod +x oakd_hand_tracker.py
```

## Running the pipeline
Simply run the provided launch file:
```
roslaunch launch/teleop_oakd_rwr.launch
```