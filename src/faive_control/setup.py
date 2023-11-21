#!/usr/bin/env python
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['faive_control',
              'faive_control.depthai_hand_tracker',
              'faive_control.low_level_control',
              'faive_control.oak_driver',
              'faive_control.utils'],     
)

setup(**d)