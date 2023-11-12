import numpy as np


# ------------------- Calculations of Tendon Lengths at single joint ------------------- #
# TODO: Add your own functions here to calculate the tendon lengths for each joint

def tendonlength_flexor_joint1(theta_joint1):
   '''Input: joint angle of joint1 in rad
      Output: total normal lengths of flexor tendon through joint1'''
   return 2*7.5+2*7.5*np.cos(np.pi/2 + theta_joint1/2)

def tendonlength_extensor_joint1(theta_joint1):
   '''Input: joint angle of joint1 in rad
      Output: total normal lengths of extensor tendon through joint1'''
   return 2*7.5-2*7.5*np.sqrt(2)*np.cos(np.pi/4 + theta_joint1/2)


def tendonlength_flexor_joint2(theta_joint2):
   '''Input: joint angle of joint2 in rad
      Output: total normal lengths of flexor tendon through joint2'''
   return 2*7.5+2*7.5*np.cos(np.pi/2 + theta_joint2/2)

def tendonlength_extensor_joint2(theta_joint2):
   '''Input: joint angle of joint2 in rad
      Output: total normal lengths of extensor tendon through joint2'''
   return 2*7.5-2*7.5*np.sqrt(2)*np.cos(np.pi/4 + theta_joint2/2)

# ------------------- Calculations of Tendon Lengths for all joints ------------------- #
# TODO: Add your own functions here to calculate the tendon lengths for all joints and for each finger (if needed)

def pose2tendon_finger1(theta_Joint1, theta_Joint2):
   '''Input: controllable joint angles
      Output: array of tendon lengths for given joint angles'''
   return [tendonlength_flexor_joint1(theta_Joint1),
            tendonlength_extensor_joint1(theta_Joint1),
            tendonlength_flexor_joint1(theta_Joint1) + tendonlength_flexor_joint2(theta_Joint2), 
            tendonlength_extensor_joint1(theta_Joint1) + tendonlength_extensor_joint2(theta_Joint2)]
