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
   return np.sqrt(8.1 ** 2 + 8.2 ** 2 - 2 * 8.1 * 8.2 * np.cos(theta_joint2))

def tendonlength_cage_joint(theta_cage_joint):
   '''Input: joint angle of cage joint in rad
      Output: total normal lengths of tendon through cage joint
      0 angle defined with spring at rest'''
   
   # Min tendon connection angle in radians (When cage angle is 0)
   THETA_MIN = 0.794
   L0 = 29.378
   
   R_CAGE_ROT = 19.98
   # Distance from palm tendon routing to cage rotation axis
   DX_ROUTE_ROT_AX = 15
   DY_ROUTE_ROT_AX = 18.91

   DZ_ROUTE_CAGE_SQ = 0.009
   dl = np.sqrt(DZ_ROUTE_CAGE_SQ + (DX_ROUTE_ROT_AX + R_CAGE_ROT * np.cos(THETA_MIN + theta_cage_joint)) * 2 + (-DY_ROUTE_ROT_AX + R_CAGE_ROT * np.sin(THETA_MIN + theta_cage_joint)) * 2)
   return L0 - dl

# ------------------- Calculations of Tendon Lengths for all joints ------------------- #
# TODO: Add your own functions here to calculate the tendon lengths for all joints and for each finger (if needed)

def pose2tendon_finger(theta_Joint1, theta_Joint2):
   '''Input: controllable joint angles
      Output: array of tendon lengths for given joint angles'''
   return [tendonlength_flexor_joint1(theta_Joint1),
            tendonlength_extensor_joint1(theta_Joint1),
            tendonlength_flexor_joint2(theta_Joint2), 
            tendonlength_extensor_joint2(theta_Joint2)]

def pose2tendon_thumb(theta_Joint1, theta_Joint2, theta_Joint3, theta_Joint4):
   return [tendonlength_cage_joint(theta_Joint1),
           theta_Joint2,
           tendonlength_flexor_joint1(theta_Joint3),
           tendonlength_extensor_joint1(theta_Joint3),
           tendonlength_flexor_joint2(theta_Joint4), 
           tendonlength_extensor_joint2(theta_Joint4)]