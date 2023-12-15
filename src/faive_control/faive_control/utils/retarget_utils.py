
from typing import Dict

import torch
import numpy as np


JOINT_PARAMETER_NAMES = [
    'palm2cage',
    'cage2base',
    'thumb_base2basevirt',
    'thumb_basevirt2pf',
    'thumb_pf2pfvirt',
    'thumb_pfvirt2mf',
    'thumb_mf2df',
    'index_base2basevirt',
    'index_basevirt2pf',
    'index_pf2pfvirt',
    'index_pfvirt2mf',
    'index_mf2df',
    'middle_base2basevirt',
    'middle_basevirt2pf',
    'middle_pf2pfvirt',
    'middle_pfvirt2mf',
    'middle_mf2df',
    'ring_base2basevirt',
    'ring_basevirt2pf',
    'ring_pf2pfvirt',
    'ring_pfvirt2mf',
    'ring_mf2df',
]

GC_TENDONS = {
    'palm2cage': {},
    'cage2base': {},
    'thumb_base2basevirt': {'thumb_basevirt2pf': 1},
    'thumb_pf2pfvirt': {'thumb_pfvirt2mf': 1, 'thumb_mf2df': 2},
    'index_base2basevirt': {'index_basevirt2pf': 1},
    'index_pf2pfvirt': {'index_pfvirt2mf': 1, 'index_mf2df': 2},
    'middle_base2basevirt': {'middle_basevirt2pf': 1},
    'middle_pf2pfvirt': {'middle_pfvirt2mf': 1, 'middle_mf2df': 2},
    'ring_base2basevirt': {'ring_basevirt2pf': 1},
    'ring_pf2pfvirt': {'ring_pfvirt2mf': 1, 'ring_mf2df': 2},
}

FINGER_CHAINS = {
    'thumb': [
        'world',
        'world2palm_fixed',
        'palm',
        'palm2cage',
        'cage2base',
        'thumb_base2basevirt',
        'thumb_basevirt2pf',
        'thumb_pf2pfvirt',
        'thumb_pfvirt2mf',
        'thumb_mf2df',
    ],
    'index': [
        'world',
        'world2palm_fixed',
        'palm',
        'palm2index_base',
        'index_base2basevirt',
        'index_basevirt2pf',
        'index_pf2pfvirt',
        'index_pfvirt2mf',
        'index_mf2df',
    ],
    'middle': [
        'world',
        'world2palm_fixed',
        'palm',
        'palm2middle_base'
        'middle_base2basevirt',
        'middle_basevirt2pf',
        'middle_pf2pfvirt',
        'middle_pfvirt2mf',
        'middle_mf2df',
    ],
    'ring': [
        'world',
        'world2palm_fixed',
        'palm',
        'palm2ring_base'
        'ring_base2basevirt',
        'ring_basevirt2pf',
        'ring_pf2pfvirt',
        'ring_pfvirt2mf',
        'ring_mf2df',
        ],
}

FINGER_TO_TIP: Dict[str, str] = {
    "thumb": "thumb_fingertip",
    "index": "index_fingertip",
    "middle": "middle_fingertip",
    "ring": "ring_fingertip",
}

FINGER_TO_BASE = {
    "thumb": "thumb_base",
    "index": "index_pp",
    "middle": "middle_pp",
    "ring": "ring_pp",
}


def get_mano_joints_dict(joints: torch.Tensor, include_wrist=False, batch_processing=False):
    # joints: 17 x 3
    # For retargeting, we don't need the wrist
    # For visualization, we need the wrist
    if not batch_processing:
        if not include_wrist:
            return {
                "thumb": joints[1:5, :],
                "index": joints[5:9, :],
                "middle": joints[9:13, :],
                "ring": joints[13:17, :],
            }
        else:
            return {
                "wrist": joints[0, :],
                "thumb": joints[1:5, :],
                "index": joints[5:9, :],
                "middle": joints[9:13, :],
                "ring": joints[13:17, :],
            }
    else:
        if not include_wrist:
            return {
                "thumb": joints[:, 1:5, :],
                "index": joints[:, 5:9, :],
                "middle": joints[:, 9:13, :],
                "ring": joints[:, 13:17, :],
            }
        else:
            return {
                "wrist": joints[:, 0, :],
                "thumb": joints[:, 1:5, :],
                "index": joints[:, 5:9, :],
                "middle": joints[:, 9:13, :],
                "ring": joints[:, 13:17, :],
            }


def get_mano_fingertips_batch(mano_joints_dict):
    return {
        "thumb": mano_joints_dict["thumb"][:, [3], :],
        "index": mano_joints_dict["index"][:, [3], :],
        "middle": mano_joints_dict["middle"][:, [3], :],
        "ring": mano_joints_dict["ring"][:, [3], :],
    }

def get_mano_pps_batch(mano_joints_dict):
    return {
        "thumb": mano_joints_dict["thumb"][:, [0], :],
        "index": mano_joints_dict["index"][:, [0], :],
        "middle": mano_joints_dict["middle"][:, [0], :],
        "ring": mano_joints_dict["ring"][:, [0], :],
    }

def get_keyvectors(fingertips: Dict[str, torch.Tensor], palm: torch.Tensor):
    return {
        'thumb2index': fingertips['index'] - fingertips['thumb'],
        'thumb2middle': fingertips['middle'] - fingertips['thumb'],
        'thumb2ring': fingertips['ring'] - fingertips['thumb'],
        'index2middle': fingertips['middle'] - fingertips['index'],
        'index2ring': fingertips['ring'] - fingertips['index'],
        'middle2ring': fingertips['ring'] - fingertips['middle'],
    }


def rotation_matrix_z(angle):
    """
    Returns a 3x3 rotation matrix about the z-axis for the given angle.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rot_mat = np.array([[cos_theta, -sin_theta, 0],
                        [sin_theta, cos_theta, 0],
                        [0, 0, 1]])
    return rot_mat


def rotation_matrix_y(angle):
    """
    Returns a 3x3 rotation matrix about the y-axis for the given angle.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rot_mat = np.array([[cos_theta, 0, sin_theta],
                        [0, 1, 0],
                        [-sin_theta, 0, cos_theta]])
    return rot_mat


def rotation_matrix_x(angle):
    """chain
    Returns a 3x3 rotation matrix about the x-axis for the given angle.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rot_mat = np.array([[1, 0, 0],
                        [0, cos_theta, -sin_theta],
                        [0, sin_theta, cos_theta]])
    return rot_mat

def get_hand_rotation_matrix(joint_pos):
    '''
    param: joint_pos, a numpy array of 3D joint positions (MANO format)

    Returns the rotation matrix that normalizes the joint orientation. 
    '''

    # normalize the translation of the hand: set the wrist point to zero
    wrist_point = joint_pos[0, :]
    joint_pos -= wrist_point

    # construct a plane from wrist, first index finger joint, first ring joint
    joint_dict = get_mano_joints_dict(joint_pos, include_wrist=True)
    wrist_point = joint_dict['wrist']
    index_point = joint_dict['index'][0]
    ring_point = joint_dict['ring'][0]

    # find basis vectors for the plane
    base_1 = index_point - wrist_point
    base_2 = ring_point - wrist_point
    normal_vec = np.cross(base_1, base_2)
    base_2 = np.cross(normal_vec, base_1)

    # normalize basis vectors
    normal_vec /= np.linalg.norm(normal_vec)
    base_1 /= np.linalg.norm(base_1)
    base_2 /= np.linalg.norm(base_2)

    # construct the matrix for the base change from the hand frame basis vectors
    base_matrix = np.zeros((3, 3))
    base_matrix[:, 0] = base_1
    base_matrix[:, 1] = base_2
    base_matrix[:, 2] = normal_vec

    return base_matrix @ rotation_matrix_z(-np.pi/2)

def normalize_points(joint_pos, flip_x_axis=True, flip_y_axis=False, add_z_rotation=0):
    '''
    param: joint_pos, a numpy array of 3D joint positions (MANO format)

    Returns the joint positions with normalized translation and rotation. 
    '''

    # normalize the translation of the hand: set the wrist point to zero
    wrist_point = joint_pos[0, :]
    joint_pos -= wrist_point

    # construct a plane from wrist, first index finger joint, first ring joint
    joint_dict = get_mano_joints_dict(joint_pos, include_wrist=True)
    wrist_point = joint_dict['wrist']
    index_point = joint_dict['index'][0]
    ring_point = joint_dict['ring'][0]

    # find basis vectors for the plane
    base_1 = index_point - wrist_point
    base_2 = ring_point - wrist_point
    normal_vec = np.cross(base_1, base_2)
    base_2 = np.cross(normal_vec, base_1)

    # normalize basis vectors
    normal_vec /= np.linalg.norm(normal_vec)
    base_1 /= np.linalg.norm(base_1)
    base_2 /= np.linalg.norm(base_2)

    # construct the matrix for the base change from the hand frame basis vectors
    base_matrix = np.zeros((3, 3))
    base_matrix[:, 0] = base_1
    base_matrix[:, 1] = base_2
    base_matrix[:, 2] = normal_vec

    # need to rotate around z axis, order of basis vectors in hand frame might be switched up
    joint_pos = joint_pos @ base_matrix @ rotation_matrix_z(-np.pi/2 + add_z_rotation)

    if flip_y_axis:
        joint_pos = joint_pos @ rotation_matrix_y(np.pi)

    if flip_x_axis:
        # flip the x axis
        joint_pos[:, 0] *= -1

    return joint_pos


def get_unoccluded_hand_joint_idx(joint_pos):
    '''
    param: joint_pos, a numpy array of 3D joint positions (MANO format), not normalized
    Returns the joint that has the least z value and should be visible in the image (y value is in the direction of the camera).
    We can then project this joint into 3D space, and then from there get the 3D position of the wrist (which may be occluded)
    '''

    # get the joint with the lowest z value (closest to camera)
    max_joint_idx = np.argmin(joint_pos[:, 2])
    return max_joint_idx


def get_wrist_translation(joint_idx, joint_pos):
    '''
    param: joint_idx, the index of the joint with the highest y value
    param: joint_pos, a numpy array of 3D joint positions (MANO format), not normalized
    Returns the translation of the wrist in the hand frame relative to the joint_idx joint
    '''

    # get the 3D position of the wrist
    joint = joint_pos[joint_idx, :]
    wrist = joint_pos[0, :]

    return wrist - joint


def rolling_average_filter(positions, new_pos):
    '''
    A rolling average filter for the wrist position.
    param: positions, a numpy array of 3D positions of the wrist
    param: new_pos, a numpy array of the new 3D position of the wrist
    '''

    positions = np.roll(positions, -1, axis=0)
    positions[-1, :] = new_pos

    return positions, np.nanmean(positions, axis=0)


# Actually not used in frankmocap default
def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
        
    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
        
    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), 2)  # batch*3*3
    return matrix

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = np.sqrt((v**2).sum(1))  # batch
    v_mag = np.maximum(v_mag, np.array([1e-8]))
    v_mag = np.broadcast_to(v_mag.reshape(batch, 1), (batch, v.shape[1]))
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = np.concatenate((i.reshape(batch, 1), j.reshape(batch, 1), k.reshape(batch, 1)), 1)
        
    return out


def normalize_points_rokoko(joint_pos, mirror_x=False, flip_y_axis=False, flip_x_axis=False):
    # normalize the translation of the hand: set the wrist point to zero
    wrist_point = joint_pos[0, :]
    joint_pos -= wrist_point

    # construct a plane from wrist, first index finger joint, first ring joint
    joint_dict = get_mano_joints_dict(joint_pos, include_wrist=True)
    wrist_point = joint_dict['wrist']
    middle_point = joint_dict['middle'][0]
    ring_point = joint_dict['ring'][0]
    # find basis vectors for the plane
    base_1 = middle_point - wrist_point
    base_2 = ring_point - wrist_point
    normal_vec = np.cross(base_1, base_2)
    base_2 = np.cross(normal_vec, base_1)

    # normalize basis vectors
    normal_vec /= np.linalg.norm(normal_vec)
    base_1 /= np.linalg.norm(base_1)
    base_2 /= np.linalg.norm(base_2)

    # construct the matrix for the base change from the hand frame basis vectors
    base_matrix = np.zeros((3, 3))
    base_matrix[:, 0] = base_1
    base_matrix[:, 1] = base_2
    base_matrix[:, 2] = normal_vec

    # need to rotate around z axis, order of basis vectors in hand frame might be switched up
    joint_pos = joint_pos @ base_matrix @ rotation_matrix_z(-np.pi/2)

    if flip_y_axis:
        joint_pos = joint_pos @ rotation_matrix_y(np.pi)

    if flip_x_axis:
        # flip the x axis
        joint_pos = joint_pos @ rotation_matrix_x(-np.pi/2)

    rot_matrix = base_matrix

    z_axis = wrist_point - middle_point
    
    rot_matrix[:,0] = -base_2
    rot_matrix[:,1] = -normal_vec
    rot_matrix[:,2] = base_1
    return joint_pos, rot_matrix