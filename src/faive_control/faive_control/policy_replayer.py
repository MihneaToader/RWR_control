import time
import numpy as np
from low_level_control.gripper_controller import GripperController
from mujoco_sim import GripperControllerMujocoSim


class PolicyReplayer:
    def __init__(self, sim=False, sub_queue_size=1) -> None:
        self.sim = sim
        if not sim:
            self.gripper_controller = GripperController("/dev/ttyUSB0", calibration=False)
        else:
            self.gripper_controller = GripperControllerMujocoSim()
        policy_npy_path = "/home/mihnea/RWR_control/src/policies/neg_z/output_-1z_7.npy"
        self.data = np.load(policy_npy_path)
        time.sleep(1)

    def start_policy(self):
        prev_time = time.time()
        for i in range(self.data.shape[0]):
            joints = np.rad2deg(self.data[i])
            if not self.sim:
                joints[0] += 70
                joints[2:] = joints[2:]*2
            self.gripper_controller.write_desired_joint_angles(joints)
            crt_time = time.time()
            while (crt_time - prev_time < 0.1):
                crt_time = time.time()
            prev_time = time.time()
        return


if __name__ == "__main__":
    replayer = PolicyReplayer(sim=False)
    time.sleep(1)
    while(True):
        replayer.start_policy()