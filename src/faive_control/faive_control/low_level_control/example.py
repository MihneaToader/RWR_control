from gripper_controller import GripperController
import time


"""
Example script to control the finger joint angles
"""

homepos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
goalpos1 = [0, 0, 0, 0, 90, 90, 90, 90, 90, 90]
goalpos2 = [0, 0, 0, 0, 0, 90, 0, 90, 0, 90]
goalpos3 = [0, 0, 0, 0, 90, 0, 90, 0, 90, 0]
goalpos4 = [45, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def main():
    global gc
    gc = GripperController(port="/dev/ttyUSB0",calibration=True)

    # print("+++++++STARTING+++++")
    # for i in range(90):
    #     gc.write_desired_joint_angles([0, 0, 0, 0, i, 0, i, 0, i, 0])
    #     time.sleep(0.1)

    # time.sleep(1)

    gc.write_desired_joint_angles(homepos)

    # gc.wait_for_motion()

    time.sleep(1)

    gc.write_desired_joint_angles(goalpos4)

    # gc.wait_for_motion()

    time.sleep(3)

    # gc.write_desired_joint_angles(homepos)

    # # gc.wait_for_motion()

    # time.sleep(1)

    # gc.write_desired_joint_angles(goalpos3)

    # # gc.wait_for_motion()

    # time.sleep(1)

    # gc.write_desired_joint_angles(homepos)

    # # gc.wait_for_motion()

    # time.sleep(1)

    # gc.write_desired_joint_angles(goalpos1)

    # # gc.wait_for_motion()

    # time.sleep(1)

    # gc.write_desired_joint_angles(goalpos2)

    # gc.wait_for_motion()

    # time.sleep(0.5)

    # gc.write_desired_joint_angles(homepos)

    # gc.wait_for_motion()

    # time.sleep(0.5)

    # gc.write_desired_joint_angles(goalpos3)

    # gc.wait_for_motion()

    # time.sleep(0.5)

    gc.write_desired_joint_angles(homepos)

    # gc.wait_for_motion()

    time.sleep(0.5)

    gc.terminate()


if __name__ == "__main__":
    main()