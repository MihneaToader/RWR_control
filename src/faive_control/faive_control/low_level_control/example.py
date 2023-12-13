from gripper_controller import GripperController
import time
import tkinter
from PIL import ImageTk, Image


"""
Example script to control the finger joint angles
"""

homepos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
goalpos1 = [0, 0, 0, 0, 90, 90, 90, 90, 90, 90]
goalpos2 = [0, 0, 0, 0, 0, 90, 0, 90, 0, 90]
goalpos3 = [-30, 0, 0, 0, 0, 0, 0, 0, 0, 0]
goalpos4 = [-70, -90, 90, 90, 90, 90, 90, 90, 90, 90]


def main():
    global gc
    gc = GripperController(port="/dev/ttyUSB0",calibration=False)

    root = tkinter.Tk()
    root.title("Debug angle screen")
    root.geometry("600x600")

    joint0 = tkinter.Scale(root, from_=0, to=90, orient="horizontal")
    joint1 = tkinter.Scale(root, from_=-180, to=0, orient="horizontal")
    joint2 = tkinter.Scale(root, from_=-20, to=120, orient="horizontal")
    joint3 = tkinter.Scale(root, from_=-20, to=120, orient="horizontal")
    joint4 = tkinter.Scale(root, from_=-20, to=120, orient="horizontal")
    joint5 = tkinter.Scale(root, from_=-20, to=120, orient="horizontal")
    joint6 = tkinter.Scale(root, from_=-20, to=120, orient="horizontal")
    joint7 = tkinter.Scale(root, from_=-20, to=120, orient="horizontal")
    joint8 = tkinter.Scale(root, from_=-20, to=120, orient="horizontal")
    joint9 = tkinter.Scale(root, from_=-20, to=120, orient="horizontal")

    joint0.pack()
    joint1.pack()
    joint2.pack()
    joint3.pack()
    joint4.pack()
    joint5.pack()
    joint6.pack()
    joint7.pack()
    joint8.pack()
    joint9.pack()

    label = tkinter.Label(root, text="Currents")
    label.pack()

    while True:
        root.update()
        goalpos = [joint0.get(), joint1.get(), joint2.get(), joint3.get(), joint4.get(), joint5.get(), joint6.get(), joint7.get(), joint8.get(), joint9.get()]
        gc.write_desired_joint_angles(goalpos)
        _, _, curr_current = gc._dxc.read_pos_vel_cur()
        current_order = [0] * 10
        for i, val in enumerate(curr_current):
            current_order[gc.motor_ids[i]] = val
        label.config(text=curr_current)

    gc.write_desired_joint_angles(homepos)

    time.sleep(1)

    print("Moving")

    gc.write_desired_joint_angles(goalpos3)

    print("Stopped moving")

    time.sleep(5)

    # gc.write_desired_joint_angles(goalpos4)

    # # gc.wait_for_motion()

    # print("Stopped moving")

    # time.sleep(60)

    gc.write_desired_joint_angles(homepos)

    # gc.wait_for_motion()

    time.sleep(0.5)

    gc.terminate()


if __name__ == "__main__":
    main()