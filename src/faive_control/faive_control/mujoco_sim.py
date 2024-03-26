import mujoco
import mujoco.viewer
import threading
import time
import numpy as np


class GripperControllerMujocoSim():
    def __init__(self) -> None:
        self.hand_xml = "/home/mihnea/RWR_control/src/faive_viz/hand_hand.xml"
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.start()
        self.running = True

    def _run_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(self.hand_xml)
        self.data = mujoco.MjData(self.model)
        self.running = True

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()
            while viewer.is_running():
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(self.model, self.data)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


    def write_desired_joint_angles(self, angles):
        if self.running:
            self.data.ctrl= np.deg2rad(angles)
            return
    
    