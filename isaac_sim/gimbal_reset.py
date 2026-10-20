"""Passive joint reset of the gimbal track model (track_model:=gimbal) for the Isaac Sim scripts.

Each wheel is one sphere on a gimbal: track body -> passive joint about the track direction (*_roller_gimbal_joint)
-> driven joint about the wheel axis (the sphere). Rolling sideways turns the passive joint by phi, which tilts the
driven axis: the sphere would then drive the track at cos(phi) of its speed. For a sphere the angle phi has no physical
effect (same shape, same contact point), so it is set back to 0 whenever |phi| exceeds THRESHOLD; the joint velocities
are kept, so the sphere keeps its rolling speeds (only the spin about the vertical, which a point contact does not feel,
changes by a negligible amount).

Runs on every physics step (pre-step callback) once constructed; does nothing on models without gimbal joints.
"""
import numpy as np

THRESHOLD = 0.05   # [rad]: drive axis tilted at most 3 deg (cos > 0.998)


class GimbalReset:
    def __init__(self, robot, threshold=THRESHOLD):
        from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

        self.robot, self.threshold = robot, threshold
        self.ids = np.array([i for i, n in enumerate(robot.dof_names) if n.endswith('_roller_gimbal_joint')])
        self.resets = 0
        if self.ids.size:
            self._cb = SimulationManager.register_callback(self._step, event=SimulationEvent.PHYSICS_PRE_STEP)

    def _step(self, dt, context=None):
        q = self.robot.get_dof_positions().numpy()[0]
        sel = self.ids[np.abs(q[self.ids]) > self.threshold]
        if sel.size:
            self.robot.set_dof_positions(np.zeros((1, sel.size), np.float32), dof_indices=sel)
            self.resets += sel.size
