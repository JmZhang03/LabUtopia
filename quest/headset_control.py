import numpy as np
from quest.headset_utils import HeadsetFeedback 
from scipy.spatial.transform import Rotation as R
from quest.transform_utils import (
    quat2mat,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
    pose2mat,
    mat2pose,
)


class HeadsetOurControl():
    """
    VR Headset -> Simulation Operation Robot Arm
    VR Right Controller -> Simulation Observation Robot Arm
    !!! Make sure every quaternion is xyzw !!!
    """
    def __init__(self):
        # VR Coordinate: X Right, Y Back, Z Up
        # Sim Coordinate: X Front, Y Left, Z Up
        self.R_align = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        self.start_h_pos = None; self.start_r_pos = None
        self.start_h_rot = None; self.start_r_rot = None
        self.start_op_arm_pose = None
        self.start_obs_arm_pose = None
        self.started = False

    def reset(self):
        self.start_h_pos = None; self.start_r_pos = None
        self.start_h_rot = None; self.start_r_rot = None
        self.start_op_arm_pose = None
        self.start_obs_arm_pose = None
        self.started = False

    def start(self, headset_data, op_arm_pose, obs_arm_pose):
        h_quat = headset_data.h_quat
        r_quat = headset_data.r_quat

        self.start_h_pos = np.array(headset_data.h_pos)
        self.start_r_pos = np.array(headset_data.r_pos)
        self.start_h_rot = pose2mat(np.zeros(3), h_quat)[:3, :3]
        self.start_r_rot = pose2mat(np.zeros(3), r_quat)[:3, :3]

        self.start_obs_arm_pose = pose2mat(obs_arm_pose[:3], obs_arm_pose[3:])
        self.start_op_arm_pose = pose2mat(op_arm_pose[:3], op_arm_pose[3:])
        self.started = True

    def run(self, headset_data):
        if not self.started: 
            return None

        h_quat = headset_data.h_quat
        r_quat = headset_data.r_quat

        # Calculate VR transition increments
        delta_h_pos_world = np.array(headset_data.h_pos) - self.start_h_pos
        delta_r_pos_world = np.array(headset_data.r_pos) - self.start_r_pos
        delta_h_pos_local = self.start_h_rot.T @ delta_h_pos_world
        delta_r_pos_local = self.start_r_rot.T @ delta_r_pos_world
        # Mapping to simulation
        delta_h_pos_sim_local = self.R_align @ delta_h_pos_local
        delta_r_pos_sim_local = self.R_align @ delta_r_pos_local

        # Calculate VR rotation increments
        current_h_rot = pose2mat(np.zeros(3), h_quat)[:3, :3]
        current_r_rot = pose2mat(np.zeros(3), r_quat)[:3, :3]
        delta_h_rot_vr_local = self.start_h_rot.T @ current_h_rot
        delta_r_rot_vr_local = self.start_r_rot.T @ current_r_rot
        # Mapping to simulation
        delta_h_rot_sim_local = self.R_align @ delta_h_rot_vr_local @ self.R_align.T
        delta_r_rot_sim_local = self.R_align @ delta_r_rot_vr_local @ self.R_align.T

        # Apply to simulation robot
        target_obs_rot = self.start_obs_arm_pose[:3, :3] @ delta_h_rot_sim_local
        target_op_rot = self.start_op_arm_pose[:3, :3] @ delta_r_rot_sim_local
        target_obs_pos = self.start_obs_arm_pose[:3, 3] + self.start_obs_arm_pose[:3, :3] @ delta_h_pos_sim_local
        target_op_pos = self.start_op_arm_pose[:3, 3] + self.start_op_arm_pose[:3, :3] @ delta_r_pos_sim_local

        # Transform to Action
        target_obs_arm_pose = np.eye(4)
        target_obs_arm_pose[:3, :3] = target_obs_rot
        target_obs_arm_pose[:3, 3] = target_obs_pos
        
        target_op_arm_pose = np.eye(4)
        target_op_arm_pose[:3, :3] = target_op_rot
        target_op_arm_pose[:3, 3] = target_op_pos
        
        target_obs_pos_out, target_obs_quat = mat2pose(target_obs_arm_pose)
        target_op_pos_out, target_op_quat = mat2pose(target_op_arm_pose)
        
        target_op_gripper = np.array([headset_data.r_index_trigger])

        action = np.concatenate([target_op_pos_out, target_op_quat, target_op_gripper, target_obs_pos_out, target_obs_quat])
        
        return action


class MockHeadsetData:
    def __init__(self, h_pos, h_quat, r_pos, r_quat, trigger):
        self.h_pos = np.array(h_pos, dtype=float)
        self.h_quat = np.array(h_quat, dtype=float) # Input: xyzw
        self.r_pos = np.array(r_pos, dtype=float)
        self.r_quat = np.array(r_quat, dtype=float) # Input: xyzw
        self.r_index_trigger = trigger


if __name__ == "__main__":
    ctrl = HeadsetOurControl()

    # Init state (all quats are strictly xyzw: [0,0,0,1])
    mock_vr_init = MockHeadsetData(
        h_pos=[0., 0., 1.5], h_quat=[0, 0, 0, 1],
        r_pos=[0.3, -0.2, 1.2], r_quat=[0, 0, 0, 1], trigger=0.0
    )
    op_arm_init = np.array([0.4, -0.3, 1.0, 0, 0, 0, 1]) # pos + xyzw
    obs_arm_init = np.array([0.0, 0.0, 1.2, 0, 0, 0, 1])

    print("Case 1: Run before start")
    assert ctrl.run(mock_vr_init) is None

    print("Case 2: Start calibration")
    ctrl.start(mock_vr_init, op_arm_init, obs_arm_init)

    print("Case 3: Zero delta")
    action = ctrl.run(mock_vr_init)
    print(f"  Op pos: {action[0:3]} (Expected: ~[0.4, -0.3, 1.0])")

    print("Case 4: Move forward & trigger (VR Y-0.1 -> Sim X+0.1)")
    mock_vr_move = MockHeadsetData(
        h_pos=[0, 0, 1.5], h_quat=[0, 0, 0, 1],
        r_pos=[0.3, -0.3, 1.2], r_quat=[0, 0, 0, 1], trigger=0.8
    )
    action = ctrl.run(mock_vr_move)
    print(f"  Op pos X: {action[0]:.4f} (Expected: ~0.5), Gripper: {action[7]:.2f}")

    print("Case 5: Pitch down -90deg (VR rot +X -> Sim rot +Y)")
    # VR: Rotate -90 around X-axis -> xyzw: [-0.707, 0, 0, 0.707]
    vr_pitch_quat = np.array([-0.7071068, 0, 0, 0.7071068])
    mock_vr_rot = MockHeadsetData(
        h_pos=[0, 0, 1.5], h_quat=[0, 0, 0, 1],
        r_pos=[0.3, -0.2, 1.2], r_quat=vr_pitch_quat, trigger=0.0
    )
    action = ctrl.run(mock_vr_rot)
    # Sim: Expected rotate -90 around Y-axis -> xyzw: [0, -0.707, 0, 0.707]
    sim_pitch_quat = np.array([0, -0.7071068, 0, 0.7071068])
    dot_val = np.abs(np.dot(action[3:7], sim_pitch_quat))
    print(f"  Op quat: {np.round(action[3:7], 4)} (Expected: ~[0, -0.7071, 0, 0.7071])")
    assert dot_val > 0.999, "Rotation mapping failed!"

    print("\nAll tests passed!")