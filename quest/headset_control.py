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
    def __init__(self, pos_scale=1.0):
        # VR / Sim World: X Front, Y Left, Z Up
        # Lula Robot Base Local: X Back, Y Left, Z Down
        self.start_h_T = None
        self.start_r_T = None
        self.start_op_arm_T = None
        self.start_obs_arm_T = None
        self.started = False
        self.pos_scale = pos_scale
        self.R_align = np.array([
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0, -1]
        ])

    def reset(self):
        self.start_h_T = None
        self.start_r_T = None
        self.start_op_arm_T = None
        self.start_obs_arm_T = None
        self.started = False

    def start(self, headset_data, op_arm_pose, obs_arm_pose):
        self.start_h_T = pose2mat(headset_data.h_pos, headset_data.h_quat)
        self.start_r_T = pose2mat(headset_data.r_pos, headset_data.r_quat)
        
        self.start_obs_arm_T = pose2mat(obs_arm_pose[:3], obs_arm_pose[3:])
        self.start_op_arm_T = pose2mat(op_arm_pose[:3], op_arm_pose[3:])
        self.started = True

    def align_delta(self, delta_T_vr):
            R_vr = delta_T_vr[:3, :3]
            t_vr = delta_T_vr[:3, 3]

            # Align Transition
            t_lula = self.R_align @ (t_vr * self.pos_scale)
            # Align Rotation: R_lula = R_align @ R_vr @ R_align^T)
            R_lula = self.R_align @ R_vr @ self.R_align.T
            
            # Return T Matrix
            delta_T_lula = np.eye(4)
            delta_T_lula[:3, :3] = R_lula
            delta_T_lula[:3, 3] = t_lula
            return delta_T_lula
    
    def run(self, headset_data):
        if not self.started: 
            return None

        # 1. VR current pose matrix
        current_h_T = pose2mat(headset_data.h_pos, headset_data.h_quat)
        current_r_T = pose2mat(headset_data.r_pos, headset_data.r_quat)

        # 2. VR increment: T_delta_world = T_start_inv @ T_current
        delta_h_T_vr = np.linalg.inv(self.start_h_T) @ current_h_T
        delta_r_T_vr = np.linalg.inv(self.start_r_T) @ current_r_T
        
        # Align and Scale
        delta_h_T_lula = self.align_delta(delta_h_T_vr)
        delta_r_T_lula = self.align_delta(delta_r_T_vr)

        # 3. Apply to sim initial pose
        target_obs_T = self.start_obs_arm_T @ delta_h_T_lula
        target_op_T = self.start_op_arm_T @ delta_r_T_lula

        target_obs_pos, target_obs_quat = mat2pose(target_obs_T)
        target_op_pos, target_op_quat = mat2pose(target_op_T)
        
        target_op_gripper = np.array([headset_data.r_index_trigger])

        action = np.concatenate([target_op_pos, target_op_quat, target_op_gripper, target_obs_pos, target_obs_quat])
        
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

    print("Case 4: Move left (VR Y-0.1)")
    mock_vr_move = MockHeadsetData(
        h_pos=[0, 0, 1.5], h_quat=[0, 0, 0, 1],
        r_pos=[0.3, -0.3, 1.2], r_quat=[0, 0, 0, 1], trigger=0.8
    )
    action = ctrl.run(mock_vr_move)
    print(f"  Op pos Y: {action[1]:.4f} (Expected: ~-0.4), Gripper: {action[7]:.2f}")

    print("Case 5: Pitch down -90deg (VR rot -X -> Sim rot -X)")
    # VR: Rotate -90 around X-axis -> xyzw: [-0.707, 0, 0, 0.707]
    vr_pitch_quat = np.array([-0.7071068, 0, 0, 0.7071068])
    mock_vr_rot = MockHeadsetData(
        h_pos=[0, 0, 1.5], h_quat=[0, 0, 0, 1],
        r_pos=[0.3, -0.2, 1.2], r_quat=vr_pitch_quat, trigger=0.0
    )
    action = ctrl.run(mock_vr_rot)
    sim_pitch_quat = np.array([-0.7071068, 0, 0, 0.7071068])
    dot_val = np.abs(np.dot(action[3:7], sim_pitch_quat))
    print(f"  Op quat: {np.round(action[3:7], 4)} (Expected: ~[-0.7071, 0, 0, 0.7071])")
    assert dot_val > 0.999, "Rotation mapping failed!"

    print("\nAll tests passed!")