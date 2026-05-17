import os
import math
import argparse
from isaacsim import SimulationApp

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='LabSim Simulation Environment')
    parser.add_argument('--backend', type=str, default='numpy', 
                       choices=['numpy', 'gpu'], 
                       help='Backend choice: numpy (CPU) or gpu')
    parser.add_argument('--headless', action='store_true', 
                       help='Run in headless mode (default is with GUI)')
    parser.add_argument('--no-video', action='store_true', 
                       help='Disable video display and saving')
    parser.add_argument('--config-name', type=str, default='level3_Heat_Liquid',
                       help='Configuration file name (without .yaml extension)')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Configuration directory path (default: config)')
    parser.add_argument('--use-vr', action='store_true', 
                       help='Enable VR teleoperation mode')
    return parser.parse_args()

# Get command line arguments
args = parse_args()

# Set up simulation app based on arguments
simulation_config = {"headless": args.headless}
simulation_app = SimulationApp(simulation_config)

import hydra
from omegaconf import OmegaConf
import cv2
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import omni
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd
from isaacsim.core.utils import extensions

extensions.enable_extension("omni.physx.bundle")
extensions.enable_extension("omni.usdphysics.ui")

from factories.robot_factory import create_robot
from lab_utils.object_utils import ObjectUtils
from factories.task_factory import create_task
from factories.controller_factory import create_controller

from quest.webrtc_headset import WebRTCHeadset
from quest.headset_control import HeadsetOurControl
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.motion_generation import (
    LulaKinematicsSolver,
    ArticulationKinematicsSolver
)
from robots.franka.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation


class IKController:
    def __init__(self, robot_articulation: Articulation):
        franka_dir = "/home/ubuntu/LabUtopia/robots/franka"
        robot_description_path = os.path.join(franka_dir, "rmpflow/robot_descriptor.yaml")
        urdf_path = os.path.join(franka_dir, "lula_franka_gen.urdf")
        
        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path
        )

        end_effector_name = "right_gripper" 
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            robot_articulation, 
            self._kinematics_solver, 
            end_effector_name
        )
        self.robot = robot_articulation
        self.last_action = None
        print("Kinematics Solver Initialized!")

    def reset(self):
        robot_base_translation, robot_base_orientation = self.robot.get_world_pose() # wxyz
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)        

    def get_current_pose(self):
        ee_pos, ee_rot = self._articulation_kinematics_solver.compute_end_effector_pose()
        ee_quat = Rotation.from_matrix(ee_rot).as_quat() # xyzw
        return ee_pos, ee_quat
    
    def forward(self, target_pos_world, target_quat_world_wxyz):
        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            target_position=target_pos_world,
            target_orientation=target_quat_world_wxyz
        )
        if not success:
            # print("[Warning] Inverse Dynamics Solving Failed!!!")
            return self.last_action
        self.last_action = action    
        return action


def main():
    hydra.initialize(config_path=args.config_dir, job_name=args.config_name)
    cfg = hydra.compose(config_name=args.config_name)
    os.makedirs(cfg.multi_run.run_dir, exist_ok=True)
    OmegaConf.save(cfg, cfg.multi_run.run_dir + "/config.yaml")

    # Set backend based on command line arguments
    if args.backend == 'gpu':
        world = World(stage_units_in_meters=1, device="cpu")
        physx_interface = omni.physx.get_physx_interface()
        physx_interface.overwrite_gpu_setting(1)
    else:
        world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
    
    # Override configuration based on command line arguments
    if args.no_video:
        save_video = False
        show_video = False
    else:
        save_video = True
        show_video = True
    
    # whether to use multiple robots
    multi_robots_mode = OmegaConf.is_list(cfg.robot)
    
    if multi_robots_mode:
        print("[Main] Running in Multi-Robot Mode.")
        robot = []
        for i, r_cfg in enumerate(cfg.robot):
            prim_path = r_cfg.get("prim_path", f"/World/Robot_{i}")
            name = r_cfg.get("name", f"robot_{i}")
            single_robot = create_robot(
                r_cfg.type,
                prim_path=prim_path,
                name=name,
                position=np.array(r_cfg.position)
            )
            robot.append(single_robot)
        assert len(robot) == 2  # Currently only use 2 robots
        robot_op = robot[0]     # First for operation
        robot_obs = robot[1]    # Second for observation
    else:
        print("[Main] Running in Single-Robot Mode.")
        robot = create_robot(
            cfg.robot.type,
            position=np.array(cfg.robot.position)
        )
    
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=os.path.abspath(cfg.usd_path), prim_path="/World")
    
    ObjectUtils.get_instance(stage)
    
    task = create_task(
        cfg.task_type if not multi_robots_mode else "active_vision",
        cfg=cfg,
        world=world,
        stage=stage,
        robot=robot,
    )
    
    task_controller = create_controller(
        cfg.controller_type if not multi_robots_mode else "active_vision",
        cfg=cfg,
        robot=robot,
    )
    
    # VR Initialize
    if args.use_vr:
        assert multi_robots_mode is True
        print("[VR] Initializing VR connection...")
        headset = WebRTCHeadset()
        ctrl = HeadsetOurControl()
        headset.run_in_thread()
        vr_connected = False
        is_calibrated = False
        ask_for_calibrate = False
        last_r_button_one = False  # Calibrate, Mark success
        last_r_thumbstick = False  # Mark failed
        last_l_button_one = False  # Switch op_paused
        op_paused = False
        vr_reset_needed = False
        final_joint_positions = None
        test_step_counter = 0

        # Prepare IK Controller
        op_ik_controller = IKController(robot_articulation=robot_op)
        obs_ik_controller = IKController(robot_articulation=robot_obs)
        print("[VR] Waiting for headset connection...")

    video_writer = None
    video_success_count = 0
    current_video_path = None
    task.reset()

    if args.use_vr:
        op_ik_controller.reset()
        obs_ik_controller.reset()

    if args.use_vr or multi_robots_mode:
        op_default_joints = robot_op.get_joint_positions()   # (9,)
        # obs_default_joints = robot_obs.get_joint_positions() # (12,)
        obs_default_joints = task.obs_initial_joints
        obs_base_default = obs_default_joints[:3]
        # print(f"Obs Initial Joint Positions: {obs_default_joints[3:10]}\n")
        
        op_initial_action = ArticulationAction(joint_positions=list(op_default_joints))
        obs_initial_action = ArticulationAction(joint_positions=list(obs_default_joints))
        last_op_action = op_initial_action
        last_obs_action = obs_initial_action
    
    while simulation_app.is_running():
        world.step(render=True)                    

        if world.is_playing():
            if task_controller.need_reset() or task.need_reset():
                if args.use_vr and not vr_reset_needed:
                    task_controller.reset_needed = False
                    task.reset_needed = False
                    continue
                
                if args.use_vr:
                    if task_controller._last_success:
                        task_controller.data_collector.write_cached_data(final_joint_positions)
                    else:
                        task_controller.data_collector.clear_cache()
                    task.on_task_complete(task_controller._last_success)
                    is_calibrated = False
                    last_r_button_one = False
                    last_r_thumbstick = False
                    last_l_button_one = False
                    vr_reset_needed = False
                    op_paused = False
                    final_joint_positions = None
                    last_op_action = op_initial_action
                    last_obs_action = obs_initial_action

                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    if task_controller._last_success:
                        video_success_count += 1
                    else:
                        if current_video_path and os.path.exists(current_video_path):
                            os.remove(current_video_path)
                    current_video_path = None 
                           
                task_controller.reset()
                if task_controller.episode_num() >= cfg.max_episodes:
                    task_controller.close()
                    simulation_app.close()
                    cv2.destroyAllWindows()
                    break
                task.reset()

                if args.use_vr:
                    op_ik_controller.reset()
                    obs_ik_controller.reset()
                
                continue
                          
            # VR Data Collection
            if args.use_vr:
                data = headset.receive_data()

                if not vr_connected:
                    if data is not None:
                        vr_connected = True
                        print("[VR] Headset Connected!")
                    else:
                        # don't move
                        robot_op.get_articulation_controller().apply_action(op_initial_action)
                        robot_obs.get_articulation_controller().apply_action(obs_initial_action)
                        continue
                
                if not is_calibrated:
                    if not ask_for_calibrate:
                        print("[VR] Press A button to calibrate!")
                        ask_for_calibrate = True

                    robot_op.get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=list(op_default_joints))
                    )
                    robot_obs.get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=list(obs_default_joints))
                    )

                    if data is not None and data.r_button_one and not last_r_button_one:
                        print("[VR] >>> START CALIBRATION...")
                        
                        # Get robots ee_pose                        
                        ee_pos_op, ee_quat_op_xyzw = op_ik_controller.get_current_pose()
                        ee_pos_obs, ee_quat_obs_xyzw = obs_ik_controller.get_current_pose()
                        
                        op_pose = np.concatenate([ee_pos_op, ee_quat_op_xyzw])
                        obs_pose = np.concatenate([ee_pos_obs, ee_quat_obs_xyzw])
                        
                        # Pass strictly xyzw into VR controller
                        ctrl.start(data, op_pose, obs_pose)
                        print("[VR] >>> CALIBRATION DONE!")

                        is_calibrated = True
                        ask_for_calibrate = False
                        last_r_button_one = data.r_button_one

                        task.reset()
                    
                    continue

                # VR Data Transformation 
                if data is not None: 
                    test_step_counter += 1 
                    state = task.step()
                    if state is None:
                        continue

                    # Record Data
                    if 'camera_data' in state:
                        task_controller.data_collector.cache_step(
                            camera_images=state['camera_data'],
                            # Concat joint states
                            # joint_angles=state['joint_positions_op'][:-1], # only op
                            joint_angles=np.concatenate([
                                state['joint_positions_op'][:-1], 
                                state['joint_positions_obs'][3:-2]
                            ]),
                        )
                    
                    # Switch op_paused
                    if data.l_button_one and not last_l_button_one:
                        if op_paused:
                            op_paused = False
                            print("[VR] OBS arm pauses! OP arm moves following controller!")
                        else:
                            op_paused = True
                            print("[VR] OP arm pauses! OBS arm moves following VR!")
                    
                    # Get Current Pose
                    ee_pos_op, ee_quat_op_xyzw = op_ik_controller.get_current_pose()
                    op_pose = np.concatenate([ee_pos_op, ee_quat_op_xyzw])
                    ee_pos_obs, ee_quat_obs_xyzw = obs_ik_controller.get_current_pose()
                    obs_pose = np.concatenate([ee_pos_obs, ee_quat_obs_xyzw])
                    if test_step_counter % 50 == 0:
                        print(f"Op current pos: {ee_pos_op} | quat: {ee_quat_op_xyzw}")
                        print(f"Obs current pos: {ee_pos_obs} | quat: {ee_quat_obs_xyzw}\n")

                    # Obs Follow VR
                    if op_paused:
                        # Op Apply Last Action
                        robot_op.get_articulation_controller().apply_action(last_op_action)

                        # Get Target Action
                        action = ctrl.run(data)
                        if action is None:
                            continue
                        
                        target_obs_pos = action[8:11]
                        target_obs_quat = action[11:15]

                        if test_step_counter % 50 == 0:
                            print(f"Obs target pos: {target_obs_pos} | quat: {target_obs_quat}\n")

                        target_obs_quat_wxyz = np.array([target_obs_quat[3], target_obs_quat[0], 
                                                            target_obs_quat[1], target_obs_quat[2]])
                        target_obs_action = obs_ik_controller.forward(target_obs_pos, target_obs_quat_wxyz)
                        if target_obs_action is None:
                            continue
                        target_obs_joints = target_obs_action.joint_positions[:7]
                        # Apply Articulation Action
                        obs_action = ArticulationAction(
                            joint_positions=list(obs_base_default) + list(target_obs_joints) + [0.04, 0.04]
                        )               
                        robot_obs.get_articulation_controller().apply_action(obs_action)
                        last_obs_action = obs_action
                    
                    # Op Follow Controller
                    else:
                        # Obs Apply Last Action
                        robot_obs.get_articulation_controller().apply_action(last_obs_action)
                        
                        # Get Op Target Action
                        action, done, is_success = task_controller.step(state)
                        if action is not None:
                            assert isinstance(action, (list, tuple))
                            action_op = action[0]
                            robot_op.get_articulation_controller().apply_action(action_op)
                            last_op_action = action_op

                        if done:
                            vr_reset_needed = True
                            task.reset_needed = True
                            task_controller.reset_needed = True
                            if is_success:
                                task_controller._last_success = True
                                # final_joint_positions = state['joint_positions_op'][:-1] # only op
                                final_joint_positions = np.concatenate([
                                    state['joint_positions_op'][:-1], 
                                    state['joint_positions_obs'][3:-2]
                                ])
                            else:
                                task_controller._last_success = False

                    # Mark Task Complete: Success
                    if data.r_button_one and not last_r_button_one:
                        print("[VR] Task Marked as SUCCESS!")
                        task_controller._last_success = True 
                        vr_reset_needed = True
                        task.reset_needed = True
                        task_controller.reset_needed = True
                        # final_joint_positions = state['joint_positions_op'][:-1] # only op
                        final_joint_positions = np.concatenate([
                            state['joint_positions_op'][:-1], 
                            state['joint_positions_obs'][3:-2]
                        ])

                    # Mark Task Complete: Failed
                    if data.r_button_thumbstick and not last_r_thumbstick:
                        print("[VR] Task Marked as FAILED!")
                        task_controller._last_success = False
                        vr_reset_needed = True
                        task.reset_needed = True 
                        task_controller.reset_needed = True

                    last_r_button_one = data.r_button_one
                    last_r_thumbstick = data.r_button_thumbstick
                    last_l_button_one = data.l_button_one

            else:
                state = task.step()
                if state is None:
                    continue
                
                action, done, is_success = task_controller.step(state)
                if action is not None:
                    if multi_robots_mode:
                        assert isinstance(action, (list, tuple))  # Need contoller to divide actions
                        action_op, action_obs = action[0], action[1]
                        # Need to combine action_obs with base and gripper
                        robot_op.get_articulation_controller().apply_action(action_op)
                        robot_obs.get_articulation_controller().apply_action(action_obs)
                    else:
                        robot.get_articulation_controller().apply_action(action)
                if done:
                    task.on_task_complete(is_success)
                    continue
            
            if save_video or show_video:
                camera_images = []
                for _, image_data in state['camera_display'].items():
                    display_img = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    camera_images.append(display_img)
                
                if camera_images:
                    combined_img = np.hstack(camera_images)
                    total_width = 0
                    for idx, img in enumerate(camera_images):
                        label = f"Camera {idx+1} ({cfg.cameras[idx].image_type})"
                        cv2.putText(combined_img, label, (total_width + 2, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                        total_width += img.shape[1]
                    if show_video:
                        pass
                        # cv2.imshow('Camera Views', combined_img)
                        # cv2.waitKey(1)
                    if save_video:
                        output_dir = os.path.join(cfg.multi_run.run_dir, "video")
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"episode_{video_success_count}.mp4")
                        current_video_path = output_path 
                        if video_writer is None:
                            height, width = combined_img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(output_path, fourcc, 60.0, (width, height))
                        video_writer.write(combined_img)


if __name__ == "__main__":
    main()
