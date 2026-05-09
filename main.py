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
from omni.isaac.core.utils.types import ArticulationAction
from robots.franka.rmpflow_controller import RMPFlowController

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
        ctrl = HeadsetOurControl(wxyz=False)
        headset.run_in_thread()
        is_calibrated = False
        vr_connected = False

        # Prepare RMPFlow Controller
        op_rmp_controller = task_controller.rmp_controller
        obs_rmp_controller = RMPFlowController(name="obs_rmp_controller",
                                               robot_articulation=robot_obs)
        task.reset()
        op_default_joints = robot_op.get_joint_positions()   # (9,)
        obs_default_joints = robot_obs.get_joint_positions() # (12,)
        obs_base_default = obs_default_joints[:3]
        # print(f"obs joint positions shape: {obs_default_joints.shape}")
        # print(obs_default_joints)
        print("[VR] Waiting for headset connection...")

    video_writer = None
    task.reset()
    # test_step_counter = 0
    
    while simulation_app.is_running():
        world.step(render=True)                    

        if world.is_playing():
            if task_controller.need_reset() or task.need_reset():
                if args.use_vr: # and not vr_connected:
                    task_controller.reset_needed = False
                    task.reset_needed = False
                    continue
                    # TODO: controller reset logic
                    # controller._last_success

                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                           
                task_controller.reset()
                if task_controller.episode_num() >= cfg.max_episodes:
                    task_controller.close()
                    simulation_app.close()
                    cv2.destroyAllWindows()
                    break
                task.reset()

                if args.use_vr:
                    is_calibrated = False
                
                continue
                
            state = task.step()
            if state is None:
                continue
            
            # VR Data Collection
            if args.use_vr:
                data = headset.receive_data()

                if not vr_connected:
                    if data is not None:
                        vr_connected = True
                        print("[VR] Headset Connected! Press A button to calibrate.")
                    else:
                        # don't move
                        robot_op.get_articulation_controller().apply_action(
                            ArticulationAction(joint_positions=list(op_default_joints))
                        )
                        robot_obs.get_articulation_controller().apply_action(
                            ArticulationAction(joint_positions=list(obs_default_joints))
                        )
                        continue
                
                if not is_calibrated:
                    robot_op.get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=list(op_default_joints))
                    )
                    robot_obs.get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=list(obs_default_joints))
                    )

                    if data is not None and data.r_button_one:
                        print(">>> START CALIBRATION...")
                        
                        # Get robots ee_pose
                        ee_pos_op = robot_op.get_gripper_position()
                        ee_quat_op = robot_op.get_gripper_orientation()
                        ee_quat_op_xyzw = np.array([ee_quat_op[1], ee_quat_op[2], ee_quat_op[3], ee_quat_op[0]])

                        ee_pos_obs = robot_obs.get_gripper_position()
                        ee_quat_obs = robot_obs.get_gripper_orientation()
                        ee_quat_obs_xyzw = np.array([ee_quat_obs[1], ee_quat_obs[2], ee_quat_obs[3], ee_quat_obs[0]])
                        
                        op_pose = np.concatenate([ee_pos_op, ee_quat_op_xyzw])
                        obs_pose = np.concatenate([ee_pos_obs, ee_quat_obs_xyzw])

                        ctrl.start(data, op_pose, obs_pose)
                        is_calibrated = True
                        print("[VR] >>> CALIBRATION DONE!")
                        
                        continue

                # VR Data Transformation 
                if data is not None: 
                    ee_pos_op = robot_op.get_gripper_position()
                    ee_quat_op = robot_op.get_gripper_orientation()
                    ee_quat_op_xyzw = np.array([ee_quat_op[1], ee_quat_op[2], ee_quat_op[3], ee_quat_op[0]])
                    op_pose = np.concatenate([ee_pos_op, ee_quat_op_xyzw])

                    ee_pos_obs = robot_obs.get_gripper_position()
                    ee_quat_obs = robot_obs.get_gripper_orientation()
                    ee_quat_obs_xyzw = np.array([ee_quat_obs[1], ee_quat_obs[2], ee_quat_obs[3], ee_quat_obs[0]])
                    obs_pose = np.concatenate([ee_pos_obs, ee_quat_obs_xyzw])

                    # Get Target Action
                    action, feedback = ctrl.run(data, op_pose, obs_pose)
                    # print(action)

                    # if feedback.right_out_of_sync:
                    #     raise RuntimeError("Warning: Right Arm Out of Sync!")

                    target_op_pos = action[0:3]
                    target_op_quat = action[3:7] # xyzw
                    target_op_gripper = action[7]
                    
                    target_obs_pos = action[8:11]
                    target_obs_quat = action[11:15]

                    target_op_quat_wxyz = np.array([target_op_quat[3], target_op_quat[0], 
                                                    target_op_quat[1], target_op_quat[2]])
                    target_obs_quat_wxyz = np.array([target_obs_quat[3], target_obs_quat[0], 
                                                        target_obs_quat[1], target_obs_quat[2]])

                    target_op_joints = op_rmp_controller.forward(target_op_pos, target_op_quat_wxyz)
                    target_obs_joints = obs_rmp_controller.forward(target_obs_pos, target_obs_quat_wxyz)

                    op_gripper_val = 0.04 * (1.0 - target_op_gripper)

                    # Apply Articulation Action
                    op_action = ArticulationAction(
                        joint_positions=list(target_op_joints.joint_positions) + [op_gripper_val, op_gripper_val]
                    )
                    obs_action = ArticulationAction(
                        joint_positions=list(obs_base_default) + list(target_obs_joints.joint_positions) + [0.04, 0.04]
                    )
                    
                    # Test Actions
                    # test_step_counter += 1
                    # op_test_joints = list(op_default_joints)
                    # op_test_joints[1] += 0.4 * math.sin(test_step_counter * 0.02)
                    # op_action = ArticulationAction(joint_positions=op_test_joints)

                    # obs_test_joints = list(obs_default_joints)
                    # obs_test_joints[4] += 0.4 * math.sin(test_step_counter * 0.02)
                    # obs_action = ArticulationAction(joint_positions=obs_test_joints)

                    robot_op.get_articulation_controller().apply_action(op_action)
                    robot_obs.get_articulation_controller().apply_action(obs_action)

            else:
                action, done, is_success = task_controller.step(state)
                if action is not None:
                    if multi_robots_mode:
                        assert isinstance(action, (list, tuple))  # Need contoller to divide actions
                        for i, act in enumerate(action):
                                if act is not None:
                                    robot[i].get_articulation_controller().apply_action(act)
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
                        output_path = os.path.join(output_dir, f"episode_{task_controller._episode_num}.mp4")
                        if video_writer is None:
                            height, width = combined_img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(output_path, fourcc, 60.0, (width, height))
                        video_writer.write(combined_img)


if __name__ == "__main__":
    main()
