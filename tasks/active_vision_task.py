from .base_task import BaseTask
from .pick_task import PickTask
from typing import Dict, Any
from omni.isaac.core.utils.types import ArticulationAction


class ActiveVisionTask(PickTask):
    def __init__(self, cfg, world, stage, robot):
        if not isinstance(robot, list) or len(robot) < 2:
            raise ValueError("Active Vision Task needs at least 2 robots!")
        self.robot_op = robot[0]  # operator
        self.robot_obs = robot[1] # observer
        self.robots = robot       # robot list
        self.obs_initial_joints = [0, 0, 0, 0.8, -1.57, 0, -1.57, 0, 1.57, -0.8, 0.04, 0.04]
        super().__init__(cfg, world, stage, self.robot_op)

    def reset(self):
        super().reset()
        self.robot_obs.initialize()
        self.robot_obs.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=list(self.obs_initial_joints))
        )
 
    def get_basic_state_info(self, object_path: str = None, target_path: str = None, 
                           additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        # operator states
        op_joints = self.robot_op.get_joint_positions()
        op_gripper = self.robot_op.get_gripper_position()
        
        # observer states
        obs_joints = self.robot_obs.get_joint_positions()
        obs_gripper = self.robot_obs.get_gripper_position()

        # get cameras
        camera_data, display_data = self.get_camera_data()
        
        state = {
            'joint_positions_op': op_joints,
            'gripper_position_op': op_gripper,
            'joint_positions_obs': obs_joints,
            'gripper_position_obs': obs_gripper,
            'camera_data': camera_data,   
            'camera_display': display_data,
            'done': self.reset_needed,
        }
        
        # Object info: copy from BaseTask
        if object_path:
            state.update({
                'object_position': self.object_utils.get_geometry_center(object_path=object_path),
                'object_size': self.object_utils.get_object_size(object_path=object_path),
                'object_path': object_path,
            })
        
        if target_path:
            state.update({
                'target_position': self.object_utils.get_geometry_center(object_path=target_path),
                'target_size': self.object_utils.get_object_size(object_path=target_path),
                'target_path': target_path,
            })
            
        if additional_info:
            state.update(additional_info)
            
        return state


class ActiveVisionTaskRaw(BaseTask):
    
    def __init__(self, cfg, world, stage, robot):
        """
        Args:
            robots (list): [operator, observer]
        """
        if not isinstance(robot, list) or len(robot) < 2:
            raise ValueError("Active Vision Task needs at least 2 robots!")
            
        self.robot_op = robot[0]  # operator
        self.robot_obs = robot[1] # observer
        self.robots = robot       # robot list
        
        # Pass robot_op to super
        super().__init__(cfg, world, stage, self.robot_op)
        
        print(f"[Active Vision Task] Initialized: Operator={self.robot_op.name}, Observer={self.robot_obs.name}")

        # Get operation task type
        # from factories.task_factory import create_task
        sub_task_type = cfg.task.get("sub_task_type", "pick")
        # self.sub_task = create_task(
        #     sub_task_type, 
        #     cfg, 
        #     world, 
        #     stage, 
        #     robot=self.robot_op
        # )
        print(f"[Active Vision Task] Wrapped sub-task: {sub_task_type}")

    def get_basic_state_info(self, object_path: str = None, target_path: str = None, 
                           additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Integrate state and vision data from multiple robots
        """
        # operator states
        op_joints = self.robot_op.get_joint_positions()
        op_gripper = self.robot_op.get_gripper_position()
        
        # observer states
        obs_joints = self.robot_obs.get_joint_positions()
        obs_gripper = self.robot_obs.get_gripper_position()

        # get cameras
        camera_data, display_data = self.get_camera_data()
        
        state = {
            'joint_positions_op': op_joints,
            'gripper_position_op': op_gripper,
            'joint_positions_obs': obs_joints,
            'gripper_position_obs': obs_gripper,
            'camera_data': camera_data,   
            'camera_display': display_data,
            'done': self.reset_needed,
        }
        
        # Object info: copy from BaseTask
        if object_path:
            state.update({
                'object_position': self.object_utils.get_geometry_center(object_path=object_path),
                'object_size': self.object_utils.get_object_size(object_path=object_path),
                'object_path': object_path,
            })
        
        if target_path:
            state.update({
                'target_position': self.object_utils.get_geometry_center(object_path=target_path),
                'target_size': self.object_utils.get_object_size(object_path=target_path),
                'target_path': target_path,
            })
            
        if additional_info:
            state.update(additional_info)
            
        return state

    def on_task_complete(self, success):
        # self.sub_task.on_task_complete(success)
        self.update_object_and_material_indices(success)

    def reset(self):
        super().reset() # BaseTask.reset -> world.reset()
        
        # multiple robots initialize
        self.robot_op.initialize()
        self.robot_obs.initialize() 
        # self.sub_task.reset()
        
        if self.material_config:
            self.apply_material_to_object(self.material_config.path)
        
        self.current_obj_path = self.place_objects_with_visibility_management(
            self.current_obj_idx, far_distance=10.0
        )

    def step(self):
        self.frame_idx += 1
        
        if not self.check_frame_limits():
            return None
            
        return self.get_basic_state_info(
            object_path=self.current_obj_path,
            additional_info={
                'object_name': self.current_obj_path.split("/")[-1]
            }
        )
