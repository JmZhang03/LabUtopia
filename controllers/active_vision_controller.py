import re
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from .inference_engines.inference_engine_factory import InferenceEngineFactory
from omni.isaac.core.utils.types import ArticulationAction


class ActiveVisionController(BaseController):
    
    def __init__(self, cfg, robot):
        if not isinstance(robot, list) or len(robot) < 2:
            raise ValueError("Active Vision Controller needs at least 2 robots!")
            
        self.robot_op = robot[0]  # operator
        self.robot_obs = robot[1] # observer
        self.robots = robot       # robot list
        
        # Pass robot_op to super
        super().__init__(cfg, self.robot_op)
        self.initial_position = None
        self.frame_idx = 0

        print(f"[Active Vision Controller] Initialized: Operator={self.robot_op.name}, Observer={self.robot_obs.name}")

    def reset(self):
        super().reset()
        
        if self.mode == "collect":
            self.pick_controller.reset()
        else:
            self.inference_engine.reset()
            
        self.initial_position = None
        self.frame_idx = 0

    def step(self, state):
        if self.initial_position is None:
            self.initial_position = state['object_position']
        self.state = state
        self.frame_idx += 1
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _check_success(self, state):
        # Same as PickTaskController: Object height increases 10cm
        return state['object_position'][2] > self.initial_position[2] + 0.1

    def get_language_instruction(self):
        # Same as PickTaskController
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').replace('  ', ' ').lower()
        self._language_instruction = f"Pick up the {object_name} from the table"
        return self._language_instruction

    def _get_observer_action(self, state) -> ArticulationAction:
        """
        Define observer action rules
        """
        # No move
        # return np.zeros(self.robot_obs.num_dof)
        
        # Some joints wave simply.
        action = np.zeros(self.robot_obs.num_dof)
        action[3] = 0.2 * np.sin(self.frame_idx * 0.1) 
        action[4] = 0.2 * np.sin(self.frame_idx * 0.1) 
        action[5] = 0.1 * np.sin(self.frame_idx * 0.1) 
        action[6] = 0.1 * np.sin(self.frame_idx * 0.1)
        return ArticulationAction(joint_positions=action)

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot=None)
        # Create operation atomic controller
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

    def _init_infer_mode(self, cfg, robot):
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=self.robot_op # robot_op
        )
        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )

    def _step_collect(self, state):
        if self._check_success(state):
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        action_op = None
        if not self.pick_controller.is_done():
            # !!! state keys: includes op & obs
            action_op = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions_op'],
                object_size=state['object_size'],
                object_name=state['object_name'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),
                gripper_position=state['gripper_position_op'],  
                pre_offset_x=0.05,
                after_offset_z=0.25
            )
        
            action_obs = self._get_observer_action(state)

            # Save both robot_op & robot_obs data
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    # Concat joint states
                    joint_angles=np.concatenate([
                        state['joint_positions_op'][:-1], 
                        state['joint_positions_obs'][:-1]
                    ]),
                    language_instruction=self.get_language_instruction()
                )
            
            return [action_op, action_obs], False, False
        
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            final_joints = np.concatenate([
                state['joint_positions_op'][:-1], 
                state['joint_positions_obs'][:-1]
            ])
            self.data_collector.write_cached_data(final_joints)
            self.reset_needed = True
            return None, True, True  
        
        self.data_collector.clear_cache()
        self._last_success = False
        self.reset_needed = True
        return None, True, False

    def _step_infer(self, state):
        state['language_instruction'] = self.get_language_instruction()
        raw_action = self.inference_engine.step_inference(state)  # action op concat obs
        action_op = raw_action[:8]
        action_obs = raw_action[8:]
        
        if self._check_success(state):
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.reset_needed = True
            return [action_op, action_obs], True, True
            
        return [action_op, action_obs], False, False
