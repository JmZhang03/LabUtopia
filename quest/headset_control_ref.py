import numpy as np
from quest.headset_utils import  convert_right_to_left_coordinates, HeadsetFeedback 
from scipy.spatial.transform import Rotation as R

from quest.transform_utils import (
    align_rotation_to_z_axis, 
    within_pose_threshold, 
    quat2mat,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
    pose2mat,
    mat2pose,
    transform_coordinates,
    our_transform_coordinates,
    calibrate_controller_ee_mapping,
)
import time
import os
import math


class HeadsetControl():
    def __init__(
            self,
            start_head_position_threshold=0.03,
            start_head_rotation_threshold=0.2,
            head_position_threshold=0.05,
            head_rotation_threshold=0.3
        ):
        self.start_middle_arm_pose = None
        self.start_headset_pose = None
        self.started = False

        self.start_head_position_threshold = start_head_position_threshold
        self.start_head_rotation_threshold = start_head_rotation_threshold

        self.head_position_threshold = head_position_threshold
        self.head_rotation_threshold = head_rotation_threshold
    
    def reset(self):
        self.start_middle_arm_pose = None
        self.start_headset_pose = None
        self.started = False

    def is_running(self):
        return self.started

    def start(self, headset_data, middle_arm_pose):
        aligned_headset_pose = np.eye(4)
        # 消除头显坐标系{H}对于上电坐标系{V}的z轴的偏移，使得vr的初始位置姿态似乎是正视图
        # 但是注意，仅仅在计算初始位置姿态的时候，存在这个纠正；在后续的采样中并不对齐
        #aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(headset_data.h_quat))
        aligned_headset_pose[:3, :3] = quat2mat(headset_data.h_quat)
        aligned_headset_pose[:3, 3] = headset_data.h_pos
        self.start_headset_pose = aligned_headset_pose

        aligned_middle_arm_pose = np.eye(4)
        aligned_middle_arm_pose[:3, :3] = quat2mat(wxyz_to_xyzw(middle_arm_pose[3:]))
        aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3]
        self.start_middle_arm_pose = aligned_middle_arm_pose

        self.started = True

    def run(self, headset_data, middle_arm_pose):
        middle_arm_pose = pose2mat(middle_arm_pose[:3], wxyz_to_xyzw(middle_arm_pose[3:]))
        headset_pose = pose2mat(headset_data.h_pos, headset_data.h_quat)

        if self.started:
            start_headset_pose = self.start_headset_pose
            start_middle_arm_pose = self.start_middle_arm_pose
        else:
            aligned_headset_pose = np.eye(4)
            aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(headset_pose[:3, :3])
            aligned_headset_pose[:3, 3] = headset_pose[:3, 3] # 位置不变
            start_headset_pose = aligned_headset_pose

            aligned_middle_arm_pose = np.eye(4)
            aligned_middle_arm_pose[:3, :3] = align_rotation_to_z_axis(middle_arm_pose[:3, :3])
            aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3, 3]
            start_middle_arm_pose = aligned_middle_arm_pose

        # calculate offset between current and saved headset pose
        # {B}_T_{E'}                                 {V}_T_{H}      {V}_T_{A}          {B}_T_{E}
        new_middle_arm_pose = transform_coordinates(headset_pose, start_headset_pose, start_middle_arm_pose)

        # convert to position and quaternion
        new_middle_arm_pos, new_middle_arm_quat = mat2pose(new_middle_arm_pose)
        new_middle_arm_quat = xyzw_to_wxyz(new_middle_arm_quat)

        # grippers 
        new_left_gripper = np.array([headset_data.l_index_trigger])
        new_right_gripper = np.array([headset_data.r_index_trigger])


        headset_action = np.concatenate([
            new_middle_arm_pos, new_middle_arm_quat
        ])

        # transform middle_arm_pose from mujoco coords to start_headset_pose coords
        unity_middle_arm_pose = transform_coordinates(middle_arm_pose, start_middle_arm_pose, start_headset_pose)
        unity_middle_arm_pos, unity_middle_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_middle_arm_pose))
        if self.started:
            headOutOfSync = not within_pose_threshold(
                middle_arm_pose[:3, 3],
                middle_arm_pose[:3, :3],
                new_middle_arm_pose[:3, 3], 
                new_middle_arm_pose[:3, :3],
                self.head_position_threshold if self.started else self.start_head_position_threshold,
                self.head_rotation_threshold if self.started else self.start_head_rotation_threshold
            )
        else:
            headOutOfSync = False
            

        feedback = HeadsetFeedback()
        feedback.info = ""
        feedback.head_out_of_sync = headOutOfSync
        feedback.left_out_of_sync = False
        feedback.right_out_of_sync = False
        feedback.left_arm_position = np.zeros(3)
        feedback.left_arm_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        feedback.right_arm_position = np.zeros(3)
        feedback.right_arm_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        feedback.middle_arm_position = unity_middle_arm_pos
        feedback.middle_arm_rotation = unity_middle_arm_quat        

        return headset_action, feedback

class HeadsetOurControl():
    """头显+右手"""
    def __init__(
            self,
            start_ctrl_position_threshold=0.06,
            start_ctrl_rotation_threshold=0.4,
            start_head_position_threshold=0.03,
            start_head_rotation_threshold=0.2,
            ctrl_position_threshold=0.04,
            ctrl_rotation_threshold=0.3,
            head_position_threshold=0.05,
            head_rotation_threshold=0.3,

            right_btv_quat=[0.0, 0.0, 0.0, 1.0],
            wxyz = True,
        ):
        self.start_middle_arm_pose = None
        self.start_right_arm_pose = None
        self.start_headset_pose = None
        self.started = False

        self.start_ctrl_position_threshold = start_ctrl_position_threshold
        self.start_ctrl_rotation_threshold = start_ctrl_rotation_threshold
        self.start_head_position_threshold = start_head_position_threshold
        self.start_head_rotation_threshold = start_head_rotation_threshold
        self.ctrl_position_threshold = ctrl_position_threshold
        self.ctrl_rotation_threshold = ctrl_rotation_threshold
        self.head_position_threshold = head_position_threshold
        self.head_rotation_threshold = head_rotation_threshold
        
        self.start_right_ee_rot = None
        self.start_right_ee_trans = None
        self.right_btv_quat = np.asarray(right_btv_quat,dtype=np.float64)

        self.wxyz = wxyz

        self.r_thumbstick_vector = None
        self.l_thumbstick_vector = None


    def reset(self):
        self.start_middle_arm_pose = None
        self.start_right_arm_pose = None
        self.start_headset_pose = None
        self.start_right_ee_rot = None
        self.start_right_ee_trans = None

        self.started = False

    def is_running(self):
        return self.started
    
 

    def thumbstick_rotation(self,headset_data):
        r_angle, l_angle = 0, 0
        current_r_thumbstick_vector = np.array([headset_data.r_thumbstick_x,headset_data.r_thumbstick_y])
        current_l_thumbstick_vector = np.array([headset_data.l_thumbstick_x,headset_data.l_thumbstick_y])   

        if np.linalg.norm(current_r_thumbstick_vector) > 0.98:
            if self.r_thumbstick_vector is None:
                self.r_thumbstick_vector = current_r_thumbstick_vector
            else:
                # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
                # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
                # 归一化向量
                v1 = self.r_thumbstick_vector / np.linalg.norm(self.r_thumbstick_vector)
                v2 = current_r_thumbstick_vector / np.linalg.norm(current_r_thumbstick_vector)
                
                # 计算点积和叉积
                dot_product = np.dot(v1, v2)
                cross_product = np.cross(v1, v2)
                
                # 使用arctan2计算角度（范围[-π, π]）
                r_angle = np.arctan2(cross_product, dot_product)
                
                # 将角度转换到[0, 2π]范围
                if r_angle < 0:
                    r_angle += 2 * np.pi
        else:
            self.r_thumbstick_vector = None

        if np.linalg.norm(current_l_thumbstick_vector) > 0.98:
            if self.l_thumbstick_vector is None:
                self.l_thumbstick_vector = current_l_thumbstick_vector
            else:
                # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
                # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
                # 归一化向量
                v1 = self.l_thumbstick_vector / np.linalg.norm(self.l_thumbstick_vector)
                v2 = current_l_thumbstick_vector / np.linalg.norm(current_l_thumbstick_vector)
                
                # 计算点积和叉积
                dot_product = np.dot(v1, v2)
                cross_product = np.cross(v1, v2)
                
                # 使用arctan2计算角度（范围[-π, π]）
                l_angle = np.arctan2(cross_product, dot_product)
                
                # 将角度转换到[0, 2π]范围
                if l_angle < 0:
                    l_angle += 2 * np.pi
        else:
            self.l_thumbstick_vector = None
        

        return r_angle, l_angle




    def start(self, headset_data, right_arm_pose, middle_arm_pose):
        if self.wxyz:
            middle_arm_pose[3:] = wxyz_to_xyzw(middle_arm_pose[3:])

        # 记录初始头显姿态{V}_T_{A}
        aligned_headset_pose = np.eye(4)
        #aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(headset_data.h_quat))
        aligned_headset_pose[:3, :3] = quat2mat(headset_data.h_quat)
        aligned_headset_pose[:3, 3] = headset_data.h_pos
        self.start_headset_pose = aligned_headset_pose

        aligned_right_pose = np.eye(4)
        #aligned_right_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(headset_data.r_quat))
        aligned_right_pose[:3, :3] = quat2mat(headset_data.r_quat)
        aligned_right_pose[:3, 3] = headset_data.r_pos
        self.start_right_pose = aligned_right_pose

        
        # 记录初始时刻的视觉机械臂位置姿态
        aligned_middle_arm_pose = np.eye(4)
        #aligned_middle_arm_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(wxyz_to_xyzw(middle_arm_pose[3:])))
        aligned_middle_arm_pose[:3, :3] = quat2mat(middle_arm_pose[3:])
        aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3]
        self.start_middle_arm_pose = aligned_middle_arm_pose

        aligned_right_arm_pose = np.eye(4)
        #aligned_right_arm_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(wxyz_to_xyzw(right_arm_pose[3:])))
        aligned_right_arm_pose[:3, :3] = quat2mat(right_arm_pose[3:])
        aligned_right_arm_pose[:3, 3] = right_arm_pose[:3]
        self.start_right_arm_pose = aligned_right_arm_pose
    
        # vtr  
        right_pose = pose2mat(headset_data.r_pos, headset_data.r_quat)
        # rse_rot  rse_trans_base
        self.start_right_ee_rot, self.start_right_ee_trans = \
            calibrate_controller_ee_mapping(right_pose, pose2mat(right_arm_pose[:3], right_arm_pose[3:]), self.right_btv_quat)

        self.started = True

    def run(self, headset_data, right_arm_pose, middle_arm_pose):
        """
        1. 将头显与控制器姿态从世界坐标系转换到以初始参考坐标系
        2. 输出机械臂动作向量 action。
        3. 计算是否“失步”（Out of Sync）以生成 feedback。
        """
        if self.wxyz:
            right_arm_pose[3:] = wxyz_to_xyzw(right_arm_pose[3:])
            middle_arm_pose[3:] = wxyz_to_xyzw(middle_arm_pose[3:])
        # 机械臂的当前位置姿态 
        middle_arm_pose = pose2mat(middle_arm_pose[:3], middle_arm_pose[3:])
        right_arm_pose = pose2mat(right_arm_pose[:3], right_arm_pose[3:])
        # 头显的目标位置姿态 {V}_T_{H}
        headset_pose = pose2mat(headset_data.h_pos, headset_data.h_quat)
        right_pose = pose2mat(headset_data.r_pos, headset_data.r_quat) # vtr

        if self.started:
            start_headset_pose = self.start_headset_pose
            start_middle_arm_pose = self.start_middle_arm_pose

            start_right_pose = self.start_right_pose
            start_right_arm_pose = self.start_right_arm_pose
            
            start_right_ee_rot = self.start_right_ee_rot 
            start_right_ee_trans = self.start_right_ee_trans

        else:
            # 不断更新初始位姿
            aligned_headset_pose = np.eye(4)
            aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(headset_pose[:3, :3])
            aligned_headset_pose[:3, 3] = headset_pose[:3, 3]
            start_headset_pose = aligned_headset_pose

            aligned_right_pose = np.eye(4)
            aligned_right_pose[:3, :3] = align_rotation_to_z_axis(right_pose[:3, :3])
            aligned_right_pose[:3, 3] = right_pose[:3, 3]
            start_right_pose = aligned_right_pose

            aligned_middle_arm_pose = np.eye(4)
            aligned_middle_arm_pose[:3, :3] = align_rotation_to_z_axis(middle_arm_pose[:3, :3])
            aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3, 3]
            start_middle_arm_pose = aligned_middle_arm_pose

            aligned_right_arm_pose = np.eye(4)
            aligned_right_arm_pose[:3, :3] = align_rotation_to_z_axis(right_arm_pose[:3, :3])
            aligned_right_arm_pose[:3, 3] = right_arm_pose[:3, 3]
            start_right_arm_pose = aligned_right_arm_pose

            start_right_ee_rot, start_right_ee_trans =  calibrate_controller_ee_mapping(right_pose,right_arm_pose,self.right_btv_quat)


        # calculate offset between current and saved headset pose
        # {B}_T_{E'}                                 {V}_T_{H}      {V}_T_{A}          {B}_T_{E}
        new_middle_arm_pose = transform_coordinates(headset_pose, start_headset_pose, start_middle_arm_pose)
        #  
        new_right_arm_pose = our_transform_coordinates(self.right_btv_quat, right_pose, start_right_ee_rot, start_right_ee_trans, np.eye(4))

        # convert to position and quaternion
        new_middle_arm_pos, new_middle_arm_quat = mat2pose(new_middle_arm_pose)
        new_right_arm_pos, new_right_arm_quat = mat2pose(new_right_arm_pose)
        new_middle_arm_quat = xyzw_to_wxyz(new_middle_arm_quat)
        new_right_arm_quat = xyzw_to_wxyz(new_right_arm_quat)

        # grippers 
        new_right_gripper = np.array([headset_data.r_index_trigger])

        # concatenate the new action
        action = np.concatenate([
            new_right_arm_pos, new_right_arm_quat, new_right_gripper,
            new_middle_arm_pos, new_middle_arm_quat
        ])

        # transform middle_arm_pose from mujoco coords to start_headset_pose coords
        # 把当前机械臂的位姿从世界坐标系转换到初始参考坐标系
        unity_middle_arm_pose = transform_coordinates(middle_arm_pose, start_middle_arm_pose, start_headset_pose)
        unity_right_arm_pose = transform_coordinates(right_arm_pose, start_middle_arm_pose, start_headset_pose)        

        # 将目标位置姿态从右手坐标系表示，转换为unity左手坐标系表示 
        unity_right_arm_pos, unity_right_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_right_arm_pose))
        unity_middle_arm_pos, unity_middle_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_middle_arm_pose))
        
        if self.started:
            headOutOfSync = not within_pose_threshold(
                middle_arm_pose[:3, 3],
                middle_arm_pose[:3, :3],
                new_middle_arm_pose[:3, 3], 
                new_middle_arm_pose[:3, :3],
                self.head_position_threshold if self.started else self.start_head_position_threshold,
                self.head_rotation_threshold if self.started else self.start_head_rotation_threshold
            )
            rightOutOfSync = not within_pose_threshold(
                right_arm_pose[:3, 3],
                right_arm_pose[:3, :3],
                new_right_arm_pose[:3, 3], 
                new_right_arm_pose[:3, :3],
                self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
                self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
            )
        else:
            headOutOfSync = False
            rightOutOfSync = False


        feedback = HeadsetFeedback()
        feedback.info = ""
        feedback.head_out_of_sync = headOutOfSync
        feedback.left_out_of_sync = False
        feedback.left_arm_position = np.zeros(3)
        feedback.left_arm_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        feedback.right_out_of_sync = rightOutOfSync
        feedback.right_arm_position = unity_right_arm_pos
        feedback.right_arm_rotation = unity_right_arm_quat
        feedback.middle_arm_position = unity_middle_arm_pos
        feedback.middle_arm_rotation = unity_middle_arm_quat        

        return action, feedback
    
class HeadsetRightControl():

    def __init__(
            self,
            start_ctrl_position_threshold=0.06,
            start_ctrl_rotation_threshold=0.4,
            ctrl_position_threshold=0.04,
            ctrl_rotation_threshold=0.3,

            right_btv_quat=[0.0, 0.0, 0.0, 1.0],
            head_btv_quat=[0.0, 0.0, 0.0, 1.0],
            wxyz = True,

            #rot_offset_axis = 'z'

        ):
        self.start_right_arm_pose = None
        self.start_head_arm_pose = None
        self.started = False

        self.start_ctrl_position_threshold = start_ctrl_position_threshold
        self.start_ctrl_rotation_threshold = start_ctrl_rotation_threshold

        self.ctrl_position_threshold = ctrl_position_threshold
        self.ctrl_rotation_threshold = ctrl_rotation_threshold

        self.start_right_ee_rot = None
        self.start_right_ee_trans = None
        
        self.start_head_ee_rot = None
        self.start_head_ee_trans = None

        self.btv_quat_right = np.asarray(right_btv_quat,dtype=np.float64)
        self.btv_quat_head = np.asarray(head_btv_quat,dtype=np.float64)
        self.wxyz = wxyz
        self.r_thumbstick_vector = None
        self.l_thumbstick_vector = None
        
        self.r_total_rot_offset = 0.0
        self.l_total_rot_offset = 0.0

        self.r_rot_offset = 0.0
        self.l_rot_offset = 0.0

        self.r_thumbstick_count = 0
        self.l_thumbstick_count = 0

        

        #self.rot_offset_axis = rot_offset_axis

    def reset(self):
        self.start_right_arm_pose = None
        self.start_head_arm_pose = None

        self.start_right_ee_rot = None
        self.start_right_ee_trans = None
        
        self.start_head_ee_rot = None
        self.start_head_ee_trans = None

        self.started = False

    def is_running(self):
        return self.started

    def thumbstick_rotating(self,headset_data,):

        current_r_thumbstick_vector = np.array([headset_data.r_thumbstick_x,headset_data.r_thumbstick_y])
        current_l_thumbstick_vector = np.array([headset_data.l_thumbstick_x,headset_data.l_thumbstick_y])   
        
        if np.linalg.norm(current_r_thumbstick_vector) > 0.98:
            self.r_thumbstick_count += 1
        else:
            self.r_thumbstick_count = 0
            
        if np.linalg.norm(current_l_thumbstick_vector) > 0.98:
            self.l_thumbstick_count += 1
        else:
            self.l_thumbstick_count = 0
        
        self.r_thumbstick_count = min(self.r_thumbstick_count,5)
        self.l_thumbstick_count = min(self.l_thumbstick_count,5)
        
        return current_r_thumbstick_vector,current_l_thumbstick_vector, self.r_thumbstick_count >= 1, self.l_thumbstick_count >= 1


    def update_offset_rotation(self,headset_data):

        current_r_thumbstick_vector,current_l_thumbstick_vector, r_thumbstick_action, l_thumbstick_action = self.thumbstick_rotating(headset_data)

        if r_thumbstick_action:
            if self.r_thumbstick_vector is None:
                self.r_thumbstick_vector = current_r_thumbstick_vector


            # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
            # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
            # 归一化向量
            v1 = self.r_thumbstick_vector / np.linalg.norm(self.r_thumbstick_vector)
            v2 = current_r_thumbstick_vector / np.linalg.norm(current_r_thumbstick_vector)
            
            # 计算点积和叉积
            dot_product = np.dot(v1, v2)
            cross_product = np.cross(v1, v2)
                
            # 使用arctan2计算角度（范围[-π, π]）
            r_angle = np.arctan2(cross_product, dot_product)

            # 将当前的角度 加上之前的总offset
            r_angle += self.r_total_rot_offset

            # 将角度转换到[0, 2π]范围
            if r_angle < 0:
                r_angle += 2 * np.pi
            # 保存摇杆当前对应的旋转量
            self.r_rot_offset = r_angle
        else:
            # 在松手后，清除起始向量
            self.r_thumbstick_vector = None
            # 叠加上次的偏移量
            if self.r_rot_offset != 0.0:
                self.r_total_rot_offset = self.r_rot_offset
            self.r_rot_offset = 0.0
            r_angle = self.r_total_rot_offset


        if l_thumbstick_action:
            
            if self.l_thumbstick_vector is None:
                # 
                self.l_thumbstick_vector = current_l_thumbstick_vector

            # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
            # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
            # 归一化向量
            v1 = self.l_thumbstick_vector / np.linalg.norm(self.l_thumbstick_vector)
            v2 = current_l_thumbstick_vector / np.linalg.norm(current_l_thumbstick_vector)
            
            # 计算点积和叉积
            dot_product = np.dot(v1, v2)
            cross_product = np.cross(v1, v2)
            
            # 使用arctan2计算角度（范围[-π, π]）
            l_angle = np.arctan2(cross_product, dot_product)

            # 将当前的角度 加上之前的总offset
            l_angle += self.l_total_rot_offset

            # 将角度转换到[0, 2π]范围
            if l_angle < 0:
                l_angle += 2 * np.pi
            # 保存摇杆当前对应的旋转量
            self.l_rot_offset = l_angle
                
                
        else:
            # 在松手后，清除起始向量
            self.l_thumbstick_vector = None
            # 叠加上次的偏移量
            if self.l_rot_offset != 0.0:
                self.l_total_rot_offset = self.l_rot_offset
            self.l_rot_offset = 0.0
            l_angle = self.l_total_rot_offset

        return r_angle, l_angle


    def start(self, headset_data, right_arm_pose, head_arm_pose):
        if self.wxyz:
            right_arm_pose[3:] = wxyz_to_xyzw(right_arm_pose[3:])
            head_arm_pose[3:] = wxyz_to_xyzw(head_arm_pose[3:])

        

        aligned_right_pose = np.eye(4)
        aligned_right_pose[:3, :3] = quat2mat(headset_data.r_quat)
        aligned_right_pose[:3, 3] = headset_data.r_pos
        self.start_right_pose = aligned_right_pose
        
        aligned_head_pose = np.eye(4)
        aligned_head_pose[:3, :3] = quat2mat(headset_data.h_quat)
        aligned_head_pose[:3, 3] = headset_data.h_pos
        self.start_head_pose = aligned_head_pose

        
        aligned_right_arm_pose = np.eye(4)
        aligned_right_arm_pose[:3, :3] = quat2mat(right_arm_pose[3:])
        aligned_right_arm_pose[:3, 3] = right_arm_pose[:3]
        self.start_right_arm_pose = aligned_right_arm_pose
    
        aligned_head_arm_pose = np.eye(4)
        aligned_head_arm_pose[:3, :3] = quat2mat(head_arm_pose[3:])
        aligned_head_arm_pose[:3, 3] = head_arm_pose[:3]
        self.start_head_arm_pose = aligned_head_arm_pose
        
        # vtr  
        right_pose = pose2mat(headset_data.r_pos, headset_data.r_quat)
        head_pose = pose2mat(headset_data.h_pos, headset_data.h_quat)
        # rse_rot  rse_trans_base
        self.start_right_ee_rot, self.start_right_ee_trans = \
            calibrate_controller_ee_mapping(right_pose, pose2mat(right_arm_pose[:3], right_arm_pose[3:]), self.btv_quat_right)
        
        self.start_head_ee_rot, self.start_head_ee_trans = \
            calibrate_controller_ee_mapping(head_pose, pose2mat(head_arm_pose[:3], head_arm_pose[3:]), self.btv_quat_head)

        self.started = True

    def run(self, headset_data, right_arm_pose, head_arm_pose):
        if self.wxyz:
            right_arm_pose[3:] = wxyz_to_xyzw(right_arm_pose[3:])
            head_arm_pose[3:] = wxyz_to_xyzw(head_arm_pose[3:])

        # 机械臂的当前位置姿态 
        right_arm_pose = pose2mat(right_arm_pose[:3], right_arm_pose[3:])
        head_arm_pose = pose2mat(head_arm_pose[:3], head_arm_pose[3:])

        right_pose = pose2mat(headset_data.r_pos, headset_data.r_quat) # vtr
        head_pose = pose2mat(headset_data.h_pos, headset_data.h_quat) # vtl

        if self.started:
            start_right_pose = self.start_right_pose
            start_right_arm_pose = self.start_right_arm_pose
            start_right_ee_rot = self.start_right_ee_rot 
            start_right_ee_trans = self.start_right_ee_trans
            
            start_head_pose = self.start_head_pose
            start_head_arm_pose = self.start_head_arm_pose
            start_head_ee_rot = self.start_head_ee_rot 
            start_head_ee_trans = self.start_head_ee_trans

        else:
            # 不断更新初始位姿
            aligned_right_pose = np.eye(4)
            aligned_right_pose[:3, :3] = align_rotation_to_z_axis(right_pose[:3, :3])
            aligned_right_pose[:3, 3] = right_pose[:3, 3]
            start_right_pose = aligned_right_pose
            
            aligned_head_pose = np.eye(4)
            aligned_head_pose[:3, :3] = align_rotation_to_z_axis(head_pose[:3, :3])
            aligned_head_pose[:3, 3] = head_pose[:3, 3]
            start_head_pose = aligned_head_pose

            aligned_right_arm_pose = np.eye(4)
            aligned_right_arm_pose[:3, :3] = align_rotation_to_z_axis(right_arm_pose[:3, :3])
            aligned_right_arm_pose[:3, 3] = right_arm_pose[:3, 3]
            start_right_arm_pose = aligned_right_arm_pose
            
            aligned_head_arm_pose = np.eye(4)
            aligned_head_arm_pose[:3, :3] = align_rotation_to_z_axis(head_arm_pose[:3, :3])
            aligned_head_arm_pose[:3, 3] = head_arm_pose[:3, 3]
            start_head_arm_pose = aligned_head_arm_pose

            start_right_ee_rot, start_right_ee_trans =  calibrate_controller_ee_mapping(right_pose,right_arm_pose,self.btv_quat_right)
            start_head_ee_rot, start_head_ee_trans =  calibrate_controller_ee_mapping(head_pose,head_arm_pose,self.btv_quat_head)
        
        r_rot_offset, l_rot_offset = self.update_offset_rotation(headset_data)
        #print(r_rot_offset)
        r_rot_offset_matrix = R.from_euler('xyz', [0.0, 0.0, r_rot_offset], degrees=False).as_matrix()
        l_rot_offset_matrix = R.from_euler('xyz', [0.0, 0.0, l_rot_offset], degrees=False).as_matrix()
        #  
        new_right_arm_pose = our_transform_coordinates(self.btv_quat_right, right_pose, start_right_ee_rot, start_right_ee_trans, r_rot_offset_matrix)
        new_head_arm_pose = our_transform_coordinates(self.btv_quat_head, head_pose, start_head_ee_rot, start_head_ee_trans, l_rot_offset_matrix)


        # convert to position and quaternion
        new_right_arm_pos, new_right_arm_quat = mat2pose(new_right_arm_pose)
        new_head_arm_pos, new_head_arm_quat = mat2pose(new_head_arm_pose)
        new_right_arm_quat = xyzw_to_wxyz(new_right_arm_quat)
        new_head_arm_quat = xyzw_to_wxyz(new_head_arm_quat)

        # grippers 
        new_right_gripper = np.array([headset_data.r_index_trigger])

        # concatenate the new action
        action = np.concatenate([
            new_right_arm_pos, new_right_arm_quat, new_right_gripper,
            new_head_arm_pos, new_head_arm_quat,
        ])

        # transform middle_arm_pose from mujoco coords to start_headset_pose coords
        
        if self.started:
            rightOutOfSync = not within_pose_threshold(
                right_arm_pose[:3, 3],
                right_arm_pose[:3, :3],
                new_right_arm_pose[:3, 3], 
                new_right_arm_pose[:3, :3],
                self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
                self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
            )
            headOutOfSync = not within_pose_threshold(
                head_arm_pose[:3, 3],
                head_arm_pose[:3, :3],
                new_head_arm_pose[:3, 3], 
                new_head_arm_pose[:3, :3],
                self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
                self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
            )
        else:
            rightOutOfSync = False
            headOutOfSync = False


        feedback = HeadsetFeedback()
        feedback.info = ""
        feedback.right_out_of_sync = rightOutOfSync
        feedback.head_out_of_sync = headOutOfSync

        return action, feedback

class HeadsetDualArmControl():

    def __init__(
            self,
            start_ctrl_position_threshold=0.06,
            start_ctrl_rotation_threshold=0.4,
            ctrl_position_threshold=0.04,
            ctrl_rotation_threshold=0.3,

            right_btv_quat=[0.0, 0.0, 0.0, 1.0],
            left_btv_quat=[0.0, 0.0, 0.0, 1.0],
            wxyz = True,
        ):
        self.start_right_arm_pose = None
        self.start_left_arm_pose = None
        self.started = False

        self.start_ctrl_position_threshold = start_ctrl_position_threshold
        self.start_ctrl_rotation_threshold = start_ctrl_rotation_threshold

        self.ctrl_position_threshold = ctrl_position_threshold
        self.ctrl_rotation_threshold = ctrl_rotation_threshold

        self.start_right_ee_rot = None
        self.start_right_ee_trans = None
        
        self.start_left_ee_rot = None
        self.start_left_ee_trans = None

        self.btv_quat_right = np.asarray(right_btv_quat,dtype=np.float64)
        self.btv_quat_left = np.asarray(left_btv_quat,dtype=np.float64)
        
        self.wxyz = wxyz

        self.r_thumbstick_vector = None
        self.l_thumbstick_vector = None
        
        self.r_total_rot_offset = 0.0
        self.l_total_rot_offset = 0.0

        self.r_rot_offset = 0.0
        self.l_rot_offset = 0.0

        self.r_thumbstick_count = 0
        self.l_thumbstick_count = 0


    def reset(self):
        self.start_right_arm_pose = None
        self.start_left_arm_pose = None

        self.start_right_ee_rot = None
        self.start_right_ee_trans = None
        
        self.start_left_ee_rot = None
        self.start_left_ee_trans = None

        self.started = False

    def is_running(self):
        return self.started
    
    def thumbstick_rotating(self,headset_data,):

        current_r_thumbstick_vector = np.array([headset_data.r_thumbstick_x,headset_data.r_thumbstick_y])
        current_l_thumbstick_vector = np.array([headset_data.l_thumbstick_x,headset_data.l_thumbstick_y])   
        
        if np.linalg.norm(current_r_thumbstick_vector) > 0.98:
            self.r_thumbstick_count += 1
        else:
            self.r_thumbstick_count = 0
            
        if np.linalg.norm(current_l_thumbstick_vector) > 0.98:
            self.l_thumbstick_count += 1
        else:
            self.l_thumbstick_count = 0
        
        self.r_thumbstick_count = min(self.r_thumbstick_count,5)
        self.l_thumbstick_count = min(self.l_thumbstick_count,5)
        
        return current_r_thumbstick_vector,current_l_thumbstick_vector, self.r_thumbstick_count >= 1, self.l_thumbstick_count >= 1


    def update_offset_rotation(self,headset_data):

        current_r_thumbstick_vector,current_l_thumbstick_vector, r_thumbstick_action, l_thumbstick_action = self.thumbstick_rotating(headset_data)

        if r_thumbstick_action:
            if self.r_thumbstick_vector is None:
                self.r_thumbstick_vector = current_r_thumbstick_vector


            # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
            # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
            # 归一化向量
            v1 = self.r_thumbstick_vector / np.linalg.norm(self.r_thumbstick_vector)
            v2 = current_r_thumbstick_vector / np.linalg.norm(current_r_thumbstick_vector)
            
            # 计算点积和叉积
            dot_product = np.dot(v1, v2)
            cross_product = np.cross(v1, v2)
                
            # 使用arctan2计算角度（范围[-π, π]）
            r_angle = np.arctan2(cross_product, dot_product)

            # 将当前的角度 加上之前的总offset
            r_angle += self.r_total_rot_offset

            # 将角度转换到[0, 2π]范围
            if r_angle < 0:
                r_angle += 2 * np.pi
            # 保存摇杆当前对应的旋转量
            self.r_rot_offset = r_angle
        else:
            # 在松手后，清除起始向量
            self.r_thumbstick_vector = None
            # 叠加上次的偏移量
            if self.r_rot_offset != 0.0:
                self.r_total_rot_offset = self.r_rot_offset
            self.r_rot_offset = 0.0
            r_angle = self.r_total_rot_offset


        if l_thumbstick_action:
            
            if self.l_thumbstick_vector is None:
                # 
                self.l_thumbstick_vector = current_l_thumbstick_vector

            # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
            # 计算从self.r_thumbstick_vector 旋转到 current_r_thumbstick_vector 向量之间的角度[0,2pi]
            # 归一化向量
            v1 = self.l_thumbstick_vector / np.linalg.norm(self.l_thumbstick_vector)
            v2 = current_l_thumbstick_vector / np.linalg.norm(current_l_thumbstick_vector)
            
            # 计算点积和叉积
            dot_product = np.dot(v1, v2)
            cross_product = np.cross(v1, v2)
            
            # 使用arctan2计算角度（范围[-π, π]）
            l_angle = np.arctan2(cross_product, dot_product)

            # 将当前的角度 加上之前的总offset
            l_angle += self.l_total_rot_offset

            # 将角度转换到[0, 2π]范围
            if l_angle < 0:
                l_angle += 2 * np.pi
            # 保存摇杆当前对应的旋转量
            self.l_rot_offset = l_angle
                
                
        else:
            # 在松手后，清除起始向量
            self.l_thumbstick_vector = None
            # 叠加上次的偏移量
            if self.l_rot_offset != 0.0:
                self.l_total_rot_offset = self.l_rot_offset
            self.l_rot_offset = 0.0
            l_angle = self.l_total_rot_offset

        return r_angle, l_angle


    def start(self, headset_data, right_arm_pose, left_arm_pose):
        if self.wxyz:
            right_arm_pose[3:] = wxyz_to_xyzw(right_arm_pose[3:])
            left_arm_pose[3:] = wxyz_to_xyzw(left_arm_pose[3:])

        aligned_right_pose = np.eye(4)
        aligned_right_pose[:3, :3] = quat2mat(headset_data.r_quat)
        aligned_right_pose[:3, 3] = headset_data.r_pos
        self.start_right_pose = aligned_right_pose
        
        aligned_left_pose = np.eye(4)
        aligned_left_pose[:3, :3] = quat2mat(headset_data.l_quat)
        aligned_left_pose[:3, 3] = headset_data.l_pos
        self.start_left_pose = aligned_left_pose

        
        aligned_right_arm_pose = np.eye(4)
        aligned_right_arm_pose[:3, :3] = quat2mat(right_arm_pose[3:])
        aligned_right_arm_pose[:3, 3] = right_arm_pose[:3]
        self.start_right_arm_pose = aligned_right_arm_pose
    
        aligned_left_arm_pose = np.eye(4)
        aligned_left_arm_pose[:3, :3] = quat2mat(left_arm_pose[3:])
        aligned_left_arm_pose[:3, 3] = left_arm_pose[:3]
        self.start_left_arm_pose = aligned_left_arm_pose
        
        # vtr  
        right_pose = pose2mat(headset_data.r_pos, headset_data.r_quat)
        left_pose = pose2mat(headset_data.l_pos, headset_data.l_quat)
        # rse_rot  rse_trans_base
        self.start_right_ee_rot, self.start_right_ee_trans = \
            calibrate_controller_ee_mapping(right_pose, pose2mat(right_arm_pose[:3], right_arm_pose[3:]), self.btv_quat_right)
        
        self.start_left_ee_rot, self.start_left_ee_trans = \
            calibrate_controller_ee_mapping(left_pose, pose2mat(left_arm_pose[:3], left_arm_pose[3:]), self.btv_quat_left)

        self.started = True

    def run(self, headset_data, right_arm_pose, left_arm_pose):
        if self.wxyz:
            right_arm_pose[3:] = wxyz_to_xyzw(right_arm_pose[3:])
            left_arm_pose[3:] = wxyz_to_xyzw(left_arm_pose[3:])

        # 机械臂的当前位置姿态 
        right_arm_pose = pose2mat(right_arm_pose[:3], right_arm_pose[3:])
        left_arm_pose = pose2mat(left_arm_pose[:3], left_arm_pose[3:])

        right_pose = pose2mat(headset_data.r_pos, headset_data.r_quat) # vtr
        left_pose = pose2mat(headset_data.l_pos, headset_data.l_quat) # vtl

        if self.started:
            start_right_pose = self.start_right_pose
            start_right_arm_pose = self.start_right_arm_pose
            start_right_ee_rot = self.start_right_ee_rot 
            start_right_ee_trans = self.start_right_ee_trans
            
            start_left_pose = self.start_left_pose
            start_left_arm_pose = self.start_left_arm_pose
            start_left_ee_rot = self.start_left_ee_rot 
            start_left_ee_trans = self.start_left_ee_trans

        else:
            # 不断更新初始位姿
            aligned_right_pose = np.eye(4)
            aligned_right_pose[:3, :3] = align_rotation_to_z_axis(right_pose[:3, :3])
            aligned_right_pose[:3, 3] = right_pose[:3, 3]
            start_right_pose = aligned_right_pose
            
            aligned_left_pose = np.eye(4)
            aligned_left_pose[:3, :3] = align_rotation_to_z_axis(left_pose[:3, :3])
            aligned_left_pose[:3, 3] = left_pose[:3, 3]
            start_left_pose = aligned_left_pose

            aligned_right_arm_pose = np.eye(4)
            aligned_right_arm_pose[:3, :3] = align_rotation_to_z_axis(right_arm_pose[:3, :3])
            aligned_right_arm_pose[:3, 3] = right_arm_pose[:3, 3]
            start_right_arm_pose = aligned_right_arm_pose
            
            aligned_left_arm_pose = np.eye(4)
            aligned_left_arm_pose[:3, :3] = align_rotation_to_z_axis(left_arm_pose[:3, :3])
            aligned_left_arm_pose[:3, 3] = left_arm_pose[:3, 3]
            start_left_arm_pose = aligned_left_arm_pose

            start_right_ee_rot, start_right_ee_trans =  calibrate_controller_ee_mapping(right_pose,right_arm_pose,self.btv_quat_right)
            start_left_ee_rot, start_left_ee_trans =  calibrate_controller_ee_mapping(left_pose,left_arm_pose,self.btv_quat_left)

        r_rot_offset, l_rot_offset = self.update_offset_rotation(headset_data)
        #print(r_rot_offset)
        r_rot_offset_matrix = R.from_euler('xyz', [0.0, 0.0, r_rot_offset], degrees=False).as_matrix()
        l_rot_offset_matrix = R.from_euler('xyz', [0.0, 0.0, l_rot_offset], degrees=False).as_matrix()

        #  
        new_right_arm_pose = our_transform_coordinates(self.btv_quat_right, right_pose, start_right_ee_rot, start_right_ee_trans,r_rot_offset_matrix)
        new_left_arm_pose = our_transform_coordinates(self.btv_quat_left, left_pose, start_left_ee_rot, start_left_ee_trans,l_rot_offset_matrix)

        # convert to position and quaternion
        new_right_arm_pos, new_right_arm_quat = mat2pose(new_right_arm_pose)
        new_left_arm_pos, new_left_arm_quat = mat2pose(new_left_arm_pose)
        new_right_arm_quat = xyzw_to_wxyz(new_right_arm_quat)
        new_left_arm_quat = xyzw_to_wxyz(new_left_arm_quat)

        # grippers 
        new_right_gripper = np.array([headset_data.r_index_trigger])
        new_left_gripper = np.array([headset_data.l_index_trigger])

        # concatenate the new action
        action = np.concatenate([
            new_right_arm_pos, new_right_arm_quat, new_right_gripper,
            new_left_arm_pos, new_left_arm_quat, new_left_gripper,
        ])

        # transform middle_arm_pose from mujoco coords to start_headset_pose coords
        
        if self.started:
            rightOutOfSync = not within_pose_threshold(
                right_arm_pose[:3, 3],
                right_arm_pose[:3, :3],
                new_right_arm_pose[:3, 3], 
                new_right_arm_pose[:3, :3],
                self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
                self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
            )
            leftOutOfSync = not within_pose_threshold(
                left_arm_pose[:3, 3],
                left_arm_pose[:3, :3],
                new_left_arm_pose[:3, 3], 
                new_left_arm_pose[:3, :3],
                self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
                self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
            )
        else:
            rightOutOfSync = False
            leftOutOfSync = False


        feedback = HeadsetFeedback()
        feedback.info = ""
        feedback.right_out_of_sync = rightOutOfSync
        feedback.left_out_of_sync = leftOutOfSync

        return action, feedback



class HeadsetFullControl():
    """将头显（HMD）和两个控制器（手柄）的实时位姿（位置 + 朝向）转换为控制机械臂
    （左右机械臂和中间机械臂）的动作指令，并判断当前设备是否“对齐”或“失步”。"""
    def __init__(
            self,
            start_ctrl_position_threshold=0.06,
            start_ctrl_rotation_threshold=0.4,
            start_head_position_threshold=0.03,
            start_head_rotation_threshold=0.2,
            ctrl_position_threshold=0.04,
            ctrl_rotation_threshold=0.3,
            head_position_threshold=0.05,
            head_rotation_threshold=0.3
        ):
        self.start_middle_arm_pose = None
        self.start_headset_pose = None
        self.started = False

        self.start_ctrl_position_threshold = start_ctrl_position_threshold
        self.start_ctrl_rotation_threshold = start_ctrl_rotation_threshold
        self.start_head_position_threshold = start_head_position_threshold
        self.start_head_rotation_threshold = start_head_rotation_threshold
        self.ctrl_position_threshold = ctrl_position_threshold
        self.ctrl_rotation_threshold = ctrl_rotation_threshold
        self.head_position_threshold = head_position_threshold
        self.head_rotation_threshold = head_rotation_threshold
    
    def reset(self):
        self.start_middle_arm_pose = None
        self.start_headset_pose = None
        self.started = False

    def is_running(self):
        return self.started

    def start(self, headset_data, middle_arm_pose):
        # 记录初始头显姿态{V}_T_{A}
        aligned_headset_pose = np.eye(4)
        aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(headset_data.h_quat))
        aligned_headset_pose[:3, 3] = headset_data.h_pos
        self.start_headset_pose = aligned_headset_pose
        # 记录初始时刻的视觉机械臂位置姿态
        aligned_middle_arm_pose = np.eye(4)
        aligned_middle_arm_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(wxyz_to_xyzw(middle_arm_pose[3:])))
        aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3]
        self.start_middle_arm_pose = aligned_middle_arm_pose

        self.started = True

    def run(self, headset_data, left_arm_pose, right_arm_pose, middle_arm_pose):
        """
        1. 将头显与控制器姿态从世界坐标系转换到以初始参考坐标系
        2. 输出机械臂动作向量 action。
        3. 计算是否“失步”（Out of Sync）以生成 feedback。
        """
        # 机械臂的当前位置姿态 
        middle_arm_pose = pose2mat(middle_arm_pose[:3], wxyz_to_xyzw(middle_arm_pose[3:]))
        left_arm_pose = pose2mat(left_arm_pose[:3], wxyz_to_xyzw(left_arm_pose[3:]))
        right_arm_pose = pose2mat(right_arm_pose[:3], wxyz_to_xyzw(right_arm_pose[3:]))
        # 头显的目标位置姿态 {V}_T_{H}
        headset_pose = pose2mat(headset_data.h_pos, headset_data.h_quat)
        left_pose = pose2mat(headset_data.l_pos, headset_data.l_quat)
        right_pose = pose2mat(headset_data.r_pos, headset_data.r_quat)

        if self.started:
            start_headset_pose = self.start_headset_pose
            start_middle_arm_pose = self.start_middle_arm_pose
        else:
            # 不断更新初始位姿
            aligned_headset_pose = np.eye(4)
            aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(headset_pose[:3, :3])
            aligned_headset_pose[:3, 3] = headset_pose[:3, 3]
            start_headset_pose = aligned_headset_pose

            aligned_middle_arm_pose = np.eye(4)
            aligned_middle_arm_pose[:3, :3] = align_rotation_to_z_axis(middle_arm_pose[:3, :3])
            aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3, 3]
            start_middle_arm_pose = aligned_middle_arm_pose

        # calculate offset between current and saved headset pose
        # {B}_T_{E'}                                 {V}_T_{H}      {V}_T_{A}          {B}_T_{E}
        new_middle_arm_pose = transform_coordinates(headset_pose, start_headset_pose, start_middle_arm_pose)
        new_left_arm_pose = transform_coordinates(left_pose, start_headset_pose, start_middle_arm_pose)
        new_right_arm_pose = transform_coordinates(right_pose, start_headset_pose, start_middle_arm_pose)

        # convert to position and quaternion
        new_middle_arm_pos, new_middle_arm_quat = mat2pose(new_middle_arm_pose)
        new_left_arm_pos, new_left_arm_quat = mat2pose(new_left_arm_pose)
        new_right_arm_pos, new_right_arm_quat = mat2pose(new_right_arm_pose)
        new_middle_arm_quat = xyzw_to_wxyz(new_middle_arm_quat)
        new_left_arm_quat = xyzw_to_wxyz(new_left_arm_quat)
        new_right_arm_quat = xyzw_to_wxyz(new_right_arm_quat)

        # grippers 
        new_left_gripper = np.array([headset_data.l_index_trigger])
        new_right_gripper = np.array([headset_data.r_index_trigger])

        # concatenate the new action
        action = np.concatenate([
            new_left_arm_pos, new_left_arm_quat, new_left_gripper,
            new_right_arm_pos, new_right_arm_quat, new_right_gripper,
            new_middle_arm_pos, new_middle_arm_quat
        ])

        # transform middle_arm_pose from mujoco coords to start_headset_pose coords
        # 把当前机械臂的位姿从世界坐标系转换到初始参考坐标系
        unity_middle_arm_pose = transform_coordinates(middle_arm_pose, start_middle_arm_pose, start_headset_pose)
        unity_left_arm_pose = transform_coordinates(left_arm_pose, start_middle_arm_pose, start_headset_pose)
        unity_right_arm_pose = transform_coordinates(right_arm_pose, start_middle_arm_pose, start_headset_pose)        

        # 将目标位置姿态从右手坐标系表示，转换为unity左手坐标系表示 
        unity_left_arm_pos, unity_left_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_left_arm_pose))
        unity_right_arm_pos, unity_right_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_right_arm_pose))
        unity_middle_arm_pos, unity_middle_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_middle_arm_pose))
        
        headOutOfSync = not within_pose_threshold(
            middle_arm_pose[:3, 3],
            middle_arm_pose[:3, :3],
            new_middle_arm_pose[:3, 3], 
            new_middle_arm_pose[:3, :3],
            self.head_position_threshold if self.started else self.start_head_position_threshold,
            self.head_rotation_threshold if self.started else self.start_head_rotation_threshold
        )
        leftOutOfSync = not within_pose_threshold(
            left_arm_pose[:3, 3],
            left_arm_pose[:3, :3],
            new_left_arm_pose[:3, 3], 
            new_left_arm_pose[:3, :3],
            self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
            self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
        )
        rightOutOfSync = not within_pose_threshold(
            right_arm_pose[:3, 3],
            right_arm_pose[:3, :3],
            new_right_arm_pose[:3, 3], 
            new_right_arm_pose[:3, :3],
            self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
            self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
        )

        feedback = HeadsetFeedback()
        feedback.info = ""
        feedback.head_out_of_sync = headOutOfSync
        feedback.left_out_of_sync = leftOutOfSync
        feedback.right_out_of_sync = rightOutOfSync
        feedback.left_arm_position = unity_left_arm_pos
        feedback.left_arm_rotation = unity_left_arm_quat
        feedback.right_arm_position = unity_right_arm_pos
        feedback.right_arm_rotation = unity_right_arm_quat
        feedback.middle_arm_position = unity_middle_arm_pos
        feedback.middle_arm_rotation = unity_middle_arm_quat        

        return action, feedback
 