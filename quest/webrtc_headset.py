import json
import asyncio
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCRtpSender
)
from aiortc import VideoStreamTrack
from av import VideoFrame
import numpy as np
import queue
import time
from quest.headset_utils import HeadsetData, HeadsetFeedback, convert_left_to_right_coordinates
from quest.headset_control import HeadsetOurControl
import os
import threading
import requests
import math

def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )

class BufferVideoStreamTrack(VideoStreamTrack):
    def __init__(self, buffer_size=1, image_format="rgb24", max_fps=60):
        super().__init__()
        self.queue = queue.Queue(maxsize=buffer_size)
        self.image_format = image_format
        self.last_frame = None
        self.max_fps = max_fps
        self.last_send_time = time.time()


    async def get_frame(self) -> np.ndarray:
        while True:
            try:
                frame = self.queue.get_nowait()
                self.last_frame = frame
                return frame
            except queue.Empty:
                if self.last_frame is not None:
                    return self.last_frame.copy()
                await asyncio.sleep(0)
        

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = await self.get_frame()
        # convert to gray scale
        frame = VideoFrame.from_ndarray(frame, format=self.image_format)
        frame.pts = pts
        frame.time_base = time_base

        # limit fps
        elapsed_time = time.time() - self.last_send_time
        await asyncio.sleep(max(1/self.max_fps - elapsed_time, 0))
        self.last_send_time = time.time()

        return frame

    def add_frame(self, frame):
        # try to put but if pull pop the oldest frame
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass

class WebRTCHeadset:
    def __init__(
        self,
        serviceAccountKeyFile='serviceAccountKey.json',
        signalingSettingsFile=None,
        video_buffer_size=1,
        data_buffer_size=1,
        send_data_freq=10,
    ):    
        """
        初始化 WebRTCHeadset 实例。

        参数:
            serviceAccountKeyFile: Firebase 服务账号密钥 JSON 文件路径。
            signalingSettingsFile: 包含信令设置的 JSON 文件路径。
            video_buffer_size: 视频帧缓冲区大小。
            data_buffer_size: 数据消息缓冲区大小。
            send_data_freq: 数据发送频率（Hz）。
        """
            
        # create firestore client
        # with open(serviceAccountKeyFile) as f:
        #     serviceAccountKey = json.load(f)
        # self.db = firestore.Client.from_service_account_info(serviceAccountKey)
        if signalingSettingsFile is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            signalingSettingsFile = os.path.join(current_dir, 'signalingSettings.json')
        # load signaling settings
        with open(signalingSettingsFile) as f:
            signalingSettings = json.load(f)
        self.robotId = signalingSettings['robotID']
        self.password = signalingSettings['password']
        self.signal_url = signalingSettings["signal_server_url"]      # ← 新增字段
        self.turn_server_url = signalingSettings['turn_server_url']
        self.turn_server_username = signalingSettings['turn_server_username']
        self.turn_server_password = signalingSettings['turn_server_password']

        # create peer connection
        self.pc = RTCPeerConnection()

        # vars for video and data
        self.channel = None
        self.left_video_track = None
        self.right_video_track = None
        self.video_buffer_size = video_buffer_size
        self.data_buffer_size = data_buffer_size
        self.send_data_freq = send_data_freq
        self.receive_data_queue = queue.Queue(maxsize=data_buffer_size)
        self.send_data_queue = queue.Queue(maxsize=data_buffer_size)
        self.receive_last_data = None

        self.thread = None
        self.event_loop = None
        self._restarting = False

    async def channel_send_loop(self):
        """
        循环发送数据：
        从 send_data_queue 中取出最近的状态反馈，通过 WebRTC data channel 发送给控制端。
        发送频率由 send_data_freq 控制。
        """
        last_data = None
        while True:
            start_time = time.time()
            
            try:
                if self.channel is not None and self.channel.readyState == "open":
                    data = self.send_data_queue.get_nowait()
                    data = json.dumps(data)
                    last_data = data
                    self.channel.send(data)
            except Exception as e:
                try:
                    if last_data is not None:
                        self.channel.send(last_data)
                except Exception as e:
                    print(f"Failed to send data: {e}")

            elapsed_time = time.time() - start_time
            await asyncio.sleep(max(1/self.send_data_freq - elapsed_time, 0))

    def receive_data(self) -> HeadsetData:
        """
        从接收缓冲区获取从控制端发来的最新一帧头显数据。
        如果当前没有数据，则返回 None。
        """
        return self.receive_last_data
        # try:
        #     data = self.receive_data_queue.get_nowait()
        #     return data
        # except queue.Empty:
        #     return None
    
    def send_images(self, left_image: np.ndarray, right_image: np.ndarray):
        """
        将左右眼图像帧发送到控制端，用于视频显示。
        left_image / right_image 为 numpy 格式图像。
        """
        try:
            if self.left_video_track is not None:
                self.left_video_track.add_frame(left_image) #TODO copy image?
            if self.right_video_track is not None:
                self.right_video_track.add_frame(right_image) #TODO copy image?
        except Exception as e:
            print(f"Failed to send image: {e}")

    def send_feedback(self, data: HeadsetFeedback):
        """
        将本地机器人状态数据（如同步标志、手臂位置、旋转等）发送给控制端。
        通常每帧由机器人端生成，供控制端用于状态回显或同步控制。
        """
        data = {
            'headOutOfSync': data.head_out_of_sync,
            'leftOutOfSync': data.left_out_of_sync,
            'rightOutOfSync': data.right_out_of_sync,
            'info': data.info,
            'leftArmPosition': data.left_arm_position.tolist(),
            'leftArmRotation': data.left_arm_rotation.tolist(),
            'rightArmPosition': data.right_arm_position.tolist(),
            'rightArmRotation': data.right_arm_rotation.tolist(),
            'middleArmPosition': data.middle_arm_position.tolist(),
            'middleArmRotation': data.middle_arm_rotation.tolist(),
        }
        try:
            self.send_data_queue.put_nowait(data)
        except queue.Full:
            try:
                self.send_data_queue.get_nowait()
                self.send_data_queue.put_nowait(data)
            except (queue.Empty, queue.Full):
                pass

    def on_message(self, message):
        """
        当控制端通过 data channel 发送 JSON 消息时触发。
        将数据解析为 HeadsetData，并放入接收队列供主线程读取。
        """
        try:
            headset_data = HeadsetData()
            data = json.loads(message)
        except json.JSONDecodeError:
            print("WebRTC: JSON decode error")
            return

        try:
            headset_data.h_pos[0] = data['HPosition']['x']
            headset_data.h_pos[1] = data['HPosition']['y']
            headset_data.h_pos[2] = data['HPosition']['z']
            headset_data.h_quat[0] = data['HRotation']['x']
            headset_data.h_quat[1] = data['HRotation']['y']
            headset_data.h_quat[2] = data['HRotation']['z']
            headset_data.h_quat[3] = data['HRotation']['w']
            headset_data.l_pos[0] = data['LPosition']['x']
            headset_data.l_pos[1] = data['LPosition']['y']
            headset_data.l_pos[2] = data['LPosition']['z']
            headset_data.l_quat[0] = data['LRotation']['x']
            headset_data.l_quat[1] = data['LRotation']['y']
            headset_data.l_quat[2] = data['LRotation']['z']
            headset_data.l_quat[3] = data['LRotation']['w']
            headset_data.l_thumbstick_x = data['LThumbstick']['x']
            headset_data.l_thumbstick_y = data['LThumbstick']['y']
            headset_data.l_index_trigger = data['LIndexTrigger']
            headset_data.l_hand_trigger = data['LHandTrigger']
            headset_data.l_button_one = data['LButtonOne']
            headset_data.l_button_two = data['LButtonTwo']
            headset_data.l_button_thumbstick = data['LButtonThumbstick']
            headset_data.r_pos[0] = data['RPosition']['x']
            headset_data.r_pos[1] = data['RPosition']['y']
            headset_data.r_pos[2] = data['RPosition']['z']
            headset_data.r_quat[0] = data['RRotation']['x']
            headset_data.r_quat[1] = data['RRotation']['y']
            headset_data.r_quat[2] = data['RRotation']['z']
            headset_data.r_quat[3] = data['RRotation']['w']
            headset_data.r_thumbstick_x = data['RThumbstick']['x']
            headset_data.r_thumbstick_y = data['RThumbstick']['y']
            headset_data.r_index_trigger = data['RIndexTrigger']
            headset_data.r_hand_trigger = data['RHandTrigger']
            headset_data.r_button_one = data['RButtonOne']
            headset_data.r_button_two = data['RButtonTwo']
            headset_data.r_button_thumbstick = data['RButtonThumbstick']
            headset_data.h_pos, headset_data.h_quat = convert_left_to_right_coordinates(headset_data.h_pos, headset_data.h_quat)
            headset_data.l_pos, headset_data.l_quat = convert_left_to_right_coordinates(headset_data.l_pos, headset_data.l_quat)
            headset_data.r_pos, headset_data.r_quat = convert_left_to_right_coordinates(headset_data.r_pos, headset_data.r_quat)
        except KeyError:
            print("[RobotWebRTC] Key error") 
            return

        self.receive_last_data = headset_data

        try:
            self.receive_data_queue.put_nowait(headset_data)
        except queue.Full:
            try:
                self.receive_data_queue.get_nowait()
                self.receive_data_queue.put_nowait(headset_data)
            except (queue.Empty, queue.Full):
                pass

    async def run_offer(self):

        # create data channel
        self.channel = self.pc.createDataChannel("control")
        @self.channel.on("open")
        def on_open():
            print("Data channel is open.")
        self.channel.on("message", self.on_message)       

        # # create video track
        # self.left_video_track = BufferVideoStreamTrack(buffer_size=self.video_buffer_size)
        # self.left_video_sender = self.pc.addTrack(self.left_video_track)
        # force_codec(self.pc, self.left_video_sender, 'video/VP8')

        # # create video track
        # self.right_video_track = BufferVideoStreamTrack(buffer_size=self.video_buffer_size)
        # self.right_video_sender = self.pc.addTrack(self.right_video_track)
        # force_codec(self.pc, self.right_video_sender, 'video/VP8')


        # create offer and place in firestore     
        print("WebRTC: Running offer...")  
        await self.pc.setLocalDescription(await self.pc.createOffer())
        print("WebRTC: Offer created, uploading…")

        # === 上传 offer 到 信令服务器 ===
        offer_payload = {
            "robot_id": self.robotId,
            "type": "offer",
            "sdp": self.pc.localDescription.sdp
        }
        requests.post(f"{self.signal_url}/webrtc/register_offer",
                      json=offer_payload, timeout=3)

        # === 4. 轮询等待 answer ===
        print("WebRTC: waiting answer…")
        while True:
            try:
                r = requests.get(f"{self.signal_url}/webrtc/answer",
                                params={"robot_id": self.robotId}, timeout=3)
                if r.status_code == 200:
                    answer = r.json()
                    if answer.get("type") == "answer":
                        break
            except requests.RequestException as e:
                print(f"signal poll err: {e}")
            await asyncio.sleep(0.5)

        print("WebRTC: Answer received.")
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"],
                                  type=answer["type"])
        )


        # add event listener for connection close
        # @self.pc.on("iceconnectionstatechange")
        # async def on_iceconnectionstatechange():
        #     if self.pc.iceConnectionState == "closed":
        #         print("WebRTC: Connection closed, restarting...")
        #         await self.restart_connection()
                
        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            state = self.pc.iceConnectionState
            print(f"[ICE STATE] -> {state}")
            if state in ("disconnected", "failed", "closed", ) and not self._restarting:
                self._restarting = True
                await self.restart_connection()
                self._restarting = False

    async def restart_connection(self):
        """
        当连接关闭时，重新初始化 PeerConnection 并重新建立连接（再次运行 offer 流程）。
        """
        # close current peer connection
        await self.pc.close()

        # create new peer connection
        self.pc = RTCPeerConnection()

        # run offer again
        await self.run_offer() 

    def run_in_thread(self):
        """
        在单独线程中运行 WebRTC 协议（用于异步事件循环与主线程并行执行）。
        """
        def run(loop: asyncio.AbstractEventLoop):  
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run_offer())  
            loop.create_task(self.channel_send_loop())
            loop.run_forever()

        self.event_loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=run, args=(self.event_loop,))
        self.thread.start()

    def close(self):
        """
        停止事件循环并安全关闭线程，终止 WebRTC 会话。
        """
        if self.thread is not None and self.thread.is_alive():
            self.event_loop.stop()
            self.thread.join()

def print_headset_data_status(data: HeadsetData):
    
    # 设置 numpy 打印格式，保留 3 位小数，去掉科学计数法
    np.set_printoptions(precision=3, suppress=True)
    
    # 清屏 (Windows用cls，Linux/Mac用clear)，让输出像仪表盘一样刷新
    # print("\033c", end="")  # 这种清屏方式在某些 IDE 里可能不生效，可以直接用 print("\n"*50)
    
    print("\n" + "="*30 + " VR DATA MONITOR " + "="*30)
    
    # 1. 头显状态
    print(f"{'[HEADSET]':<15} Pos: {data.h_pos} | Quat: {data.h_quat}")
    
    # 2. 左手柄状态
    print(f"{'[LEFT CTRL]':<15} Pos: {data.l_pos} | Quat: {data.l_quat}")
    print(f"{'':<15} Stick: ({data.l_thumbstick_x:.2f}, {data.l_thumbstick_y:.2f}) | Trigger: {data.l_index_trigger:.2f}")
    btn_l = []
    if data.l_button_one: btn_l.append("A")
    if data.l_button_two: btn_l.append("B")
    if data.l_button_thumbstick: btn_l.append("Stick")
    print(f"{'':<15} Buttons Pressed: {btn_l if btn_l else 'None'}")

    # 3. 右手柄状态 (重点测试对象)
    print(f"{'[RIGHT CTRL]':<15} Pos: {data.r_pos} | Quat: {data.r_quat}")
    print(f"{'':<15} Stick: ({data.r_thumbstick_x:.2f}, {data.r_thumbstick_y:.2f}) | Trigger: {data.r_index_trigger:.2f}")
    
    btn_r = []
    if data.r_button_one: btn_r.append("A")
    if data.r_button_two: btn_r.append("B")
    if data.r_button_thumbstick: btn_r.append("Stick")
    print(f"{'':<15} Buttons Pressed: {btn_r if btn_r else 'None'}")
    
    print("="*78)

if __name__ == "__main__":
    
    # 关键参数：确认你的 VR 端发来的是 Unity 标准四元数 (x,y,z,w)
    # 如果是，这里一定要设为 False，否则初始化会出错
    ctrl = HeadsetOurControl() 
    
    # 虚拟机器人的当前状态 (格式: [x, y, z, qx, qy, qz, qw])
    # 假设初始位置在右前方，无旋转
    # 注意：因为 wxyz=False，我们这里用 xyzw 格式存储，和控制器输入保持一致
    init_right_pose = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]) 
    init_middle_pose = np.array([0.0, 0.0, 1.6, 0.0, 0.0, 0.0, 1.0])
    
    current_right_arm_pose = init_right_pose.copy()
    current_middle_arm_pose = init_middle_pose.copy()
    
    is_calibrated = False

    #  这里模拟一个虚拟机器人，接收头显控制数据，并给头显发送图像信息等
    try:
        headset = WebRTCHeadset()
        headset.run_in_thread()
        print("System Ready. Waiting for VR data...")
        print("Press 'A' button (r_button_one) on the RIGHT controller to CALIBRATE.")
        
        while True:
            
            # 接收从“头显控制端”发送过来的用户数据（位置、角度等）
            data = headset.receive_data()
            if data is not None:
                # print_headset_data_status(data)
                if not is_calibrated and data.r_button_one:
                    print("\n>>> START CALIBRATION...")
                    # 调用 start，传入初始时刻的 VR 数据 和 虚拟机器人当前位姿
                    # 注意：HeadsetOurControl 内部会处理坐标系记录
                    ctrl.start(data, current_right_arm_pose, current_middle_arm_pose)
                    is_calibrated = True
                    print(">>> CALIBRATION DONE! You can now move the controller.")
                    continue # 这一帧只做标定，下一帧开始控制

                if is_calibrated:
                    # 计算目标动作
                    # 输入：当前VR数据 + 虚拟机器人当前位姿
                    # action, feedback = ctrl.run(data, current_right_arm_pose, current_middle_arm_pose)
                    action = ctrl.run(data)
                    
                    # 解析 Action
                    # action 结构: [右臂pos(3), 右臂quat(4), 右臂gripper(1), 中臂pos(3), 中臂quat(4)]
                    target_right_pos = action[0:3]
                    target_right_quat = action[3:7] # 注意：run() 输出的四元数是 wxyz (根据代码逻辑)
                    target_gripper = action[7]
                    target_middle_pos = action[8:11]
                    target_middle_quat = action[11:15]

                    # --- 验证逻辑 (调试用) ---
                    # 打印看看：VR手柄移动量 vs 机器人目标移动量
                    # print(f"Target Right Pos: {target_right_pos}")

                    # 模拟机器人运动 (闭环)
                    # 让“虚拟机器人”的当前位姿 直接等于 目标位姿
                    # 这样下一帧 ctrl.run 就会基于这个新位姿计算
                    current_right_arm_pose = np.concatenate([target_right_pos, target_right_quat])
                    current_middle_arm_pose = np.concatenate([target_middle_pos, target_middle_quat])

                     # --- 处理失步反馈 ---
                    # if feedback.right_out_of_sync:
                    #     print("Warning: Right Arm Out of Sync!")

                # feedback_msg = HeadsetFeedback()
                # if is_calibrated:
                #     feedback_msg.info = f"Calibrated. Gripper: {action[7]:.2f}"
                # else:
                #     feedback_msg.info = "Press 'A' to Calibrate"
                
                # headset.send_feedback(feedback_msg)
                
                # d = math.sqrt(data.r_thumbstick_x**2 + data.r_thumbstick_y**2)
                # print(f"Received data: {round(d,3)}, {data.r_thumbstick_x}, {data.r_thumbstick_y}")

            # feedback = HeadsetFeedback()
            # # 显示在头显视野中的一段信息字符
            # feedback.info = f"Hello from python: {time.time()}"
            # headset.send_feedback(feedback)

            # headset.send_images(left_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            #                     right_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            
            time.sleep(1/120)

    except KeyboardInterrupt:
        print("Shutting down...")
        os._exit(42)
