#!/home/vision/anaconda3/envs/aloha/bin/python

# Author: Chemin Ahn (chemx3937@gmail.com)
# Use of this source code is governed by the MIT, see LICENSE

# Isaac Sim Camera + Dual Arm Control
# Camera Topics:
# 1. cam_front: /isaac_camera_front/rgb/image_raw
# 2. cam_head: /isaac_camera_head/rgb/image_raw

import numpy as np
import time
import threading

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState, Image
    from rclpy.logging import get_logger
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from cv_bridge import CvBridge
    import cv2
    _ROS2_AVAILABLE = True

except Exception:
    _ROS2_AVAILABLE = False

# Inference시
from custom_constants import DT

# custom_robot_utils.py만 실행시
# from custom_constants import DT


class ImageRecorder:
    """
    Isaac Sim 카메라에서 ROS2 토픽으로 이미지 수신
    - cam_front: /isaac_camera_front/rgb/image_raw
    - cam_head: /isaac_camera_head/rgb/image_raw
    """
    def __init__(self, 
                 front_topic: str = '/isaac_camera_front/rgb/image_raw',
                 head_topic: str = '/isaac_camera_head/rgb/image_raw',
                 node_name: str = 'isaac_camera_recorder'):
        
        self._lock = threading.Lock()
        self._front_image = None
        self._head_image = None
        self._stop_evt = threading.Event()
        
        # CV Bridge for ROS Image conversion
        self.bridge = CvBridge()
        
        # ROS2 초기화
        try:
            rclpy.init(args=None)
        except RuntimeError as e:
            if "must only be called once" in str(e):
                pass
            else:
                raise
                
        self.node = Node(node_name)
        
        # Image subscribers
        self._front_sub = self.node.create_subscription(
            Image, front_topic, self._front_image_callback, 10
        )
        self._head_sub = self.node.create_subscription(
            Image, head_topic, self._head_image_callback, 10
        )
        
        # Executor for spinning
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        
        print(f"[INFO] ImageRecorder initialized with topics:")
        print(f"  - Front camera: {front_topic}")
        print(f"  - Head camera: {head_topic}")

    def _spin(self):
        try:
            while not self._stop_evt.is_set():
                self._exec.spin_once(timeout_sec=0.05)
        except Exception as e:
            self.node.get_logger().error(f'Image recorder spin error: {e}')

    def _front_image_callback(self, msg: Image):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._front_image = cv_image.copy()
        except Exception as e:
            self.node.get_logger().error(f'Front image conversion error: {e}')

    def _head_image_callback(self, msg: Image):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._head_image = cv_image.copy()
        except Exception as e:
            self.node.get_logger().error(f'Head image conversion error: {e}')

    def get_images(self) -> dict:
        """
        현재 캐시된 이미지들을 반환
        Returns:
            dict: {'cam_front': np.ndarray, 'cam_head': np.ndarray}
        """
        with self._lock:
            return {
                'cam_front': self._front_image.copy() if self._front_image is not None else None,
                'cam_head': self._head_image.copy() if self._head_image is not None else None
            }

    def wait_for_images(self, timeout: float = 5.0) -> dict:
        """
        모든 카메라에서 이미지가 준비될 때까지 대기
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            images = self.get_images()
            if images['cam_front'] is not None and images['cam_head'] is not None:
                return images
            time.sleep(0.1)
        
        raise TimeoutError("Timeout waiting for camera images")

    def shutdown(self):
        try:
            self._stop_evt.set()
            if self._spin_thread.is_alive():
                self._spin_thread.join(timeout=2.0)
            if self._exec is not None:
                self._exec.shutdown()
            if self.node is not None:
                self.node.destroy_node()
        except Exception as e:
            print(f"ImageRecorder shutdown error: {e}")


class Recorder:
    """
    Dual-arm version (L6 + R6).
    - 현재 조인트값 읽기: /isaac_joint_states 구독 (left_joint_1..6, right_joint_1..6)
    - 특정 조인트로 이동: /isaac_joint_command 퍼블리시 (듀얼암 동시)
    """
    JOINT_NAMES_LEFT  = [f'left_joint_{i}'  for i in range(1, 7)]
    JOINT_NAMES_RIGHT = [f'right_joint_{i}' for i in range(1, 7)]

    def __init__(self,
                 joint_state_topic: str = '/isaac_joint_states',
                 joint_command_topic: str = '/isaac_joint_command',
                 node_name: str = 'act_dualarm_recorder'):

        # 내부 상태
        self._lock = threading.Lock()
        self._left = None   # np.ndarray (6,) in radians
        self._right = None  # np.ndarray (6,) in radians
        self._stop_evt = threading.Event()

        # ROS2 초기화 & 노드/통신자원
        try:
            rclpy.init(args=None)
        except RuntimeError as e:
            if "must only be called once" in str(e):
                pass  # Already initialized, continue
            else:
                raise  # Re-raise if it's a different error
        
        self.node = Node(node_name)

        # QoS 설정 (servo_isaacsim_ultimate.py와 동일)
        qos_profile = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # 현재 qpos 받기
        # subscriber: /isaac_joint_states
        self._sub = self.node.create_subscription(
            JointState, joint_state_topic, self._joint_cb, 10
        )

        # Action값으로 Joint 이동
        # publisher: /isaac_joint_command (듀얼암 동시 퍼블리시)
        self.joint_pub = self.node.create_publisher(
            JointState, joint_command_topic, qos_profile
        )

        # spin: executor 스레드
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    # ---------- ROS2 spin ----------
    def _spin(self):
        try:
            while not self._stop_evt.is_set():
                self._exec.spin_once(timeout_sec=0.05)
        except Exception as e:
            self.node.get_logger().error(f'Spin error: {e}')

    # ---------- /isaac_joint_states callback ----------
    def _joint_cb(self, msg: JointState):
        """
        servo_isaacsim_ultimate.py의 joint_state_callback과 동일한 방식:
        msg.name/msg.position에서 우리가 원하는 12개만 순서대로 추출.
        """
        joint_map = dict(zip(msg.name, msg.position))

        # 원하는 순서로 재배열
        all_names = self.JOINT_NAMES_LEFT + self.JOINT_NAMES_RIGHT
        try:
            positions = [joint_map[n] for n in all_names]
            L = np.asarray(positions[:6], dtype=np.float64)
            R = np.asarray(positions[6:], dtype=np.float64)

            with self._lock:
                self._left = L
                self._right = R
        except KeyError as e:
            # Joint가 모두 준비되지 않은 경우
            pass

    # ---------- API: 현재 qpos 읽기 ----------
    def get_qpos(self, block: bool = True, timeout: float = 1.0) -> np.ndarray:
        """
        현재 듀얼암 조인트(rad)를 (12,)로 반환. 순서: [L1..L6, R1..R6]
        block=True면 timeout까지 대기.
        """
        t0 = time.time()
        while True:
            with self._lock:
                if self._left is not None and self._right is not None:
                    curr_qpos = np.concatenate([self._left, self._right], axis=0)
                    curr_L, curr_R = curr_qpos[:6], curr_qpos[6:]
                    print(f"[DEBUG] L: {np.round(curr_L, 3)}  \n R: {np.round(curr_R, 3)}")
                    return curr_qpos
            if not block:
                raise RuntimeError("Joint states not ready yet (non-blocking).")
            if time.time() - t0 > timeout:
                raise TimeoutError("Timeout waiting for /isaac_joint_states.")
            time.sleep(0.05)

    def set_joint_positions(self, joint12_rad: np.ndarray) -> bool:
        """
        servo_isaacsim_ultimate.py의 publish_joint_command와 동일한 방식으로
        ACT Policy 출력 action(12,)을 받아 /isaac_joint_command로 퍼블리시.
        joint12_rad = [L1..L6, R1..R6] (rad)
        """
        try:
            joint12_rad = np.asarray(joint12_rad, dtype=np.float64).reshape(12)

            # servo_isaacsim_ultimate.py와 동일한 메시지 구성
            joint_msg = JointState()
            joint_msg.header.stamp = self.node.get_clock().now().to_msg()
            joint_msg.name = [
                'left_joint_1', 'left_joint_2', 'left_joint_3',
                'left_joint_4', 'left_joint_5', 'left_joint_6',
                'right_joint_1', 'right_joint_2', 'right_joint_3',
                'right_joint_4', 'right_joint_5', 'right_joint_6'
            ]
            joint_msg.position = joint12_rad.tolist()
            joint_msg.velocity = []
            joint_msg.effort = []

            # /isaac_joint_command로 퍼블리시
            self.joint_pub.publish(joint_msg)
            return True

        except Exception as e:
            self.node.get_logger().error(f"set_joint_positions failed: {e}")
            return False

    # reset 시킬때만 Joint 움직이게 하는 메소드
    def move_arm(self,
                target_pose,
                arm: str = 'both',
                kp: float = 1.3,
                pos_tol_deg: float = 3,
                steady_cycles: int = 5,
                max_duration: float = 20.0,
                delta_theta_max: float = np.deg2rad(4.5)):
        """
        목표 조인트(target_pose)에 '도달할 때까지' 반복 제어 (라디안).
        - arm='left'  : target_pose.shape == (6,)
        - arm='right' : target_pose.shape == (6,)
        - arm='both'  : target_pose.shape == (12,)
        - kp: P 이득 (0~1 권장)
        - pos_tol_deg: 도달 판정 각도 허용오차(도) (기본 3° ≈ 0.05236 rad)
        - steady_cycles: 연속 N회 오차 이하일 때 도달로 판정 (노이즈 억제)
        - max_duration: 최대 동작 시간(sec)
        - delta_theta_max: 1주기당 관절 변화 최대값(rad) (급격한 점프 방지)
        """
        target_pose = np.asarray(target_pose, dtype=np.float64)
        tol = np.deg2rad(pos_tol_deg)

        # 12차 목표 벡터 구성
        curr_all = self.get_qpos(timeout=1.0)  # (12,)
        if arm == 'left':
            if target_pose.shape != (6,):
                raise ValueError("arm='left'일 때 target_pose는 (6,) 이어야 합니다.")
            target12 = np.concatenate([target_pose, curr_all[6:]], axis=0)
            mask = np.r_[np.ones(6, dtype=bool), np.zeros(6, dtype=bool)]
        elif arm == 'right':
            if target_pose.shape != (6,):
                raise ValueError("arm='right'일 때 target_pose는 (6,) 이어야 합니다.")
            target12 = np.concatenate([curr_all[:6], target_pose], axis=0)
            mask = np.r_[np.zeros(6, dtype=bool), np.ones(6, dtype=bool)]
        elif arm == 'both':
            if target_pose.shape != (12,):
                raise ValueError("arm='both'일 때 target_pose는 (12,) 이어야 합니다.")
            target12 = target_pose
            mask = np.ones(12, dtype=bool)
        else:
            raise ValueError("arm must be one of {'left','right','both'}")

        t0 = time.time()
        ok_count = 0

        while True:
            # 1) 현재 상태 읽기
            curr = self.get_qpos(timeout=1.0)  # (12,)

            # 2) 선택된 관절의 오차 계산
            err = (target12 - curr)
            sel_err = err[mask]

            # 3) 도달 판정(연속 steady_cycles회)
            if np.all(np.abs(sel_err) <= tol):
                ok_count += 1
            else:
                ok_count = 0

            if ok_count >= steady_cycles:
                # 마지막으로 목표값 한 번 더 보내 안정화 (선택)
                self.set_joint_positions(target12)
                return True  # 성공

            # 4) 시간 초과 방지
            if time.time() - t0 > max_duration:
                # 필요시 로깅만 하고 False 반환
                self.node.get_logger().warn("move_arm: max_duration 초과, 중단합니다.")
                return False

            # 5) P-제어 + Δθ 클리핑
            step = kp * err
            # 선택 안 된 관절은 움직이지 않도록 0으로
            step[~mask] = 0.0
            # per-joint change 제한
            step = np.clip(step, -delta_theta_max, delta_theta_max)
            cmd = curr + step

            # 6) 명령 전송 (/isaac_joint_command로 퍼블리시)
            self.set_joint_positions(cmd)

            time.sleep(DT)
            
    # ---------- 종료 ----------
    def shutdown(self):
        try:
            self._stop_evt.set()
            if self._spin_thread.is_alive():
                self._spin_thread.join(timeout=2.0)
            if self._exec is not None:
                self._exec.shutdown()
            if self.node is not None:
                self.node.destroy_node()
        finally:
            try:
                rclpy.shutdown()
            except Exception:
                pass

##################################################################################

# Isaac Sim 카메라 + 듀얼암 테스트: 실시간 미리보기 + J6 동시 90도 회전 (1회)
if __name__ == "__main__":
    import cv2

    # 1) Isaac Sim 카메라 열기
    image_recorder = ImageRecorder(
        front_topic='/isaac_camera_front/rgb/image_raw',
        head_topic='/isaac_camera_head/rgb/image_raw'
    )

    # 2) 듀얼암 Recorder 시작 (servo_isaacsim_ultimate.py와 동일한 토픽 사용)
    recorder = Recorder(
        joint_state_topic='/isaac_joint_states',
        joint_command_topic='/isaac_joint_command',
        node_name='act_dualarm_recorder_main'
    )

    # 3) 시작 직후 J6(왼/오) 90도 동시 회전 (백그라운드 1회)
    def rotate_both_j6_once():
        try:
            q = recorder.get_qpos(timeout=5.0)     # (12,) rad
            target = q.copy()
            target[5]  += np.deg2rad(90.0)         # left_joint_6
            target[11] += np.deg2rad(90.0)         # right_joint_6
            # 목표에 도달할 때까지 수렴 이동
            recorder.move_arm(target, arm='both', kp=0.6, pos_tol_deg=0.8,
                               steady_cycles=3, max_duration=10.0,
                               delta_theta_max=np.deg2rad(5.0))
            print("[INFO] Both J6 rotated by +90 deg.")
        except Exception as e:
            print(f"[WARN] Rotate J6 failed: {e}")

    t = threading.Thread(target=rotate_both_j6_once, daemon=True)
    t.start()

    # 4) 실시간 미리보기 루프
    try:
        print("[INFO] Waiting for camera images...")
        
        # 첫 이미지가 올 때까지 대기
        try:
            image_recorder.wait_for_images(timeout=10.0)
            print("[INFO] Camera images ready!")
        except TimeoutError:
            print("[WARN] Camera images not ready, continuing anyway...")
        
        while True:
            images = image_recorder.get_images()
            cam_front = images['cam_front']
            cam_head  = images['cam_head']

            if cam_front is not None:
                frame_f = cam_front.copy()
                cv2.putText(frame_f, "cam_front (Isaac Sim)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow('cam_front (Isaac Sim)', frame_f)

            if cam_head is not None:
                frame_h = cam_head.copy()
                cv2.putText(frame_h, "cam_head (Isaac Sim)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow('cam_head (Isaac Sim)', frame_h)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                # 필요 시 언제든 다시 90도 회전 재시도
                threading.Thread(target=rotate_both_j6_once, daemon=True).start()
            if key == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        image_recorder.shutdown()
        recorder.shutdown()
        print("[INFO] Shutdown complete.")

