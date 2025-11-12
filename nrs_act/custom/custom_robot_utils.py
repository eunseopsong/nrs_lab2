#!/home/eunseop/anaconda3/envs/env_isaaclab/bin/python
# Author: Eunseop Song (based on Chemin Ahn)
# MIT License

import numpy as np
import time
import threading

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState, Image
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from cv_bridge import CvBridge
    import cv2
    _ROS2_AVAILABLE = True
except Exception:
    _ROS2_AVAILABLE = False

from custom_constants import DT  # 제어 주기


class ImageRecorder:
    """
    Isaac Sim 카메라에서 ROS2 토픽으로 이미지 수신
    - /front_camera/rgb
    - /top_camera/rgb
    """
    def __init__(self,
                 front_topic='/front_camera/rgb',
                 top_topic='/top_camera/rgb',
                 node_name='ur10e_camera_recorder'):
        self._lock = threading.Lock()
        self._front = None
        self._top = None
        self._stop_evt = threading.Event()
        self.bridge = CvBridge()

        try:
            rclpy.init(args=None)
        except RuntimeError as e:
            if "must only be called once" not in str(e):
                raise

        self.node = Node(node_name)
        self._front_sub = self.node.create_subscription(
            Image, front_topic, self._front_cb, 10)
        self._top_sub = self.node.create_subscription(
            Image, top_topic, self._top_cb, 10)

        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        print(f"[INFO] ImageRecorder initialized:")
        print(f"  - Front : {front_topic}")
        print(f"  - Top   : {top_topic}")

    def _spin(self):
        while not self._stop_evt.is_set():
            self._exec.spin_once(timeout_sec=0.05)

    def _front_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._front = img.copy()
        except Exception as e:
            print(f"[WARN] front image error: {e}")

    def _top_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._top = img.copy()
        except Exception as e:
            print(f"[WARN] top image error: {e}")

    def get_images(self):
        with self._lock:
            return {
                'cam_front': self._front.copy() if self._front is not None else None,
                'cam_top': self._top.copy() if self._top is not None else None
            }

    def wait_for_images(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            imgs = self.get_images()
            if imgs['cam_front'] is not None and imgs['cam_top'] is not None:
                return imgs
            time.sleep(0.1)
        raise TimeoutError("Timeout waiting for camera images")

    def shutdown(self):
        self._stop_evt.set()
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._exec.shutdown()
        self.node.destroy_node()
        print("[INFO] Camera recorder shutdown complete.")


class Recorder:
    """
    UR10e Inference Controller
    - JointState: /isaac_joint_states
    - Command: /isaac_joint_commands
    """
    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]

    def __init__(self,
                 joint_state_topic='/isaac_joint_states',
                 joint_command_topic='/isaac_joint_commands',
                 node_name='act_ur10e_recorder'):
        self._lock = threading.Lock()
        self._q = None
        self._stop_evt = threading.Event()

        try:
            rclpy.init(args=None)
        except RuntimeError as e:
            if "must only be called once" not in str(e):
                raise

        self.node = Node(node_name)
        qos = QoSProfile(depth=10,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)

        self.sub = self.node.create_subscription(
            JointState, joint_state_topic, self._cb_joint, 10)
        self.pub = self.node.create_publisher(
            JointState, joint_command_topic, qos)

        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self):
        while not self._stop_evt.is_set():
            self._exec.spin_once(timeout_sec=0.05)

    def _cb_joint(self, msg):
        joint_map = dict(zip(msg.name, msg.position))
        try:
            q = [joint_map[n] for n in self.JOINT_NAMES]
            with self._lock:
                self._q = np.asarray(q, dtype=np.float64)
        except KeyError:
            pass

    def get_qpos(self, timeout=1.0):
        t0 = time.time()
        while True:
            with self._lock:
                if self._q is not None:
                    return self._q.copy()
            if time.time() - t0 > timeout:
                raise TimeoutError("Timeout waiting for joint states.")
            time.sleep(0.05)

    def set_joint_positions(self, q_target):
        try:
            q_target = np.asarray(q_target, dtype=np.float64).reshape(6)
            msg = JointState()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = self.JOINT_NAMES
            msg.position = q_target.tolist()
            self.pub.publish(msg)
        except Exception as e:
            print(f"[ERROR] set_joint_positions failed: {e}")

    def move_to_target(self, q_target,
                       kp=0.8,
                       pos_tol_deg=2,
                       steady_cycles=4,
                       max_duration=15.0,
                       delta_theta_max=np.deg2rad(5)):
        q_target = np.asarray(q_target, dtype=np.float64)
        tol = np.deg2rad(pos_tol_deg)
        ok_count = 0
        t0 = time.time()

        while True:
            q_curr = self.get_qpos()
            err = q_target - q_curr
            if np.all(np.abs(err) < tol):
                ok_count += 1
            else:
                ok_count = 0

            if ok_count >= steady_cycles:
                self.set_joint_positions(q_target)
                print("[INFO] Target reached.")
                return True

            if time.time() - t0 > max_duration:
                print("[WARN] move_to_target timeout.")
                return False

            step = np.clip(kp * err, -delta_theta_max, delta_theta_max)
            self.set_joint_positions(q_curr + step)
            time.sleep(DT)

    def shutdown(self):
        self._stop_evt.set()
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._exec.shutdown()
        self.node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("[INFO] Recorder shutdown complete.")


##################################################################################
# 테스트 실행
if __name__ == "__main__":
    rec_img = ImageRecorder()
    rec_joint = Recorder()

    def move_once():
        q = rec_joint.get_qpos(timeout=5)
        target = q.copy()
        target[5] += np.deg2rad(90)
        rec_joint.move_to_target(target,
                                 kp=0.6,
                                 pos_tol_deg=1.0,
                                 steady_cycles=3,
                                 max_duration=10)
        print("[INFO] wrist_3 rotated +90deg")

    t = threading.Thread(target=move_once, daemon=True)
    t.start()

    print("[INFO] Waiting for camera...")
    try:
        rec_img.wait_for_images(timeout=10)
        print("[INFO] Camera images ready!")
    except TimeoutError:
        print("[WARN] Camera images not ready.")

    try:
        while True:
            imgs = rec_img.get_images()
            if imgs['cam_front'] is not None:
                f = imgs['cam_front'].copy()
                cv2.imshow('front_camera', f)
            if imgs['cam_top'] is not None:
                t = imgs['cam_top'].copy()
                cv2.imshow('top_camera', t)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        rec_img.shutdown()
        rec_joint.shutdown()
        print("[INFO] UR10e Inference Recorder terminated.")
