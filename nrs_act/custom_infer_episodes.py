#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from tqdm import tqdm
from act.policy import Policy
from act.utils import set_seed, load_checkpoint
from act.dataset import ActDataset
from act.detr.models.detr_vae import build as build_act_model
from act.detr.util.misc import NestedTensor, is_main_process

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


# ================================================================
# Isaac Sim Inference Runner for ACT Policy
# ================================================================
class ACTInferenceNode(Node):
    def __init__(self, ckpt_dir, task_name):
        super().__init__('act_inference_node')

        # --------------------------
        # 1) Load Model
        # --------------------------
        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model architecture
        class Args:
            hidden_dim = 512
            dim_feedforward = 3200
            nheads = 8
            enc_layers = 4
            dropout = 0.1
            pre_norm = False
            num_queries = 100
            camera_names = ['cam_front', 'cam_head']
        args = Args()

        self.model = build_act_model(args)
        self.policy = Policy(self.model, device=device)
        self.policy.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.policy.eval()
        print(f"[INFO] Loaded policy from {ckpt_path}")

        # --------------------------
        # 2) ROS2 Setup
        # --------------------------
        self.joint_sub = self.create_subscription(
            JointState, '/isaac_joint_states', self.joint_callback, 10
        )

        self.joint_pub = self.create_publisher(
            JointState, '/isaac_joint_command', 10
        )

        self.latest_joint = np.zeros(6)
        self.received_joint = False
        self.timer = self.create_timer(0.05, self.control_loop)  # 20Hz

        print(f"[INFO] ACT Inference Node initialized for task: {task_name}")

    # --------------------------
    # JointState callback
    # --------------------------
    def joint_callback(self, msg):
        joint_map = dict(zip(msg.name, msg.position))
        joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint",
            "elbow_joint", "wrist_1_joint",
            "wrist_2_joint", "wrist_3_joint"
        ]
        try:
            q = [joint_map[n] for n in joint_names]
            self.latest_joint = np.array(q)
            self.received_joint = True
        except KeyError:
            pass

    # --------------------------
    # Control loop (inference)
    # --------------------------
    def control_loop(self):
        if not self.received_joint:
            return

        with torch.no_grad():
            qpos = torch.tensor(self.latest_joint, dtype=torch.float32).unsqueeze(0)
            # Dummy placeholders for image inputs
            dummy_image = torch.zeros((1, 2, 3, 224, 224))
            dummy_env = torch.zeros((1, 7))

            action_hat, _, _ = self.model(qpos, dummy_image, dummy_env)
            action = action_hat.squeeze(0).cpu().numpy()

        # Publish as joint command
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            "shoulder_pan_joint", "shoulder_lift_joint",
            "elbow_joint", "wrist_1_joint",
            "wrist_2_joint", "wrist_3_joint"
        ]
        joint_msg.position = action.tolist()
        self.joint_pub.publish(joint_msg)

# ================================================================
# Main
# ================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    args = parser.parse_args()

    rclpy.init()
    node = ACTInferenceNode(args.ckpt_dir, args.task_name)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
