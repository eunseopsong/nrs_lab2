import nrs_ik_core as nrs_ik

def load_txt(path, use_degrees=False):
    poses = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 6:
                continue
            x, y, z, r, p, yaw = map(float, cols[:6])
            if use_degrees:
                import math
                D2R = math.pi / 180.0
                r *= D2R
                p *= D2R
                yaw *= D2R
            pose = nrs_ik.PoseRPY()
            pose.line_no = line_no
            pose.x, pose.y, pose.z = x, y, z
            pose.r, pose.p, pose.yaw = r, p, yaw
            poses.append(pose)
    return poses


def main():
    print("============== Running IK for all poses =================")

    solver = nrs_ik.IKSolver(0.239, False)  # tool_z=0.239, use_degrees=True

    # ✅ 실제 파일 경로 반영
    txt_path = "/home/eunseop/nrs_ws/src/rtde_handarm2/data/hand_g_recording.txt"

    poses = load_txt(txt_path, use_degrees=False)
    print(f"TXT 파일 로드 성공: 총 {len(poses)} 개 pose")

    for i, pose in enumerate(poses, 1):
        ok, q = solver.compute(pose)
        if ok:
            print(f"[Pose {i}/{len(poses)}] IK 성공 → {q}")
        else:
            print(f"[Pose {i}/{len(poses)}] IK 실패")


if __name__ == "__main__":
    main()
