#include <Eigen/Dense>
#include <cmath>
#include <utility>
#include "Kinematics.h"
#include "Arm_class.h"

// =============== 헬퍼 함수들 ===============
static inline Eigen::Matrix<double, 6, 1>
to_rad(const Eigen::Matrix<double, 6, 1>& q_in, bool use_degrees) {
    if (!use_degrees) return q_in;
    constexpr double D2R = M_PI / 180.0;
    Eigen::Matrix<double, 6, 1> q = q_in;
    for (int i = 0; i < 6; ++i) q(i) *= D2R;
    return q;
}

// ZYX 기준 RPY 추출 (네 Kinematics::Rotation2RPY와 동일 수식)
static inline Eigen::Vector3d rpy_from_R_zyx(const Eigen::Matrix3d& R) {
    Eigen::Vector3d rpy;
    const double yaw = std::atan2(R(1,0), R(0,0));              // Z
    const double cy  = std::cos(yaw),  sy = std::sin(yaw);
    const double pitch = std::atan2(-R(2,0), R(0,0)*cy + R(1,0)*sy); // Y
    const double roll  = std::atan2(-(R(1,2)*cy + R(0,2)*sy),
                                    (R(1,1)*cy - R(0,1)*sy));       // X
    rpy << roll, pitch, yaw;
    return rpy;
}

// =============== FKSolver 본체 ===============
class FKSolver {
public:
    FKSolver(double tool_z, bool use_degrees)
        : tool_z_(tool_z), use_degrees_(use_degrees) {}

    // 4x4 TCP 변환행렬
    Eigen::Matrix4d transform(const Eigen::Matrix<double, 6, 1>& q_in) const {
        const auto qrad = to_rad(q_in, use_degrees_);

        // Eigen::VectorXd 로 변환 (Kinematics 시그니처 맞추기)
        Eigen::VectorXd q(6);
        for (int i = 0; i < 6; ++i) q(i) = qrad(i);

        // 1) EE 포즈
        Kinematic_func kin;
        Eigen::Matrix4d T_ee = Eigen::Matrix4d::Identity();
        kin.iForwardK_T(q, T_ee, /*endlength*/ 0.0);

        // 2) EE→TCP 오프셋(+Z 방향 tool_z)
        Eigen::Matrix4d EE2TCP = Eigen::Matrix4d::Identity();
        EE2TCP(2,3) = tool_z_;

        // 최종 TCP 포즈
        return T_ee * EE2TCP;
    }

    // (x,y,z, roll, pitch, yaw) 반환
    std::pair<bool, Eigen::Matrix<double, 6, 1>>
    compute(const Eigen::Matrix<double, 6, 1>& q_in, bool as_degrees = false) const {
        const Eigen::Matrix4d T = transform(q_in);

        if (!T.allFinite()) {
            return {false, Eigen::Matrix<double, 6, 1>::Zero()};
        }

        const Eigen::Vector3d p = T.block<3,1>(0,3);
        const Eigen::Matrix3d R = T.block<3,3>(0,0);
        Eigen::Vector3d rpy = rpy_from_R_zyx(R);

        Eigen::Matrix<double, 6, 1> out;
        if (as_degrees) {
            constexpr double R2D = 180.0 / M_PI;
            out << p(0), p(1), p(2), rpy(0)*R2D, rpy(1)*R2D, rpy(2)*R2D;
        } else {
            out << p(0), p(1), p(2), rpy(0), rpy(1), rpy(2);
        }
        return {true, out};
    }

private:
    double tool_z_;
    bool   use_degrees_;
};
