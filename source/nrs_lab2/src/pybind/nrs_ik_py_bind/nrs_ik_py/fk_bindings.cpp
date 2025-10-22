#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <utility>
#include <Eigen/Dense>

// fk_solver.cpp 에 정의된 FKSolver 선언부(헤더 없이 사용하기 위해 동일 시그니처로 선언)
class FKSolver {
public:
    FKSolver(double tool_z, bool use_degrees);
    Eigen::Matrix4d transform(const Eigen::Matrix<double, 6, 1>& q_in) const;
    std::pair<bool, Eigen::Matrix<double, 6, 1>>
    compute(const Eigen::Matrix<double, 6, 1>& q_in, bool as_degrees = false) const;
};

namespace py = pybind11;

PYBIND11_MODULE(nrs_fk_core, m) {
    m.doc() = "Forward Kinematics (UR10/UR10e) Python bindings via pybind11";

    py::class_<FKSolver>(m, "FKSolver")
        .def(py::init<double, bool>(), py::arg("tool_z"), py::arg("use_degrees"),
             R"pbdoc(
FKSolver(tool_z, use_degrees)

Args:
  tool_z      : EE +Z 방향의 TCP 오프셋 [m] (스핀들/툴 길이)
  use_degrees : 입력 q 가 deg 단위면 True, rad 이면 False
)pbdoc")
        .def("transform", &FKSolver::transform, py::arg("q"),
             R"pbdoc(
Compute 4x4 TCP homogeneous transform.

Args:
  q : shape (6,) joint values [deg if use_degrees=True, else rad]
)pbdoc")
        .def("compute",
             &FKSolver::compute,
             py::arg("q"),
             py::arg("as_degrees") = false,
             R"pbdoc(
Compute (x, y, z, roll, pitch, yaw) with ZYX convention.

Args:
  q           : shape (6,) joint values [deg if use_degrees=True, else rad]
  as_degrees  : R/P/Y 를 deg 로 받을지 여부

Returns:
  (ok: bool, pose: ndarray shape (6,))  where pose = [x, y, z, roll, pitch, yaw]
)pbdoc");
}
