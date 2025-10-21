from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "nrs_ik_core",   # ✅ 모듈 이름은 core
        ["nrs_ik_py/ik_bindings.cpp", "nrs_ik_py/ik_solver.cpp", "nrs_ik_py/Kinematics.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/usr/include/eigen3",
            "nrs_ik_py"
        ],
        language="c++"
    )
]

setup(
    name="nrs_ik_py",
    version="0.1",
    packages=["nrs_ik_py"],  # ✅ Python 패키지
    ext_modules=ext_modules,
)
