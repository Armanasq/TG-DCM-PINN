# Implementation audit: fallow24/ros_imu_calib

**Revision:** 918a19236cecd6a367921e44ac73958ab9fbb9c8
**Repository:** https://github.com/fallow24/ros_imu_calib/tree/918a19236cecd6a367921e44ac73958ab9fbb9c8
**License/provenance:** no root license in tree; inspected C++ source/header contains BSD-style notice (2014 Pretto; 2025 Arzberger)
**Method:** remotely inspected source files at pinned commit. Local clone/build/tests **not** executed.

## Inspected architecture and methodology

C++ Eigen/Ceres multi-position accelerometer and gyroscope calibration utilities; static-interval detection and residual optimization, intrinsic scale/bias/misalignment; used as paper-linked source context for motion acceleration compensation.

## Actual source files inspected

- [src/calibration.cpp](https://github.com/fallow24/ros_imu_calib/blob/918a19236cecd6a367921e44ac73958ab9fbb9c8/src/calibration.cpp)
- [src/filters.cpp](https://github.com/fallow24/ros_imu_calib/blob/918a19236cecd6a367921e44ac73958ab9fbb9c8/src/filters.cpp)
- [include/imu_tk/calibration.h](https://github.com/fallow24/ros_imu_calib/blob/918a19236cecd6a367921e44ac73958ab9fbb9c8/include/imu_tk/calibration.h)

## Experimental and evidence caveats

The audited C++ files include inherited imu_tk toolkit code; do not attribute all routines solely to the 2026 paper or assume all paper compensation experiments reproduced.

Do not conflate code availability with reproduced research results. See [architecture review](../review/01_architectures.md) and [full synthesis](../review/08_literature_review.md).
