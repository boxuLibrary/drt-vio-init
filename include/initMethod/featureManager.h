//
// Created by xubo on 23-8-30.
//

#ifndef DERVIO_FEATUREMANAGER_H
#define DERVIO_FEATUREMANAGER_H

#include "utils/eigenTypes.h"
using namespace Eigen;
using FeatureID = int;
using TimeFrameId = double;
using FeatureTrackerResulst = Eigen::aligned_map<
        int, Eigen::aligned_vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;

struct MotionData {
    double timestamp;
    Eigen::Vector3d imu_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d imu_gyro = Eigen::Vector3d::Zero();
};

class FeaturePerFrame {
public:
    FeaturePerFrame() = default;

    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td) {
        normalpoint.x() = _point(0);
        normalpoint.y() = _point(1);
        normalpoint.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        cur_td = td;
    }

    double cur_td;
    Vector3d normalpoint;
    Vector2d uv;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

class SFMFeature {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SFMFeature() = default;

    SFMFeature(FeatureID _feature_id, TimeFrameId _start_frame)
            : feature_id(_feature_id), kf_id(_start_frame) {}

    FeatureID feature_id{};
    TimeFrameId kf_id{};

    bool state = false; // sfm使用：被三角化: true, 未被三角化：false
    Vector3d p3d; // sfm使用：世界坐标系下的3D位置
    // TODO: 观测值
    Eigen::aligned_map<TimeFrameId, FeaturePerFrame> obs;
};
#endif //DERVIO_FEATUREMANAGER_H
