//
// Created by xubo on 22-8-21.
//

#ifndef VIO_INIT_SYS_ROBUST_INITIALIZE_VIO_H
#define VIO_INIT_SYS_ROBUST_INITIALIZE_VIO_H

#include <string>
#include <unordered_map>
#include <glog/logging.h>
#include <pangolin/pangolin.h>

#include "utils/eigenTypes.h"
#include "utils/ticToc.h"
#include "IMU/basicTypes.hpp"
#include "IMU/imuPreintegrated.hpp"
#include "featureManager.h"
#include "initMethod/optimization.hpp"
#include "featureTracker/parameters.h"

namespace DRT {

    using namespace Eigen;
    using namespace std;

    class drtVioInit {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        drtVioInit(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &tbc);

        virtual ~drtVioInit() = default;

        virtual bool process() = 0;

        bool gravityRefine(const Eigen::MatrixXd &M,
                           const Eigen::VectorXd &m,
                           double Q,
                           double gravity_mag,
                           Eigen::VectorXd &rhs);


        // 用于标定gyr bias
        bool gyroBiasEstimator();

        bool addFeatureCheckParallax(TimeFrameId frame_id, const FeatureTrackerResulst &image, double td);


        double compensatedParallax2(const Eigen::Vector3d &p_i, const Eigen::Vector3d &p_j);

        void addImuMeasure(const vio::IMUPreintegrated &imuData);

        void recomputeFrameId();

        bool checkAccError();

        inline Eigen::Matrix3d cross_product_matrix(const Eigen::Vector3d &x) {
            Eigen::Matrix3d X;
            X << 0, -x(2), x(1),
                    x(2), 0, -x(0),
                    -x(1), x(0), 0;
            return X;
        }

        using Ptr = std::shared_ptr<drtVioInit>;
    public:

        std::set<double> local_active_frames;
        std::map<int, TimeFrameId> int_frameid2_time_frameid;
        std::map<TimeFrameId, int> time_frameid2_int_frameid;
        Eigen::aligned_map<TimeFrameId, Eigen::Matrix3d> frame_rot;

        Eigen::Vector3d biasg;
        Eigen::Vector3d biasa;
        Eigen::Vector3d gravity;
        double avg_observation;
        std::vector<Eigen::Vector3d> velocity;
        std::vector<Eigen::Vector3d> position;
        std::vector<Eigen::Matrix3d> rotation;

        Eigen::Matrix3d Rbc_;
        Eigen::Vector3d pbc_;

        Eigen::aligned_unordered_map<FeatureID, SFMFeature>
                SFMConstruct;
        Eigen::aligned_vector<vio::IMUPreintegrated> imu_meas;

        TimeFrameId last_image_t_ns;

        int frame_num_;
    };
}

#endif //VIO_INIT_SYS_ROBUST_INITIALIZE_VIO_H
