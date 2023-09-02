//
// Created by ubuntu on 2020/9/1.
//

#ifndef DATASET_IO_H
#define DATASET_IO_H

#include <array>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "utils/sophus_utils.hpp"

namespace struct_vio {

    inline bool file_exists(const std::string &name) {
        std::ifstream f(name.c_str());

        return f.good();
    }

    using TimeCamId = std::pair<int64_t, size_t>;

    struct ImageData {
        cv::Mat image;
    };

    struct GyroData {
        int64_t timestamp_ns;
        Eigen::Vector3d data;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct AccelData {
        int64_t timestamp_ns;
        Eigen::Vector3d data;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct PoseData {
        int64_t timestamp_ns;
        Sophus::SE3d data;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct GtData {
        int64_t timestamp_ns;
        Eigen::Vector3d position;
        Eigen::Quaterniond rotation;
        Eigen::Vector3d velocity;
        Eigen::Vector3d bias_gyr;
        Eigen::Vector3d bias_acc;
    };

    class VioDataset {
    public:

        virtual ~VioDataset() {};

        virtual size_t get_num_cams() const = 0;

        virtual std::vector<int64_t> &get_image_timestamps() = 0;

        virtual const Eigen::vector<AccelData> &get_accel_data() const = 0;

        virtual const Eigen::vector<GyroData> &get_gyro_data() const = 0;

        virtual const std::vector<int64_t> &get_gt_timestamps() const = 0;

        // virtual const Eigen::vector<Sophus::SE3d> &get_gt_pose_data() const = 0;

        virtual const Eigen::map<double, GtData> &get_gt_state_data() const = 0;

        virtual std::vector<ImageData> get_image_data(int64_t t_ns) = 0;

    };

    typedef std::shared_ptr<VioDataset> VioDatasetPtr;

    class DatasetIoInterface {
    public:
        virtual void read(const std::string &path) = 0;

        virtual void reset() = 0;

        virtual VioDatasetPtr get_data() = 0;

        virtual ~DatasetIoInterface() {};
    };

    typedef std::shared_ptr<DatasetIoInterface> DatasetIoInterfacePtr;

    class DatasetIoFactory {
    public:
        static DatasetIoInterfacePtr getDatasetIo(const std::string &dataset_type,
                                                  bool load_mocap_as_gt = false);
    };


} // namespace struct_vio

#endif // DATASET_IO_H
