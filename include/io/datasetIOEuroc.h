//
// Created by ubuntu on 2020/9/1.
//

#ifndef DATASET_IO_EUROC_H
#define DATASET_IO_EUROC_H

#include <experimental/filesystem>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <glog/logging.h>

#include "io/datasetIO.h"

namespace fs = std::experimental::filesystem;

namespace struct_vio {

    class EurocVioDataset : public VioDataset {
    public:

        size_t num_cams;

        std::string dataset_base_path;
        std::vector<std::string> image_folder;
        std::string imu_path;

        std::vector<int64_t> image_timestamps;
        std::unordered_map<int64_t, std::string> image_path;

        // vector of images for every timestamp
        // assumes vectors size is num_cams for every timestamp with null pointers for
        // missing frames
        // std::unordered_map<int64_t, std::vector<ImageData>> image_data;

        Eigen::vector<AccelData> accel_data;
        Eigen::vector<GyroData> gyro_data;

        std::vector<int64_t> gt_timestamps;       // ordered gt timestamps
        Eigen::map<double, GtData> gt_data;            // TODO: change to eigen aligned


    public:
        ~EurocVioDataset() {};

        size_t get_num_cams() const { return num_cams; }

        std::vector<int64_t> &get_image_timestamps() { return image_timestamps; }

        const Eigen::vector<AccelData> &get_accel_data() const { return accel_data; }

        const Eigen::vector<GyroData> &get_gyro_data() const { return gyro_data; }

        const std::vector<int64_t> &get_gt_timestamps() const {
            return gt_timestamps;
        }

        const Eigen::map<double, GtData> &get_gt_state_data() const {
            return gt_data;
        }

        std::vector<ImageData> get_image_data(int64_t t_ns) {
            std::vector<ImageData> res(num_cams);

            for (size_t i = 0; i < num_cams; i++) {
                std::string full_image_path =
                        dataset_base_path + image_folder[i] + "/data/" + image_path[t_ns];

                if (file_exists(full_image_path)) {
                    cv::Mat image = cv::imread(full_image_path, -1);
                    // LOG(INFO) << image.rows << "," << image.cols;
                    res[i].image = image;
                }
            }

            return res;
        } // funcation get_image_data
    };

    class EurocIO : public DatasetIoInterface {
    public:
        EurocIO() {};

        void read(const std::string &config_path) {

            cv::FileStorage fsSettings(config_path, cv::FileStorage::READ);
            if (!fsSettings.isOpened()) {
                LOG(ERROR) << "ERROR: Wrong path to settings";
            }

            std::string data_path = fsSettings["data_path"];

            LOG(INFO) << "data_path: " << data_path << std::endl;

            if (!fs::exists(data_path)) {
                LOG(ERROR) << "No dataset found in " << data_path << std::endl;
            }

            int camera_num = fsSettings["camera_num"];

            std::vector<std::string> image_names;

            for (const auto &node: fsSettings["camera_name"]) {
                image_names.emplace_back(node);
            }
            LOG(INFO) << "image_names size: " << image_names.size();

            std::string imu_path = fsSettings["imu_path"];

            data.reset(new EurocVioDataset);

            data->dataset_base_path = data_path;
            data->num_cams = camera_num;
            data->image_folder = image_names;

            // 读取IMU以及camera的数据路径
            read_image_timestamps(data_path + image_names[0] + "/");
            read_imu_data(data_path + imu_path + "/");

            // TODO: 读取真值
            bool has_ground_gtruth = int(fsSettings["has_ground_truth"]);
            if (has_ground_gtruth) {
                std::string ground_truth_path = fsSettings["ground_truth_path"];
                read_gt_data_state(data_path + ground_truth_path + "/");

            }
        }

        void reset() { data.reset(); }

        VioDatasetPtr get_data() { return data; }

    private:
        void read_image_timestamps(const std::string &path) {

            std::ifstream f(path + "data.csv");
            if (!f) {
                LOG(INFO) << "fail to open the file: " << path + "data.csv";
            }

            std::string line;
            while (std::getline(f, line)) {
                if (line[0] == '#')
                    continue;

                std::stringstream ss(line);

                char tmp;
                int64_t t_ns;
                std::string path;

                ss >> t_ns >> tmp >> path;

                data->image_timestamps.emplace_back(t_ns);
                data->image_path[t_ns] = path;

                // LOG(INFO) << t_ns << "-->" << path;
            }
        } // function read_image_timestamps

        void read_imu_data(const std::string &path) {
            data->accel_data.clear();
            data->gyro_data.clear();

            std::ifstream f(path + "data.csv");
            if (!f) {
                LOG(INFO) << "fail to open the file: " << path + "data.csv";
            }

            std::string line;
            while (std::getline(f, line)) {
                if (line[0] == '#')
                    continue;

                std::stringstream ss(line);

                char tmp;
                uint64_t timestamp;
                Eigen::Vector3d gyro, accel;

                ss >> timestamp >> tmp >> gyro[0] >> tmp >> gyro[1] >> tmp >> gyro[2] >>
                   tmp >> accel[0] >> tmp >> accel[1] >> tmp >> accel[2];

                data->accel_data.emplace_back();
                data->accel_data.back().timestamp_ns = timestamp;
                data->accel_data.back().data = accel;

                data->gyro_data.emplace_back();
                data->gyro_data.back().timestamp_ns = timestamp;
                data->gyro_data.back().data = gyro;


//                LOG(INFO) << accel.transpose() << "---" << gyro.transpose();
//                char tmp1;
//                std::cin >> tmp1;
            }
        }

        void read_gt_data_state(const std::string &path) {
            data->gt_timestamps.clear();
            data->gt_data.clear();

            std::ifstream f(path + "data.csv");
            if (!f) {
                LOG(INFO) << "fail to open the file: " << path + "data.csv";
            }

            std::string line;
            while (std::getline(f, line)) {
                if (line[0] == '#')
                    continue;

                std::stringstream ss(line);

                char tmp;
                uint64_t timestamp;
                Eigen::Quaterniond q;
                Eigen::Vector3d pos, vel, accel_bias, gyro_bias;

//                std::cout << ss.str() << std::endl;

                ss >> timestamp >> tmp >> pos[0] >> tmp >> pos[1] >> tmp >> pos[2] >>
                   tmp >> q.w() >> tmp >> q.x() >> tmp >> q.y() >> tmp >> q.z() >> tmp >>
                   vel[0] >> tmp >> vel[1] >> tmp >> vel[2] >> tmp >> gyro_bias[0] >>
                   tmp >> gyro_bias[1] >> tmp >> gyro_bias[2] >> tmp >> accel_bias[0] >>
                   tmp >> accel_bias[1] >> tmp >> accel_bias[2];

//                std::cout << "gt: " << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
//                std::cout << "gt: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
//                std::cout << "gt: " << vel[0] << " " << vel[1] << " " << vel[2] << std::endl;
//                std::cout << "gt: " << gyro_bias[0] << " " << gyro_bias[1] << " " << gyro_bias[2] << std::endl;
//                std::cout << "gt: " << accel_bias[0] << " " << accel_bias[1] << " " << accel_bias[2] << std::endl;
//
//                char tmp1;
//                std::cin >> tmp1;

                data->gt_timestamps.emplace_back(timestamp);

                GtData gt;
                gt.rotation = q;
                gt.position = pos;
                gt.velocity = vel;
                gt.bias_gyr = gyro_bias;
                gt.bias_acc = accel_bias;
                data->gt_data[timestamp * 1e-9] = gt;
            }
        }


        std::shared_ptr<EurocVioDataset> data;

    };

} // namespace struct_vio

#endif // DATASET_IO_EUROC_H
