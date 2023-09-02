//
// Created by ubuntu on 2020/9/1.
//

#include "io/datasetIO.h"
#include "io/datasetIOEuroc.h"
#include "featureTracker/featureTracker.h"
#include "featureTracker/parameters.h"
#include "IMU/imuPreintegrated.hpp"
#include "initMethod/drtVioInit.h"
#include "initMethod/drtLooselyCoupled.h"
#include "initMethod/drtTightlyCoupled.h"
#include "utils/eigenUtils.hpp"
#include "utils/ticToc.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <glog/logging.h>
#include <string>

using namespace std;
using namespace cv;


int main(int argc, char **argv) {


    bool use_single_ligt = false;
    bool use_ligt_vins = false;

    if (argc != 3) {
        std::cout << "Usage: code" << " code type" << " data type"  << "\n";
        return -1;
    }

    char *codeType = argv[1];
    char *dataType = argv[2];
    std::ofstream save_file("../result/" + string(codeType) + "_" + string(dataType) + ".txt");

    auto dataset_io = struct_vio::EurocIO();
    dataset_io.read("../config/euroc.yaml");
    auto vio_dataset = dataset_io.get_data();
    readParameters("../config/euroc.yaml");
    FeatureTracker trackerData;
    trackerData.readIntrinsicParameter("../config/euroc.yaml");
    PUB_THIS_FRAME = true;
    double sf = std::sqrt(double(IMU_FREQ));

    for (int i = 0; i < vio_dataset->get_image_timestamps().size() - 100; i += 10) {

        DRT::drtVioInit::Ptr  pDrtVioInit;
        if (string(codeType) == "drtTightly")
        {
            pDrtVioInit.reset(new DRT::drtTightlyCoupled(RIC[0], TIC[0]));
        }

        if (string(codeType) == "drtLoosely")
        {
            pDrtVioInit.reset(new DRT::drtLooselyCoupled(RIC[0], TIC[0]));
        }


        std::vector<int> idx;

        // 40 4HZ, 0.25s
        for (int j = i; j < 100 + i; j += 1)
            idx.push_back(j);

        double last_img_t_s, cur_img_t_s;
        bool first_img = true;
        bool init_feature = true;


        trackerData.reset();

        std::vector<double> idx_time;
        std::vector<cv::Mat> imgs;

        for (int i: idx) {

            int64_t t_ns = vio_dataset->get_image_timestamps()[i];

            cur_img_t_s = t_ns * 1e-9;


            cv::Mat img = vio_dataset->get_image_data(t_ns)[0].image;

            trackerData.readImage(img, t_ns * 1e-9);

            for (unsigned int i = 0;; i++) {
                bool completed = false;
                completed |= trackerData.updateID(i);
                if (!completed)
                    break;
            }

            auto &un_pts = trackerData.cur_un_pts;
            auto &cur_pts = trackerData.cur_pts;
            auto &ids = trackerData.ids;
            auto &pts_velocity = trackerData.pts_velocity;

            Eigen::aligned_map<int, Eigen::aligned_vector<pair<int, Eigen::Matrix<double, 7, 1 >> >>
                    image;
            for (unsigned int i = 0; i < ids.size(); i++) {
                if (trackerData.track_cnt[i] > 1) {
                    int v = ids[i];
                    int feature_id = v / NUM_OF_CAM;
                    int camera_id = v % NUM_OF_CAM;
                    double x = un_pts[i].x;
                    double y = un_pts[i].y;
                    double z = 1;
                    double p_u = cur_pts[i].x;
                    double p_v = cur_pts[i].y;
                    double velocity_x = pts_velocity[i].x;
                    double velocity_y = pts_velocity[i].y;
                    assert(camera_id == 0);
                    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                    image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
                }
            }

            if (init_feature) {
                init_feature = false;
                continue;
            }

            if (pDrtVioInit->addFeatureCheckParallax(cur_img_t_s, image, 0.0)) {

                idx_time.push_back(cur_img_t_s);

                std::cout << "add image is: " << fixed << cur_img_t_s << " image number is: " << idx_time.size()
                          << std::endl;

                if (first_img) {
                    last_img_t_s = cur_img_t_s;
                    first_img = false;
                    continue;
                }

                auto GyroData = vio_dataset->get_gyro_data();
                auto AccelData = vio_dataset->get_accel_data();

                std::vector<MotionData> imu_segment;

                for (size_t i = 0; i < GyroData.size(); i++) {
                    double timestamp = GyroData[i].timestamp_ns * 1e-9;

                    MotionData imu_data;
                    imu_data.timestamp = timestamp;
                    imu_data.imu_acc = AccelData[i].data;
                    imu_data.imu_gyro = GyroData[i].data;

                    if (timestamp > last_img_t_s && timestamp <= cur_img_t_s) {
                        imu_segment.push_back(imu_data);
                    }
                    if (timestamp > cur_img_t_s) {
                        imu_segment.push_back(imu_data);
                        break;
                    }
                }

                vio::IMUBias bias;
                vio::IMUCalibParam
                        imu_calib(RIC[0], TIC[0], GYR_N * sf, ACC_N * sf, GYR_W / sf, ACC_W / sf);
                vio::IMUPreintegrated imu_preint(bias, &imu_calib, last_img_t_s, cur_img_t_s);

                int n = imu_segment.size() - 1;

                for (int i = 0; i < n; i++) {
                    double dt;
                    Eigen::Vector3d gyro;
                    Eigen::Vector3d acc;

                    if (i == 0 && i < (n - 1))               // [start_time, imu[0].time]
                    {
                        float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                        float tini = imu_segment[i].timestamp - last_img_t_s;
                        CHECK(tini >= 0);
                        acc = (imu_segment[i + 1].imu_acc + imu_segment[i].imu_acc -
                               (imu_segment[i + 1].imu_acc - imu_segment[i].imu_acc) * (tini / tab)) * 0.5f;
                        gyro = (imu_segment[i + 1].imu_gyro + imu_segment[i].imu_gyro -
                                (imu_segment[i + 1].imu_gyro - imu_segment[i].imu_gyro) * (tini / tab)) * 0.5f;
                        dt = imu_segment[i + 1].timestamp - last_img_t_s;
                    } else if (i < (n - 1))      // [imu[i].time, imu[i+1].time]
                    {
                        acc = (imu_segment[i].imu_acc + imu_segment[i + 1].imu_acc) * 0.5f;
                        gyro = (imu_segment[i].imu_gyro + imu_segment[i + 1].imu_gyro) * 0.5f;
                        dt = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                    } else if (i > 0 && i == n - 1) {
                        // std::cout << " n : " << i + 1 << " " << n << " " << imu_segment[i + 1].timestamp << std::endl;
                        float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                        float tend = imu_segment[i + 1].timestamp - cur_img_t_s;
                        CHECK(tend >= 0);
                        acc = (imu_segment[i].imu_acc + imu_segment[i + 1].imu_acc -
                               (imu_segment[i + 1].imu_acc - imu_segment[i].imu_acc) * (tend / tab)) * 0.5f;
                        gyro = (imu_segment[i].imu_gyro + imu_segment[i + 1].imu_gyro -
                                (imu_segment[i + 1].imu_gyro - imu_segment[i].imu_gyro) * (tend / tab)) * 0.5f;
                        dt = cur_img_t_s - imu_segment[i].timestamp;
                    } else if (i == 0 && i == (n - 1)) {
                        acc = imu_segment[i].imu_acc;
                        gyro = imu_segment[i].imu_gyro;
                        dt = cur_img_t_s - last_img_t_s;
                    }

                    CHECK(dt >= 0);
                    imu_preint.integrate_new_measurement(gyro, acc, dt);
                }
                // std::cout << fixed << "cur time: " << cur_img_t_s << " " << "last time: " << last_img_t_s << std::endl;

                pDrtVioInit->addImuMeasure(imu_preint);

                last_img_t_s = cur_img_t_s;


            }

            if (idx_time.size() >= 10) break;

            if (SHOW_TRACK) {
                cv::Mat show_img;
                cv::cvtColor(img, show_img, CV_GRAY2RGB);
                for (unsigned int j = 0; j < trackerData.cur_pts.size(); j++) {
                    double len = min(1.0, 1.0 * trackerData.track_cnt[j] / WINDOW_SIZE);
                    cv::circle(show_img, trackerData.cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                }

                cv::namedWindow("IMAGE", WINDOW_AUTOSIZE);
                cv::imshow("IMAGE", show_img);
                cv::waitKey(1);
            }

            cv::Mat show_img;
            cv::cvtColor(img, show_img, CV_GRAY2RGB);
            for (unsigned int j = 0; j < trackerData.cur_pts.size(); j++) {
                double len = min(1.0, 1.0 * trackerData.track_cnt[j] / WINDOW_SIZE);
                cv::circle(show_img, trackerData.cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
            }

            imgs.push_back(show_img);
        }

        if (idx_time.size() < 10) continue;

        bool is_good = pDrtVioInit->checkAccError();

        if (!is_good)
        {
            continue;
        }

        if ( !pDrtVioInit->process()) {
            save_file << "time: " << fixed << idx_time[0] << " other_reason" << std::endl;
            save_file << "scale_error: " << "nan" << std::endl;
            save_file << "pose_error: " << "nan" << std::endl;
            save_file << "biasg_error: " << "nan" << std::endl;
            save_file << "velo_error: " << "nan" << std::endl;
            save_file << "gravity_error: " << "nan" << " " << "nan" << std::endl;
            save_file << "v0_error: "  << "nan" << std::endl;
            save_file << "gt_vel_rot: " << "nan" << " " << "nan" << std::endl;
            LOG(INFO) << "---scale: ";
            std::cout << "time: " << fixed << idx_time[0] << std::endl;
            std::cout << "scale_error: " << 100 << std::endl;
            std::cout << "pose_error: " << 100 << std::endl;
            std::cout << "biasg_error: " << 100 << std::endl;
            std::cout << "velo_error: " << 100 << std::endl;
            std::cout << "rot_error: " << "nan" << std::endl;
            continue;
        }

        // 获取真实值
        std::vector<Eigen::Vector3d> gt_pos;
        std::vector<Eigen::Matrix3d> gt_rot;
        std::vector<Eigen::Vector3d> gt_vel;
        std::vector<Eigen::Vector3d> gt_g_imu;
        std::vector<Eigen::Vector3d> gt_angluar_vel;
        Eigen::Vector3d avgBg;
        avgBg.setZero();

        auto get_traj = [&](double timeStamp, struct_vio::GtData &rhs) -> bool {
            Eigen::map<double, struct_vio::GtData> gt_data = vio_dataset->get_gt_state_data();

            for (const auto &traj: gt_data) {
                if (std::abs((traj.first - timeStamp)) < 1e-3) {
                    rhs = traj.second;
                    return true;
                }
            }
            return false;
        };

        try {
            for (auto &t: idx_time) {
                struct_vio::GtData rhs;
                if (get_traj(t, rhs)) {
                    gt_pos.emplace_back(rhs.position);
                    gt_vel.emplace_back(rhs.velocity);
                    gt_rot.emplace_back(rhs.rotation.toRotationMatrix());
                    gt_g_imu.emplace_back(rhs.rotation.inverse() * G);

                    avgBg += rhs.bias_gyr;
                } else {
                    std::cout << "no gt pose,fail" << std::endl;
                    throw -1;
                }
            }
        } catch (...) {
            save_file << "time: " << fixed << idx_time[0] << " other_reason" << std::endl;
            save_file << "scale_error: " << "nan" << std::endl;
            save_file << "pose_error: " << "nan" << std::endl;
            save_file << "biasg_error: " << "nan" << std::endl;
            save_file << "velo_error: " << "nan" << std::endl;
            save_file << "gravity_error: " << "nan" << " " << "nan" << std::endl;
            save_file << "v0_error: " << "nan" << std::endl;
            LOG(INFO) << "---scale: ";
            std::cout << "time: " << fixed << idx_time[0] << std::endl;
            std::cout << "scale_error: " << 100 << std::endl;
            std::cout << "pose_error: " << 100 << std::endl;
            std::cout << "biasg_error: " << 100 << std::endl;
            std::cout << "velo_error: " << 100 << std::endl;
            std::cout << "rot_error: " << "nan" << std::endl;
            continue;
        }

        avgBg /= idx_time.size();

        double rot_rmse = 0;

        // rotation accuracy estimation
        for (int i = 0; i < idx_time.size() - 1; i++) {
            int j = i + 1;
            Eigen::Matrix3d rij_est = pDrtVioInit->rotation[i].transpose() * pDrtVioInit->rotation[j];
            Eigen::Matrix3d rij_gt = gt_rot[i].transpose() * gt_rot[j];
            Eigen::Quaterniond qij_est = Eigen::Quaterniond(rij_est);
            Eigen::Quaterniond qij_gt = Eigen::Quaterniond(rij_gt);
            double error =
                    std::acos(((qij_gt * qij_est.inverse()).toRotationMatrix().trace() - 1.0) / 2.0) * 180.0 / M_PI;
            rot_rmse += error * error;
        }
        rot_rmse /= (idx_time.size() - 1);
        rot_rmse = std::sqrt(rot_rmse);

        // translation accuracy estimation
        Eigen::Matrix<double, 3, Eigen::Dynamic> est_aligned_pose(3, idx_time.size());
        Eigen::Matrix<double, 3, Eigen::Dynamic> gt_aligned_pose(3, idx_time.size());

        for (int i = 0; i < idx_time.size(); i++) {
            est_aligned_pose(0, i) = pDrtVioInit->position[i](0);
            est_aligned_pose(1, i) = pDrtVioInit->position[i](1);
            est_aligned_pose(2, i) = pDrtVioInit->position[i](2);

            gt_aligned_pose(0, i) = gt_pos[i](0);
            gt_aligned_pose(1, i) = gt_pos[i](1);
            gt_aligned_pose(2, i) = gt_pos[i](2);
        }


        Eigen::Matrix4d Tts = Eigen::umeyama(est_aligned_pose, gt_aligned_pose, true);
        Eigen::Matrix3d cR = Tts.block<3, 3>(0, 0);
        Eigen::Vector3d t = Tts.block<3, 1>(0, 3);
        double s = cR.determinant();
        s = pow(s, 1.0 / 3);
        Eigen::Matrix3d R = cR / s;

        double pose_rmse = 0;
        for (int i = 0; i < idx_time.size(); i++) {
            Eigen::Vector3d target_pose = R * est_aligned_pose.col(i) + t;
            pose_rmse += (target_pose - gt_aligned_pose.col(i)).dot(target_pose - gt_aligned_pose.col(i));

        }
        pose_rmse /= idx_time.size();
        pose_rmse = std::sqrt(pose_rmse);

        std::cout << "vins sfm pose rmse: " << pose_rmse << std::endl;

        // gravity accuracy estimation
        double gravity_error =
                180. * std::acos(pDrtVioInit->gravity.normalized().dot(gt_g_imu[0].normalized())) / EIGEN_PI;

        // gyroscope bias accuracy estimation
        Eigen::Vector3d Bgs = pDrtVioInit->biasg;

        LOG(INFO) << "calculate bias: " << Bgs.x() << " " << Bgs.y() << " " << Bgs.z();
        LOG(INFO) << "gt bias: " << avgBg.x() << " " << avgBg.y() << " " << avgBg.z();

        const double scale_error = std::abs(s - 1.);
        const double gyro_bias_error = 100. * std::abs(Bgs.norm() - avgBg.norm()) / avgBg.norm();
        const double gyro_bias_error2 = 180. * std::acos(Bgs.normalized().dot(avgBg.normalized())) / EIGEN_PI;
        const double pose_error = pose_rmse;
        const double rot_error = rot_rmse;


        // velocity accuracy estimation
        double velo_norm_rmse = 0;
        double mean_velo = 0;
        for (int i = 0; i < idx_time.size(); i++) {
            velo_norm_rmse += (gt_vel[i].norm() - pDrtVioInit->velocity[i].norm()) *
                              (gt_vel[i].norm() - pDrtVioInit->velocity[i].norm());
            mean_velo += gt_vel[i].norm();
        }

        velo_norm_rmse /= idx_time.size();
        velo_norm_rmse = std::sqrt(velo_norm_rmse);
        mean_velo = mean_velo / idx_time.size();

        // the initial velocity accuracy estimation
        double v0_error = std::abs(gt_vel[0].norm() - pDrtVioInit->velocity[0].norm());

        std::cout << "integrate time: " << fixed << *idx_time.begin() << " " << *idx_time.rbegin() << " "
                  << *idx_time.rbegin() - *idx_time.begin() << std::endl;
        std::cout << "pose error: " << pose_error << " m" << std::endl;
        std::cout << "biasg error: " << gyro_bias_error << " %" << std::endl;
        std::cout << "gravity_error: " << gravity_error << std::endl;
        std::cout << "scale error: " << scale_error * 100 << " %" << std::endl;
        std::cout << "velo error: " << velo_norm_rmse << " m/s" << std::endl;
        std::cout << "v0_error: " << v0_error << std::endl;
        std::cout << "rot error: " << rot_error << std::endl;

        if (std::abs(s - 1) > 0.5 or std::abs(gravity_error) > 10) {
            LOG(INFO) << "===scale: " << s << " " << "gravity error: " << gravity_error;
            save_file << "time: " << fixed << idx_time[0] << " scale_gravity_fail" << std::endl;
            save_file << "scale_error: " << scale_error << std::endl;
            save_file << "pose_error: " << pose_error << std::endl;
            save_file << "biasg_error: " << gyro_bias_error << std::endl;
            save_file << "velo_error: " << velo_norm_rmse << std::endl;
            save_file << "gravity_error: " << gravity_error << " " << rot_error  << std::endl;
            save_file << "v0_error: " << v0_error << std::endl;
        } else {
            LOG(INFO) << "***scale: " << s << " " << "gravity error: " << gravity_error;
            save_file << "time: " << fixed << idx_time[0] << " good" << std::endl;
            save_file << "scale_error: " << scale_error << std::endl;
            save_file << "pose_error: " << pose_error << std::endl;
            save_file << "biasg_error: " << gyro_bias_error << std::endl;
            save_file << "velo_error: " << velo_norm_rmse << std::endl;
            save_file << "gravity_error: " << gravity_error << " " << rot_error << std::endl;
            save_file << "v0_error: " << v0_error << std::endl;
        }

    }

}
