/*
 * @Author: Yijia He 
 * @Date: 2021-11-06 19:39:23 
 * @Last Modified by: Yijia He
 * @Last Modified time: 2021-11-07 21:25:55
 */
#ifndef BasicTypes_hpp
#define BasicTypes_hpp

#include <Eigen/Eigen>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include <condition_variable>
#include <deque>
#include <mutex>

#include "sophus/se3.hpp"

template<typename T>
Eigen::Matrix<T, 3, 3>  skew_matrix(const Eigen::Matrix<T, 3, 1>& a)
{
    Eigen::Matrix<T,3,3> m;
    m << 0, -a(2), a(1),
                a(2), 0, -a(0),
                -a(1), a(0),0;
    return m;
}

namespace vio {

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 15,15> Matrix15d;   // imu cov matrix
typedef Eigen::Matrix<double, 6, 1> Vector6d;
const double GRAVITY_VALUE=9.81;                  // imu acc  

// IMU biases (gyro and accelerometer)
class IMUBias {
public:
    IMUBias(): bg_(Eigen::Vector3d::Zero()), ba_(Eigen::Vector3d::Zero()){}
    IMUBias(const Eigen::Vector3d &bg, const Eigen::Vector3d &ba): bg_(bg), ba_(ba){}
    Eigen::Vector3d bg_;
    Eigen::Vector3d ba_;
};

// IMU intrinsic and extrinsic (Tbc, Tcb, IMU noise)
class IMUCalibParam
{
public:
    IMUCalibParam(const Eigen::Matrix3d& Rbc, const Eigen::Vector3d& tbc, const double ng, const double na, const double ngw, const double naw)
    : Tbc_(Rbc, tbc), Tcb_(Tbc_.inverse())
    {
        cov_.setIdentity();
        cov_walk_.setIdentity();

        const double ng2 = ng * ng;
        const double na2 = na * na;
        cov_.diagonal().head(3) *= ng2;
        cov_.diagonal().tail(3) *= na2;
        
        const double ngw2 = ngw * ngw;
        const double naw2 = naw * naw;
        cov_walk_.diagonal().head(3) *= ngw2;
        cov_walk_.diagonal().tail(3) *= naw2;

        // std::cout << cov_ << std::endl;
        // std::cout << cov_walk_ << std::endl;

        Eigen::Vector3d omega = Tbc_.so3().log();
        pose_param_[0] = omega.x();
        pose_param_[1] = omega.y();
        pose_param_[2] = omega.z();
        pose_param_[3] = Tbc_.translation().x();
        pose_param_[4] = Tbc_.translation().y();
        pose_param_[5] = Tbc_.translation().z();
    }

    Sophus::SE3d Tbc() { return Tbc_;}
    Eigen::Matrix3d Rbc() {return Tbc_.rotationMatrix();}
    Eigen::Vector3d tbc() {return Tbc_.translation();}
    double* pose_param(){ return pose_param_;}
    void updata_with_param()
    {
        Eigen::Vector3d omega(pose_param_[0], pose_param_[1], pose_param_[2]);
        Eigen::Vector3d tbc(pose_param_[3], pose_param_[4], pose_param_[5]);

        Tbc_ = Sophus::SE3d(Sophus::SO3d::exp(omega), tbc);
        Tcb_ = Tbc_.inverse();
    }

    Matrix6d cov_;         // Gauss Noise
    Matrix6d cov_walk_;    // Random Walk
    
    Sophus::SE3d Tbc_;
    Sophus::SE3d Tcb_;

    double pose_param_[6];  // ceres param: Tbc
};

class CameraPose {
  public:
    CameraPose(): Twc_(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()), Tcw_(Twc_.inverse()), vio_mode_(false) {}
    CameraPose(Eigen::Matrix3d Rwc, Eigen::Vector3d twc): vio_mode_(false) {set_pose(Rwc, twc);}
    CameraPose(IMUCalibParam* calib, Eigen::Matrix3d Rwb, Eigen::Vector3d twb, Eigen::Vector3d vel, IMUBias bias){
        imu_calib_ = calib;
        vio_mode_ = true;
        set_pose(Rwb, twb);
        set_vel_bias(vel, bias);
    }
    inline void set_pose(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
        if(!vio_mode_)
        {
            Twc_ = Sophus::SE3d(R, t);
        }else
        {
            Twb_ = Sophus::SE3d(R, t);
            Twc_ = Twb_ * imu_calib_->Tbc();  // Twb_ * Tbc;
        }

        Tcw_ = Twc_.inverse();
        Eigen::Vector3d omega = Sophus::SO3d(R).log();
        pose_param_[0] = omega.x();
        pose_param_[1] = omega.y();
        pose_param_[2] = omega.z();
        pose_param_[3] = t.x();
        pose_param_[4] = t.y();
        pose_param_[5] = t.z();
    }
    inline void set_vel_bias(const Eigen::Vector3d &vel, const IMUBias &bias)
    {
        vel_ = vel;
        bias_ = bias;
        velbgba_param_[0] = vel_.x(); velbgba_param_[1] = vel_.y(); velbgba_param_[2] = vel_.z(); 
        velbgba_param_[3] = bias_.bg_.x(); velbgba_param_[4] = bias_.bg_.y(); velbgba_param_[5] = bias_.bg_.z(); 
        velbgba_param_[6] = bias_.ba_.x(); velbgba_param_[7] = bias_.ba_.y(); velbgba_param_[8] = bias_.ba_.z(); 
    }

    inline void set_rotation(const Sophus::SO3d &Rwc) {
        if(!vio_mode_)
        {
            Twc_.so3() = Rwc;
            Tcw_ = Twc_.inverse();

            Eigen::Vector3d omega = Twc_.so3().log();
            pose_param_[0] = omega.x();
            pose_param_[1] = omega.y();
            pose_param_[2] = omega.z();
        }
    }

    void update_with_motion(const Sophus::SE3d &vel) {
        if(!vio_mode_)
        {
            Sophus::SE3d Twc = Twc_ * vel;
            set_pose(Twc.rotationMatrix(), Twc.translation());
        }
    }

    void updata_with_param()
    {
        Eigen::Vector3d omega(pose_param_[0], pose_param_[1], pose_param_[2]);
        Eigen::Vector3d t(pose_param_[3], pose_param_[4], pose_param_[5]);
        if(!vio_mode_)
        {
            Twc_ = Sophus::SE3d(Sophus::SO3d::exp(omega), t);
        }else
        {
            Twb_ = Sophus::SE3d(Sophus::SO3d::exp(omega), t);
            Twc_ = Twb_ * imu_calib_->Tbc();                        // Twb_ * Tbc;

            vel_ = Eigen::Vector3d(velbgba_param_[0], velbgba_param_[1], velbgba_param_[2]);
            bias_.bg_ = Eigen::Vector3d(velbgba_param_[3], velbgba_param_[4], velbgba_param_[5]);
            bias_.ba_ = Eigen::Vector3d(velbgba_param_[6], velbgba_param_[7], velbgba_param_[8]);
        }
        Tcw_ = Twc_.inverse();
    }

    Eigen::Matrix3d Rwc() const { return Twc_.rotationMatrix(); }
    Eigen::Matrix3d Rcw() const { return Tcw_.rotationMatrix(); }
    Eigen::Vector3d twc() const { return Twc_.translation(); }
    Eigen::Vector3d tcw() const { return Tcw_.translation(); }

    Sophus::SO3d Rwb() const { return Twb_.so3();}
    Eigen::Vector3d twb() const { return Twb_.translation();}
    Eigen::Vector3d Vwb() const { return vel_;}
    Eigen::Vector3d bias_gyro() const  { return bias_.bg_;}
    Eigen::Vector3d bias_acc() const  { return bias_.ba_;}

    double* pose_param(){ return pose_param_;}
    double* velbgba_param() {return velbgba_param_;}

    void rescale(double scale) {
        if(!vio_mode_)
        {
            Eigen::Vector3d twc_scaled = Twc_.translation() * scale;
            set_pose(Rwc(), twc_scaled);
        }
    }

    inline Eigen::Vector3d world2cam(const Eigen::Vector3d &p3d_world) {
      return Tcw_ * p3d_world;
    }

    Sophus::SE3d Twc_;
    Sophus::SE3d Tcw_;
    double pose_param_[6];      // ceres param: 1. visual mode: Twc, 2. vio mode: Twb

    // vio mode
    Sophus::SE3d Twb_;
    Eigen::Vector3d vel_;       // body velocity
    IMUBias bias_;              // gyro and acc bias
    double velbgba_param_[9];
    IMUCalibParam* imu_calib_;  // Tbc camera frame to body frame.  
    bool vio_mode_ = false;
};

struct IMUMeasurement {
public:
    IMUMeasurement(){}
    IMUMeasurement(const double t, const Eigen::Vector3d &gyro, const Eigen::Vector3d &acc):timestamp_(t), gyro_(gyro), acc_(acc){}
    double timestamp_;
    Eigen::Vector3d gyro_;
    Eigen::Vector3d acc_;
};

struct Observation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector2d uv;
    // Eigen::Vector3d bearing; //normlized
    Eigen::Vector3d feat_norm; // the third row is 1, for vins-mono

};
class MapPoint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapPoint(const Eigen::Vector3d &pw);
    MapPoint(const Eigen::Vector3d &pw, int id);
    void update_observation(int frame_id, const Observation &obs);
    std::unordered_map<int, Observation> filter_obs(const std::vector<int> &indices);

    static u_int64_t mp_id_;

    inline Eigen::Vector3d &pw() {return pw_;}
    inline std::unordered_map<int, Observation> &obs() {return obs_;}
    inline u_int64_t &id() {return id_;}

private:
    Eigen::Vector3d pw_;
    std::unordered_map<int, Observation> obs_;
    u_int64_t id_;
};

} // namespace vio

#endif