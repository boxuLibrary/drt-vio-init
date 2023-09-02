#include "factor/imuIntegFactor.h"
#include "utils/sophusExtUtils.hpp"

namespace vio {

/// @cite Christian Forster, Luca Carlone, Frank Dellaert, Davide Scaramuzza, “On-Manifold Preintegration for Real-Time Visual-Inertial Odometry”, in
/// IEEE Transactions on Robotics, 2016.
bool ImuIntegFactor::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    /// step 1: prepare data
    Eigen::Vector3d omegai(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d Pi(parameters[0][3], parameters[0][4], parameters[0][5]);
    Sophus::SO3d Ri = Sophus::SO3d::exp(omegai);
    Sophus::SO3d invRi = Ri.inverse();

    Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d Bgi(parameters[1][3], parameters[1][4], parameters[1][5]);
    Eigen::Vector3d Bai(parameters[1][6], parameters[1][7], parameters[1][8]);

    Eigen::Vector3d omegaj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Vector3d Pj(parameters[2][3], parameters[2][4], parameters[2][5]);
    Sophus::SO3d Rj = Sophus::SO3d::exp(omegaj);

    Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
    Eigen::Vector3d Bgj(parameters[3][3], parameters[3][4], parameters[3][5]);
    Eigen::Vector3d Baj(parameters[3][6], parameters[3][7], parameters[3][8]);

    /// step 2: compute residual
    IMUBias bias(Bgi, Bai);
    Eigen::Vector3d g(0, 0, -GRAVITY_VALUE);
    float dt = pre_integration_->sum_dt_;
    Sophus::SO3d erSO3 = pre_integration_->get_delta_rotation(bias).inverse().cast<double>() * invRi * Rj;
    Eigen::Vector3d er = erSO3.log();  // eq.(45)  er = Log(dR^{T} * Rwbi^{T} * Rwbj)
    Eigen::Vector3d ev =
        invRi * (Vj - Vi - g * dt) - pre_integration_->get_delta_velocity(bias).cast<double>();  // eq.(45)  ev = Ri^{T}*(Vj-Vi-g*dt) - dV
    Eigen::Vector3d ep = invRi * (Pj - Pi - Vi * dt - 0.5 * g * dt * dt) -
                         pre_integration_->get_delta_position(bias).cast<double>();  // eq.(45)  ep = Ri^{T}*(Pj-Pi-Vi*dt-0.5*g*dt^2) - dP
    Eigen::Vector3d ebg = Bgj - Bgi;
    Eigen::Vector3d eba = Baj - Bai;
    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
    residual << er, ev, ep, ebg, eba;
    residual = sqrt_info_ * residual;

    // std::cout <<"imu preint err: " << er.transpose() << " " << ev.transpose() << " " << ep.transpose() << std::endl;
    // std::cout <<"imu preint sqt: "<< residual.transpose() << std::endl;
    // std::cout << "bgba: " << Bgi.transpose() << " " << Bai.transpose() << std::endl;

    /// step 3: compute jacobian
    if (jacobians) {
        Eigen::Matrix3d invJr;
        Sophus::rightJacobianInvSO3(er, invJr);
        Eigen::Matrix3d invRi_matrixd = invRi.matrix();
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.setZero();
            // rotation
            jacobian_pose_i.block<3, 3>(0, 0) = -invJr * (Rj.inverse() * Ri).matrix();                               // eq.(78)  der/dphi
            jacobian_pose_i.block<3, 3>(3, 0) = Sophus::SO3d::hat(invRi * (Vj - Vi - g * dt));                       // eq.(77)  dev/dphi
            jacobian_pose_i.block<3, 3>(6, 0) = Sophus::SO3d::hat(invRi * (Pj - Pi - Vi * dt - 0.5 * g * dt * dt));  // eq.(74)  dep/dphi
            // translation
            jacobian_pose_i.block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity();  // eq.(71)  dep/dp

//             std::cout << "~~~~~~~~~~ jacobian_pose_i ~~~~~~~~~~~\n";
//             std::cout << jacobian_pose_i << std::endl;

            jacobian_pose_i = sqrt_info_ * jacobian_pose_i;

            // jacobian check
            {
//                 Eigen::Matrix<double, 9, 6> jacobian_check; jacobian_check.setZero();
//                 const double eps = 1e-6;
//                 for (size_t i = 0; i < 6; i++)
//                 {
//                     Eigen::Vector3d omegai_new(parameters[0][0], parameters[0][1], parameters[0][2]);
//                     Eigen::Vector3d Pi_new(parameters[0][3], parameters[0][4], parameters[0][5]);
//                     int a = i / 3, b = i % 3;
//                     Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
//                     if (a == 0)
//                         omegai_new = (Sophus::SO3d::exp(omegai_new) * Sophus::SO3d::exp(delta)).log();
//                     else if (a == 1)
//                         Pi_new += Sophus::SO3d::exp(omegai_new) * delta;
//
//                     Sophus::SO3d Ri_new = Sophus::SO3d::exp(omegai_new);
//                     Sophus::SO3d erSO3_new = (pre_integration_->get_delta_rotation(bias).inverse()) * Ri_new.inverse() * Rj;
//                     Eigen::Vector3d er_new = erSO3_new.log();                                                                              //
//                     // eq.(45)  er = Log(dR^{T} * Rwbi^{T} * Rwbj)
//                     Eigen::Vector3d ev_new = Ri_new.inverse()*(Vj-Vi-g*dt) - pre_integration_->get_delta_velocity(bias);
//                     // eq.(45)  ev = Ri^{T}*(Vj-Vi-g*dt) - dV
//                     Eigen::Vector3d ep_new = Ri_new.inverse()*(Pj - Pi_new - Vi*dt - 0.5*g*dt*dt) - pre_integration_->get_delta_position(bias); //
//                     // eq.(45)  ep = Ri^{T}*(Pj-Pi-Vi*dt-0.5*g*dt^2) - dP
//                     Eigen::Vector3d ebg_new = Bgj - Bgi;
//                     Eigen::Vector3d eba_new = Baj - Bai;
//
//                     jacobian_check.col(i) << (er_new-er).cast<double>()/eps,
//                                              (ev_new-ev).cast<double>()/eps,
//                                              (ep_new-ep).cast<double>()/eps;
//                 }
//                 std::cout << "check:\n";
//                 std::cout << jacobian_check << std::endl;
            }
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
            jacobian_speedbias_i.setZero();
            // velocity
            jacobian_speedbias_i.block<3, 3>(3, 0) = -invRi_matrixd;               // eq.(75) dev/dv
            jacobian_speedbias_i.block<3, 3>(6, 0) = -invRi_matrixd * (double)dt;  // eq.(73) dep/dv
            // bg
            Eigen::Vector3d dbg = (bias.bg_ - pre_integration_->bias_.bg_).cast<double>();
            Eigen::Matrix3d Jrb;
            Sophus::rightJacobianSO3(pre_integration_->JRg_.cast<double>() * dbg, Jrb);
            jacobian_speedbias_i.block<3, 3>(0, 3) =
                -invJr * erSO3.matrix().transpose() * Jrb * pre_integration_->JRg_.cast<double>();  // eq.(80)  der/dbg
            jacobian_speedbias_i.block<3, 3>(3, 3) = -(pre_integration_->JVg_).cast<double>();      // eq.()  dev/dbg
            jacobian_speedbias_i.block<3, 3>(6, 3) = -(pre_integration_->JPg_).cast<double>();      // eq.()  dep/dbg
            jacobian_speedbias_i.block<3, 3>(9, 3) = -Eigen::Matrix3d::Identity();                  //        debg/dbg
            // ba
            jacobian_speedbias_i.block<3, 3>(3, 6) = -(pre_integration_->JVa_).cast<double>();  // eq.()  dev/dba
            jacobian_speedbias_i.block<3, 3>(6, 6) = -(pre_integration_->JPa_).cast<double>();  // eq.()  dep/dba
            jacobian_speedbias_i.block<3, 3>(12, 6) = -Eigen::Matrix3d::Identity();             //        deba/ba

            // std::cout << "~~~~~~~~~~ jacobian_speedbias_i ~~~~~~~~~~~\n";
            // std::cout << jacobian_speedbias_i << std::endl;

            jacobian_speedbias_i = sqrt_info_ * jacobian_speedbias_i;

            // jacobian check
            {
//                 Eigen::Matrix<double, 15, 9> jacobian_check; jacobian_check.setZero();
//                 const double eps = 1e-2;
//                 for (size_t i = 0; i < 9; i++)
//                 {
//                     Eigen::Vector3d Vi_new(parameters[1][0], parameters[1][1], parameters[1][2]);
//                     Eigen::Vector3d Bgi_new(parameters[1][3], parameters[1][4], parameters[1][5]);
//                     Eigen::Vector3d Bai_new(parameters[1][6], parameters[1][7], parameters[1][8]);
//                     int a = i / 3, b = i % 3;
//                     Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
//                     if (a == 0)
//                         Vi_new += delta;
//                     else if (a == 1)
//                         Bgi_new += delta;
//                     else if (a == 2)
//                         Bai_new += delta;
//
//                     IMUBias bias_new(Bgi_new, Bai_new);
//                     Sophus::SO3d erSO3_new = pre_integration_->get_delta_rotation(bias_new).inverse()*invRi* Rj;
//                     Eigen::Vector3d er_new = erSO3_new.log();                                                                              //
//                     // eq.(45)  er = Log(dR^{T} * Rwbi^{T} * Rwbj)
//                     Eigen::Vector3d ev_new = invRi*(Vj-Vi_new-g*dt) - pre_integration_->get_delta_velocity(bias_new);
//                     // eq.(45)  ev = Ri^{T}*(Vj-Vi-g*dt) - dV
//                     Eigen::Vector3d ep_new = invRi*(Pj - Pi - Vi_new*dt - 0.5*g*dt*dt) - pre_integration_->get_delta_position(bias_new);
//                     // eq.(45)  ep = Ri^{T}*(Pj-Pi-Vi*dt-0.5*g*dt^2) - dP
//                     Eigen::Vector3d ebg_new = Bgj - Bgi_new;
//                     Eigen::Vector3d eba_new = Baj - Bai_new;
//
//                     jacobian_check.col(i) << (er_new-er).cast<double>()/eps,
//                                              (ev_new-ev).cast<double>()/eps,
//                                              (ep_new-ep).cast<double>()/eps,
//                                              (ebg_new-ebg).cast<double>()/eps,
//                                              (eba_new-eba).cast<double>()/eps;
//                 }
//                 std::cout << "check:\n";
//                 std::cout << jacobian_check << std::endl;
            }
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
            jacobian_pose_j.setZero();
            // rotation
            jacobian_pose_j.block<3, 3>(0, 0) = invJr;  // eq.(79)  der/dphi
            // translation
            jacobian_pose_j.block<3, 3>(6, 3) = (invRi * Rj).matrix();  // eq.(72)  dep/dp

            jacobian_pose_j = sqrt_info_ * jacobian_pose_j;
        }

        if (jacobians[3]) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
            jacobian_speedbias_j.setZero();
            // velocity
            jacobian_speedbias_j.block<3, 3>(3, 0) = invRi_matrixd;  // eq.(76)  dev/dv
            // bg
            jacobian_speedbias_j.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();  //         debg/bg
            // ba
            jacobian_speedbias_j.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();  //         deba/ba

            jacobian_speedbias_j = sqrt_info_ * jacobian_speedbias_j;
        }
    }

    return true;
}

bool ImuIntegGdirFactor::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    /// step 1: prepare data
    Eigen::Vector3d omegai(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d Pi(parameters[0][3], parameters[0][4], parameters[0][5]);
    Sophus::SO3d Ri = Sophus::SO3d::exp(omegai);
    Sophus::SO3d invRi = Ri.inverse();

    Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d Bgi(parameters[1][3], parameters[1][4], parameters[1][5]);
    Eigen::Vector3d Bai(parameters[1][6], parameters[1][7], parameters[1][8]);

    Eigen::Vector3d omegaj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Vector3d Pj(parameters[2][3], parameters[2][4], parameters[2][5]);
    Sophus::SO3d Rj = Sophus::SO3d::exp(omegaj);

    Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
    Eigen::Vector3d Bgj(parameters[3][3], parameters[3][4], parameters[3][5]);
    Eigen::Vector3d Baj(parameters[3][6], parameters[3][7], parameters[3][8]);

    Eigen::Vector3d omegaw(parameters[4][0], parameters[4][1], parameters[4][2]);
    Sophus::SO3d Rw = Sophus::SO3d::exp(omegaw);

    /// step 2: compute residual
    IMUBias bias(Bgi, Bai);
    Eigen::Vector3d g(0, 0, -GRAVITY_VALUE);
    Eigen::Vector3d g_new = Rw * g;
    float dt = pre_integration_->sum_dt_;
    Sophus::SO3d erSO3 = pre_integration_->get_delta_rotation(bias).inverse().cast<double>() * invRi * Rj;
    Eigen::Vector3d er = erSO3.log();  // eq.(45)  er = Log(dR^{T} * Rwbi^{T} * Rwbj)
    Eigen::Vector3d ev =
        invRi * (Vj - Vi - g_new * dt) - pre_integration_->get_delta_velocity(bias).cast<double>();  // eq.(45)  ev = Ri^{T}*(Vj-Vi-g*dt) - dV
    Eigen::Vector3d ep = invRi * (Pj - Pi - Vi * dt - 0.5 * g_new * dt * dt) -
                         pre_integration_->get_delta_position(bias).cast<double>();  // eq.(45)  ep = Ri^{T}*(Pj-Pi-Vi*dt-0.5*g*dt^2) - dP
    Eigen::Vector3d ebg = Bgj - Bgi;
    Eigen::Vector3d eba = Baj - Bai;
    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
    residual << er, ev, ep, ebg, eba;
    residual = sqrt_info_ * residual;

    // std::cout <<"imu preint err: " << er.transpose() << " " << ev.transpose() << " " << ep.transpose() << std::endl;
    // std::cout <<"imu preint sqt: "<< residual.transpose() << std::endl;
    // std::cout << "bgba: " << Bgi.transpose() << " " << Bai.transpose() << std::endl;

    /// step 3: compute jacobian
    if (jacobians) {
        Eigen::Matrix3d invJr;
        Sophus::rightJacobianInvSO3(er, invJr);
        Eigen::Matrix3d invRi_matrixd = invRi.matrix();
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.setZero();
            // rotation
            jacobian_pose_i.block<3, 3>(0, 0) = -invJr * (Rj.inverse() * Ri).matrix();                               // eq.(78)  der/dphi
            jacobian_pose_i.block<3, 3>(3, 0) = Sophus::SO3d::hat(invRi * (Vj - Vi - g_new * dt));                       // eq.(77)  dev/dphi
            jacobian_pose_i.block<3, 3>(6, 0) = Sophus::SO3d::hat(invRi * (Pj - Pi - Vi * dt - 0.5 * g_new * dt * dt));  // eq.(74)  dep/dphi
            // translation
            jacobian_pose_i.block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity();  // eq.(71)  dep/dp

            // std::cout << "~~~~~~~~~~ jacobian_pose_i ~~~~~~~~~~~\n";
            // std::cout << jacobian_pose_i << std::endl;

            jacobian_pose_i = sqrt_info_ * jacobian_pose_i;

            // jacobian check
            {
                // Eigen::Matrix<double, 9, 6> jacobian_check; jacobian_check.setZero();
                // const double eps = 1e-2;
                // for (size_t i = 0; i < 6; i++)
                // {
                //     Eigen::Vector3d omegai_new(parameters[0][0], parameters[0][1], parameters[0][2]);
                //     Eigen::Vector3d Pi_new(parameters[0][3], parameters[0][4], parameters[0][5]);
                //     int a = i / 3, b = i % 3;
                //     Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
                //     if (a == 0)
                //         omegai_new = (Sophus::SO3d::exp(omegai_new) * Sophus::SO3d::exp(delta)).log();
                //     else if (a == 1)
                //         Pi_new += Sophus::SO3d::exp(omegai_new) * delta;

                //     Sophus::SO3f Ri_new = Sophus::SO3f::exp(omegai_new.cast<float>());
                //     Sophus::SO3f erSO3_new = pre_integration_->get_delta_rotation(bias).inverse() * Ri_new.inverse() * Rj;
                //     Eigen::Vector3f er_new = erSO3_new.log();                                                                              //
                //     eq.(45)  er = Log(dR^{T} * Rwbi^{T} * Rwbj) Eigen::Vector3f ev_new = Ri_new.inverse()*(Vj-Vi-g*dt) -
                //     pre_integration_->get_delta_velocity(bias);                          // eq.(45)  ev = Ri^{T}*(Vj-Vi-g*dt) - dV Eigen::Vector3f
                //     ep_new = Ri_new.inverse()*(Pj - Pi_new.cast<float>() - Vi*dt - 0.5*g*dt*dt) - pre_integration_->get_delta_position(bias); //
                //     eq.(45)  ep = Ri^{T}*(Pj-Pi-Vi*dt-0.5*g*dt^2) - dP Eigen::Vector3f ebg_new = Bgj - Bgi; Eigen::Vector3f eba_new = Baj - Bai;

                //     jacobian_check.col(i) << (er_new-er).cast<double>()/eps,
                //                              (ev_new-ev).cast<double>()/eps,
                //                              (ep_new-ep).cast<double>()/eps;
                // }
                // std::cout << "check:\n";
                // std::cout << jacobian_check << std::endl;
            }
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
            jacobian_speedbias_i.setZero();
            // velocity
            jacobian_speedbias_i.block<3, 3>(3, 0) = -invRi_matrixd;               // eq.(75) dev/dv
            jacobian_speedbias_i.block<3, 3>(6, 0) = -invRi_matrixd * (double)dt;  // eq.(73) dep/dv
            // bg
            Eigen::Vector3d dbg = (bias.bg_ - pre_integration_->bias_.bg_).cast<double>();
            Eigen::Matrix3d Jrb;
            Sophus::rightJacobianSO3(pre_integration_->JRg_.cast<double>() * dbg, Jrb);
            jacobian_speedbias_i.block<3, 3>(0, 3) =
                -invJr * erSO3.matrix().transpose() * Jrb * pre_integration_->JRg_.cast<double>();  // eq.(80)  der/dbg
            jacobian_speedbias_i.block<3, 3>(3, 3) = -(pre_integration_->JVg_).cast<double>();      // eq.()  dev/dbg
            jacobian_speedbias_i.block<3, 3>(6, 3) = -(pre_integration_->JPg_).cast<double>();      // eq.()  dep/dbg
            jacobian_speedbias_i.block<3, 3>(9, 3) = -Eigen::Matrix3d::Identity();                  //        debg/dbg
            // ba
            jacobian_speedbias_i.block<3, 3>(3, 6) = -(pre_integration_->JVa_).cast<double>();  // eq.()  dev/dba
            jacobian_speedbias_i.block<3, 3>(6, 6) = -(pre_integration_->JPa_).cast<double>();  // eq.()  dep/dba
            jacobian_speedbias_i.block<3, 3>(12, 6) = -Eigen::Matrix3d::Identity();             //        deba/ba

            // std::cout << "~~~~~~~~~~ jacobian_speedbias_i ~~~~~~~~~~~\n";
            // std::cout << jacobian_speedbias_i << std::endl;

            jacobian_speedbias_i = sqrt_info_ * jacobian_speedbias_i;

            // jacobian check
            {
                // Eigen::Matrix<double, 15, 9> jacobian_check; jacobian_check.setZero();
                // const double eps = 1e-2;
                // for (size_t i = 0; i < 9; i++)
                // {
                //     Eigen::Vector3d Vi_new(parameters[1][0], parameters[1][1], parameters[1][2]);
                //     Eigen::Vector3d Bgi_new(parameters[1][3], parameters[1][4], parameters[1][5]);
                //     Eigen::Vector3d Bai_new(parameters[1][6], parameters[1][7], parameters[1][8]);
                //     int a = i / 3, b = i % 3;
                //     Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
                //     if (a == 0)
                //         Vi_new += delta;
                //     else if (a == 1)
                //         Bgi_new += delta;
                //     else if (a == 2)
                //         Bai_new += delta;

                //     IMUBias bias_new(Bgi_new.cast<float>(), Bai_new.cast<float>());
                //     Sophus::SO3f erSO3_new = pre_integration_->get_delta_rotation(bias_new).inverse()*invRi* Rj;
                //     Eigen::Vector3f er_new = erSO3_new.log();                                                                              //
                //     eq.(45)  er = Log(dR^{T} * Rwbi^{T} * Rwbj) Eigen::Vector3f ev_new = invRi*(Vj-Vi_new.cast<float>()-g*dt) -
                //     pre_integration_->get_delta_velocity(bias_new);                          // eq.(45)  ev = Ri^{T}*(Vj-Vi-g*dt) - dV
                //     Eigen::Vector3f ep_new = invRi*(Pj - Pi - Vi_new.cast<float>()*dt - 0.5*g*dt*dt) -
                //     pre_integration_->get_delta_position(bias_new);       // eq.(45)  ep = Ri^{T}*(Pj-Pi-Vi*dt-0.5*g*dt^2) - dP Eigen::Vector3f
                //     ebg_new = Bgj - Bgi_new.cast<float>(); Eigen::Vector3f eba_new = Baj - Bai_new.cast<float>();

                //     jacobian_check.col(i) << (er_new-er).cast<double>()/eps,
                //                              (ev_new-ev).cast<double>()/eps,
                //                              (ep_new-ep).cast<double>()/eps,
                //                              (ebg_new-ebg).cast<double>()/eps,
                //                              (eba_new-eba).cast<double>()/eps;
                // }
                // std::cout << "check:\n";
                // std::cout << jacobian_check << std::endl;
            }
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
            jacobian_pose_j.setZero();
            // rotation
            jacobian_pose_j.block<3, 3>(0, 0) = invJr;  // eq.(79)  der/dphi
            // translation
            jacobian_pose_j.block<3, 3>(6, 3) = (invRi * Rj).matrix();  // eq.(72)  dep/dp

            jacobian_pose_j = sqrt_info_ * jacobian_pose_j;
        }

        if (jacobians[3]) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
            jacobian_speedbias_j.setZero();
            // velocity
            jacobian_speedbias_j.block<3, 3>(3, 0) = invRi_matrixd;  // eq.(76)  dev/dv
            // bg
            jacobian_speedbias_j.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();  //         debg/bg
            // ba
            jacobian_speedbias_j.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();  //         deba/ba

            jacobian_speedbias_j = sqrt_info_ * jacobian_speedbias_j;
        }

        if(jacobians[4]){
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor> > jacobian_gdir(jacobians[4]);
            jacobian_gdir.setZero();
            // velocity
            jacobian_gdir.block<3, 3>(3, 0) = invRi.matrix() * Rw.matrix() * Sophus::SO3d::hat( g * dt);
            // translation
            jacobian_gdir.block<3, 3>(6, 0) = invRi.matrix() * Rw.matrix() * Sophus::SO3d::hat( 0.5 * g * dt * dt);

        }

    }

    return true;
}

}
