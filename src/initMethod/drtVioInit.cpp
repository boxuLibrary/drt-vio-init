#include <iostream>
#include <set>


#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <chrono>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "utils/eigenUtils.hpp"

#include "initMethod/drtVioInit.h"
#include "initMethod/geometry.hpp"
#include "initMethod/polynomial.h"


namespace DRT {

    using namespace vio;

    template<typename T1, typename T2>
    void reduceVector(vector<T1> &v, vector<T2> status) {
        int j = 0;
        for (int i = 0; i < int(v.size()); i++)
            if (status[i])
                v[j++] = v[i];
        v.resize(j);
    }


// ====================   LiGT Problem ========================

    drtVioInit::drtVioInit(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &pbc): Rbc_(Rbc),pbc_(pbc)
    {

        biasa.setZero();
        biasg.setZero();
        time_frameid2_int_frameid.clear();
        int_frameid2_time_frameid.clear();
    }

    void drtVioInit::recomputeFrameId() {

        int_frameid2_time_frameid.clear();
        time_frameid2_int_frameid.clear();

        int localwindow_id = 0;
        std::string output;
        for (const auto tid: local_active_frames) {
            int_frameid2_time_frameid[localwindow_id] = tid;
            time_frameid2_int_frameid[tid] = localwindow_id;
            output +=
                    std::to_string(localwindow_id) + " ->" + std::to_string(tid) + "\n";

            localwindow_id++;
        }
        // LOG(INFO) << "WINDOW Frame: \n" << output;

    } // recomputeFrameId


    void drtVioInit::addImuMeasure(const vio::IMUPreintegrated &imuData)
    {
        imu_meas.push_back(imuData);
    }

    bool drtVioInit::addFeatureCheckParallax(TimeFrameId frame_id, const FeatureTrackerResulst &image,
                                             double td) {

        bool insert_image_frame = false;
        if (local_active_frames.size() == 0)
        {
            insert_image_frame = true;
        } else {

            double parallax_sum = 0;
            int parallax_num = 0;
            for (const auto &pts: image) {
                if (SFMConstruct.find(pts.first) != SFMConstruct.end()) {

                    Eigen::Vector3d cur_pt{pts.second[0].second(0),
                                           pts.second[0].second(1),
                                           pts.second[0].second(2)};

                    Eigen::Vector3d last_pt = SFMConstruct.at(pts.first).obs.at(last_image_t_ns).normalpoint;

                    parallax_sum += compensatedParallax2(cur_pt, last_pt);
                    ++parallax_num;
                }
            }

            if (std::abs(frame_id - last_image_t_ns) >= 0.22) {
                insert_image_frame = true;
            } else
            {
                insert_image_frame = false;
            }
        }

        if (insert_image_frame) {
            local_active_frames.insert(frame_id);

            recomputeFrameId();

            for (auto &id_pts: image) {
                // 特征点的观测，0代表左目特征
                FeaturePerFrame kpt_obs(id_pts.second[0].second, td);
                // 每个特征点的点号
                FeatureID feature_id = id_pts.first;

                if (SFMConstruct.find(feature_id) == SFMConstruct.end()) {
                    SFMConstruct[feature_id] = SFMFeature(feature_id, frame_id);
                    CHECK(frame_id != 0) << "frame_id == 0";
                    SFMConstruct[feature_id].obs[frame_id] = kpt_obs;
                } else {
                    SFMConstruct[feature_id].obs[frame_id] = kpt_obs;
                }
            }

            last_image_t_ns = frame_id;
            return true;
        } else {
            return false;
        }
    }

    bool drtVioInit::checkAccError() {

        bool check_success = false;

        for (int i = 0; i < int_frameid2_time_frameid.size() - 1; i++) {
            int j = i + 1;
            CHECK((int_frameid2_time_frameid.at(j) - int_frameid2_time_frameid.at(i)) == imu_meas[i].sum_dt_) <<
            int_frameid2_time_frameid.at(j) << " " << int_frameid2_time_frameid.at(i) << imu_meas[i].sum_dt_;
        }

        Eigen::Vector3d avgA;
        avgA.setZero();

        std::vector<int> is_bad(imu_meas.size(), 0);
        for (int i = 0; i < imu_meas.size(); i++) {
            Eigen::Vector3d acc = imu_meas[i].dV_ / imu_meas[i].sum_dt_;
            if (std::abs(acc.norm() - G.norm()) / G.norm() < 5e-3)
                is_bad[i] = 1;
            avgA += imu_meas[i].dV_ / imu_meas[i].sum_dt_;
        }

        int scoreSum = std::accumulate(is_bad.begin(), is_bad.end(), 0.0);


        avgA /= static_cast<double>(imu_meas.size());
        const double avgA_error = std::abs(avgA.norm() - G.norm()) / G.norm();

        if (avgA_error > 5e-3 and scoreSum <= 1)
            check_success = true;

        return check_success;
    }


    double drtVioInit::compensatedParallax2(const Eigen::Vector3d &p_i, const Eigen::Vector3d &p_j) {

        double ans = 0;
        double u_j = p_j(0);
        double v_j = p_j(1);

        Vector3d p_i_comp;

        //int r_i = frame_count - 2;
        //int r_j = frame_count - 1;
        //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
        p_i_comp = p_i;
        double dep_i = p_i(2);
        double u_i = p_i(0) / dep_i;
        double v_i = p_i(1) / dep_i;
        double du = u_i - u_j, dv = v_i - v_j;

        double dep_i_comp = p_i_comp(2);
        double u_i_comp = p_i_comp(0) / dep_i_comp;
        double v_i_comp = p_i_comp(1) / dep_i_comp;
        double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

        ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

        return ans;
    }


    bool drtVioInit::gravityRefine(const Eigen::MatrixXd &M,
                                   const Eigen::VectorXd &m,
                                   double Q,
                                   double gravity_mag,
                                   Eigen::VectorXd &rhs) {

        // Solve
        int q = M.rows() - 3;

        Eigen::MatrixXd A = 2. * M.block(0, 0, q, q);
        //LOG(INFO) << StringPrintf("A: %.16f", A);

        // TODO Check if A is invertible!!
        //Eigen::Matrix3d A_ = A.block<3, 3>(1, 1);
        //Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> svdA_(A_, Eigen::EigenvaluesOnly);
        //result.svA_ = svdA_.eigenvalues();
        //result.detA_ = A_.determinant();

        Eigen::MatrixXd Bt = 2. * M.block(q, 0, 3, q);
        Eigen::MatrixXd BtAi = Bt * A.inverse();

        Eigen::Matrix3d D = 2. * M.block(q, q, 3, 3);
        Eigen::Matrix3d S = D - BtAi * Bt.transpose();

        // TODO Check if S is invertible!
        //Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> svdS(S, Eigen::EigenvaluesOnly);
        //result.svS = svdS.eigenvalues();
        //result.detS = S.determinant();
        //LOG(INFO) << StringPrintf("det(S): %.16f", S.determinant());
        //LOG(INFO) << StringPrintf("eigenvalues(S): %.16f %.16f %.16f",
        //                          c[0], svd.eigenvalues()[1], svd.eigenvalues()[2]);

        Eigen::Matrix3d Sa = S.determinant() * S.inverse();
        Eigen::Matrix3d U = S.trace() * Eigen::Matrix3d::Identity() - S;

        Eigen::Vector3d v1 = BtAi * m.head(q);
        Eigen::Vector3d m2 = m.tail<3>();

        Eigen::Matrix3d X;
        Eigen::Vector3d Xm2;

        // X = I
        const double c4 = 16. * (v1.dot(v1) - 2. * v1.dot(m2) + m2.dot(m2));

        X = U;
        Xm2 = X * m2;
        const double c3 = 16. * (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

        X = 2. * Sa + U * U;
        Xm2 = X * m2;
        const double c2 = 4. * (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

        X = Sa * U + U * Sa;
        Xm2 = X * m2;
        const double c1 = 2. * (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

        X = Sa * Sa;
        Xm2 = X * m2;
        const double c0 = (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

        const double s00 = S(0, 0), s01 = S(0, 1), s02 = S(0, 2);
        const double s11 = S(1, 1), s12 = S(1, 2), s22 = S(2, 2);

        const double t1 = s00 + s11 + s22;
        const double t2 = s00 * s11 + s00 * s22 + s11 * s22
                          - std::pow(s01, 2) - std::pow(s02, 2) - std::pow(s12, 2);
        const double t3 = s00 * s11 * s22 + 2. * s01 * s02 * s12
                          - s00 * std::pow(s12, 2) - s11 * std::pow(s02, 2) - s22 * std::pow(s01, 2);

        Eigen::VectorXd coeffs(7);
        coeffs << 64.,
                64. * t1,
                16. * (std::pow(t1, 2) + 2. * t2),
                16. * (t1 * t2 + t3),
                4. * (std::pow(t2, 2) + 2. * t1 * t3),
                4. * t3 * t2,
                std::pow(t3, 2);

        const double G2i = 1. / std::pow(gravity_mag, 2);

        coeffs(2) -= c4 * G2i;
        coeffs(3) -= c3 * G2i;
        coeffs(4) -= c2 * G2i;
        coeffs(5) -= c1 * G2i;
        coeffs(6) -= c0 * G2i;

        Eigen::VectorXd real, imag;
        if (!FindPolynomialRootsCompanionMatrix(coeffs, &real, &imag)) {
            LOG(ERROR) << "Failed to find roots\n";
            printf("%.16f %.16f %.16f %.16f %.16f %.16f %.16f",
                   coeffs[0], coeffs[1], coeffs[2], coeffs[3],
                   coeffs[4], coeffs[5], coeffs[6]);

            return false;
        }

        Eigen::VectorXd lambdas = real_roots(real, imag);
        if (lambdas.size() == 0) {
            LOG(ERROR) << "No real roots found\n";
            printf("%.16f %.16f %.16f %.16f %.16f %.16f %.16f",
                   coeffs[0], coeffs[1], coeffs[2], coeffs[3],
                   coeffs[4], coeffs[5], coeffs[6]);

            return false;
        }

        Eigen::MatrixXd W(M.rows(), M.rows());
        W.setZero();
        W.block<3, 3>(q, q) = Eigen::Matrix3d::Identity();

        Eigen::VectorXd solution;
        double min_cost = std::numeric_limits<double>::max();
        for (Eigen::VectorXd::Index i = 0; i < lambdas.size(); ++i) {
            const double lambda = lambdas(i);

            Eigen::FullPivLU<Eigen::MatrixXd> lu(2. * M + 2. * lambda * W);
            Eigen::VectorXd x_ = -lu.inverse() * m;

            double cost = x_.transpose() * M * x_;
            cost += m.transpose() * x_;
            cost += Q;

            if (cost < min_cost) {
                solution = x_;
                min_cost = cost;
            }
        }


        const double constraint = solution.transpose() * W * solution;

//        if (solution[0] < 1e-3 || constraint < 0.
//            || std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag > 1e-3) { // TODO
//            LOG(WARNING) << "Discarding bad solution...\n";
//            printf("constraint: %.16f\n", constraint);
//            printf("constraint error: %.2f\n",
//                   100. * std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag);
//            return false;
//        }

        if (constraint < 0.
            || std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag > 1e-3) { // TODO
            LOG(WARNING) << "Discarding bad solution...\n";
            printf("constraint: %.16f\n", constraint);
            printf("constraint error: %.2f\n",
                   100. * std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag);
            return false;
        }

        rhs = solution;

        return true;

    }

    bool drtVioInit::gyroBiasEstimator() {

        ticToc t_optimize;

        ceres::Problem problem;

        ceres::LossFunction *loss_function;

        loss_function = new ceres::CauchyLoss(1e-5);

        std::cout << "before bg: " << biasg.transpose() << std::endl;

        int num_obs = 0;

        for (int i = 0; i < int_frameid2_time_frameid.size() - 1; i++) {

            auto target1_tid = int_frameid2_time_frameid.at(i);
            auto target2_tid = int_frameid2_time_frameid.at(i + 1);

            std::vector<Eigen::Vector3d> fis;
            std::vector<Eigen::Vector3d> fjs;

            std::vector<Eigen::Vector2d> fis_img;
            std::vector<Eigen::Vector2d> fjs_img;

            for (const auto &pts: SFMConstruct) {

                // if(pts.second.obs.size() < 3) continue;

                if (pts.second.obs.find(target1_tid) != pts.second.obs.end() &&
                    pts.second.obs.find(target2_tid) != pts.second.obs.end()) {

                    ++num_obs;

                    fis.push_back(pts.second.obs.at(target1_tid).normalpoint);
                    fjs.push_back(pts.second.obs.at(target2_tid).normalpoint);

                    fis_img.push_back(pts.second.obs.at(target1_tid).uv);
                    fjs_img.push_back(pts.second.obs.at(target2_tid).uv);
                }
            }

            vio::IMUPreintegrated imu1 = imu_meas[i];

            CHECK(imu1.start_t_ns == target1_tid) << "imu meas error" << fixed << imu1.start_t_ns << " " << target1_tid;
            CHECK(imu1.end_t_ns == target2_tid) << "imu meas error" << fixed << imu1.end_t_ns << " " << target2_tid;

            //自动求导
            ceres::CostFunction *eigensolver_cost_function = BiasSolverCostFunctor::Create(fis, fjs,
                                                                                           Eigen::Quaterniond(Rbc_),
                                                                                           imu1);
            problem.AddResidualBlock(eigensolver_cost_function, loss_function, biasg.data());

        } // end frame to frame loop



        std::cout << "number observation: " << num_obs << std::endl;

        avg_observation = num_obs / local_active_frames.size();

        if(num_obs / local_active_frames.size() < 30) {
            std::cout << "invalid number observation: " << num_obs / local_active_frames.size() << std::endl;
            // throw -1;
        }
        ceres::Solver::Options options;
        options.max_num_iterations = 200;
        // options.min_linear_solver_iterations = 10;
        options.gradient_tolerance = 1e-20;
        options.function_tolerance = 1e-20;
        options.parameter_tolerance = 1e-20;
        // options.jacobi_scaling = false;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;

        try
        {
            ceres::Solve(options, &problem, &summary);
        }
        catch(...)
        {
            return false;
        }

        if (summary.termination_type != ceres::TerminationType::CONVERGENCE) {
            std::cout << "not converge" << std::endl;
            return false;
        }
        std::cout << "after bg: " << biasg.transpose() << std::endl;

        return true;
    }


}

