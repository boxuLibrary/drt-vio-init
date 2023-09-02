//
// Created by xubo on 23-8-30.
//

#ifndef DRT_VIO_DRTLOOSELYCOUPLED_H
#define DRT_VIO_DRTLOOSELYCOUPLED_H
#include "drtVioInit.h"
#include "utils/eigenUtils.hpp"
#include <memory>
namespace DRT {


    class drtLooselyCoupled : public drtVioInit
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        drtLooselyCoupled(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &pbc);

        virtual bool process();

        void build_LTL(Eigen::MatrixXd &LTL, Eigen::MatrixXd &A_lr);

        bool solve_LTL(const Eigen::MatrixXd &LTL, Eigen::VectorXd &evectors);

        void identify_sign(const Eigen::MatrixXd &A_lr, Eigen::VectorXd &evectors);

        void select_base_views(const Eigen::aligned_map<TimeFrameId, FeaturePerFrame> &track,
                               TimeFrameId &lbase_view_id,
                               TimeFrameId &rbase_view_id);

        bool linearAlignment();

        using Ptr = std::shared_ptr<drtLooselyCoupled>;
    };


}
#endif //DRT_VIO_DRTLOOSELYCOUPLED_H
