//
// Created by xubo on 23-8-30.
//

#ifndef DRT_VIO_DRTTIGHTLYCOUPLED_H
#define DRT_VIO_DRTTIGHTLYCOUPLED_H
#include "drtVioInit.h"
#include <memory>
namespace DRT {


    class drtTightlyCoupled : public drtVioInit
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        drtTightlyCoupled(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &pbc);

        virtual bool process();

        void select_base_views(const Eigen::aligned_map<TimeFrameId, FeaturePerFrame> &track,
                                                  TimeFrameId &lbase_view_id,
                                                  TimeFrameId &rbase_view_id);

        using Ptr = std::shared_ptr<drtTightlyCoupled>;
    };


}
#endif //DRT_VIO_DRTTIGHTLYCOUPLED_H
