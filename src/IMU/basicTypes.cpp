/*
 * @Author: ouyangzhanpeng
 * @Date: 2022-07-16 12:37:11
 * @Description: 
 */
#include "IMU/basicTypes.hpp"

namespace vio {
// 多次运行有风险 一直累加 暂时注释掉
// uint64_t MapPoint::mp_id_ = 0;

MapPoint::MapPoint(const Eigen::Vector3d &pw) : pw_(pw){
    // id_ = mp_id_++;
}

MapPoint::MapPoint(const Eigen::Vector3d &pw, int id) : pw_(pw), id_(id){}

void MapPoint::update_observation(int frame_id, const Observation &obs) {
    obs_[frame_id] = obs;
}

std::unordered_map<int, Observation> MapPoint::filter_obs(const std::vector<int> &indices) {
    std::unordered_map<int, Observation> filtered_obs;
    for (int frame_id : indices) {
        if (obs_.find(frame_id) != obs_.end()) {
            filtered_obs.insert({frame_id, obs_[frame_id]});
        }
    }
    return filtered_obs;
}

} //namespace vio