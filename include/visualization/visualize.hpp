
#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <pangolin/pangolin.h>
#include <Eigen/Geometry>
#include <iostream>
namespace visualize {


void DrawCurrentPose(pangolin::OpenGlMatrix &Twc);

void DrawKeyFrames(const std::vector<pangolin::OpenGlMatrix>& poses);

void GetCurrentOpenGLPoseMatrix(pangolin::OpenGlMatrix &M, const Eigen::Isometry3d &Tbw);

void GetSwOpenGLPoseMatrices(std::vector<pangolin::OpenGlMatrix>& vPoses_SW, const std::vector<Eigen::Isometry3d>& swPoses);

void DrawSlideWindow(const std::vector<pangolin::OpenGlMatrix>& vPoses_SW);

void DrawActiveMapPoints(const std::map<int,Eigen::Vector3d>& mMapPoints);

void DrawStableMapPoints(const std::map<int, Eigen::Vector3d>& mMapPoints);

}


#endif