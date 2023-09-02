cmake_minimum_required(VERSION 2.8.3)
# set project's name
project(drt_vio)

set(CMAKE_BUILD_TYPE "Release")  # Debug
#set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
# eigen 3.3以上版本加-mavx -mfma
#set(CMAKE_CXX_FLAGS_RELEASE "-O3  -g -DEIGEN_NO_DEBUG -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DEIGEN_NO_DEBUG -g")

set(STD_CXX_FS stdc++fs)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_definitions("-D SOPHUS_USE_BASIC_LOGGING")
add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)

find_package(Gflags)
find_package(Glog)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
  include
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${TBB_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS})

add_library(camera_model SHARED
        src/camera_models/chessboard/Chessboard.cc
        src/camera_models/calib/CameraCalibration.cc
        src/camera_models/camera_models/Camera.cc
        src/camera_models/camera_models/CameraFactory.cc
        src/camera_models/camera_models/CostFunctionFactory.cc
        src/camera_models/camera_models/PinholeCamera.cc
        src/camera_models/camera_models/CataCamera.cc
        src/camera_models/camera_models/EquidistantCamera.cc
        src/camera_models/camera_models/ScaramuzzaCamera.cc
        src/camera_models/sparse_graph/Transform.cc
        src/camera_models/gpl/gpl.cc
        src/camera_models/gpl/EigenQuaternionParameterization.cc)

target_link_libraries(camera_model ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

ADD_LIBRARY(MyVio SHARED
        src/featureTracker/featureTracker.cpp
        src/featureTracker/parameters.cpp
        src/IMU/basicTypes.cpp
        src/IMU/imuPreintegrated.cpp
        src/initMethod/drtVioInit.cpp
        src/initMethod/drtLooselyCoupled.cpp
        src/initMethod/drtTightlyCoupled.cpp
        src/initMethod/polynomial.cc
        src/visualization/visualize.cpp
        src/factor/imuIntegFactor.cc

        )

target_link_libraries(MyVio
        ${TBB_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${Boost_LIBRARIES}
        ${STD_CXX_FS}
        camera_model
        )



add_executable(run_euroc app/main.cpp)
#add_executable(run_euroc app/imu_aided_vp_detection2_static.cpp)


target_link_libraries(run_euroc
        MyVio
        ${Pangolin_LIBRARIES}
        -lpthread)
