
# drt-vio-initialization
## Decoupled Rotation and Translation VIO initialization
An accurate and robust initialization is crucial for visual inertial odometry (VIO). Existing loosely-coupled VIO initialization methods suffer from poor stability from structure-from-motion (SfM). Whereas tightly-copupled methods often ignore the gyroscope bias in the closed-form solution, resulting in limited accuracy. Moreover, the aforementioned two classes of methods are computationally expensive, because 3D point clouds need to be reconstructed simultaneously. We propose a novel VIO initialization method, which decouples rotation and translation estimation, and achieves higher efficiency and better robustness. This code is the implementation of our proposed method, which runs on **Linux**. We also provide the code of loosely coupled method and tightly coupled method for comparision as described in the paper. Since the drt-vio-initialization and all comparison algorithms will be released, I am still busy cleaning up the code. **So it will be released in August**.

![pipeline](doc/image/pipline.jpg)

## 1. Prerequisites
1.1 **Ubuntu** 
* Ubuntu 16.04 or Ubuntu 18.04

1.2. **Dependency**

* C++14 or C++17 Compiler
* Eigen 3.3.7
* OpenCV 3.4.9
* Boost 1.58.0
* Cere-solver 1.14.0: [Ceres Installation](http://ceres-solver.org/installation.html), remember to **sudo make install**.

## 2. Build Project with Cmake
Clone the repository and compile the project:
```
git clone https://github.com/boxuLibrary/drt-vio-init.git
cd ~/drt-vio-init/
mkdir build
cd build
cmake ..
make -j4
```

## 3. Run with Simulation Dataset

We have provided the simulation data, you can test drt-vio-initialization with it, where green line represents trajectory, and blue points represent landmarks.



<div align=center><img src=doc/image/simulation.png width="250"></div>

Notice: if you want to generate your own data, you can refer to the code below.
```
https://github.com/HeYijia/vio_data_simulation
```


## 4.Performance on EuRoC dataset


#### 4.1 Download [EuRoC MAV Dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). Although it contains stereo cameras, we only use one camera and IMU data.

#### 4.1 You can run different initialization method on the dataset via configuration parameter. The methods for comparision include:
* Open-VINS initialization 
* VINS-Mono initialization
* An improved work of ORB-SLAM3 initializaiton
* Our method in a tightly coupled manner
* Our method in a loosely coupled manner

## 5 Related Papers

- **A Rotation-Translation-Decoupled Solution for Robust and Efficient Visual-Inertial Initialization**, Yijia He, Bo Xu, Zhanpeng Ouyang and Hongdong Li.

```
@InProceedings{He_2023_CVPR,
    author    = {He, Yijia and Xu, Bo and Ouyang, Zhanpeng and Li, Hongdong},
    title     = {A Rotation-Translation-Decoupled Solution for Robust and Efficient Visual-Inertial Initialization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {739-748}
}
```

*If you use drt-vio-initialization for your academic research, please cite our related papers.*

<!-- ## 6. Acknowledgements -->


## 6. Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.


