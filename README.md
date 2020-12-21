[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="50%" src="https://upload.wikimedia.org/wikipedia/en/thumb/d/dd/OpenVX_logo.svg/1920px-OpenVX_logo.svg.png" /></p>

<a href="https://www.khronos.org/openvx/" target="_blank">Khronos OpenVX™</a> is an open, royalty-free standard for cross-platform acceleration of computer vision applications. OpenVX enables performance and power-optimized computer vision processing, especially important in embedded and real-time use cases such as face, body, and gesture tracking, smart video surveillance, advanced driver assistance systems (ADAS), object and scene reconstruction, augmented reality, visual inspection, robotics and more.

This repository has developer help, tools, and examples to make development with OpenVX simple.

* [Digits Classification Application](#digits-classification-application)
* [OpenVX for Raspberry Pi](#openvx-implementation-for-raspberry-pi)

## Digits Classification Application
[Digits Classification](digitClassification) is a sample tutorial program for those who are new to OpenVX. It runs inference on handwritten digits with the [MNIST](http://yann.lecun.com/exdb/mnist/) NNEF model using OpenVX [NNEF Import Kernel](https://www.khronos.org/registry/OpenVX/specs/1.3/vx_khr_feature_sets/1.1/html/vx_khr_feature_sets_1_1.html#sec_nnef) conformance profile.

<p align="center">
 <img src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/DGtest.gif">
</p>

## OpenVX Implementation for Raspberry Pi

<p align="center"> &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; <img width="10%" src="https://www.raspberrypi.org/app/uploads/2018/03/RPi-Logo-Reg-SCREEN.png" /> &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; <img width="8%" src="https://svgsilh.com/svg/156116.svg"/> &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; <img width="40%" src="https://upload.wikimedia.org/wikipedia/en/thumb/d/dd/OpenVX_logo.svg/1920px-OpenVX_logo.svg.png"/> </p>

### Identify Raspberry Pi

* Check hardware version
```
cat /proc/device-tree/model && pinout
```

* Check Raspbian version on Raspberry Pi
```
cat /etc/os-release
```
* Check Kernel version on Raspberry Pi
```
uname -a
```

### Vision, Enhanced Vision, & Neural Net Conformance Feature Set

The [OpenVX 1.3 implementation](https://github.com/KhronosGroup/OpenVX-sample-impl/tree/openvx_1.3) is available on GitHub. To build and install the library follow the instructions below.

#### Build OpenVX 1.3 on Raspberry Pi

* Git Clone project with the recursive flag to get submodules.

````
git clone --recursive https://github.com/KhronosGroup/OpenVX-sample-impl.git
````
**Note:** The API Documents and Conformance Test Suite are set as submodules in the sample implementation project 

* Use Build.py script to build and install OpenVX 1.3

````
cd OpenVX-sample-impl/
python Build.py --os=Linux --venum --conf=Debug --conf_vision --enh_vision --conf_nn
````

* Build and run the conformance

````
export OPENVX_DIR=$(pwd)/install/Linux/x32/Debug
export VX_TEST_DATA_PATH=$(pwd)/cts/test_data/
mkdir build-cts
cd build-cts
cmake -DOPENVX_INCLUDES=$OPENVX_DIR/include -DOPENVX_LIBRARIES=$OPENVX_DIR/bin/libopenvx.so\;$OPENVX_DIR/bin/libvxu.so\;pthread\;dl\;m\;rt -DOPENVX_CONFORMANCE_VISION=ON -DOPENVX_USE_ENHANCED_VISION=ON -DOPENVX_CONFORMANCE_NEURAL_NETWORKS=ON ../cts/
cmake --build .
LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance
````

### NNEF Import Conformance Feature Set

The [OpenVX 1.3 implementation](https://github.com/KhronosGroup/OpenVX-sample-impl/tree/openvx_1.3) is available on GitHub. To build and install the library follow the instructions below.

#### Build OpenVX 1.3 on Raspberry Pi

* Git Clone project with the recursive flag to get submodules.

````
git clone --recursive https://github.com/KhronosGroup/OpenVX-sample-impl.git
````
**Note:** The API Documents and Conformance Test Suite are set as submodules in the sample implementation project 

* Use Build.py script to build and install OpenVX 1.3

````
cd OpenVX-sample-impl/
python Build.py --os=Linux --venum --conf=Debug --conf_nnef
````

* Build and run the conformance

````
export OPENVX_DIR=$(pwd)/install/Linux/x32/Debug
export VX_TEST_DATA_PATH=$(pwd)/cts/test_data/
mkdir build-nnef-cts
cd build-nnef-cts
cmake -DOPENVX_INCLUDES=$OPENVX_DIR/include -DOPENVX_LIBRARIES=$OPENVX_DIR/bin/libopenvx.so\;$OPENVX_DIR/bin/libvxu.so\;$OPENVX_DIR/bin/libnnef-lib.a\;pthread\;dl\;m\;rt -DOPENVX_CONFORMANCE_NNEF_IMPORT=ON ../cts/
cmake --build .
LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance --filter=*TensorNNEF*
````
