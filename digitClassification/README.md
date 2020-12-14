# Digit Classification

DGtest is a tutorial program for those who are new to OpenVX. It runs inference on handwritten digits with the [MNIST](http://yann.lecun.com/exdb/mnist/) NNEF model using OpenVX NNEF Import Kernel.

<p align="center">
 <img src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/DGtest.gif">
</p>

### Pre-requisites

* [Conformant OpenVX 1.3 Implementation with NNEF Import Profile](https://github.com/KhronosGroup/Khronosdotorg/blob/master/api/openvx/resources.md)

* [OpenCV](https://github.com/opencv/opencv/releases/tag/3.4.0)

### Build using Cmake on Linux

* **Step - 1:** Build and install [Conformant OpenVX Implementation](https://github.com/KhronosGroup/OpenVX-sample-impl). In this example we will use the OpenVX Sample Implementation available on [GitHub](https://github.com/KhronosGroup/OpenVX-sample-impl)

```
Build OpenVX on Linux

* Git Clone project with a recursive flag to get submodules

      git clone --recursive https://github.com/KhronosGroup/OpenVX-sample-impl.git

* Use Build.py script

      cd OpenVX-sample-impl/
      python Build.py --os=Linux --arch=64 --conf=Debug --conf_nnef
```

* **Step - 2:** Export OpenVX Directory Path

```
export OPENVX_DIR=$(pwd)/install/Linux/x64/Debug
```
* **Step - 3:** Clone the OpenVX project and build the Digit Classification application

```
cd ~/ && mkdir OpenVXSample-nnef
cd OpenVXSample-nnef/
git clone https://github.com/kiritigowda/openvx.git
```

* **Step - 4:** CMake and Build the Digit Classification application

```
mkdir nnef-build && cd nnef-build
cmake -DOPENVX_INCLUDES=$OPENVX_DIR/include -DOPENVX_LIBRARIES=$OPENVX_DIR/bin/libopenvx.so ../openvx/digitClassification/
make
```

### Usage

 Usage: 
 
 ```
 ./DGtest [MNIST NNEF MODEL URL]
 ```
