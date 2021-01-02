# Digit Classification

Digits Classification is a sample tutorial program for those who are new to OpenVX. It runs inference on handwritten digits with the [MNIST](http://yann.lecun.com/exdb/mnist/) NNEF model using OpenVX [NNEF Import](https://www.khronos.org/registry/OpenVX/specs/1.3/vx_khr_feature_sets/1.1/html/vx_khr_feature_sets_1_1.html#sec_nnef) conformance feature set.

The **NNEF Import** Conformance Feature Set defines a minimum set of functions to import and execute neural networks described in the NNEF standard format. 

Applications using this feature set will use the `vxImportKernelFromURL` function to import an `NNEF` file at the location of the `URL` to create an `OpenVX kernel` representing the neural network. This kernel can subsequently be used to create a `node` in an `OpenVX graph`, which can be executed using the normal OpenVX functions from the Base Feature Set. The inputs and outputs of the neural network node will be `vx_tensor` objects.

This feature set is dependent on the Base feature set and the tensor data object, which must also be supported in order to support this feature set.

The name of this feature set is `vx_khr_nnef_import`.

<p align="center">
 <img src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/DGtest.gif">
</p>

### Pre-requisites

* Conformant OpenVX 1.3 [Implementation with NNEF Import Feature Set](https://github.com/KhronosGroup/Khronosdotorg/blob/master/api/openvx/resources.md)

* Linux `X86_64` / `Raspberry Pi` system

* [OpenCV](https://github.com/opencv/opencv/releases/tag/3.4.0) - For GUI

### Build Instructions 

* **Step - 1:** Build and install [Conformant OpenVX Implementation](https://github.com/KhronosGroup/OpenVX-sample-impl). In this example we will use the OpenVX Sample Implementation available on [GitHub](https://github.com/KhronosGroup/OpenVX-sample-impl)

    + Git Clone project with a recursive flag to get submodules
    ```
      git clone --recursive https://github.com/KhronosGroup/OpenVX-sample-impl.git
    ```
    + Use Build.py script on **x86_64** system
    ```
      cd OpenVX-sample-impl/
      python Build.py --os=Linux --arch=64 --conf=Debug --conf_nnef
    ```
    + Use Build.py script on **Raspberry Pi** system
    ```
    cd OpenVX-sample-impl/
    python Build.py --os=Linux --venum --conf=Debug --conf_nnef
    ```

* **Step - 2:** Export OpenVX Directory Path

    + **x86_64** system
    ```
    export OPENVX_DIR=$(pwd)/install/Linux/x64/Debug
    ```
    + **Raspberry Pi** system
    ```
    export OPENVX_DIR=$(pwd)/install/Linux/x32/Debug
    ```

* **Step - 3:** Clone the project `Digit Classification` application

```
cd ~/ && mkdir OpenVXSample-nnef
cd OpenVXSample-nnef/
git clone https://github.com/kiritigowda/openvx.git
```

* **Step - 4:** CMake and Build `Digit Classification` application

```
mkdir nnef-build && cd nnef-build
cmake -DOPENVX_INCLUDES=$OPENVX_DIR/include -DOPENVX_LIBRARIES=$OPENVX_DIR/bin/libopenvx.so\;$OPENVX_DIR/bin/libvxu.so\;$OPENVX_DIR/bin/libnnef-lib.a\;pthread\;dl\;m\;rt ../openvx/digitClassification/
make
```

* **Step - 5:** Run `Digit Classification` application

    + Usage
    ```
    ./DGtest [NNEF Model URL]
    ```
    + MNIST NNEF
    ``` 
    ./DGtest ../openvx/digitClassification/mnist-nnef
    ```
 
