<p align="center"><img src="https://www.khronos.org/assets/uploads/ceimg/made/assets/uploads/apis/NNEF_500px_Apr17_165_75.png" /></p>

# Neural Network Exchange Format (NNEF)
NNEF reduces machine learning deployment fragmentation by enabling a rich mix of neural network training tools and inference engines to be used by applications across a diverse range of devices and platforms.

## Steps to setup [NNEF-Tools](https://github.com/KhronosGroup/NNEF-Tools) on RPi

* Note: Python3 version >= 3.5.2 is supported

### Step 1: Download and install NNEF Tools:
```
git clone https://github.com/KhronosGroup/NNEF-Tools
cd NNEF-Tools
cd parser/python
python setup.py install
cd ../..
```

### Step 2: Install dependencies
```
pip install future typing six numpy protobuf onnx onnx-simplifier onnxruntime
```

### Step 3: Convert model from ONNX to NNEF
```
python -m nnef_tools.convert --help
usage: convert.py [-h] --input-model INPUT_MODEL [--output-model OUTPUT_MODEL]
                  --input-format {tf,tflite,onnx,nnef,caffe2,caffe}
                  --output-format {tf,tflite,onnx,nnef,caffe2}
                  [--input-shapes INPUT_SHAPES]
                  [--io-transpose [IO_TRANSPOSE [IO_TRANSPOSE ...]]]
                  [--fold-constants] [--optimize]
                  [--custom-converters CUSTOM_CONVERTERS [CUSTOM_CONVERTERS ...]]
                  [--custom-shapes CUSTOM_SHAPES [CUSTOM_SHAPES ...]]
                  [--custom-fragments CUSTOM_FRAGMENTS [CUSTOM_FRAGMENTS ...]]
                  [--custom-optimizers CUSTOM_OPTIMIZERS [CUSTOM_OPTIMIZERS ...]]
                  [--mirror-unsupported] [--generate-custom-fragments]
                  [--keep-io-names] [--decompose [DECOMPOSE [DECOMPOSE ...]]]
                  [--input-names INPUT_NAMES [INPUT_NAMES ...]]
                  [--output-names OUTPUT_NAMES [OUTPUT_NAMES ...]]
                  [--tensor-mapping [TENSOR_MAPPING]] [--annotate-shapes]
                  [--compress [COMPRESS]]
```
* Command:
```
python -m nnef_tools.convert --input-format onnx --output-format nnef --input-model <PATH-TO-MNIST-ONNX-MODEL>/mnist.onnx --output-model <PATH-TO-SAVE-NNEF-MODEL>/mnist.nnef
```
