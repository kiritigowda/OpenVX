#include <iostream>
#include <opencv2/opencv.hpp>
#include "DGtest.h"

using namespace cv;
using namespace std;

#define ERROR_CHECK_OBJECT(obj)                                                                                                      \
    {                                                                                                                                \
        vx_status status = vxGetStatus((vx_reference)(obj));                                                                         \
        if (status != VX_SUCCESS)                                                                                                    \
        {                                                                                                                            \
            vxAddLogEntry((vx_reference)context, status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); \
            return status;                                                                                                           \
        }                                                                                                                            \
    }
#define ERROR_CHECK_STATUS(call)                                                               \
    {                                                                                          \
        vx_status status = (call);                                                             \
        if (status != VX_SUCCESS)                                                              \
        {                                                                                      \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); \
            exit(-1);                                                                          \
        }                                                                                      \
    }

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0)
    {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

DGtest::DGtest(const char *model_url)
{

    // create context
    mContext = vxCreateContext();
    vx_status status;
    status = vxGetStatus((vx_reference)mContext);
    if (status)
    {
        printf("ERROR: vxCreateContext() failed\n");
        exit(-1);
    }

    // register log
    vxRegisterLogCallback(mContext, log_callback, vx_false_e);

    // create graph
    mGraph = vxCreateGraph(mContext);
    status = vxGetStatus((vx_reference)mGraph);
    if (status)
    {
        printf("ERROR: vxCreateGraph(...) failed (%d)\n", status);
        exit(-1);
    }

    // create and initialize input tensor data
    vx_size input_dims[4] = {28, 28, 1, 1};
    mInputTensor = vxCreateTensor(mContext, 4, input_dims, VX_TYPE_FLOAT32, 0);
    if (vxGetStatus((vx_reference)mInputTensor))
    {
        printf("ERROR: vxCreateTensor() failed\n");
        exit(-1);
    }

    // create and initialize output tensor data
    vx_size output_dims[4] = {1, 1, 10, 1};
    mOutputTensor = vxCreateTensor(mContext, 4, output_dims, VX_TYPE_FLOAT32, 0);
    if (vxGetStatus((vx_reference)mOutputTensor))
    {
        printf("ERROR: vxCreateTensor() failed for mOutputTensor\n");
        exit(-1);
    }

    //Get nnef kernel
    char nn_type[5] = "nnef";
    vx_char *nnef_type = nn_type;
    mNN_kernel = vxImportKernelFromURL(mContext, nnef_type, model_url);
    if (vxGetStatus((vx_reference)mNN_kernel))
    {
        printf("ERROR: vxImportKernelFromURL() failed for NN_kernel\n");
        exit(-1);
    }
    else
    {
        printf("STATUS: vxImportKernelFromURL() for %s model: %s successful\n", nnef_type, model_url);
    }

    // query number of parameters in imported kernel
    vx_int32 num_params = 0;
    ERROR_CHECK_STATUS(vxQueryKernel(mNN_kernel, VX_KERNEL_PARAMETERS, &num_params, sizeof(vx_uint32)));
    vx_int32 direction, i;
    vx_int32 input_num = 0, output_num = 0;
    for (i = 0; i < num_params; i++)
    {
        vx_parameter prm = vxGetKernelParameterByIndex(mNN_kernel, i);

        ERROR_CHECK_STATUS(vxQueryParameter(prm, VX_PARAMETER_DIRECTION, &direction, sizeof(enum vx_type_e)));

        if (direction == VX_INPUT)
        {
            input_num++;
            vx_int32 param_type;
            ERROR_CHECK_STATUS(vxQueryParameter(prm, VX_PARAMETER_TYPE, &param_type, sizeof(enum vx_type_e)));
            if (VX_TYPE_TENSOR == param_type)
            {
                vx_meta_format meta;
                vx_int32 tensor_type = 0;
                vx_size num_of_dims = 0;
                vx_size dims[4] = {1, 1, 1, 1};
                ERROR_CHECK_STATUS(vxQueryParameter(prm, VX_PARAMETER_META_FORMAT, &meta, sizeof(vx_meta_format)));
                ERROR_CHECK_STATUS(vxQueryMetaFormatAttribute(meta, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(vx_size)));
                ERROR_CHECK_STATUS(vxQueryMetaFormatAttribute(meta, VX_TENSOR_DIMS, &dims, sizeof(dims[0]) * num_of_dims));
                ERROR_CHECK_STATUS(vxQueryMetaFormatAttribute(meta, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(vx_int32)));
                printf("STATUS: InputTensor:%d TENSOR_DATA_TYPE:%d Num Dimensions:%zu  Dimensions - [%zu, %zu, %zu, %zu])\n", input_num, tensor_type, num_of_dims, dims[0], dims[1], dims[2], dims[3]);
            }
        }
        else if (direction == VX_OUTPUT)
        {
            output_num++;
            vx_int32 param_type;
            ERROR_CHECK_STATUS(vxQueryParameter(prm, VX_PARAMETER_TYPE, &param_type, sizeof(enum vx_type_e)));
            if (VX_TYPE_TENSOR == param_type)
            {
                vx_meta_format meta;
                vx_int32 tensor_type = 0;
                vx_size num_of_dims = 0;
                vx_size dims[4] = {1, 1, 1, 1};
                ERROR_CHECK_STATUS(vxQueryParameter(prm, VX_PARAMETER_META_FORMAT, &meta, sizeof(vx_meta_format)));
                ERROR_CHECK_STATUS(vxQueryMetaFormatAttribute(meta, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(vx_size)));
                ERROR_CHECK_STATUS(vxQueryMetaFormatAttribute(meta, VX_TENSOR_DIMS, &dims, sizeof(dims[0]) * num_of_dims));
                ERROR_CHECK_STATUS(vxQueryMetaFormatAttribute(meta, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(vx_int32)));
                printf("STATUS: OutputTensor:%d TENSOR_DATA_TYPE:%d Num Dimensions:%zu  Dimensions - [%zu, %zu, %zu, %zu])\n", output_num, tensor_type, num_of_dims, dims[0], dims[1], dims[2], dims[3]);
            }
        }

        ERROR_CHECK_STATUS(vxReleaseParameter(&prm));
    }
    printf("STATUS: NN Import Kernel -- Num Params:%d Num Inputs:%d Num Outputs:%d\n", num_params, input_num, output_num);

    // create nn node for the graph
    mNode = vxCreateGenericNode(mGraph, mNN_kernel);
    if (vxGetStatus((vx_reference)mNode))
    {
        printf("ERROR: vxCreateGenericNode() failed for mNode\n");
        exit(-1);
    }

    //add input and output tensors to the node
    ERROR_CHECK_STATUS(vxSetParameterByIndex(mNode, 0, (vx_reference)mInputTensor));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(mNode, 1, (vx_reference)mOutputTensor));

    //verify the graph
    status = vxVerifyGraph(mGraph);
    if (status)
    {
        printf("ERROR: vxVerifyGraph(...) failed (%d)\n", status);
        exit(-1);
    }
    else
    {
        printf("STATUS: vxVerifyGraph Passed for NNEF Import Kernel Graph\n");
    }
};

DGtest::~DGtest()
{
    //release the tensors
    ERROR_CHECK_STATUS(vxReleaseTensor(&mInputTensor));
    ERROR_CHECK_STATUS(vxReleaseTensor(&mOutputTensor));
    //release node
    ERROR_CHECK_STATUS(vxReleaseNode(&mNode));
    //release Kernel
    ERROR_CHECK_STATUS(vxReleaseKernel(&mNN_kernel));
    //release the graph
    ERROR_CHECK_STATUS(vxReleaseGraph(&mGraph));
    // release context
    ERROR_CHECK_STATUS(vxReleaseContext(&mContext));
};

int DGtest::runInference(Mat &image)
{
    Mat img = image.clone();

    // convert to grayscale image
    cvtColor(img, img, CV_BGR2GRAY);

    // resize to 24 x 24
    resize(img, img, Size(24, 24));

    // dilate image
    dilate(img, img, Mat::ones(2, 2, CV_8U));

    // add border to the image so that the digit will go center and become 28 x 28 image
    copyMakeBorder(img, img, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(0, 0, 0));

    // set result to invalid
    mDigit = -1;

    vx_status status;
    vx_size num_of_dims;
    vx_size dims[4] = {1, 1, 1, 1};
    // query input tensor for the num dimensions
    ERROR_CHECK_STATUS(vxQueryTensor(mInputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    // query input tensor for the dimension
    ERROR_CHECK_STATUS(vxQueryTensor(mInputTensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0]) * num_of_dims));
    printf("STATUS: InputTensor: Num Dimensions: %zu  Dimensions - [%zu, %zu, %zu, %zu])\n", num_of_dims, dims[0], dims[1], dims[2], dims[3]);

    // convert image to tensor
    vx_size inputTensorSize = (dims[0] * dims[1] * dims[2] * dims[3]);
    float *localInputTensor = new float[inputTensorSize];
    memset(localInputTensor, 0, (sizeof(vx_float32) * inputTensorSize));

    vx_size inputViewStart[4] = {0};
    vx_size inputTensorStride[4] = {0};
    inputTensorStride[0] = sizeof(vx_float32);
    for (int j = 1; j < num_of_dims; j++)
    {
        inputTensorStride[j] = inputTensorStride[j - 1] * dims[j - 1];
    }

    for (vx_size y = 0; y < dims[1]; y++)
    {
        unsigned char *src = img.data + y * dims[0] * dims[2];
        float *dst = localInputTensor + ((y * inputTensorStride[1]) >> 2);
        for (vx_size x = 0; x < dims[0]; x++, src++)
        {
            *dst++ = src[0];
        }
    }
    printf("STATUS: Image to Tensor Conversion Successful\n");

    ERROR_CHECK_STATUS(vxCopyTensorPatch(mInputTensor, num_of_dims, inputViewStart, dims,
                                         inputTensorStride, (void **)&localInputTensor, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyTensorPatch Passed for Input Tensor\n");

    //process the graph
    status = vxProcessGraph(mGraph);
    if (status != VX_SUCCESS)
    {
        printf("ERROR: vxProcessGraph failed (%d)\n", status);
        return status;
    }
    else
    {
        printf("STATUS: vxProcessGraph Successful\n");
    }

    // query input tensor for the num dimensions
    ERROR_CHECK_STATUS(vxQueryTensor(mOutputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    // query input tensor for the dimension
    ERROR_CHECK_STATUS(vxQueryTensor(mOutputTensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0]) * num_of_dims));
    printf("STATUS: OutputTensor: Num Dimensions: %zu  Dimensions - [%zu, %zu, %zu, %zu])\n", num_of_dims, dims[0], dims[1], dims[2], dims[3]);

    // copy output tensor
    vx_size outputTensorSize = (dims[0] * dims[1] * dims[2] * dims[3]);
    float *localOutputTensor = new float[outputTensorSize];
    memset(localOutputTensor, 0, (sizeof(vx_float32) * outputTensorSize));

    vx_size outputViewStart[4] = {0};
    vx_size outputTensorStride[4] = {0};
    outputTensorStride[0] = sizeof(vx_float32);
    for (int j = 1; j < num_of_dims; j++)
    {
        outputTensorStride[j] = outputTensorStride[j - 1] * dims[j - 1];
    }

    ERROR_CHECK_STATUS(vxCopyTensorPatch(mOutputTensor, num_of_dims, outputViewStart, dims,
                                         outputTensorStride, (void **)&localOutputTensor, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyTensorPatch Passed for Output Tensor\n");

    for (int i = 0; i < outputTensorSize; i++)
    {
        printf("STATUS: Output[%d] - %f\n", i, localOutputTensor[i]);
    }

    mDigit = std::distance(localOutputTensor, std::max_element(localOutputTensor, (localOutputTensor + 10)));
    printf("STATUS: Output Tensor Analysis Passed %d\n", mDigit);

    delete localInputTensor;
    delete localOutputTensor;

    return 0;
}

int DGtest::getResult()
{
    return mDigit;
}
