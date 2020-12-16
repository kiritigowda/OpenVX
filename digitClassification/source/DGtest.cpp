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

    mNode = vxCreateGenericNode(mGraph, mNN_kernel);
    if (vxGetStatus((vx_reference)mNode))
    {
        printf("ERROR: vxCreateGenericNode() failed for mNode\n");
        exit(-1);
    }

    status = vxQueryKernel(mNN_kernel, VX_KERNEL_PARAMETERS, &mNum_params, sizeof(vx_uint32));
    if (status)
    {
        printf("ERROR: vxQueryKernel(...) failed (%d)\n", status);
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

    vx_size dims[4] = {1, 1, 1, 1}, stride[4];
    vx_status status;
    vx_map_id map_id;
    float *ptr;

    // query tensor for the dimension
    ERROR_CHECK_STATUS(vxQueryTensor(mInputTensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0]) * 4));

    // convert image to tensor
    vx_size inputTensorSize = (dims[0] * dims[1] * dims[2] * dims[3]);
    float *localInputTensor = new float[inputTensorSize];
    memset(localInputTensor, 0, sizeof(float) * inputTensorSize);
    for (vx_size y = 0; y < dims[1]; y++)
    {
        unsigned char *src = img.data + y * dims[0] * dims[2];
        float *dst = localInputTensor + ((y * stride[1]) >> 2);
        for (vx_size x = 0; x < dims[0]; x++, src++)
        {
            *dst++ = src[0];
        }
    }
    vx_size *tensorStride;
    vx_size *viewStart;
    vx_size *viewEnd;
    *(viewStart) = 0;
    *(viewEnd) = (dims[0] * dims[1] * dims[2]);
    *(tensorStride) = sizeof(vx_float32);
    ERROR_CHECK_STATUS(vxCopyTensorPatch(mInputTensor, 4, viewStart, viewEnd,
                                         tensorStride, (vx_char *)&localInputTensor, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    delete[] localInputTensor;

    // copy image to input tensor
    /*
    status = vxMapTensorPatch(mInputTensor, 4, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if (status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for mInputTensor" << std::endl;
        return -1;
    }

    status = vxUnmapTensorPatch(mInputTensor, map_id);
    if (status)
    {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for mInputTensor" << std::endl;
        return -1;
    }
    */

    //process the graph
    status = vxProcessGraph(mGraph);
    if (status != VX_SUCCESS)
    {
        std::cerr << "ERROR: vxProcessGraph() failed" << std::endl;
        return -1;
    }

    // get the output result from output tensor
    /*
    status = vxMapTensorPatch(mOutputTensor, 4, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if (status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for mOutputTensor" << std::endl;
        return -1;
    }

    mDigit = std::distance(ptr, std::max_element(ptr, ptr + 10));

    status = vxUnmapTensorPatch(mOutputTensor, map_id);
    if (status)
    {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for mOutputTensor" << std::endl;
        return -1;
    }
    */

    // query tensor for the dimension
    ERROR_CHECK_STATUS(vxQueryTensor(mOutputTensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0]) * 4));

    // copy output tensor
    vx_size outputTensorSize = (dims[0] * dims[1] * dims[2] * dims[3]);
    float *localOutputTensor = new float[outputTensorSize];
    memset(localOutputTensor, 0, sizeof(float) * outputTensorSize);
    vx_size *tensorStride;
    vx_size *viewStart;
    vx_size *viewEnd;
    *(viewStart) = 0;
    *(viewEnd) = (dims[0] * dims[1] * dims[2]);
    *(tensorStride) = sizeof(vx_float32);
    ERROR_CHECK_STATUS(vxCopyTensorPatch(mInputTensor, 4, viewStart, viewEnd,
                                         tensorStride, (vx_char *)&localOutputTensor, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    mDigit = std::distance(localOutputTensor, std::max_element(localOutputTensor, localOutputTensor + 10));

    delete[] localOutputTensor;

    return 0;
}

int DGtest::getResult()
{
    return mDigit;
}
