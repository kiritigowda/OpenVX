#pragma once
#include <VX/vx_khr_import_kernel.h>

/**
 *  Class to run the inference
 */
class DGtest
{
public:
    /**
     * Constructor
     */
    DGtest(const char* model_url);

    /**
     * Destructor
     */
    ~DGtest();

    /**
     *  Run the inference
     */
    int runInference(cv::Mat &image); 

    /**
     *  Get the inference result
     */
    int getResult(); 

private:

    /**
     *  Inference result
     */
    int mDigit;

    /**
     *  Context that will be used for the inference
     */
    vx_context mContext;
    
    /**
     *  Graph that will be used for the inference
     */
    vx_graph mGraph;

    /**
     *  Graph that will be used for the inference
     */
    vx_tensor mInputTensor;
    
    /**
     *  Graph that will be used for the inference
     */
    vx_tensor mOutputTensor;

};