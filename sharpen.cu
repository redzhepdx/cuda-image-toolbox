#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

// Sharpening Kernel function
__global__ void sharpen(unsigned char* d_img, unsigned char* d_out, const int ROWS, const int COLS) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;

    if(idy > 0 && idy < ROWS - 1 && idx > 0 && idx < COLS - 1)
        d_out[idy*COLS + idx] = 5*d_img[(y)*COLS + (x)] - d_img[(y)*COLS + (x+1)] -
                                d_img[(y)*COLS + (x-1)] - d_img[(y+1)*COLS + (x)] -
                                d_img[(y-1)*COLS + (x)];
}

// Apply Sharpening function
int applySharpening(Mat* in_img, Mat* out_img){
    // Create an empty image with same dimensions
    const int ROWS = in_img->rows;
    const int COLS = in_img->cols;

    // Allocate memories in host and device
    unsigned char* h_img = (unsigned char*)malloc(ROWS*COLS*sizeof(unsigned char));
    if(h_img == NULL){
        printf("ERROR: malloc - Couldn't allocate memory for the image channel input, on the host.");
        return -1;
    }

    unsigned char* h_out = (unsigned char*)malloc(ROWS*COLS*sizeof(unsigned char));
    if(h_out == NULL){
        printf("ERROR: malloc - Couldn't allocate memory for the image channel output, on the host.");
        return -1;
    }

    // Allocate memory on device for the image input
    unsigned char* d_img;
    checkCudaErrors(cudaMalloc((void**)&d_img, ROWS*COLS*sizeof(unsigned char)));

    // Allocate memory on device for the image output
    unsigned char* d_out;
    checkCudaErrors(cudaMalloc((void**)&d_out, ROWS*COLS*sizeof(unsigned char)));

    // Determine the block and grid sizes
    dim3 blockSize(4, 4, 1);
    dim3 gridSize((int)((COLS - 1) / blockSize.x) + 1, (int)((ROWS - 1) / blockSize.y) + 1, 1);

    // Loop for each channel
    for(int channel = 0; channel < 3; ++channel){
        // Put Mat's current channel to the host channel
        for(int i = 0; i < ROWS; ++i)
            for(int j = 0; j < COLS; ++j)
                h_img[i*COLS + j] = in_img.at<cv::Vec3b>(i, j)[channel];

        // Transfer the channel to the device
        checkCudaErrors(cudaMemcpy(d_img, h_img, ROWS*COLS*sizeof(unsigned char), cudaMemcpyHostToDevice));

        // Run the sharpen algorithm on the image
        sharpen<<<gridSize, blockSize>>>(d_img, d_out, ROWS, COLS);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // Transfer the channel back to host
        checkCudaErrors(cudaMemcpy(h_out, d_out, ROWS*COLS*sizeof(unsigned char), cudaMemcpyDeviceToHost));

        // Save the channel to the Mat
        for(int i = 0; i < ROWS; ++i)
            for(int j = 0; j < COLS; ++j)
                out_img->at<cv::Vec3b>(i, j)[channel] = h_out[i*COLS + j];
    }

    // Cleanup
    if(h_img != NULL) free(h_img);
    if(h_out != NULL) free(h_out);
    checkCudaErrors(cudaFree(d_img));
    checkCudaErrors(cudaFree(d_out));
    return 0;
}

// Main function
int main() {
    // Load the image
    Mat in_img = imread("lana.jpg", CV_LOAD_IMAGE_COLOR);
    if(!in_img.data) {
        printf("ERROR: Couldn't open the image.");
        return -1;
    }

    // Display the input image
    imshow("Input Image", in_img);
    waitKey(0);

    // Create an empty image with same dimensions
    Mat out_img = Mat::zeros(in_img.rows, in_img.cols, CV_8UC3);

    // Sharpen the image
    int result = applySharpening(&in_img, &out_img);

    // Display the output image
    if(result == 0){
        imshow("Output Image", out_img);
        waitKey(0);
    }

    return result;
}
