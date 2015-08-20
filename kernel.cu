#include "opencv2/highgui/highgui.hpp"
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

using namespace cv;
using namespace std;

__global__ void filter(unsigned char * d_input, int *d_filter, unsigned char * d_output,
	int rows, int cols, int filterWidth) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int globalId = idy * cols + idx;

	if (idx < cols && idy < rows){
		
		unsigned int result = 0;
		for (int i = -filterWidth / 2; i <= filterWidth / 2; i++) {
			for (int j = -filterWidth / 2; j <= filterWidth / 2; j++) {
				int x = idx + i;
				int y = idy + j;
				int filterIndex = (x + filterWidth / 2)*filterWidth + (y + filterWidth / 2);
				if (y >= 0 && x >= 0 && y < rows && x < cols) {
					result += d_input[y * cols + x];
				}
			}
		}
		result = result / 9;
		d_output[globalId] = (unsigned char)result;
	}
}

int main(int argc, const char** argv){
		const int filterWidth = 3;

		unsigned char *d_img, *d_out;
		int *d_filter;
		Mat img;
		Mat image;
		img = imread("img4.jpg", CV_LOAD_IMAGE_COLOR);
		image = Mat::zeros(img.rows, img.cols, CV_8UC3);
		int Rows = img.rows;
		int Cols = img.cols;
		const int imgRow = Rows;
		const int imgCol = Cols;
		dim3 DimGrid( (int)((Cols - 1) / 32) + 1 , (int)((Rows - 1) / 32) + 1 , 1);
		dim3 DimBlock(32, 32, 1);

		unsigned char *imgR = (unsigned char*)malloc(Rows * Cols * sizeof(unsigned char));
		unsigned char *imgG = (unsigned char*)malloc(Rows * Cols * sizeof(unsigned char));
		unsigned char *imgB = (unsigned char*)malloc(Rows * Cols * sizeof(unsigned char));

		for (int i = 0; i < Rows; i++){
			for (int j = 0; j < Cols; j++){
				imgB[i * Cols + j] = img.at<cv::Vec3b>(i, j)[0];
				imgG[i * Cols + j] = img.at<cv::Vec3b>(i, j)[1];
				imgR[i * Cols + j] = img.at<cv::Vec3b>(i, j)[2];
			}
		}
		int h_filter[][filterWidth] = { { 1, 1, 1 },
										{ 1, 1, 1 },
										{ 1, 1, 1 } };
		unsigned char *h_out = (unsigned char*)malloc(Rows * Cols * sizeof(unsigned char));


		cudaMalloc((void**)&d_img, sizeof(unsigned char)*imgRow*imgCol);
		cudaMemcpy(d_img, imgR, sizeof(unsigned char)*imgRow*imgCol, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_filter, sizeof(int)*filterWidth*filterWidth);
		cudaMemcpy(d_filter, h_filter, sizeof(int)*filterWidth*filterWidth, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_out, sizeof(unsigned char)*imgRow*imgCol);

		filter <<<DimGrid, DimBlock>>>(d_img, d_filter, d_out, imgRow, imgCol, filterWidth);
		cudaMemcpy(h_out, d_out, sizeof(unsigned char)*imgRow*imgCol, cudaMemcpyDeviceToHost);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
					image.at<cv::Vec3b>(i, j)[2] = h_out[i*Cols+j];
			}
			puts("");
			}
		imshow("Display window", image);
		waitKey(0);
		
		cudaMalloc((void**)&d_img, sizeof(unsigned char)*imgRow*imgCol);
		cudaMemcpy(d_img, imgG, sizeof(unsigned char)*imgRow*imgCol, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_filter, sizeof(int)*filterWidth*filterWidth);
		cudaMemcpy(d_filter, h_filter, sizeof(int)*filterWidth*filterWidth, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_out, sizeof(unsigned char)*imgRow*imgCol);

		filter <<< DimGrid, DimBlock >>>(d_img, d_filter, d_out, imgRow, imgCol, filterWidth);
		cudaMemcpy(h_out, d_out, sizeof(unsigned char)*imgRow*imgCol, cudaMemcpyDeviceToHost);
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
					image.at<cv::Vec3b>(i, j)[1] = h_out[i*Cols+j];
			}
			puts("");
		}
		
		imshow("Display window", image);
		waitKey(0);

		cudaMalloc((void**)&d_img, sizeof(unsigned char)*imgRow*imgCol);
		cudaMemcpy(d_img, imgB, sizeof(unsigned char)*imgRow*imgCol, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_filter, sizeof(int)*filterWidth*filterWidth);
		cudaMemcpy(d_filter, h_filter, sizeof(int)*filterWidth*filterWidth, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_out, sizeof(unsigned char)*imgRow*imgCol);
		
		filter <<< DimGrid, DimBlock >>>(d_img, d_filter, d_out, imgRow, imgCol, filterWidth);
		cudaMemcpy(h_out, d_out, sizeof(unsigned char)*imgRow*imgCol, cudaMemcpyDeviceToHost);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				image.at<cv::Vec3b>(i, j)[0] = h_out[i*Cols+j];
			}
			puts("");
		}
		
		imshow("Display window", image);
		waitKey(0);
	return 0;
}

