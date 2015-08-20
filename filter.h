#ifndef FILTER_H
#define FILTER_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

__global__ void filter(uchar* d_input, uchar* d_output,\
  int rows, int cols, int filterWidth) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  //int grid_width = gridDim.x * blockDim.x;
  //int globalId = idy * grid_width + idx;
  int globalId = idy * cols + idx;

<<<<<<< Updated upstream
  d_output[globalId] = d_input[idy * cols + idx];
  return;

  if (idx >= cols) {
    if (idy >= rows) {
      return; //Invalid location, do nothing.
    }
=======
  //d_output[globalId] = d_input[idy * cols + idx];
  //return;
  if (idx >= cols || idy >= rows) {
      return; //Invalid location, do nothing.
>>>>>>> Stashed changes
  }

  int result = 0;
  for (int i = -filterWidth/2; i <= filterWidth/2; i++) {
    for (int j = -filterWidth/2; j <= filterWidth/2; j++) {
      int x = idx + i;
      int y = idy + j;
<<<<<<< Updated upstream
=======
      int filterIndex = (i + filterWidth/2)*filterWidth + (j + filterWidth/2);
>>>>>>> Stashed changes
      if (y >= 0 && x >= 0 && y < rows && x < cols) {
        result += d_filter[filterIndex] * d_input[y * cols + x];
      }
    }
  }
  result = result / 9;
  if (result < 0) {
    result *= -1;
  }
  d_output[globalId] = (uchar) result;
}

void transform(){
	const int filterWidth=3;

	uchar *d_img,*d_filter,*d_out;
  Mat img;
  Mat image;
  img = imread("./atlitest.png", CV_LOAD_IMAGE_COLOR);
  image = Mat::zeros(img.rows, img.cols, CV_8UC3);
  int Rows = img.rows;
	int Cols = img.cols;
  int imgRow = Rows;
  int imgCol = Cols;
<<<<<<< Updated upstream
	uchar** imgR = (uchar**)malloc(Rows * sizeof(uchar*));
	uchar** imgG = (uchar**)malloc(Rows * sizeof(uchar*));
	uchar** imgB = (uchar**)malloc(Rows * sizeof(uchar*));
=======
	uchar* imgR = (uchar*)malloc(Rows * Cols * sizeof(uchar));
	uchar* imgG = (uchar*)malloc(Rows * Cols * sizeof(uchar));
	uchar* imgB = (uchar*)malloc(Rows * Cols * sizeof(uchar));
>>>>>>> Stashed changes

	for (int i = 0; i < Rows; i++)
	{
		imgR[i] = (uchar*)malloc(Cols * sizeof(uchar));
		imgG[i] = (uchar*)malloc(Cols * sizeof(uchar));
		imgB[i] = (uchar*)malloc(Cols * sizeof(uchar));
		for (int j = 0; j < img.cols; j++)
		{
			imgB[i][j] = img.at<cv::Vec3b>(i, j)[0];
			imgG[i][j] = img.at<cv::Vec3b>(i, j)[1];
			imgR[i][j] = img.at<cv::Vec3b>(i, j)[2];
		}
	}
<<<<<<< Updated upstream
	uchar h_filter[][filterWidth]={{1,1,1},
								 {1,1,1},
								 {1,1,1}};
=======
	int h_filter[][filterWidth]={-1,-2,-1,0,0,0,1,2,1};
>>>>>>> Stashed changes
	uchar h_out[img.rows][img.cols];
	cudaMalloc((void**)&d_img,sizeof(uchar)*imgRow*imgCol);
	cudaMemcpy(d_img,imgR,sizeof(uchar)*imgRow*imgCol,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_filter,sizeof(uchar)*filterWidth*filterWidth);
	cudaMemcpy(d_filter,h_filter,sizeof(uchar)*filterWidth*filterWidth,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_out,sizeof(uchar)*imgRow*imgCol);

<<<<<<< Updated upstream
  filter<<<dim3(1,imgRow,1),dim3(imgCol,1,1)>>>(d_img,d_out,imgRow,imgCol,filterWidth);
=======
  filter<<<dim3(1, imgRow, 1),dim3(imgCol,1,1)>>>(d_img,d_filter,d_out,imgRow,imgCol,filterWidth);
>>>>>>> Stashed changes
	cudaMemcpy(h_out,d_out,sizeof(uchar)*imgRow*imgCol,cudaMemcpyDeviceToHost);
  /*for (int i=0; i < img.rows; i++) {
    for (int j=0; j < img.cols; j++) {
      printf("%u\t", imgR[i][j]);
    }
    puts("");
  }
  printf("********************\n");
  for (int i=0; i < img.rows; i++) {
    for (int j=0; j < img.cols; j++) {
      //printf("%u\t", h_out[i][j]);
      image.at<cv::Vec3b>(i, j)[2] = h_out[i][j];
    }
    puts("");
  }*/
  imshow("Display window", image);
  waitKey(0);
  printf("********************\n");
  /**********/
  cudaMalloc((void**)&d_img,sizeof(uchar)*imgRow*imgCol);
  cudaMemcpy(d_img,imgG,sizeof(uchar)*imgRow*imgCol,cudaMemcpyHostToDevice);
<<<<<<< Updated upstream
  cudaMalloc((void**)&d_filter,sizeof(uchar)*filterWidth*filterWidth);
  cudaMemcpy(d_filter,h_filter,sizeof(uchar)*filterWidth*filterWidth,cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_out,sizeof(uchar)*imgRow*imgCol);

  filter<<<dim3(1,imgRow,1),dim3(imgCol,1,1)>>>(d_img,d_out,imgRow,imgCol,filterWidth);
=======
  cudaMalloc((void**)&d_out,sizeof(uchar)*imgRow*imgCol);

  filter<<<dim3(1, imgRow, 1),dim3(imgCol,1,1)>>>(d_img,d_filter,d_out,imgRow,imgCol,filterWidth);
>>>>>>> Stashed changes

  cudaMemcpy(h_out,d_out,sizeof(uchar)*imgRow*imgCol,cudaMemcpyDeviceToHost);
  for (int i=0; i < img.rows; i++) {
    for (int j=0; j < img.cols; j++) {
      //printf("%u\t", h_out[i][j]);
      image.at<cv::Vec3b>(i,j)[1] = h_out[i][j];
    }
    puts("");
  }
  //printf("********************\n");
  imshow("Display window", image);
  waitKey(0);

  cudaMalloc((void**)&d_img,sizeof(uchar)*imgRow*imgCol);
  cudaMemcpy(d_img,imgB,sizeof(uchar)*imgRow*imgCol,cudaMemcpyHostToDevice);
<<<<<<< Updated upstream
  cudaMalloc((void**)&d_filter,sizeof(uchar)*filterWidth*filterWidth);
  cudaMemcpy(d_filter,h_filter,sizeof(uchar)*filterWidth*filterWidth,cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_out,sizeof(uchar)*imgRow*imgCol);

  filter<<<dim3(1,imgRow,1),dim3(imgCol,1,1)>>>(d_img,d_out,imgRow,imgCol,filterWidth);
  cudaMemcpy(h_out,d_out,sizeof(uchar)*imgRow*imgCol,cudaMemcpyDeviceToHost);
  for (int i=0; i < img.rows; i++) {
    for (int j=0; j < img.cols; j++) {
      printf("%u\t", imgR[i][j]);
    }
    puts("");
  }
  printf("********************\n");
=======
  cudaMalloc((void**)&d_out,sizeof(uchar)*imgRow*imgCol);

  filter<<<dim3(1, imgRow, 1),dim3(imgCol,1,1)>>>(d_img,d_filter,d_out,imgRow,imgCol,filterWidth);
  cudaMemcpy(h_out,d_out,sizeof(uchar)*imgRow*imgCol,cudaMemcpyDeviceToHost);
  //printf("********************\n");
>>>>>>> Stashed changes
  for (int i=0; i < img.rows; i++) {
    for (int j=0; j < img.cols; j++) {
      //printf("%u\t", h_out[i][j]);
      image.at<cv::Vec3b>(i,j)[0] = h_out[i][j];
    }
    puts("");
  }
  imshow("Display window", image);
  waitKey(0);

}

void foo() {

}
#endif
