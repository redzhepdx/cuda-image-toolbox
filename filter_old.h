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

__global__ void filterTest(uchar *d_img, uchar *d_filter, uchar *d_out,
						   int imgRow, int imgCol ,int filterWidth){
						   int idx=threadIdx.x+blockDim.x*blockIdx.x;
						   int idy=threadIdx.y+blockDim.y*blockIdx.y;
						   int gid=idy*imgCol+idx;

						   if(idx>=imgCol || idy>=imgRow)
							   return;

						   int result = 0;

						   for(int a = -filterWidth/2; a <= filterWidth/2; ++a){
								for(int b = -filterWidth/2; b <= filterWidth/2; ++b){
									int newIdx = idx + b;
									int newIdy = idy + a;
									int filterIndex = (a + filterWidth/2)*filterWidth + (b + filterWidth/2);
									if( newIdy >= 0 && newIdy < imgRow && newIdx >= 0 && newIdx < imgCol)
									result += d_filter[filterIndex] * d_img[newIdy * imgCol + newIdx];
							    }
						   }

						   d_out[gid]=(uchar) (result/9);
}


void transform(){
	const int filterWidth=3;

	uchar *d_img,*d_filter,*d_out;
  Mat img;
  Mat image;
  image = imread("./test.png", CV_LOAD_IMAGE_COLOR);
  img = imread("./test.png", CV_LOAD_IMAGE_COLOR);
  int Rows = img.rows;
	int Cols = img.cols;
  int imgRow = Rows;
  int imgCol = Cols;
	uchar** imgR = (uchar**)malloc(Rows * sizeof(uchar*));
	uchar** imgG = (uchar**)malloc(Rows * sizeof(uchar*));
	uchar** imgB = (uchar**)malloc(Rows * sizeof(uchar*));

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
	uchar h_filter[][filterWidth]={{1,1,1},
								 {1,1,1},
								 {1,1,1}};
	uchar h_out[img.rows][img.cols];
	cudaMalloc((void**)&d_img,sizeof(uchar)*imgRow*imgCol);
	cudaMemcpy(d_img,imgR,sizeof(uchar)*imgRow*imgCol,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_filter,sizeof(uchar)*filterWidth*filterWidth);
	cudaMemcpy(d_filter,h_filter,sizeof(uchar)*filterWidth*filterWidth,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_out,sizeof(uchar)*imgRow*imgCol);

	filterTest<<<dim3(1,imgRow,1),dim3(imgCol,1,1)>>>(d_img,d_filter,d_out,imgRow,imgCol,filterWidth);

	cudaMemcpy(h_out,d_out,sizeof(uchar)*imgRow*imgCol,cudaMemcpyDeviceToHost);

  for (int i=0; i < img.rows; i++) {
    for (int j=0; j < img.cols; j++) {
      image.at<cv::Vec3b>(i, j)[2] = h_out[i][j];
    }
  }

  /**********/
  cudaMalloc((void**)&d_img,sizeof(uchar)*imgRow*imgCol);
  cudaMemcpy(d_img,imgG,sizeof(uchar)*imgRow*imgCol,cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_filter,sizeof(uchar)*filterWidth*filterWidth);
  cudaMemcpy(d_filter,h_filter,sizeof(uchar)*filterWidth*filterWidth,cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_out,sizeof(uchar)*imgRow*imgCol);

  filterTest<<<dim3(1,imgRow,1),dim3(imgCol,1,1)>>>(d_img,d_filter,d_out,imgRow,imgCol,filterWidth);

  cudaMemcpy(h_out,d_out,sizeof(uchar)*imgRow*imgCol,cudaMemcpyDeviceToHost);
  for (int i=0; i < img.rows; i++) {
    for (int j=0; j < img.cols; j++) {
      image.at<cv::Vec3b>(i,j)[1] = h_out[i][j];
    }
  }

  cudaMalloc((void**)&d_img,sizeof(uchar)*imgRow*imgCol);
  cudaMemcpy(d_img,imgB,sizeof(uchar)*imgRow*imgCol,cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_filter,sizeof(uchar)*filterWidth*filterWidth);
  cudaMemcpy(d_filter,h_filter,sizeof(uchar)*filterWidth*filterWidth,cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_out,sizeof(uchar)*imgRow*imgCol);

  filterTest<<<dim3(1,imgRow,1),dim3(imgCol,1,1)>>>(d_img,d_filter,d_out,imgRow,imgCol,filterWidth);

  cudaMemcpy(h_out,d_out,sizeof(uchar)*imgRow*imgCol,cudaMemcpyDeviceToHost);

  for (int i=0; i < img.rows; i++) {
    for (int j=0; j < img.cols; j++) {
      image.at<cv::Vec3b>(i,j)[0] = h_out[i][j];
    }
  }
  imshow("Display window", image);
  waitKey(0);

}
#endif
