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

__global__ void filterTest(	int *d_img, int *d_filter, int *d_out,
						   int imgRow, int imgCol ,int filterWidth ){
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

							   d_out[gid]=result/9;

                 printf("%d %d %d | %d %d %d | %d %d %d -> %d, Thread: %d,%d\n", d_img[(idy-1)*imgCol+idx-1], d_img[(idy-1)*imgCol+idx], d_img[(idy-1)*imgCol+idx+1], \
                  d_img[(idy)*imgCol+idx-1], d_img[(idy)*imgCol+idx], d_img[(idy)*imgCol+idx+1], d_img[(idy+1)*imgCol+idx-1], d_img[(idy+1)*imgCol+idx], d_img[(idy+1)*imgCol+idx+1], result, idx, idy);
}


void transform(){
  int satir = 0;
	const int filterWidth=3;

	int *d_img,*d_filter,*d_out;
  Mat img;
  Mat image;
  image = imread("./test.png", CV_LOAD_IMAGE_COLOR);
  img = imread("./test.png", CV_LOAD_IMAGE_COLOR);
  int Rows = img.rows;
	int Cols = img.cols;
  int imgRow = Rows;
  int imgCol = Cols;
  printf("wtf %d\n", satir++);
	int** imgR = (int**)malloc(Rows * sizeof(int*));
	int** imgG = (int**)malloc(Rows * sizeof(int*));
	int** imgB = (int**)malloc(Rows * sizeof(int*));

	for (int i = 0; i < Rows; i++)
	{
		imgR[i] = (int*)malloc(Cols * sizeof(int));
		imgG[i] = (int*)malloc(Cols * sizeof(int));
		imgB[i] = (int*)malloc(Cols * sizeof(int));
		for (int j = 0; j < img.cols; j++)
		{
			imgB[i][j] = img.at<cv::Vec3b>(i, j)[0];
			imgG[i][j] = img.at<cv::Vec3b>(i, j)[1];
			imgR[i][j] = img.at<cv::Vec3b>(i, j)[2];
		}
	}
	int h_filter[][filterWidth]={{-1,-2,-1},
								 {0,0,0},
								 {1,2,1}};
	int h_out[img.rows][img.cols];
  printf("wtf %d\n", satir++);
  image = Mat::zeros(img.rows, img.cols, CV_32FC2);
  printf("wtf %d\n", satir++);
	cudaMalloc((void**)&d_img,sizeof(int)*imgRow*imgCol);
	cudaMemcpy(d_img,imgR,sizeof(int)*imgRow*imgCol,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_filter,sizeof(int)*filterWidth*filterWidth);
	cudaMemcpy(d_filter,h_filter,sizeof(int)*filterWidth*filterWidth,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_out,sizeof(int)*imgRow*imgCol);
  printf("wtf %d\n", satir++);
	filterTest<<<dim3(1,imgRow,1),dim3(imgCol,1,1)>>>(d_img,d_filter,d_out,imgRow,imgCol,filterWidth);
  printf("wtf %d\n", satir++);
	cudaMemcpy(h_out,d_out,sizeof(int)*imgRow*imgCol,cudaMemcpyDeviceToHost);
    printf("wtf %d\n", satir++);
  for (int i=0; i < img.rows; i++) {
    for (int j=0; j < img.cols; j++) {
      image.at<cv::Vec3b>(i, j)[2] = h_out[i][j];
    }
  }
    printf("wtf %d\n", satir++);
  /**********/
  cudaMalloc((void**)&d_img,sizeof(int)*imgRow*imgCol);
  cudaMemcpy(d_img,imgG,sizeof(int)*imgRow*imgCol,cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_filter,sizeof(int)*filterWidth*filterWidth);
  cudaMemcpy(d_filter,h_filter,sizeof(int)*filterWidth*filterWidth,cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_out,sizeof(int)*imgRow*imgCol);
  printf("wtf %d\n", satir++);
  filterTest<<<dim3(1,imgRow,1),dim3(imgCol,1,1)>>>(d_img,d_filter,d_out,imgRow,imgCol,filterWidth);
  printf("wtf %d\n", satir++);
  cudaMemcpy(h_out,d_out,sizeof(int)*imgRow*imgCol,cudaMemcpyDeviceToHost);
  for (int i=0; i < img.rows; i++) {
    for (int j=0; j < img.cols; j++) {
      image.at<cv::Vec3b>(i,j)[1] = h_out[i][j];
    }
  }
    printf("wtf %d\n", satir++);
  /***********/
  cudaMalloc((void**)&d_img,sizeof(int)*imgRow*imgCol);
  cudaMemcpy(d_img,imgB,sizeof(int)*imgRow*imgCol,cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_filter,sizeof(int)*filterWidth*filterWidth);
  cudaMemcpy(d_filter,h_filter,sizeof(int)*filterWidth*filterWidth,cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_out,sizeof(int)*imgRow*imgCol);
  printf("wtf %d\n", satir++);
  filterTest<<<dim3(1,imgRow,1),dim3(imgCol,1,1)>>>(d_img,d_filter,d_out,imgRow,imgCol,filterWidth);
  printf("wtf %d\n", satir++);
  cudaMemcpy(h_out,d_out,sizeof(int)*imgRow*imgCol,cudaMemcpyDeviceToHost);
  printf("wtf %d\n", satir++);
  for (int i=0; i < img.rows; i++) {
    for (int j=0; j < img.cols; j++) {
      printf("%d\t", h_out[i][j]);
      image.at<cv::Vec3b>(i,j)[0] = h_out[i][j];
    }
    puts("");
  }
  printf("%s\n", "SONRAKI BITTI");
  printf("%s\n", "SONRAKI BITTI");
  printf("%s\n", "SONRAKI BITTI");
  for (int i=0; i < img.rows; i++) {
      for (int j=0; j < img.cols; j++) {
        printf("%d\t", imgB[i][j]);
      }
      puts("");
  }
  printf("wtf %d\n", satir++);
  namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
  imshow("2", img);
  waitKey(0);
  imshow("Display window", image);
  printf("wtf %d\n", satir++);
  waitKey(0);

}
#endif
