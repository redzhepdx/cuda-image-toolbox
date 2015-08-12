#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "misc.h"
#include "filter.h"

using namespace cv;
using namespace std;

int main (int argc, char** argv) {
  transform();
  /*if (argc != 2) {
    cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
    return -1;
  }

  Mat* imageMat = new Mat;
  *imageMat = imread(argv[1], CV_LOAD_IMAGE_COLOR); // Read the file.
  if (!imageMat->data) {
    cout << "Could not open or find the image." << endl;
    return -1;
  }
  imageVector = matToVector(*imageMat);
  delete imageMat;

  namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
  imshow("Display window", *imageMat);

  waitKey(0);*/
  return 0;
}
