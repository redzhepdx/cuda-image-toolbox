#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "misc.h"
#include "filter.h"

using namespace cv;
using namespace std;

int main (int argc, char** argv) {
  if (argc != 2) {
    cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
    return -1;
  }

  Mat imageMat;
  imageMat = imread(argv[1], CV_LOAD_IMAGE_COLOR); // Read the file.
  if (!imageMat.data) {
    cout << "Could not open or find the image." << endl;
    return -1;
  }

  /*
    TODO: Add image processing functions here.
  */

  transform();

  namedWindow("Filtered Image", WINDOW_AUTOSIZE); // Create a window for display.
  imshow("Filtered Image", imageMat);

  waitKey(0);
  return 0;
}
