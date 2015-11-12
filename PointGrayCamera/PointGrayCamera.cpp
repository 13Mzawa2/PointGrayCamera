#include "FlyCap2CVWrapper.h"

using namespace FlyCapture2;

int main(void)
{
	FlyCap2CVWrapper cam;

	// capture loop
	char key = 0;
	while (key != 'q')
	{
		// Get the image
		cv::Mat image = cam.readImage();
		flip(image, image, 1);
		cv::imshow("image", image);
		key = cv::waitKey(30);
	}

	return 0;
}