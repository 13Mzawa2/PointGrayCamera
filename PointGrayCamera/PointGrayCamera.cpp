#include "FlyCap2CVWrapper.h"

using namespace FlyCapture2;

int main(void)
{
	FlyCap2CVWrapper cam;
	int count = 0;
	double time100 = 0;
	int waitparam = 1;
	// capture loop
	char key = 0;
	while (key != 'q')
	{

		// Get the image
		cv::Mat image = cam.readImage();
		flip(image, image, 1);
		cv::imshow("image", image);

		double f = 1000.0 / cv::getTickFrequency();
		int64 time = cv::getTickCount();

		key = cv::waitKey(waitparam);

		if (count == 1000)
		{// TickCount‚Ì•Ï‰»‚ð[ms]’PˆÊ‚Å•\Ž¦ 1000‰ñ•½‹Ï
			std::cout << "param = " << waitparam << ", time = " << time100 / (count+1) << " [ms]" << std::endl;
			time100 = 0;
			count = 0;
			waitparam++;
		}
		else
		{
			count++;
			time100 += (cv::getTickCount() - time)*f;
		}
	}

	return 0;
}