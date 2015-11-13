#pragma once

#include <opencv2/opencv.hpp>

class GrayCodePatternProjection
{
public:
	std::vector<cv::Mat> patternsW, patternsH, patternsWN, patternsHN; 
	cv::Mat patternListW, patternListH;			//	s”rows‚ªpatterns‚Ì—v‘f”‚Æˆê’v
	cv::Mat mask;
	cv::Mat camera2ProjectorMap;
	cv::Size projectorSize, cameraSize;

	GrayCodePatternProjection();
	GrayCodePatternProjection(cv::Size projSize, cv::Size camSize);
	~GrayCodePatternProjection();
	int bin2gray(int bin);
	int gray2bin(int gray);
	void makeGrayCodePatternLists(void);
	void makeGrayCodeImages();
	void init(cv::Size projectorSize, cv::Size cameraSize);
};

