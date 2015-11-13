#pragma once

#include <opencv2/opencv.hpp>

class GrayCodePatternProjection
{
public:
	std::vector<cv::Mat> patternsW, patternsH, patternsWN, patternsHN; 
	cv::Mat patternListW, patternListH;
	cv::Mat mask;
	cv::Mat camera2ProjectorMap;
	cv::Size projectorSize, cameraSize;

	GrayCodePatternProjection();
	GrayCodePatternProjection(cv::Size projSize, cv::Size camSize);
	~GrayCodePatternProjection();
	int bin2gray(int bin);
	int gray2bin(int gray);
	std::vector<cv::Mat> makeGrayCodePatternLists(cv::Size projSize);
	std::vector<cv::Mat> makeGrayCodeImages(cv::Size projSize, cv::Mat patternList, bool isVertical);
	void init(cv::Size projectorSize, cv::Size cameraSize);
};

