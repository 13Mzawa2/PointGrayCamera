#pragma once

#include <opencv2/opencv.hpp>

class GrayCodePatternProjection
{
public:
	std::vector<cv::Mat> patternsW, patternsH, patternsWN, patternsHN; 
	std::vector<cv::Mat> captureW, captureH, captureWN, captureHN;
	cv::Mat patternListW, patternListH;			//	行数rowsがpatternsの要素数と一致
	cv::Mat mapX, mapY, mask;				//	結果をここに保存
	cv::Mat camera2ProjectorMap;
	cv::Size projectorSize, cameraSize;

	GrayCodePatternProjection();
	GrayCodePatternProjection(cv::Size projSize, cv::Size camSize);
	~GrayCodePatternProjection();
	int bin2gray(int bin);
	int gray2bin(int gray);
	void makeGrayCodePatternLists(void);
	void makeGrayCodeImages(void);
	void init(cv::Size projectorSize, cv::Size cameraSize);
	void getMask(int thresh = 20);
	void loadCapPatterns(std::vector<cv::Mat> cap);
	void decodePatterns(void);
	void showMaps(void);
};

