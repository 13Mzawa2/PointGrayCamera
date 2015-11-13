#include "GrayCodePatternProjection.h"

using namespace cv;
using namespace std;

GrayCodePatternProjection::GrayCodePatternProjection()
{
}

GrayCodePatternProjection::GrayCodePatternProjection(Size projSize, Size camSize)
{
	init(projSize, camSize);
}

//	����������
void GrayCodePatternProjection::init(cv::Size projSize, cv::Size camSize)
{
	projectorSize = projSize; cameraSize = camSize;

}

GrayCodePatternProjection::~GrayCodePatternProjection()
{
}

//	Convert Binary Code to Gray Code
//	Example:
//	bin = 0x4c = 0b01001100
//
//	  0b01001100
//	^ 0b00100110 (= bin >> 1)
//	-------------------------
//	  0b01101010 = 0x6a = gray
int GrayCodePatternProjection::bin2gray(int bin)
{
	return bin ^ (bin >> 1);
}

//	Convert Gray Code to Binary Code
//	Example:
//	gray = 0x6a = 0b01101010
//	
//	bin = gray       = 0b01101010
//	mask = gray >> 1 = 0b00110101
//	bin = bin ^ mask = 0b01011111
//	mask = mask >> 1 = 0b00011010
//	bin = bin ^ mask = 0b01000101
//	mask = mask >> 1 = 0b00001101
//	bin = bin ^ mask = 0b01001000
//	mask = mask >> 1 = 0b00000110
//	bin = bin ^ mask = 0b01001110
//	mask = mask >> 1 = 0b00000011
//	bin = bin ^ mask = 0b01001101
//	mask = mask >> 1 = 0b00000001(End)
//	bin = bin ^ mask = 0b01001100 = 0x4c = bin
int GrayCodePatternProjection::gray2bin(int gray)
{
	int bin = gray, mask = gray;
	for (mask >> 1; mask != 0; mask = mask >> 1)
		bin = bin ^ mask;
	return bin;
}

//	�O���C�R�[�h�p�^�[����\���s��̍쐬
//	�s��̊e�s���O���C�R�[�h�p�^�[����\��
//
//	Output = {patternListW, patternListH}
//	Example:
//	projSize = [16, 6]
//	
//	patternListW =
//		[0000000011111111] -> pattern 1
//		[0000111111110000] -> pattern 2
//		[0011110000111100] -> pattern 3
//		[0110011001100110] -> pattern 4
//
//	patternListH = 
//		[000011] -> pattern 1
//		[001111] -> pattern 2
//		[011001] -> pattern 3
vector<Mat> GrayCodePatternProjection::makeGrayCodePatternLists(Size projSize)
{
	//	�ő�r�b�g��l�̃J�E���g
	int lw = 0, lh = 0;		//	�v���W�F�N�^�摜�T�C�Y���ꂼ��̍ő�r�b�g��
	for (int x = projSize.width-1; x > 0; x = x >> 1) lw++;
	for (int y = projSize.height-1; y > 0; y = y >> 1) lh++;
	//	0 ~ projSize.width (heigt) �̃O���C�R�[�h���r�b�g�����čs��Ɋi�[
	//	�s�����o���΃p�^�[���ɂȂ�悤�ɂ���
	Mat patternListW(Size(projSize.width, lw), CV_8UC1),
		patternListH(Size(projSize.height, lh), CV_8UC1);
	for (int i = 0; i < projSize.width; i++){
		for (int j = lw; j > 0; j--){			//	�ŏ��bit���璲�ׂĂ���
			//	�O���C�R�[�h�� i �� j bit�ڂ�1�ł����1���C�����łȂ����0������
			patternListW.at<uchar>(lw - j, i) = (bin2gray(i) & (1 << (j - 1))) && 1;
		}
	}
	for (int i = 0; i < projSize.height; i++){
		for (int j = lh; j > 0; j--){			//	�ŏ��bit���璲�ׂĂ���
			//	�O���C�R�[�h�� i �� j bit�ڂ�1�ł����1���C�����łȂ����0������
			patternListH.at<uchar>(lh - j, i) = (bin2gray(i) & (1 << (j - 1))) && 1;
		}
	}
	vector<Mat> patternLists;
	patternLists.push_back(patternListW);
	patternLists.push_back(patternListH);

	return patternLists;
}

//	�p�^�[�����X�g����O���C�R�[�h�p�^�[���𐶐�
//	isVertical: true = �c�����Cfalse = ������
//	1�Ԗڂ��ŏ�ʃr�b�g
std::vector<cv::Mat> GrayCodePatternProjection::makeGrayCodeImages(cv::Size projSize, cv::Mat patternList, bool isVertical)
{
	vector<Mat> patternImages;
	if (isVertical)
	{
		for (int i = 0; i < patternList.rows; i++)
		{
			Mat pattern(projSize, CV_8UC1);
			resize(255 * patternList.row(i), pattern, projSize, INTER_NEAREST);
			patternImages.push_back(pattern);
			imshow("test", pattern);
			waitKey(0);
		}
	}
	else
	{
		for (int i = 0; i < patternList.rows; i++)
		{
			Mat pattern(projSize, CV_8UC1);
			resize(255 * patternList.row(i).t(), pattern, projSize, INTER_NEAREST);
			patternImages.push_back(pattern);
			imshow("test", pattern);
			waitKey(0);
		}
	}

	return patternImages;
}
