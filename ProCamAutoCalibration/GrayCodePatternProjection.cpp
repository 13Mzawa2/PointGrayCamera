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
	makeGrayCodePatternLists();
	makeGrayCodeImages();
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
void GrayCodePatternProjection::makeGrayCodePatternLists(void)
{
	//	�ő�r�b�g��l�̃J�E���g
	int lw = 0, lh = 0;		//	�v���W�F�N�^�摜�T�C�Y���ꂼ��̍ő�r�b�g��
	for (int x = projectorSize.width-1; x > 0; x = x >> 1) lw++;
	for (int y = projectorSize.height - 1; y > 0; y = y >> 1) lh++;
	//	0 ~ projSize.width (heigt) �̃O���C�R�[�h���r�b�g�����čs��Ɋi�[
	//	�s�����o���΃p�^�[���ɂȂ�悤�ɂ���
	patternListW = Mat(Size(projectorSize.width, lw), CV_8UC1);
	patternListH = Mat(Size(projectorSize.height, lh), CV_8UC1);
	for (int i = 0; i < projectorSize.width; i++){
		for (int j = lw; j > 0; j--){			//	�ŏ��bit���璲�ׂĂ���
			//	�O���C�R�[�h�� i �� j bit�ڂ�1�ł����1���C�����łȂ����0������
			patternListW.at<uchar>(lw - j, i) = (bin2gray(i) & (1 << (j - 1))) && 1;
		}
	}
	for (int i = 0; i < projectorSize.height; i++){
		for (int j = lh; j > 0; j--){			//	�ŏ��bit���璲�ׂĂ���
			//	�O���C�R�[�h�� i �� j bit�ڂ�1�ł����1���C�����łȂ����0������
			patternListH.at<uchar>(lh - j, i) = (bin2gray(i) & (1 << (j - 1))) && 1;
		}
	}
	
}

//	�p�^�[�����X�g����O���C�R�[�h�p�^�[���𐶐�
//	isVertical: true = �c�����Cfalse = ������
//	1�Ԗڂ��ŏ�ʃr�b�g
void GrayCodePatternProjection::makeGrayCodeImages(void)
{
	for (int i = 0; i < patternListW.rows; i++)
	{
		Mat pattern(projectorSize, CV_8UC1);
		resize(255 * patternListW.row(i), pattern, projectorSize, INTER_NEAREST);
		patternsW.push_back(pattern);
		patternsWN.push_back(~pattern);		//	�f�R�[�h���艻�̂��߂̃l�K
	}
	for (int i = 0; i < patternListH.rows; i++)
	{
		Mat pattern(projectorSize, CV_8UC1);
		resize(255 * patternListH.row(i).t(), pattern, projectorSize, INTER_NEAREST);
		patternsH.push_back(pattern); 
		patternsHN.push_back(~pattern);		//	�f�R�[�h���艻�̂��߂̃l�K
	}

}
