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

//	初期化処理
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

//	グレイコードパターンを表す行列の作成
//	行列の各行がグレイコードパターンを表す
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
	//	最大ビット長lのカウント
	int lw = 0, lh = 0;		//	プロジェクタ画像サイズそれぞれの最大ビット数
	for (int x = projectorSize.width-1; x > 0; x = x >> 1) lw++;
	for (int y = projectorSize.height - 1; y > 0; y = y >> 1) lh++;
	//	0 ~ projSize.width (heigt) のグレイコードをビット化して行列に格納
	//	行を取り出せばパターンになるようにする
	patternListW = Mat(Size(projectorSize.width, lw), CV_8UC1);
	patternListH = Mat(Size(projectorSize.height, lh), CV_8UC1);
	for (int i = 0; i < projectorSize.width; i++){
		for (int j = lw; j > 0; j--){			//	最上位bitから調べていく
			//	グレイコードの i の j bit目が1であれば1を，そうでなければ0を入れる
			patternListW.at<uchar>(lw - j, i) = (bin2gray(i) & (1 << (j - 1))) && 1;
		}
	}
	for (int i = 0; i < projectorSize.height; i++){
		for (int j = lh; j > 0; j--){			//	最上位bitから調べていく
			//	グレイコードの i の j bit目が1であれば1を，そうでなければ0を入れる
			patternListH.at<uchar>(lh - j, i) = (bin2gray(i) & (1 << (j - 1))) && 1;
		}
	}
	
}

//	パターンリストからグレイコードパターンを生成
//	isVertical: true = 縦方向，false = 横方向
//	1番目が最上位ビット
void GrayCodePatternProjection::makeGrayCodeImages(void)
{
	for (int i = 0; i < patternListW.rows; i++)
	{
		Mat pattern(projectorSize, CV_8UC1);
		resize(255 * patternListW.row(i), pattern, projectorSize, INTER_NEAREST);
		patternsW.push_back(pattern);
		patternsWN.push_back(~pattern);		//	デコード安定化のためのネガ
	}
	for (int i = 0; i < patternListH.rows; i++)
	{
		Mat pattern(projectorSize, CV_8UC1);
		resize(255 * patternListH.row(i).t(), pattern, projectorSize, INTER_NEAREST);
		patternsH.push_back(pattern); 
		patternsHN.push_back(~pattern);		//	デコード安定化のためのネガ
	}

}
