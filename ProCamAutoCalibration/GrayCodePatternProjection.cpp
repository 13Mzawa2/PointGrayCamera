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
	//	パターン配列を初期化
	patternsW.clear(); patternsWN.clear();
	patternsH.clear(); patternsHN.clear();
	
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

//	マスク画像を作成する
void GrayCodePatternProjection::getMask(Mat white, Mat black, int thresh)
{
	Mat wg, bg;
	cvtColor(white, wg, CV_BGR2GRAY);
	cvtColor(black, bg, CV_BGR2GRAY);
	Mat m = wg.clone();
	for (int i = 0; i < m.rows; i++){
		for (int j = 0; j < m.cols; j++)
			m.at<uchar>(i, j) = ((wg.at<uchar>(i, j) - bg.at<uchar>(i, j)) > thresh) ? 1 : 0;
	}
	mask = m.clone();
}

//	撮影したパターンを読み込む
//	capは w[0], wn[0], w[1], wn[1], ..., wn[wrows], h[0], hn[0], ..., hn[hrows]　の順
//	capはCV_8UC3なので注意
void GrayCodePatternProjection::loadCapPatterns(vector<Mat> cap)
{
	//	パターン配列の初期化
	captureW.clear(); captureWN.clear();
	captureH.clear(); captureHN.clear();
	//	投影画像の読み込み
	for (int i = 0; i < patternListW.rows; i++)
	{
		captureW.push_back(cap[2 * i]);
		captureWN.push_back(cap[2 * i + 1]);
	}
	for (int i = 0; i < patternListH.rows; i++)
	{
		captureH.push_back(cap[patternListW.rows + 2 * i]);
		captureHN.push_back(cap[patternListW.rows + 2 * i + 1]);
	}
}

//	読み込んだパターンを解読してマップ化
//	unsigned int16で座標値を入力
void GrayCodePatternProjection::decodePatterns()
{
	Mat temp(cameraSize, CV_16UC2);
	for (int i = 0; i < temp.rows; i++)
	{
		for (int j = 0; j < temp.cols; j++)
		{	//	画素毎に実行
			int it = temp.step * i + temp.channels() * j;
			//	x方向
			int xgray = 0;
			for (int k = 0; k < patternListW.rows; k++)
			{	//	k枚目の画素の輝度がネガより大きければ1，そうでなければ0を対応するビットで立てる
				Mat capWk, capWNk;
				cvtColor(captureW[k], capWk, CV_BGR2GRAY);
				cvtColor(captureWN[k], capWNk, CV_BGR2GRAY);
				xgray = xgray | ((capWk.data[it] - capWNk.data[it]) > 0 ? 1 : 0) << (patternListW.rows - 1 - k);
			}
			int xbin = gray2bin(xgray);
			//	y方向
			int ygray = 0;
			for (int k = 0; k < patternListH.rows; k++)
			{	//	k枚目の画素の輝度がネガより大きければ1，そうでなければ0を対応するビットで立てる
				Mat capHk, capHNk;
				cvtColor(captureH[k], capHk, CV_BGR2GRAY);
				cvtColor(captureHN[k], capHNk, CV_BGR2GRAY);
				ygray = ygray | ((capHk.data[it] - capHNk.data[it]) > 0 ? 1 : 0) << (patternListH.rows - 1 - k);
			}
			int ybin = gray2bin(ygray);
			// tempへデコードした座標値を代入
			temp.data[it + 0] = xbin; 
			temp.data[it + 1] = ybin;
		}
	}
}