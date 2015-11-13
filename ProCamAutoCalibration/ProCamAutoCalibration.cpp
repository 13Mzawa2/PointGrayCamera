/*******************************************
		OpenCV Semi-Auto Calibration
********************************************/

#include "FlyCap2CVWrapper.h"
#include "GrayCodePatternProjection.h"

using namespace std;
using namespace cv;

//	Constants
const int imgNum = 30;			//	画像数
const Size patternSize(7, 10);
const int allPoints = imgNum * patternSize.width * patternSize.height;
const double chessSize = 22.5;		//	mm
const double interval = 300.0;
const Size projSize(1024, 768);

//	Results
Mat cameraMatrix;		//	カメラ内部行列
Mat distCoeffs;			//	レンズ歪みベクトル
vector<Mat> rvecs;		//	個々のチェスボードから見たカメラの回転ベクトル
vector<Mat> tvecs;		//	個々のチェスボードから見たカメラの並進ベクトル
Mat map1, map2;			//	歪み補正マップ

int main(void)
{
	FlyCap2CVWrapper cam;
	cv::Mat img = cam.readImage();

	GrayCodePatternProjection gcp(projSize, img.size());
	namedWindow("GrayCodePattern", CV_WINDOW_FREERATIO);
	moveWindow("GrayCodePattern", -1024, 0);
	setWindowProperty("GrayCodePattern", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	waitKey(0);
	Mat imgWhite, imgBlack;
	Mat whiteLight(projSize, CV_8UC1, Scalar(255));
	Mat blackLight(projSize, CV_8UC1, Scalar(0));
	imshow("GrayCodePattern", blackLight);
	waitKey(500);
	imgBlack = cam.readImage();
	imshow("Mask", imgBlack);
	waitKey(500);
	imshow("GrayCodePattern", whiteLight);
	waitKey(500);
	imgWhite = cam.readImage();
	imshow("Mask", imgWhite);
	waitKey(500);
	gcp.getMask(imgWhite, imgBlack, 10);
	imshow("Mask", 255*gcp.mask);
	waitKey(0);

	for (int i = 0; i < gcp.patternListW.rows; i++)
	{
		imshow("GrayCodePattern", gcp.patternsW[i]);
		waitKey(0);
		imshow("GrayCodePattern", gcp.patternsWN[i]);
		waitKey(0);
	}
	for (int i = 0; i < gcp.patternListH.rows; i++)
	{
		imshow("GrayCodePattern", gcp.patternsH[i]);
		waitKey(0);
		imshow("GrayCodePattern", gcp.patternsHN[i]);
		waitKey(0);
	}

	vector<vector<Point3f>> corners3d(imgNum);		//	チェスボード座標系での3次元点
	vector<vector<Point2f>> corners2d;				//	検出されたカメラ座標コーナー点

	//	チェスボード座標上の3次元点
	for (int i = 0; i < imgNum; i++){
		for (int j = 0; j < patternSize.height; j++){
			for (int k = 0; k < patternSize.width; k++)
				corners3d[i].push_back(Point3f(k * chessSize, j * chessSize, 0.0));
		}
	}
	//	チェスボード自動検出
	//	画角内でチェスボードを動かすと自動的にとれる
	for (int imgFound = 0; imgFound < imgNum;)
	{
		//	タイマー
		static double timer = 0.0;
		static int64 count = getTickCount();
		//	画像取得
		img = cam.readImage();
		flip(img, img, 1);
		//	チェスボード検出
		//	1000msのインターバルを設けている
		vector<Point2f> corners;
		bool patternfound = findChessboardCorners(img, patternSize, corners,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		if (patternfound && timer > interval)
		{	//	見つかった場合
			imgFound++;
			timer = 0.0; count = getTickCount();		//	インターバルタイマーを初期化
			cout << "Chessboard No. " << imgFound << " / " << imgNum << " is found at " << corners[0] << endl;
			corners2d.push_back(corners);
			drawChessboardCorners(img, patternSize, Mat(corners), patternfound);
		}
		imshow("Chessboard", img);
		if (waitKey(1) == 'q') exit(0);
		timer = (getTickCount() - count)*1000.0 / getTickFrequency();
	}
	//	Start Calibration
	//	パラメータ推定
	cout << "Start Calibration..." << endl;
	cv::calibrateCamera(
		corners3d, corners2d,
		img.size(),
		cameraMatrix, distCoeffs,
		rvecs, tvecs);
	cout << "Camera Matirx = \n" << cameraMatrix << "\nDistortion Coeffs = \n" << distCoeffs << endl;
	//	歪み補正マップ計算
	cout << "Making Undistort Map..." << endl;
	initUndistortRectifyMap(
		cameraMatrix, distCoeffs, 
		Mat(), cameraMatrix, img.size(), CV_16SC2,
		map1, map2);
	cout << "Calibration Ended." << endl;

	// capture loop
	char key = 0;
	while (key != 'q')
	{	// Get the image
		cv::Mat image = cam.readImage();
		flip(image, image, 1);
		Mat imgUndistorted;
		remap(image, imgUndistorted, map1, map2, INTER_CUBIC);
		cv::imshow("calibrated", imgUndistorted);
		key = cv::waitKey(1);
	}

	//	save as XML, PNG
	FileStorage fs("calibdata.xml", FileStorage::WRITE);
	fs << "calibration"
		<< "{"
		<< "CameraMatrix" << cameraMatrix
		<< "DistCoeffs" << distCoeffs
		<< "}";

	return 0;
}