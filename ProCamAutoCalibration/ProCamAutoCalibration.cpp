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
const double chessboardInterval = 300.0;		//	チェスボード検出インターバル
const Size projSize(1024, 768);
const int projectionInterval = 100;			//	グレイコードパターンの撮影インターバル
const char* projectorWindowName = "Projector";		//	全画面表示するプロジェクタ画面のID
const char* cameraWindowName = "Camera";			//	カメラ画面のID

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

	//	プロジェクタの準備
	GrayCodePatternProjection gcp(projSize, img.size());
	namedWindow(projectorWindowName, CV_WINDOW_FREERATIO);
	moveWindow(projectorWindowName, -1024, 0);
	setWindowProperty(projectorWindowName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	Mat whiteLight(projSize, CV_8U, Scalar(255));
	imshow(projectorWindowName, whiteLight);

	//-----------------------------------------
	//	1. カメラキャリブレーション
	//-----------------------------------------
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
	std::cout << "Please Move " << patternSize << " Chessboard in your projection area." << endl;
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
		if (patternfound && timer > chessboardInterval)
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
	//	Calibration Start
	//	パラメータ推定
	cout << "Calculating Camera Inner Matrix and Distortion Parameters..." << endl;
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
		Mat(), cameraMatrix, img.size(), CV_32FC1,
		map1, map2);
	cout << "Camera Calibration Ended.\n"
		<< "Press any key, and projector-camera calibration will begin." << endl;
	cv::waitKey(0);

	//-----------------------------------------
	//	2.	カメラ・プロジェクタ画素対応取得
	//		（グレイコードパターン投影法）
	//		歪み補正済みカメラとプロジェクタ
	//-----------------------------------------
	cout << "Projector-Camera Calibration is starting..." << endl;

	//	グレイコードパターン投影
	vector<Mat> captures;
	for (int i = 0; i < gcp.patternListW.rows; i++)
	{	//	W, WN
		imshow(projectorWindowName, gcp.patternsW[i]);
		waitKey(projectionInterval);
		img = cam.readImage();
		flip(img, img, 1);
		remap(img, img, map1, map2, INTER_CUBIC);
		captures.push_back(img);
		imshow(projectorWindowName, gcp.patternsWN[i]);
		waitKey(projectionInterval);
		img = cam.readImage();
		flip(img, img, 1);
		remap(img, img, map1, map2, INTER_CUBIC);
		captures.push_back(img);
	}
	for (int i = 0; i < gcp.patternListH.rows; i++)
	{	//	H, HN
		imshow(projectorWindowName, gcp.patternsH[i]);
		waitKey(projectionInterval);
		img = cam.readImage();
		flip(img, img, 1);
		remap(img, img, map1, map2, INTER_CUBIC);
		captures.push_back(img);
		imshow(projectorWindowName, gcp.patternsHN[i]);
		waitKey(projectionInterval);
		img = cam.readImage();
		flip(img, img, 1);
		remap(img, img, map1, map2, INTER_CUBIC);
		captures.push_back(img);
	}

	//	グレイコードパターン解析
	cout << "Decoding gray code patterns..." << endl;
	gcp.loadCapPatterns(captures);
	gcp.decodePatterns();
	cout << "Correspondence maps are generated." << endl;
	gcp.showMaps();

	waitKey(0);

	//-----------------------------------------
	//	3. プロジェクタキャリブレーション
	//-----------------------------------------
	//	取得済みのカメラ座標2次元コーナーの歪みを除去
	vector<vector<Point2f>> corners2dUndistorted;
	for (int i = 0; i < imgNum; i++)
	{
		vector<Point2f> points;
		for (int j = 0; j < patternSize.height; j++){
			for (int k = 0; k < patternSize.width; k++)
			{	//	getRectSubPix()を3x3の矩形サイズで利用すことでピクセルをバイリニア補間
				Point2f p = (corners2d[i])[j*patternSize.width + k];		//	歪み補正前
				Size patchSize(3, 3);
				Mat patchX, patchY;
				getRectSubPix(map1, patchSize, p, patchX);
				getRectSubPix(map2, patchSize, p, patchY);
				Point2f pu(patchX.at<float>(1, 1), patchY.at<float>(1, 1));			//	歪み補正後
				points.push_back(pu);
			}
		}
		corners2dUndistorted.push_back(points);
	}
	//	マップを使ってプロジェクタ座標に変換
	vector<vector<Point2f>> corners2dProj;
	for (int i = 0; i < imgNum; i++)
	{
		vector<Point2f> points;
		for (int j = 0; j < patternSize.height; j++){
			for (int k = 0; k < patternSize.width; k++)
			{	//	getRectSubPix()を3x3の矩形サイズで利用すことでピクセルをバイリニア補間
				Point2f pcam = (corners2dUndistorted[i])[j*patternSize.width + k];		//	カメラのサブピクセル座標
				Size patchSize(3, 3);
				Mat patchX, patchY;
				getRectSubPix(gcp.mapX, patchSize, pcam, patchX);
				getRectSubPix(gcp.mapY, patchSize, pcam, patchY);
				Point2f pproj(patchX.at<float>(1, 1), patchY.at<float>(1, 1));		//	プロジェクタのサブピクセル座標
				points.push_back(pproj);
			}
		}
		corners2dProj.push_back(points);
	}
		//	コーナー点を投影(テスト用)
		Mat mapped(projSize, CV_8UC3, Scalar::all(255));
		drawChessboardCorners(mapped, patternSize, Mat(corners2dProj[imgNum-1]), true);
		imshow(projectorWindowName, mapped);
		waitKey(0);

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