/*******************************************
		OpenCV Semi-Auto Calibration
********************************************/

#include "FlyCap2CVWrapper.h"
#include "GrayCodePatternProjection.h"

using namespace std;
using namespace cv;

//	Constants
const int imgNum = 30;			//	�摜��
const Size patternSize(7, 10);
const int allPoints = imgNum * patternSize.width * patternSize.height;
const double chessSize = 22.5;		//	mm
const double chessboardInterval = 300.0;		//	�`�F�X�{�[�h���o�C���^�[�o��
const Size projSize(1024, 768);
const int projectionInterval = 100;			//	�O���C�R�[�h�p�^�[���̎B�e�C���^�[�o��
const char* projectorWindowName = "Projector";		//	�S��ʕ\������v���W�F�N�^��ʂ�ID
const char* cameraWindowName = "Camera";			//	�J������ʂ�ID

//	Results
Mat cameraMatrix;		//	�J���������s��
Mat distCoeffs;			//	�����Y�c�݃x�N�g��
Mat cameraMatrixProj;	//	�v���W�F�N�^�����s��
Mat distCoeffsProj;		//	�v���W�F�N�^�̃����Y�c�݃x�N�g��
vector<Mat> rvecs, rvecsProj;		//	�X�̃`�F�X�{�[�h���猩���J�����̉�]�x�N�g��
vector<Mat> tvecs, tvecsProj;		//	�X�̃`�F�X�{�[�h���猩���J�����̕��i�x�N�g��
Mat map1, map2;				//	�c�ݕ␳�}�b�v
Mat map1Proj, map2Proj;		//	�v���W�F�N�^�̘c�ݕ␳�}�b�v
Mat RProCam, TProCam;	//	�J����-�v���W�F�N�^�Ԃ̉�]�s��E���i�x�N�g��
Mat EProCam, FProCam;	//	�J����-�v���W�F�N�^�Ԃ̊�{�s��C��b�s��

Point3d operator*(Mat M, const Point3d& p);

int main(void)
{
	FlyCap2CVWrapper cam;
	cv::Mat img = cam.readImage();

	//	�v���W�F�N�^�̏���
	GrayCodePatternProjection gcp(projSize, img.size());
	namedWindow(projectorWindowName, CV_WINDOW_FREERATIO);
	moveWindow(projectorWindowName, -1024, 0);
	setWindowProperty(projectorWindowName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	Mat whiteLight(projSize, CV_8U, Scalar(255));
	imshow(projectorWindowName, whiteLight);

	//-----------------------------------------
	//	1. �J�����L�����u���[�V����
	//-----------------------------------------
	vector<vector<Point3f>> corners3d(imgNum);		//	�`�F�X�{�[�h���W�n�ł�3�����_
	vector<vector<Point2f>> corners2d;				//	���o���ꂽ�J�������W�R�[�i�[�_

	//	�`�F�X�{�[�h���W���3�����_
	for (int i = 0; i < imgNum; i++){
		for (int j = 0; j < patternSize.height; j++){
			for (int k = 0; k < patternSize.width; k++)
				corners3d[i].push_back(Point3f(k * chessSize, j * chessSize, 0.0));
		}
	}
	//	�`�F�X�{�[�h�������o
	//	��p���Ń`�F�X�{�[�h�𓮂����Ǝ����I�ɂƂ��
	std::cout << "Please Move " << patternSize << " Chessboard in your projection area." << endl;
	for (int imgFound = 0; imgFound < imgNum;)
	{
		//	�^�C�}�[
		static double timer = 0.0;
		static int64 count = getTickCount();
		//	�摜�擾
		img = cam.readImage();
		flip(img, img, 1);
		//	�`�F�X�{�[�h���o
		//	1000ms�̃C���^�[�o����݂��Ă���
		vector<Point2f> corners;
		bool patternfound = findChessboardCorners(img, patternSize, corners,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		if (patternfound && timer > chessboardInterval)
		{	//	���������ꍇ
			imgFound++;
			timer = 0.0; count = getTickCount();		//	�C���^�[�o���^�C�}�[��������
			cout << "Chessboard No. " << imgFound << " / " << imgNum << " is found at " << corners[0] << endl;
			corners2d.push_back(corners);
			drawChessboardCorners(img, patternSize, Mat(corners), patternfound);
		}
		imshow("Chessboard", img);
		if (waitKey(1) == 'q') exit(0);
		timer = (getTickCount() - count)*1000.0 / getTickFrequency();
	}
	//	Calibration Start
	//	�p�����[�^����
	cout << "Calculating Camera Inner Matrix and Distortion Parameters..." << endl;
	cv::calibrateCamera(
		corners3d, corners2d,
		img.size(),
		cameraMatrix, distCoeffs,
		rvecs, tvecs);
	cout << "Camera Matirx = \n" << cameraMatrix << "\nDistortion Coeffs = \n" << distCoeffs << endl;
	//	�c�ݕ␳�}�b�v�v�Z
	cout << "Making Undistort Map..." << endl;
	initUndistortRectifyMap(
		cameraMatrix, distCoeffs,
		Mat(), cameraMatrix, img.size(), CV_32FC1,
		map1, map2);
	cout << "Camera Calibration Ended.\n"
		<< "Press any key, and projector-camera calibration will begin." << endl;
	cv::waitKey(0);

	//-----------------------------------------
	//	2.	�J�����E�v���W�F�N�^��f�Ή��擾
	//		�i�O���C�R�[�h�p�^�[�����e�@�j
	//		�c�ݕ␳�ς݃J�����ƃv���W�F�N�^
	//-----------------------------------------
	cout << "Projector-Camera Calibration is starting..." << endl;

	//	�O���C�R�[�h�p�^�[�����e
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

	//	�O���C�R�[�h�p�^�[�����
	cout << "Decoding gray code patterns..." << endl;
	gcp.loadCapPatterns(captures);
	gcp.decodePatterns();
	cout << "Correspondence maps are generated." << endl;
	gcp.showMaps();

	//-----------------------------------------
	//	3. �v���W�F�N�^�L�����u���[�V����
	//-----------------------------------------
	cout << "Calculating Chessboard Corners from Projector..." << endl;
	//	�擾�ς݂̃J�������W2�����R�[�i�[�̘c�݂�����
	vector<vector<Point2f>> corners2dUndistorted;
	for (int i = 0; i < imgNum; i++)
	{
		vector<Point2f> points;
		for (int j = 0; j < patternSize.height; j++){
			for (int k = 0; k < patternSize.width; k++)
			{	//	getRectSubPix()��3x3�̋�`�T�C�Y�ŗ��p�����ƂŃs�N�Z�����o�C���j�A���
				Point2f p = (corners2d[i])[j*patternSize.width + k];		//	�c�ݕ␳�O
				Size patchSize(3, 3);
				Mat patchX, patchY;
				getRectSubPix(map1, patchSize, p, patchX);
				getRectSubPix(map2, patchSize, p, patchY);
				Point2f pu(patchX.at<float>(1, 1), patchY.at<float>(1, 1));			//	�c�ݕ␳��
				points.push_back(pu);
			}
		}
		corners2dUndistorted.push_back(points);
	}
	//	�}�b�v���g���ăv���W�F�N�^���W�ɕϊ�
	vector<vector<Point2f>> corners2dProj;
	for (int i = 0; i < imgNum; i++)
	{
		vector<Point2f> points;
		for (int j = 0; j < patternSize.height; j++){
			for (int k = 0; k < patternSize.width; k++)
			{	//	getRectSubPix()��3x3�̋�`�T�C�Y�ŗ��p�����ƂŃs�N�Z�����o�C���j�A���
				Point2f pcam = (corners2dUndistorted[i])[j*patternSize.width + k];		//	�J�����̃T�u�s�N�Z�����W
				Size patchSize(3, 3);
				Mat patchX, patchY;
				getRectSubPix(gcp.mapX, patchSize, pcam, patchX);
				getRectSubPix(gcp.mapY, patchSize, pcam, patchY);
				Point2f pproj(patchX.at<float>(1, 1), patchY.at<float>(1, 1));		//	�v���W�F�N�^�̃T�u�s�N�Z�����W
				points.push_back(pproj);
			}
		}
		corners2dProj.push_back(points);
	}
	//	�R�[�i�[�_�𓊉e(�e�X�g�p)
	cout << "Please check whether this chess corners are correct." << endl;
	Mat mapped(projSize, CV_8UC3, Scalar::all(255));
	drawChessboardCorners(mapped, patternSize, Mat(corners2dProj[imgNum-1]), true);
	imshow(projectorWindowName, mapped);
	waitKey(0);

	//	�v���W�F�N�^�L�����u���[�V����
	//	�p�����[�^����
	cout << "Calculating Projector Inner Matrix and Distortion Parameters..." << endl;
	cv::calibrateCamera(
		corners3d, corners2dProj,
		img.size(),
		cameraMatrixProj,distCoeffsProj,
		rvecsProj, tvecsProj);
	cout << "Projector Camera Matirx = \n" << cameraMatrix << "\nProjector Distortion Coeffs = \n" << distCoeffs << endl;
	//	�c�ݕ␳�}�b�v�v�Z
	cout << "Making Projector Undistort Map..." << endl;
	initUndistortRectifyMap(
		cameraMatrixProj, distCoeffsProj,
		Mat(), cameraMatrixProj, img.size(), CV_32FC1,
		map1Proj, map2Proj);
	//	�J���� - �v���W�F�N�^�O���s��̐���
	cout << "Caluculating Camera-Projector outer parameters..." << endl;
	cv::stereoCalibrate(
		corners3d, corners2d, corners2dProj,		//	�R�[�i�[�_�Q
		cameraMatrix, distCoeffs,					//	�J�����p�����[�^
		cameraMatrixProj, distCoeffsProj,			//	�v���W�F�N�^�p�����[�^	
		img.size(),									//	�摜�T�C�Y
		RProCam, TProCam, EProCam, FProCam,			//	�J�����ɑ΂���v���W�F�N�^�̊O���p�����[�^
		CALIB_FIX_INTRINSIC,						//	�X�ɋ��߂������p�����[�^�ŌŒ�i�O���p�����[�^��������j
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6)
		);
	cout << "R = \n" << RProCam
		<< "\nT = \n" << TProCam
		<< "\nE = \n" << EProCam
		<< "\nF = \n" << FProCam
		<< "\nProjector Calibration Ended." << endl;
	cv::waitKey(0);


	// capture loop
	char key = 0;
	while (key != 'q')
	{	// Get the image
		cv::Mat image = cam.readImage();
		flip(image, image, 1);
		Mat imgUndistorted;
		remap(image, imgUndistorted, map1, map2, INTER_CUBIC);
		//	�`�F�X�{�[�h���o
		vector<Point2d> corners;
		bool patternfound = findChessboardCorners(imgUndistorted, patternSize, corners,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		if (patternfound)
		{	//	���������ꍇ
			Mat rvec1, tvec1, rvec2, tvec2, rvec12, tvec12;
			solvePnP(corners3d[0], corners, cameraMatrix, Mat(), rvec1, tvec1);	//	��]�x�N�g���C���i�x�N�g�����v�Z
			Rodrigues(RProCam, rvec12); tvec12 = TProCam.clone();			//	ProCam
			composeRT(rvec1, tvec1, rvec12, tvec12, rvec2, tvec2);			//	RTp = RTpc * RTc
			vector<Point2f> cornersProj;
			for (int i = 0; i < patternSize.width * patternSize.height; i++)
			{
				projectPoints(corners3d[0], rvec2, tvec2, cameraMatrixProj, distCoeffsProj, cornersProj);
			}
			//	�`�F�X�R�[�i�[��`�� ��肭�����Ώd�Ȃ�͂�
			Mat drawPattern(projSize, CV_8UC3, Scalar::all(255));
			drawChessboardCorners(drawPattern, patternSize, Mat(cornersProj), patternfound);
			imshow(projectorWindowName, drawPattern);
		}
		cv::imshow("calibrated", imgUndistorted);
		key = cv::waitKey(1);
	}

	//	save as XML, PNG
	FileStorage fs("calibdata.xml", FileStorage::WRITE);
	fs << "Camera"
		<< "{"
		<< "CameraMatrix" << cameraMatrix
		<< "DistCoeffs" << distCoeffs
		<< "}"
		<< "Projector"
		<< "{"
		<< "CameraMatrix" << cameraMatrixProj
		<< "Distcoeffs" << distCoeffsProj
		<< "}"
		<< "ProCam"
		<< "{"
		<< "R" << RProCam
		<< "T" << TProCam
		<< "E" << EProCam
		<< "F" << FProCam
		<< "}";

	return 0;
}

//	Point3f = Mat * Point3f�@�̍s�񉉎Z�I�y���[�^
Point3d operator*(Mat M, const Point3d& p)
{
	Mat src(3/*rows*/, 1 /* cols */, CV_64F);

	src.at<double>(0, 0) = p.x;
	src.at<double>(1, 0) = p.y;
	src.at<double>(2, 0) = p.z;

	Mat dst = M*src; //USE MATRIX ALGEBRA
	return Point3d(dst.at<double>(0, 0), dst.at<double>(1, 0), dst.at<double>(2, 0));
}