/*******************************************
		OpenCV Semi-Auto Calibration
********************************************/

#include "FlyCap2CVWrapper.h"
#include "GrayCodePatternProjection.h"

using namespace std;
using namespace cv;

//	Constants
const int imgNum = 15;			//	�摜��
const Size patternSize(7, 10);
const int allPoints = imgNum * patternSize.width * patternSize.height;
const double chessSize = 22.5;		//	mm
const double chessboardInterval = 600.0;		//	�`�F�X�{�[�h���o�C���^�[�o��
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
	//	��p���Ń`�F�X�{�[�h��3�����I�ɓ������Ǝ����I�ɂƂ��
	std::cout << "Please Move " << patternSize << " Chessboard in your projection area." << endl;
	for (int imgFound = 0; imgFound < imgNum;)
	{
		//	�^�C�}�[
		static double timer = 0.0;
		static int64 count = getTickCount();
		//	�摜�擾
		img = cam.readImage();
		flip(img, img, 1);
		vector<Mat> imgChannels;
		split(img, imgChannels);
		//	�`�F�X�{�[�h���o
		//	1000ms�̃C���^�[�o����݂��Ă���
		vector<Point2f> corners;
		bool patternfound = findChessboardCorners(imgChannels[2], patternSize, corners,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		if (patternfound && timer > chessboardInterval)
		{	//	���������ꍇ
			imgFound++;
			timer = 0.0; count = getTickCount();		//	�C���^�[�o���^�C�}�[��������
			cout << "Chessboard No. " << imgFound << " / " << imgNum << " is found at " << corners[0] << endl;
			corners2d.push_back(corners);
			drawChessboardCorners(imgChannels[2], patternSize, Mat(corners), patternfound);
		}
		imshow("Chessboard", imgChannels[2]);
		if (waitKey(1) == 'q') exit(0);
		timer = (getTickCount() - count)*1000.0 / getTickFrequency();
	}
	//	Calibration Start
	//	�p�����[�^����
	cout << "All Chessboard is found!\n";
	cout << "Calculating Camera Inner Matrix and Distortion Parameters..." << endl;
	cv::calibrateCamera(
		corners3d, corners2d,
		img.size(),
		cameraMatrix, distCoeffs,
		rvecs, tvecs);
	cout << "Camera Matirx = \n" << cameraMatrix
		<< "\nDistortion Coeffs = \n" << distCoeffs
		<< "\n" << endl;
	//	�c�ݕ␳�}�b�v�v�Z
	cout << "Making Undistort Map..." << endl;
	initUndistortRectifyMap(
		cameraMatrix, distCoeffs,
		Mat(), cameraMatrix, img.size(), CV_32FC1,
		map1, map2);
	cout << "\nCamera Calibration finished!.\n\n"
		<< "Press any key, and projector-camera pixel coordination will begin." << endl;
	cv::waitKey(0);

	//-----------------------------------------
	//	2.	�J�����E�v���W�F�N�^��f�Ή��擾
	//		�i�O���C�R�[�h�p�^�[�����e�@�j
	//		�c�ݕ␳�O�J�����ƃv���W�F�N�^
	//-----------------------------------------
	vector<vector<Point2f>> corners2dCam;				//	���o���ꂽ�J�������W�R�[�i�[�_
	vector<vector<Point2f>> corners2dProj;				//	���̃v���W�F�N�^���W�R�[�i�[�_

	for (int imgFound = 0; imgFound < imgNum;)
	{
		//	�摜�擾
		img = cam.readImage();
		flip(img, img, 1);
		//remap(img, img, map1, map2, INTER_CUBIC);
		int c = waitKey(1);
		imshow("Chessboard", img);
		//	Space�L�[�������ꂽ�琅�F�`�F�X�{�[�h�F���J�n
		if (c == ' ')
		{
			vector<Mat> imgChannels;
			split(img, imgChannels);
			//	�`�F�X�{�[�h���o
			vector<Point2f> corners;
			bool patternfound = findChessboardCorners(imgChannels[2], patternSize, corners,
				CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FILTER_QUADS + CALIB_CB_FAST_CHECK);
			if (patternfound)
			{
				imgFound++;
				cout << "Chessboard No. " << imgFound << " / " << imgNum << " is found at " << corners[0] << endl;
				corners2dCam.push_back(corners);			//	�c�ݕ␳�O�J�������W
				//	�F���ł�����O���C�R�[�h�p�^�[�����e
				vector<Mat> captures;
				for (int i = 0; i < gcp.patternListW.rows; i++)
				{	//	W, WN
					imshow(projectorWindowName, gcp.patternsW[i]);
					waitKey(projectionInterval);
					img = cam.readImage();
					flip(img, img, 1);
					//remap(img, img, map1, map2, INTER_CUBIC);
					captures.push_back(img);
					imshow(projectorWindowName, gcp.patternsWN[i]);
					waitKey(projectionInterval);
					img = cam.readImage();
					flip(img, img, 1);
					//remap(img, img, map1, map2, INTER_CUBIC);
					captures.push_back(img);
				}
				for (int i = 0; i < gcp.patternListH.rows; i++)
				{	//	H, HN
					imshow(projectorWindowName, gcp.patternsH[i]);
					waitKey(projectionInterval);
					img = cam.readImage();
					flip(img, img, 1);
					//remap(img, img, map1, map2, INTER_CUBIC);
					captures.push_back(img);
					imshow(projectorWindowName, gcp.patternsHN[i]);
					waitKey(projectionInterval);
					img = cam.readImage();
					flip(img, img, 1);
					//remap(img, img, map1, map2, INTER_CUBIC);
					captures.push_back(img);
				}
				//	�O���C�R�[�h�p�^�[�����
				gcp.loadCapPatterns(captures);
				gcp.decodePatterns();
				gcp.showMaps();

				//	����ꂽ�}�b�v���g���ăT�u�s�N�Z�����x�Ńv���W�F�N�^���W�ɕϊ�
				vector<Point2f> points;
				for (int j = 0; j < patternSize.height; j++){
					for (int k = 0; k < patternSize.width; k++)
					{	//	getRectSubPix()��3x3�̋�`�T�C�Y�ŗ��p�����ƂŃs�N�Z�����o�C���j�A���
						Point2f pcam = (corners)[j*patternSize.width + k];		//	�J�����̃T�u�s�N�Z�����W
						Size patchSize(3, 3);
						Mat patchX, patchY;
						getRectSubPix(gcp.mapX, patchSize, pcam, patchX);
						getRectSubPix(gcp.mapY, patchSize, pcam, patchY);
						Point2f pproj(patchX.at<float>(1, 1), patchY.at<float>(1, 1));		//	�v���W�F�N�^�̃T�u�s�N�Z�����W
						points.push_back(pproj);
					}
				}
				corners2dProj.push_back(points);			//	�v���W�F�N�^���W

				imshow(projectorWindowName, whiteLight);
			}
			else
				cout << "Not Found." << endl;
		}		//	�`�F�X�{�[�h���o�I��
		if (c == 'q') exit(0);
	}

	//-----------------------------------------
	//	3. �v���W�F�N�^�L�����u���[�V����
	//-----------------------------------------
	//	�p�����[�^����
	cout << "Calculating Projector Inner Matrix and Distortion Parameters..." << endl;
	cv::calibrateCamera(
		corners3d, corners2dProj,
		projSize,
		cameraMatrixProj,distCoeffsProj,
		rvecsProj, tvecsProj);
	cout << "Projector Camera Matirx = \n" << cameraMatrix
		<< "\nProjector Distortion Coeffs = \n" << distCoeffs
		<< "\n" << endl;
	//	�c�ݕ␳�}�b�v�v�Z
	cout << "Making Projector Undistort Map..." << endl;
	initUndistortRectifyMap(
		cameraMatrixProj, distCoeffsProj,
		Mat(), cameraMatrixProj, projSize, CV_32FC1,
		map1Proj, map2Proj);
	//	�J���� - �v���W�F�N�^�O���s��̐���
	cout << "Caluculating Camera-Projector outer parameters..." << endl;
	//Mat cameraMatrixProjCamSize = cv::getDefaultNewCameraMatrix(cameraMatrixProj, img.size(), true);
	cv::stereoCalibrate(
		corners3d, corners2dCam, corners2dProj,		//	�R�[�i�[�_�Q
		cameraMatrix, distCoeffs,					//	�J�����p�����[�^
		cameraMatrixProj, distCoeffsProj,			//	�v���W�F�N�^�p�����[�^	
		img.size(),									//	�摜�T�C�Y
		RProCam, TProCam, EProCam, FProCam,			//	�J�����ɑ΂���v���W�F�N�^�̊O���p�����[�^
		CALIB_FIX_INTRINSIC,					//	�X�ɋ��߂������p�����[�^�ŌŒ�i�O���p�����[�^��������j
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
		Mat imgUndistorted, imgGray;
		remap(image, imgUndistorted, map1, map2, INTER_CUBIC);
		cvtColor(imgUndistorted, imgGray, CV_BGR2GRAY);
		//	�`�F�X�{�[�h���o
		vector<Point2d> corners;
		bool patternfound = findChessboardCorners(imgGray, patternSize, corners,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FILTER_QUADS + CALIB_CB_FAST_CHECK);
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
			remap(drawPattern, drawPattern, map1Proj, map2Proj, INTER_CUBIC);
			imshow(projectorWindowName, drawPattern);
		}
		cv::imshow("calibrated", imgUndistorted);
		key = cv::waitKey(1);
	}

	//	save as XML, PNG
	FileStorage fs("calibdata.xml", FileStorage::WRITE);
	fs << "Camera"
		<< "{"
		<< "size" << img.size()
		<< "CameraMatrix" << cameraMatrix
		<< "DistCoeffs" << distCoeffs
		<< "}"
		<< "Projector"
		<< "{"
		<< "size" << projSize
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
	imwrite("UndistortMap1Cam.png", map1);
	imwrite("UndistortMap2Cam.png", map2);
	imwrite("UndistortMap1Pro.png", map1Proj);
	imwrite("UndistortMap2Pro.png", map2Proj);

	return 0;
}
