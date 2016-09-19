
#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>


#ifdef _DEBUG
	#pragma comment(lib, "opencv_core310d.lib")
	#pragma comment(lib, "opencv_imgcodecs310d.lib")
	#pragma comment(lib, "opencv_videoio310d.lib")
	#pragma comment(lib, "opencv_highgui310d.lib")
	#pragma comment(lib, "opencv_imgproc310d.lib")
	#pragma comment(lib, "opencv_calib3d310d.lib")
	#pragma comment(lib, "opencv_aruco310d.lib")
#else
	#pragma comment(lib, "opencv_core310.lib")
	#pragma comment(lib, "opencv_imgcodecs310.lib")
	#pragma comment(lib, "opencv_videoio310.lib")
	#pragma comment(lib, "opencv_highgui310.lib")
	#pragma comment(lib, "opencv_imgproc310.lib")
	#pragma comment(lib, "opencv_calib3d310.lib")
	#pragma comment(lib, "opencv_aruco310.lib")
#endif


using namespace graphic;
using namespace cv;


class cViewer : public framework::cGameMain
{
public:
	cViewer();
	virtual ~cViewer();

	virtual bool OnInit() override;
	virtual void OnUpdate(const float elapseT) override;
	virtual void OnRender(const float elapseT) override;
	virtual void OnShutdown() override;
	virtual void OnMessageProc(UINT message, WPARAM wParam, LPARAM lParam) override;


private:
	LPD3DXSPRITE m_sprite;
	graphic::cSprite *m_videoSprite;
	graphic::cCharacter m_character;
	graphic::cLine m_line[3];

	VideoCapture m_inputVideo;
	Mat m_camImage;

	Ptr<aruco::DetectorParameters> m_detectorParams;
	Ptr<aruco::Dictionary> m_dictionary;
	Mat m_camMatrix;
	Mat m_distCoeffs;
	float m_markerLength = 75;

	Matrix44 m_zealotCameraView;

	Vector3 m_pos;
	POINT m_curPos;
	bool m_LButtonDown;
	bool m_RButtonDown;
	bool m_MButtonDown;
	Matrix44 m_rotateTm;

	Vector3 m_boxPos;
};

INIT_FRAMEWORK(cViewer);

const int WINSIZE_X = 640;
const int WINSIZE_Y = 480;

cViewer::cViewer() :
	m_character(1000)
{
	m_windowName = L"ARUCO Zealot";
	const RECT r = { 0, 0, WINSIZE_X, WINSIZE_Y };
	m_windowRect = r;
	m_LButtonDown = false;
	m_RButtonDown = false;
	m_MButtonDown = false;
}

cViewer::~cViewer()
{
	SAFE_DELETE(m_videoSprite);
	m_sprite->Release();
	graphic::ReleaseRenderer();
}


static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}


static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["doCornerRefinement"] >> params->doCornerRefinement;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}


bool cViewer::OnInit()
{
	// start craft 2
	// zealot
	{
		m_character.Create(m_renderer, "zealot.dat");
		if (graphic::cMesh* mesh = m_character.GetMesh("Sphere001"))
			mesh->SetRender(false);
		m_character.SetShader(graphic::cResourceManager::Get()->LoadShader(m_renderer, 
			"hlsl_skinning_using_texcoord_sc2.fx"));
		m_character.SetRenderShadow(true);

		vector<sActionData> actions;
		actions.reserve(16);
		actions.push_back(sActionData(CHARACTER_ACTION::NORMAL, "zealot_stand.ani"));
		actions.push_back(sActionData(CHARACTER_ACTION::RUN, "zealot_walk.ani"));
		actions.push_back(sActionData(CHARACTER_ACTION::ATTACK, "zealot_attack.ani"));
		m_character.SetActionData(actions);
		m_character.Action( CHARACTER_ACTION::RUN );
	}

	D3DXCreateSprite(m_renderer.GetDevice(), &m_sprite);
	m_videoSprite = new graphic::cSprite(m_sprite, 0);
	m_videoSprite->SetTexture(m_renderer, "kim.jpg");
	m_videoSprite->SetPos(Vector3(0, 0, 0));


	GetMainCamera()->Init(&m_renderer);
	GetMainCamera()->SetCamera(Vector3(10, 10, -10), Vector3(0, 0, 0), Vector3(0, 1, 0));
	GetMainCamera()->SetProjection(D3DX_PI / 4.f, (float)WINSIZE_X / (float)WINSIZE_Y, 1.f, 10000.0f);

	GetMainLight().Init(cLight::LIGHT_DIRECTIONAL);
	GetMainLight().SetPosition(Vector3(5, 5, 5));
	GetMainLight().SetDirection(Vector3(1, -1, 1).Normal());
	GetMainLight().Bind(m_renderer, 0);

	m_renderer.GetDevice()->SetRenderState(D3DRS_NORMALIZENORMALS, TRUE);
	m_renderer.GetDevice()->LightEnable(0, true);

	m_line[0].SetLine(m_renderer, Vector3(0, 0, 0), Vector3(1, 0, 0), 0.1f);
	m_line[0].GetMaterial().InitRed();
	m_line[1].SetLine(m_renderer, Vector3(0, 0, 0), Vector3(0, 1, 0), 0.1f);
	m_line[1].GetMaterial().InitGreen();
	m_line[2].SetLine(m_renderer, Vector3(0, 0, 0), Vector3(0, 0, 1), 0.1f);
	m_line[2].GetMaterial().InitBlue();


	m_detectorParams = aruco::DetectorParameters::create();
	readDetectorParameters("detector_params.yml", m_detectorParams);
	m_detectorParams->doCornerRefinement = true; // do corner refinement in markers
	m_dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);

	readCameraParameters("camera.yml", m_camMatrix, m_distCoeffs);

	m_inputVideo.open(0);

	return true;
}


// 모델뷰 행렬(OpenGL) -> 뷰 행렬(Direct3D)
// http://www.cg-ya.net/imedia/ar/artoolkit_directx/
void D3DConvMatrixView(float* src, D3DXMATRIXA16* dest)
{
	// src의 값을 dest로 1:1 복사를 수행한다.
	dest->_11 = src[0];
	dest->_12 = -src[1];
	dest->_13 = src[2];
	dest->_14 = src[3];

	dest->_21 = src[8];
	dest->_22 = -src[9];
	dest->_23 = src[10];
	dest->_24 = src[7];

	dest->_31 = src[4];
	dest->_32 = -src[5];
	dest->_33 = src[6];
	dest->_34 = src[11];

	dest->_41 = src[12];
	dest->_42 = -src[13];
	dest->_43 = src[14];
	dest->_44 = src[15];
}


// 투영 행렬(OpenGL) -> 투영 행렬(Direct3D)
// http://www.cg-ya.net/imedia/ar/artoolkit_directx/
void D3DConvMatrixProjection(float* src, D3DXMATRIXA16* dest)
{
	dest->_11 = src[0];
	dest->_12 = src[1];
	dest->_13 = src[2];
	dest->_14 = src[3];
	dest->_21 = src[4];
	dest->_22 = src[5];
	dest->_23 = src[6];
	dest->_24 = src[7];
	dest->_31 = -src[8];
	dest->_32 = -src[9];
	dest->_33 = -src[10];
	dest->_34 = -src[11];
	dest->_41 = src[12];
	dest->_42 = src[13];
	dest->_43 = src[14];
	dest->_44 = src[15];
}



void cViewer::OnUpdate(const float elapseT)
{
	static float incT = 0;
	incT += elapseT;
	if (incT < 0.033f)
		return;

	GetMainCamera()->Update();
	m_character.Update(incT);

	incT = 0;

	Mat image;
	m_inputVideo >> image;
	if (!image.data)
		return;

	if (1)
	{
		vector< int > ids;
		vector< vector< Point2f > > corners, rejected;
		vector< Vec3d > rvecs, tvecs;

		// detect markers and estimate pose
		aruco::detectMarkers(image, m_dictionary, corners, ids, m_detectorParams, rejected);

		if (1 && ids.size() > 0)
		{
			aruco::estimatePoseSingleMarkers(corners, m_markerLength, m_camMatrix, m_distCoeffs, rvecs, tvecs);
			aruco::drawDetectedMarkers(image, corners, ids);

			for (unsigned int i = 0; i < ids.size(); i++)
			{
				aruco::drawAxis(image, m_camMatrix, m_distCoeffs, rvecs[i], tvecs[i], m_markerLength * 0.5f);

				// change aruco space to direct x space
				Mat rot;
				Rodrigues(rvecs[i], rot);

				Mat viewMatrix = cv::Mat::zeros(4, 4, CV_32F);
				for (unsigned int row = 0; row < 3; ++row)
				{
					for (unsigned int col = 0; col < 3; ++col)
					{
						viewMatrix.at<float>(row, col) = (float)rot.at<double>(row, col);
					}
					viewMatrix.at<float>(row, 3) = 0;// (float)tvecs[i][row] * 0.01f;
					//viewMatrix.at<float>(row, 3) = (float)tvecs[i][row] * 0.1f;
				}
				viewMatrix.at<float>(3, 3) = 1.0f;

				cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
				cvToGl.at<float>(0, 0) = 1.0f;
				cvToGl.at<float>(1, 1) = -1.0f; // Invert the y axis 
				cvToGl.at<float>(2, 2) = -1.0f; // invert the z axis 
				cvToGl.at<float>(3, 3) = 1.0f;
				viewMatrix = cvToGl * viewMatrix;
				
				Matrix44 m(viewMatrix.ptr<float>());
				Matrix44 rm;
				rm.SetRotationX(ANGLE2RAD(180));
				m_zealotCameraView = m;

				

// 				m_zealotCameraView._41 = -.2f;
// 				m_zealotCameraView._42 = .2f;
// 				m_zealotCameraView._43 = .8f;

				// 			cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
				// 			cvToGl.at<float>(0, 0) = 1.0f;
				// 			cvToGl.at<float>(1, 1) = 1.0f; // Invert the y axis 
				// 			cvToGl.at<float>(2, 2) = 1.0f; // invert the z axis 
				// 			cvToGl.at<float>(3, 3) = 1.0f;
				// 			viewMatrix = cvToGl * viewMatrix;
				// 			//viewMatrix = viewMatrix * cvToGl;
				// 
				//  			m_zealotCameraView.SetIdentity();
				// 			float *vm = viewMatrix.ptr<float>();
				// 			m_zealotCameraView.m[0][0] = vm[0];
				// 			m_zealotCameraView.m[0][1] = vm[1];
				// 			m_zealotCameraView.m[0][2] = vm[2];
				// 			m_zealotCameraView.m[0][3] = vm[3];
				// 
				// 			m_zealotCameraView.m[1][0] = vm[4];
				// 			m_zealotCameraView.m[1][1] = vm[5];
				// 			m_zealotCameraView.m[1][2] = vm[6];
				// 			m_zealotCameraView.m[1][3] = vm[7];
				// 
				// 			m_zealotCameraView.m[2][0] = vm[8];
				// 			m_zealotCameraView.m[2][1] = vm[9];
				// 			m_zealotCameraView.m[2][2] = vm[10];
				// 			m_zealotCameraView.m[2][3] = vm[11];

				// 			m_zealotCameraView.m[3][0] = (float)tvecs[i][0] * 0.01f;
				// 			m_zealotCameraView.m[3][1] = (float)tvecs[i][1] * 0.01f;
				// 			m_zealotCameraView.m[3][2] = (float)tvecs[i][2] * 0.01f;
				// 			m_zealotCameraView.m[3][3] = vm[15];

				// 			m_zealotCameraView.m[3][0] = 500;// vm[12];
				// 			m_zealotCameraView.m[3][1] = 500;// vm[13];
				// 			m_zealotCameraView.m[3][2] = 500;// vm[14];
				// 			m_zealotCameraView.m[3][3] = vm[15];

				//m_zealotCameraView  = m_zealotCameraView.Inverse();

				//cv::transpose(viewMatrix, viewMatrix);

				// 			D3DConvMatrixView((float*)viewMatrix.data, (D3DXMATRIXA16*)&m_zealotCameraView);


				// 			Mat invRot;
				// 			transpose(rot, invRot); // inverse matrix
				// 			double *pinvR = invRot.ptr<double>();
				// 
				// 			Matrix44 tm;
				// 			tm.m[0][0] = -(float)pinvR[0];
				// 			tm.m[0][1] = (float)pinvR[1];
				// 			tm.m[0][2] = -(float)pinvR[2];
				// 			
				// 			tm.m[1][0] = -(float)pinvR[3];
				// 			tm.m[1][1] = (float)pinvR[4];
				// 			tm.m[1][2] = -(float)pinvR[5];
				// 
				// 			tm.m[2][0] = -(float)pinvR[6];
				// 			tm.m[2][1] = (float)pinvR[7];
				// 			tm.m[2][2] = -(float)pinvR[8];
				// 
				// 			Matrix44 rot2;
				// 			rot2.SetRotationX(ANGLE2RAD(-90)); // y-z axis change
				// 
				// 			Matrix44 trans;
				// 			trans.SetPosition( Vector3((float)tvecs[i][0], -(float)tvecs[i][1], (float)tvecs[i][2]) * 0.01f );
				// 
				// 			m_zealotCameraView = rot2 * tm * trans;
			}
		}

		// display camera image to DirectX Texture
		D3DLOCKED_RECT lockRect;
		m_videoSprite->GetTexture()->Lock(lockRect);
		if (lockRect.pBits)
		{
			Mat BGRA = image.clone();
			cvtColor(image, BGRA, CV_BGR2BGRA, 4);
			const size_t sizeInBytes2 = BGRA.step[0] * BGRA.rows;
			memcpy(lockRect.pBits, BGRA.data, sizeInBytes2);
			m_videoSprite->GetTexture()->Unlock();
		}

	}

}


void cViewer::OnRender(const float elapseT)
{
	//화면 청소
	if (SUCCEEDED(m_renderer.GetDevice()->Clear(0,NULL,D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL,D3DCOLOR_XRGB(150, 150, 150),1.0f,0)))
	{
		m_renderer.GetDevice()->BeginScene();

		m_renderer.GetDevice()->SetRenderState(D3DRS_ZENABLE, 0);
		m_videoSprite->Render(m_renderer, Matrix44::Identity);
		m_renderer.GetDevice()->SetRenderState(D3DRS_ZENABLE, 1);

		m_renderer.RenderGrid();
		m_renderer.RenderAxis();

		m_renderer.GetDevice()->LightEnable(0, true);
		m_line[0].Render(m_renderer, m_zealotCameraView);
		m_line[1].Render(m_renderer, m_zealotCameraView);
		m_line[2].Render(m_renderer, m_zealotCameraView);
		m_renderer.GetDevice()->LightEnable(0, true);



//  		GetMainCamera()->SetViewMatrix(m_zealotCameraView);
//  		m_renderer.GetDevice()->SetTransform(D3DTS_VIEW, (D3DMATRIX*)&m_zealotCameraView);
		//m_character.Render(m_renderer, Matrix44::Identity);

		m_renderer.RenderFPS();

		//랜더링 끝
		m_renderer.GetDevice()->EndScene();
		m_renderer.GetDevice()->Present(NULL, NULL, NULL, NULL);
	}
}


void cViewer::OnShutdown()
{

}


void cViewer::OnMessageProc(UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_MOUSEWHEEL:
	{
		int fwKeys = GET_KEYSTATE_WPARAM(wParam);
		int zDelta = GET_WHEEL_DELTA_WPARAM(wParam);
		dbg::Print("%d %d", fwKeys, zDelta);

		const float len = graphic::GetMainCamera()->GetDistance();
		float zoomLen = (len > 100) ? 50 : (len / 4.f);
		if (fwKeys & 0x4)
			zoomLen = zoomLen / 10.f;

		graphic::GetMainCamera()->Zoom((zDelta < 0) ? -zoomLen : zoomLen);
	}
	break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_BACK:
			break;
		case VK_TAB:
		{
			static bool flag = false;
			m_renderer.GetDevice()->SetRenderState(D3DRS_CULLMODE, flag ? D3DCULL_CCW : D3DCULL_NONE);
			m_renderer.GetDevice()->SetRenderState(D3DRS_FILLMODE, flag ? D3DFILL_SOLID : D3DFILL_WIREFRAME);
			flag = !flag;
		}
		break;

		case VK_LEFT: m_boxPos.x -= 10.f; break;
		case VK_RIGHT: m_boxPos.x += 10.f; break;
		case VK_UP: m_boxPos.z += 10.f; break;
		case VK_DOWN: m_boxPos.z -= 10.f; break;
		}
		break;

	case WM_LBUTTONDOWN:
	{
		m_LButtonDown = true;
		m_curPos.x = LOWORD(lParam);
		m_curPos.y = HIWORD(lParam);
	}
	break;

	case WM_LBUTTONUP:
		m_LButtonDown = false;
		break;

	case WM_RBUTTONDOWN:
	{
		m_RButtonDown = true;
		m_curPos.x = LOWORD(lParam);
		m_curPos.y = HIWORD(lParam);
	}
	break;

	case WM_RBUTTONUP:
		m_RButtonDown = false;
		break;

	case WM_MBUTTONDOWN:
		m_MButtonDown = true;
		m_curPos.x = LOWORD(lParam);
		m_curPos.y = HIWORD(lParam);
		break;

	case WM_MBUTTONUP:
		m_MButtonDown = false;
		break;

	case WM_MOUSEMOVE:
	{
		if (m_RButtonDown)
		{
			POINT pos = { LOWORD(lParam), HIWORD(lParam) };
			const int x = pos.x - m_curPos.x;
			const int y = pos.y - m_curPos.y;
			m_curPos = pos;

			graphic::GetMainCamera()->Yaw2(x * 0.005f);
			graphic::GetMainCamera()->Pitch2(y * 0.005f);
		}
		else if (m_MButtonDown)
		{
			const POINT point = { LOWORD(lParam), HIWORD(lParam) };
			const POINT pos = { point.x - m_curPos.x, point.y - m_curPos.y };
			m_curPos = point;

			const float len = graphic::GetMainCamera()->GetDistance();
			graphic::GetMainCamera()->MoveRight(-pos.x * len * 0.001f);
			graphic::GetMainCamera()->MoveUp(pos.y * len * 0.001f);
		}

	}
	break;
	}
}

