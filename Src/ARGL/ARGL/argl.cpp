
// 참조 코드
//http://www.songho.ca/opengl/gl_matrix.html#example1

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <GL/glut.h>
#include <thread>
#include "Matrices.h"
#include <mmsystem.h>

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

#pragma comment(lib, "winmm.lib")

using namespace std;
using namespace cv;

GLfloat light_diffuse[] = { 1.0, 0.0, 0.0, 1.0 };  /* Red diffuse light. */
GLfloat light_position[] = { 10.0, 10.0, 10.0, 0.0 };  /* Infinite light location. */
GLfloat normals[6][3] = {  /* Normals for the 6 faces of a cube. */
	{ -1.0, 0.0, 0.0 },{ 0.0, 1.0, 0.0 },{ 1.0, 0.0, 0.0 },
	{ 0.0, -1.0, 0.0 },{ 0.0, 0.0, 1.0 },{ 0.0, 0.0, -1.0 } };
GLint faces[6][4] = {  /* Vertex indices for the 6 faces of a cube. */
	{ 0, 1, 2, 3 },{ 3, 2, 6, 7 },{ 7, 6, 5, 4 },
	{ 4, 5, 1, 0 },{ 5, 6, 2, 1 },{ 7, 4, 0, 3 } };
GLfloat vertices[8][3];  /* Will be filled in with X,Y,Z vertexes. */

int screenWidth = 640;
int screenHeight = 480;
bool mouseLeftDown;
bool mouseRightDown;
float mouseX, mouseY;
float cameraAngleX = 45;
float cameraAngleY;
float cameraDistance = 10.f;
Matrix4 matrixView;
Matrix4 matrixModel;
Matrix4 matrixModelView;    // = matrixView * matrixModel
Matrix4 matrixProjection;
GLuint textureID = -1;

VideoCapture m_inputVideo;
Mat m_camImage;

Ptr<aruco::DetectorParameters> m_detectorParams;
Ptr<aruco::Dictionary> m_dictionary;
Mat m_camMatrix;
Mat m_distCoeffs;
float m_markerLength = 75;

void drawGrid(float size = 10.0f, float step = 1.0f);
void mouseCB(int button, int stat, int x, int y);



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


///////////////////////////////////////////////////////////////////////////////
// draw a grid on XZ-plane
///////////////////////////////////////////////////////////////////////////////
void drawGrid(float size, float step)
{
	// disable lighting
	glDisable(GL_LIGHTING);

	// 20x20 grid
	glBegin(GL_LINES);

	glColor3f(0.5f, 0.5f, 0.5f);
	for (float i = step; i <= size; i += step)
	{
		glVertex3f(-size, 0, i);   // lines parallel to X-axis
		glVertex3f(size, 0, i);
		glVertex3f(-size, 0, -i);   // lines parallel to X-axis
		glVertex3f(size, 0, -i);

		glVertex3f(i, 0, -size);   // lines parallel to Z-axis
		glVertex3f(i, 0, size);
		glVertex3f(-i, 0, -size);   // lines parallel to Z-axis
		glVertex3f(-i, 0, size);
	}

	// x-axis
	glColor3f(1, 0, 0);
	glVertex3f(-size, 0, 0);
	glVertex3f(size, 0, 0);

	// z-axis
	glColor3f(0, 0, 1);
	glVertex3f(0, 0, -size);
	glVertex3f(0, 0, size);

	glEnd();

	// enable lighting back
	glEnable(GL_LIGHTING);
}


void drawQuad()
{
	const GLfloat width = (GLfloat)glutGet(GLUT_WINDOW_WIDTH);
	const GLfloat height = (GLfloat)glutGet(GLUT_WINDOW_HEIGHT);
	GLfloat quad[4][3] = { { 0,height,0 },{ 0,0,0},{ width,0,0},{ width,height,0} };
	GLfloat normal[3] = { 0,1,0 };

	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, width, 0.0, height, -1.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	glLoadIdentity();
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, textureID);

	glDisable(GL_LIGHTING);
	glBegin(GL_QUADS);
	glColor3f(1, 1, 1);
	glNormal3fv(normal);
	glTexCoord2f(0, 1);  glVertex3fv(quad[0]);
	glTexCoord2f(0, 0);  glVertex3fv(quad[1]);
	glTexCoord2f(1, 0);  glVertex3fv(quad[2]);
	glTexCoord2f(1, 1);  glVertex3fv(quad[3]);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glEnable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	glEnable(GL_DEPTH_TEST);
}


void drawBox(void)
{
	for (int i = 0; i < 6; i++) 
	{
		glBegin(GL_LINE_LOOP);
		//glBegin(GL_QUADS);
		glNormal3fv(&normals[i][0]);
		glVertex3fv(&vertices[faces[i][0]][0]);
		glVertex3fv(&vertices[faces[i][1]][0]);
		glVertex3fv(&vertices[faces[i][2]][0]);
		glVertex3fv(&vertices[faces[i][3]][0]);
		glEnd();
	}
}

void display(void)
{
	Mat image;
	m_inputVideo >> image;
	if (!image.data)
		return;

	vector< int > ids;
	vector< vector< Point2f > > corners, rejected;
	vector< Vec3d > rvecs, tvecs;

	// detect markers and estimate pose
	aruco::detectMarkers(image, m_dictionary, corners, ids, m_detectorParams, rejected);

	Matrix4 markerTm;
	markerTm.identity();

	if (ids.size() > 0)
	{
		aruco::estimatePoseSingleMarkers(corners, m_markerLength, m_camMatrix, m_distCoeffs, rvecs, tvecs);
		aruco::drawDetectedMarkers(image, corners, ids);

		for (unsigned int i = 0; i < ids.size(); i++)
		{
			aruco::drawAxis(image, m_camMatrix, m_distCoeffs, rvecs[i], tvecs[i], m_markerLength * 0.5f);

			Mat rot;
			Rodrigues(rvecs[i], rot);

			Mat viewMatrix = cv::Mat::zeros(4, 4, CV_32F);
			for (unsigned int row = 0; row < 3; ++row)
			{
				for (unsigned int col = 0; col < 3; ++col)
				{
					viewMatrix.at<float>(row, col) = (float)rot.at<double>(row, col);
				}
				viewMatrix.at<float>(row, 3) = (float)tvecs[i][row] * 0.1f;
			}
			viewMatrix.at<float>(3, 3) = 1.0f;

			cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F); 
			cvToGl.at<float>(0, 0) = 1.0f; 
			cvToGl.at<float>(1, 1) = -1.0f; // Invert the y axis 
			cvToGl.at<float>(2, 2) = -1.0f; // invert the z axis 
			cvToGl.at<float>(3, 3) = 1.0f;
			viewMatrix = cvToGl * viewMatrix;

			// 행기준 행렬을 OpenGL 에 맞게 열기준 행렬로 바꾼다.
			cv::transpose(viewMatrix, viewMatrix);
			markerTm = Matrix4((float*)viewMatrix.data);
		}
	}

	// -----------------------------------------------------------------------------------------
	// 카메라 영상을 텍스쳐에 복사
	Mat texImage;
	flip(image, texImage, 0);

	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
		texImage.cols, texImage.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, texImage.data);
	glBindTexture(GL_TEXTURE_2D, 0);
	// -----------------------------------------------------------------------------------------

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	drawQuad();

	glMatrixMode(GL_MODELVIEW);

	// save the initial ModelView matrix before modifying ModelView matrix
	glPushMatrix();
	glLoadIdentity();

	Matrix4 scalM;
	scalM.scale(4);

	Matrix4 tranM;
	tranM.translate(Vector3(0, 0, 1));

	// compute modelview matrix
	matrixModelView = markerTm * scalM * tranM;

	// copy modelview matrix to OpenGL
	glLoadMatrixf(matrixModelView.get());

	if (ids.size() > 0)
 		drawBox();

	glPopMatrix();

	glutSwapBuffers();
}

void init(void)
{
	/* Setup cube vertex data. */
	vertices[0][0] = vertices[1][0] = vertices[2][0] = vertices[3][0] = -1;
	vertices[4][0] = vertices[5][0] = vertices[6][0] = vertices[7][0] = 1;
	vertices[0][1] = vertices[1][1] = vertices[4][1] = vertices[5][1] = -1;
	vertices[2][1] = vertices[3][1] = vertices[6][1] = vertices[7][1] = 1;
	vertices[0][2] = vertices[3][2] = vertices[4][2] = vertices[7][2] = 1;
	vertices[1][2] = vertices[2][2] = vertices[5][2] = vertices[6][2] = -1;

	/* Enable a single OpenGL light. */
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);

	/* Use depth buffering for hidden surface elimination. */
	glEnable(GL_DEPTH_TEST);

	/* Setup the view of the cube. */
	glMatrixMode(GL_PROJECTION);
	gluPerspective( 40.0, (float)screenWidth / (float)screenHeight, 1.0, 100.0);
}


void keyboardCB(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27: // ESCAPE
		exit(0);
		break;
	default:
		;
	}
}


void mouseCB(int button, int state, int x, int y)
{
	mouseX = (float)x;
	mouseY = (float)y;

	if (button == GLUT_LEFT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			mouseLeftDown = true;
		}
		else if (state == GLUT_UP)
			mouseLeftDown = false;
	}

	else if (button == GLUT_RIGHT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			mouseRightDown = true;
		}
		else if (state == GLUT_UP)
			mouseRightDown = false;
	}
}


void mouseMotionCB(int x, int y)
{
	if (mouseLeftDown)
	{
		cameraAngleY += (x - mouseX);
		cameraAngleX += (y - mouseY);
		mouseX = (float)x;
		mouseY = (float)y;
	}
	if (mouseRightDown)
	{
		cameraDistance -= (y - mouseY) * 0.2f;
		mouseY = (float)y;
	}
}

void glutTimer(int value)
{
	glutPostRedisplay();
	glutTimerFunc(33, glutTimer, 33);
}


int main(int argc, char **argv)
{
	m_detectorParams = aruco::DetectorParameters::create();
	readDetectorParameters("detector_params.yml", m_detectorParams);
	m_detectorParams->doCornerRefinement = true; // do corner refinement in markers
	m_dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);
	readCameraParameters("camera.yml", m_camMatrix, m_distCoeffs);
	m_inputVideo.open(0);

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &textureID);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(screenWidth, screenHeight);  // window size
	glutCreateWindow("Grid and Box");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboardCB);
	glutMouseFunc(mouseCB);
	glutMotionFunc(mouseMotionCB);
	glutTimerFunc(33, glutTimer, 33);
	init();

	glutMainLoop();

	return 0;
}
