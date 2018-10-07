#pragma once

#include <cv.h>
//#include <windows.h>
//
//struct circle_found
//{
//	float score;	//匹配分数
//	cv::Vec3f circle;	//找到的圆
//};
//
//namespace ExtendCV
//{
//	//_image——输入的图像，必须为8位单通道，_circles——找到的圆，dp——cv::houghcircles中的dp，min_dist——cv::houghcircles中的minDist两圆最小距离
//	//low_threshold——将_image预处理提取轮廓的canny低阈值，high_threshold——将_image预处理提取轮廓的canny高阈值
//	//acc_threshold——cv::houghcircles中的param2累加器值，minRadius——圆的最小半径，maxRadius——圆的最大半径
//	//minScore——找出的圆与现有的轮廓的重合率，作为分数
//	//_contour_image——可选的输入轮廓图，如果这里非空，则将low_threshold与high_threshold忽略（方便轮廓图的进一步预处理）
//	void FindCircles( cv::InputArray _image, cv::vector<circle_found>& _circles,float dp, int min_dist,
//	int low_threshold, int high_threshold,int acc_threshold,int minRadius, int maxRadius,
//	float minScore, cv::InputArray _contour_image=cv::Mat() );
//
//}

#define HOUGH_CIRCLE_RADIUS_MIN_DIST				3
#define HOUGH_CIRCLE_INTEGRITY_DEGREE				0.6
#define HOUGH_CIRCLE_ACCUM_NORMALIZE_MAX			256
#define HOUGH_CIRCLE_RADIUS_MIN						10
#define HOUGH_CIRCLE_SAMEDIRECT_DEGREE				0.99
#define HOUGH_CIRCLE_GRADIENT_INTEGRITY_DEGREE		0.9

#define HOUGH_MATH_PI								3.14159265358979
// 幂律变化γ的值，一般大于1保证暗的更暗
#define HOUGH_MATH_GAMMA					2.5
#define HOUGH_MATH_SEGFUN_R1				0.3
#define HOUGH_MATH_SEGFUN_S1				0.2
#define HOUGH_MATH_SEGFUN_R2				0.5
#define HOUGH_MATH_SEGFUN_S2				0.8

void myHoughCircles(cv::InputArray _image, cv::OutputArray _circles,
	int method, double dp, double min_dist,
	double param1, double param2,
	int minRadius, int maxRadius);