#pragma once

#include <cv.h>
//#include <windows.h>
//
//struct circle_found
//{
//	float score;	//ƥ�����
//	cv::Vec3f circle;	//�ҵ���Բ
//};
//
//namespace ExtendCV
//{
//	//_image���������ͼ�񣬱���Ϊ8λ��ͨ����_circles�����ҵ���Բ��dp����cv::houghcircles�е�dp��min_dist����cv::houghcircles�е�minDist��Բ��С����
//	//low_threshold������_imageԤ������ȡ������canny����ֵ��high_threshold������_imageԤ������ȡ������canny����ֵ
//	//acc_threshold����cv::houghcircles�е�param2�ۼ���ֵ��minRadius����Բ����С�뾶��maxRadius����Բ�����뾶
//	//minScore�����ҳ���Բ�����е��������غ��ʣ���Ϊ����
//	//_contour_image������ѡ����������ͼ���������ǿգ���low_threshold��high_threshold���ԣ���������ͼ�Ľ�һ��Ԥ����
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
// ���ɱ仯�õ�ֵ��һ�����1��֤���ĸ���
#define HOUGH_MATH_GAMMA					2.5
#define HOUGH_MATH_SEGFUN_R1				0.3
#define HOUGH_MATH_SEGFUN_S1				0.2
#define HOUGH_MATH_SEGFUN_R2				0.5
#define HOUGH_MATH_SEGFUN_S2				0.8

void myHoughCircles(cv::InputArray _image, cv::OutputArray _circles,
	int method, double dp, double min_dist,
	double param1, double param2,
	int minRadius, int maxRadius);