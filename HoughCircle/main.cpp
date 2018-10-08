#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#include "ExtendHoughCircle.h"

using namespace cv;

/** @function main */
int main(int argc, char** argv)
{
	Mat src, src_gray;

	/// Read the image
	src = imread("D:\\Project\\VsProject\\HoughCircle\\12.jpg", 1);

	if (!src.data)
	{
		return -1;
	}

	/// Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

	imshow("guass", src_gray);
	cv::waitKey(0);

	std::vector<Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	//myHoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1.0, 5, 80, 100, 0, 0);
	houghcircles(src_gray, circles, 5, 80, 100, 0, 0);

	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(src, center, radius, Scalar(0, 0, 255), 1, 8, 0);
		printf("x:%d, y:%d, r:%d\n", cvRound(circles[i][0]), cvRound(circles[i][1]), radius);
	}

	/// Show your results
	namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	imshow("Hough Circle Transform Demo", src);

	waitKey(0);
	return 0;
}