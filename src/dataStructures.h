#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>
#include <cstdio>

struct DataFrame { // represents the available sensor information at the same time instance

	cv::Mat cameraImg; // camera image

	std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
	cv::Mat descriptors; // keypoint descriptors
	std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
	DataFrame()
	{
		//std::cout << "DataFrame create\n";
	}
	~DataFrame()
	{
		//std::cout << "DataFrame destroy\n";
	}

};


#endif /* dataStructures_h */
