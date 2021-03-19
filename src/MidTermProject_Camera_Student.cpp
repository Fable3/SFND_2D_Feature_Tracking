/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

void get_response_mean_and_std(const vector<cv::KeyPoint> &keypoints, double &mean, double &std_deviation)
{
	if (keypoints.size() == 0)
	{
		mean = std_deviation = 0;
		return;
	}
	double sum = 0;
	for (auto &kp : keypoints) sum += kp.size;
	mean = sum / keypoints.size();
	double std_dev_square = 0;
	for (auto &kp : keypoints) std_dev_square += (kp.size-mean)*(kp.size - mean);
	std_deviation = sqrt(std_dev_square/ keypoints.size());
}

/* MAIN PROGRAM */

void run(string detectorType, string descriptorType, string stat_type, double &total_time, double &average_match, FILE *fLogFile = NULL)
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
	bool bVisDetection = false;            // visualize detection results
	bool bVisMatch = false;            // visualize matching results

	// for logging statistics:
	//string stat_type = "keypoint_count";
	vector<cv::KeyPoint> all_keypoints; // for neightborhood_size stat only
	int total_keypoint_count = 0;
	int total_match_count = 0;
	double total_keypoint_time = 0;
	total_time = 0;
	int image_count = 0;
	if (fLogFile)
	{
		fprintf(fLogFile, "%s | %s | ", detectorType.c_str(), descriptorType.c_str());
	}
	if (descriptorType == "AKAZE" && detectorType != "AKAZE")
	{
		average_match = 0;
		// invalid combination: @details AKAZE descriptors can only be used with KAZE or AKAZE keypoints.
		return;
	}
	if (descriptorType == "ORB" && detectorType == "SIFT")
	{
		average_match = 0;
		// invalid combination: keypoint octave is too high, it'll cause memory full
		return;
	}

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
		if (dataBuffer.size() >= dataBufferSize) dataBuffer.pop_front(); // using deque

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        //string detectorType = "SHITOMASI";
		//string detectorType = "FAST";
		//string detectorType = "HARRIS_GFT";
		//string detectorType = "HARRIS";
		//string detectorType = "BRISK";

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
		double t = (double)cv::getTickCount();
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
			detKeypointsHarris(keypoints, imgGray, false);
		}
		else if (detectorType.compare("HARRIS_GFT") == 0)
		{
			detKeypointsHarrisWithGoodFeaturesToTrack(keypoints, imgGray, false);
		}
		else
		{
			detKeypointsModern(keypoints, img, detectorType, false);
		}
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
		total_time += t;
		if (bVisDetection /*&& image_count==1*/)
		{
			// visualize results
			cv::Mat visImage = img.clone();
			cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			string windowName = "Detector Results";
			cv::namedWindow(windowName, 6);
			imshow(windowName, visImage);
			cv::waitKey(0);
		}
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect2f vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
			// faster to create new vector than erase elements one by one
			vector<cv::KeyPoint> filtered_keypoints;
			for (auto &kp : keypoints)
			{
				if (vehicleRect.contains(kp.pt)) filtered_keypoints.push_back(kp);
			}
			keypoints = filtered_keypoints;
        }

		image_count++;
		if (fLogFile != NULL)
		{
			if (stat_type == "keypoint_count")
			{
				fprintf(fLogFile, "%d | ", keypoints.size());
				total_keypoint_count += keypoints.size();
			} else if (stat_type == "neightborhood_size")
			{
				double resp_mean, resp_stddev;
				get_response_mean_and_std(keypoints, resp_mean, resp_stddev);
				cout << "mean " << resp_mean << " stddev " << resp_stddev << endl;

				fprintf(fLogFile, "%.2f (%.2f) | ", resp_mean, resp_stddev);
				all_keypoints.insert(all_keypoints.end(), keypoints.begin(), keypoints.end());
			} else if (stat_type == "keypoint_time")
			{
				fprintf(fLogFile, "%.3f | ", 1000 * t);
				total_keypoint_time += 1000 * t;
			}
		}


        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        //string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
		t = (double)cv::getTickCount();
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
		total_time += t;
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string matcherDescriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
			if (descriptorType == "SIFT") matcherDescriptorType = "DES_HOG"; // SIFT uses float
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
			double t = (double)cv::getTickCount();
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, matcherDescriptorType, matcherType, selectorType);
			t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
			cout << matcherType << " " << selectorType << " with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

			if (fLogFile != NULL)
			{
				if (stat_type == "match_count")
				{
					fprintf(fLogFile, "%d | ", matches.size());
				}
			}
			total_match_count += matches.size();

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            if (bVisMatch)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
        }

    } // eof loop over all images

	if (fLogFile != NULL)
	{
		if (stat_type == "keypoint_count")
		{
			fprintf(fLogFile, "%.1f\n", double(total_keypoint_count)/image_count);
		}
		else if (stat_type == "neightborhood_size")
		{
			double resp_mean, resp_stddev;
			get_response_mean_and_std(all_keypoints, resp_mean, resp_stddev);
			fprintf(fLogFile, "%.2f (%.2f)\n", resp_mean, resp_stddev);
		}
		else if (stat_type == "keypoint_time")
		{
			fprintf(fLogFile, "%.3f\n", total_keypoint_time/image_count);
		}
		else if (stat_type == "match_count")
		{
			fprintf(fLogFile, "%.1f\n", total_match_count / double(image_count-1));
		}
	}
	average_match = total_match_count / double(image_count - 1);
}

int main(int argc, const char *argv[])
{
	//string stat_type = "keypoint_count";
	string stat_type = "keypoint_time";
	//string stat_type = "neightborhood_size";
	//string stat_type = "match_count";
	vector<string> all_detectors = { "SHITOMASI", "HARRIS", "HARRIS_GFT", "FAST", "BRISK", "ORB", "AKAZE", "SIFT" };
	vector<string> all_descriptors = { "BRISK", "BRIEF","ORB", "FREAK", "AKAZE", "SIFT"};

	bool measure_detectors = true;
	bool measure_combinations = false;

	if (measure_detectors)
	{
		// Detector statistics
		FILE *fLogFile;
		fopen_s(&fLogFile, (stat_type + ".log").c_str(), "wt");
		for (auto det : all_detectors)
		{
			double total_time, average_match;
			run(det, "BRISK", stat_type, total_time, average_match, fLogFile);
		}
		fclose(fLogFile);
	}
	
	if (measure_combinations)
	{
		// Total Time and Match count calculation for all det/desc combinations
		FILE *fLogFile[2];

		fopen_s(&(fLogFile[0]), "avg_match.log", "wt");
		fopen_s(&(fLogFile[1]), "total_time.log", "wt");
		// headers:
		for (int nf = 0; nf < 2; nf++)
		{
			FILE *f = fLogFile[nf];
			for (auto desc : all_descriptors)
			{
				fprintf(f, "| %s", desc.c_str());
			}
			fprintf(f, "\n---");
			for (auto desc : all_descriptors)
			{
				fprintf(f, "|---");
			}
			fprintf(f, "\n");
		}
		for (auto det : all_detectors)
		{
			fprintf(fLogFile[0], "%s ", det.c_str());
			fprintf(fLogFile[1], "%s ", det.c_str());
			for (auto desc : all_descriptors)
			{
				double total_time = 0, average_match = 0;
				run(det, desc, "", total_time, average_match, NULL);
				fprintf(fLogFile[0], "| %d", (int)average_match);
				fprintf(fLogFile[1], "| %.3f", total_time);
			}
			fprintf(fLogFile[0], "\n");
			fprintf(fLogFile[1], "\n");
		}
		fclose(fLogFile[0]);
		fclose(fLogFile[1]);
	}
	return 0;
}
