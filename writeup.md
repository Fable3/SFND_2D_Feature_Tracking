# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

## MP.1 Data Buffer Optimization

The ring buffer is implemented by using an STL container: deque<DataFrame> dataBuffer
The interface is similar to vector, but calling dataBuffer.pop_front() just advances the head pointer. After that the memory is reused when calling dataBuffer.push_back(frame) instead of growing the container.

## MP.2 Keypoint Detection

HARRIS uses the cv::cornerHarris function. It's also possible to invoke Harris detection through cv::goodFeaturesToTrack, I've added a HARRIS_GFT type for that to have both options.
SHITOMASI and HARRIS_GFT have the same interface in OpenCV, so I created a detKeypointsShiTomasiOrHarris in which bUseHarris is a parameter passed to the cv::goodFeaturesToTrack.
For FAST, BRISK, ORB, AKAZE, and SIFT I use detKeypointsModern, as suggested by the provided header file. There's a class for each of the detectors, implementing the common parent cv::FeatureDetector detect function.
For each detectors, the setup is different and can be optimized separately. I started out with the default parameters.

## MP.3 Keypoint Removal

The rectangle for the preceding vehicle was provided, however, I changed the type to Rect2f since all the points were also using float coordinates.
One possibility would be to erase all points from the vector.
That would have been slow, since each time an element is removed from a vector, the rest is shifted, moving all remaining items in the memory.
The more optimal solution is to create a separate vector for the result and only copy points which are inside the rectangle, using the Rect2f::contains operator.

## MP.4 Keypoint Descriptors

Similar to detection, there's a common interface for descriptors, cv::DescriptorExtractor::compute.
Since both cv::DescriptorExtractor and cv::FeatureDetector are just typedefs to cv::Feature2D, the detector and descriptor selection could use the same code, returning Feature2D type.
BRIEF, ORB, FREAK, AKAZE and SIFT
There're some limitations in OpenCV:
```
@details AKAZE descriptors can only be used with KAZE or AKAZE keypoints.
```
Another issue is that SIFT detector fills high octave values for keypoints and ORB descriptor uses it to allocate memory:
```
level = keypoints[i].octave;
```

## MP.5 Descriptor Matching

There are 3 parameters for matching:
- matcherType supports MAT_BF for Brute Force matching, and MAT_FLANN for FLANN based matching.
- descriptorType can be DES_BINARY for binary descriptors and DES_HOG for float based descriptors. This is only used for BF matching, because FLANN only supports float type descriptors, and thus binary descriptors are converted to float as a workaround.
- selectorType is either SEL_NN for nearest neighbor, or SEL_KNN (see next paragraph)

From the OpenCV documentation:
```
@param normType One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are
    preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and
    BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor
    description).
```
In our case, SIFT uses L2, the rest uses HAMMING.

## MP.6 Descriptor Distance Ratio

If selectorType is SEL_KNN, the knnMatch will be used with k=2. This is for a more advanced selection where the best match is only accepted if the second best is much worse, using a 0.8 ratio.

## MP.7 Performance Evaluation 1

table test:
Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

Number of keypoints detected on preceding car:
(ORB is limited by parameter)

detector | \#1 | \#2| \#3| \#4| \#5| \#6| \#7| \#8| \#9| \#10 | Average
---|---|---|---|---|---|---|---|---|---|---|---
SHITOMASI |  125 | 118 | 123 | 120 | 120 | 113 | 114 | 123 | 111 | 112 | 117.9
HARRIS |  17 | 14 | 17 | 20 | 25 | 40 | 18 | 28 | 24 | 33 | 23.6
HARRIS_GFT |  50 | 54 | 53 | 55 | 56 | 58 | 57 | 61 | 59 | 57 | 56.0
FAST |  149 | 152 | 150 | 155 | 149 | 149 | 156 | 150 | 138 | 143 | 149.1
BRISK | 254 | 274 | 276 | 275 | 293 | 275 | 289 | 268 | 259 | 250 | 271.3
ORB |  91 | 102 | 106 | 113 | 109 | 124 | 129 | 127 | 124 | 125 | 115.0
AKAZE |  162 | 157 | 159 | 154 | 162 | 163 | 173 | 175 | 175 | 175 | 165.5
SIFT |  137 | 131 | 121 | 135 | 134 | 139 | 136 | 147 | 156 | 135 | 137.1

Time for detection only (ms):

detector | \#1 | \#2| \#3| \#4| \#5| \#6| \#7| \#8| \#9| \#10 | Average
---|---|---|---|---|---|---|---|---|---|---|---
SHITOMASI |  15.311 | 13.451 | 11.597 | 11.788 | 11.806 | 12.345 | 12.163 | 11.553 | 11.416 | 13.018 | 12.445
HARRIS |  12.066 | 12.252 | 10.560 | 10.404 | 13.702 | 15.939 | 10.612 | 11.835 | 11.950 | 13.230 | 12.255
HARRIS_GFT |  12.657 | 10.729 | 12.631 | 12.608 | 11.838 | 10.787 | 11.090 | 13.647 | 13.835 | 11.453 | 12.127
FAST |  1.086 | 1.073 | 1.102 | 1.053 | 1.089 | 1.146 | 1.064 | 1.440 | 1.116 | 1.059 | 1.123
BRISK | 239.208 | 244.529 | 245.291 | 233.253 | 236.359 | 236.963 | 231.385 | 231.344 | 241.744 | 245.554 | 238.563
ORB |  277.560 | 6.907 | 6.729 | 7.708 | 6.963 | 7.808 | 7.078 | 6.814 | 7.503 | 7.088 | 34.216
AKAZE |  72.456 | 74.631 | 74.749 | 76.185 | 75.209 | 78.532 | 70.483 | 80.414 | 75.498 | 72.332 | 75.049
SIFT |  84.392 | 87.303 | 85.522 | 86.260 | 84.681 | 84.621 | 76.301 | 82.530 | 81.760 | 86.132 | 83.950

Mean of neighbour size with standard deviation per image and also for all images:

detector | \#1 | \#2| \#3| \#4| \#5| \#6| \#7| \#8| \#9| \#10 | All images
---|---|---|---|---|---|---|---|---|---|---|---
SHITOMASI |  4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00)
HARRIS |  3.00 (0.00) | 3.00 (0.00) | 3.00 (0.00) | 3.00 (0.00) | 3.00 (0.00) | 3.00 (0.00) | 3.00 (0.00) | 3.00 (0.00) | 3.00 (0.00) | 3.00 (0.00) | 3.00 (0.00)
HARRIS_GFT |  4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00) | 4.00 (0.00)
FAST |  7.00 (0.00) | 7.00 (0.00) | 7.00 (0.00) | 7.00 (0.00) | 7.00 (0.00) | 7.00 (0.00) | 7.00 (0.00) | 7.00 (0.00) | 7.00 (0.00) | 7.00 (0.00) | 7.00 (0.00)
BRISK |  21.44 (14.44) | 21.95 (14.66) | 21.84 (13.91) | 20.41 (12.65) | 22.78 (14.93) | 23.01 (15.89) | 21.81 (14.67) | 22.28 (15.11) | 22.75 (15.29) | 22.22 (14.71) | 22.05 (14.66)
ORB |  57.11 (25.85) | 57.23 (26.09) | 56.49 (25.93) | 55.14 (25.10) | 56.74 (25.01) | 56.58 (24.52) | 56.61 (25.46) | 55.27 (24.85) | 54.51 (25.48) | 54.41 (23.95) | 55.95 (25.21)
AKAZE |  7.77 (3.96) | 7.49 (3.52) | 7.41 (3.50) | 7.48 (3.39) | 7.75 (3.44) | 7.70 (3.39) | 7.74 (3.43) | 7.83 (3.51) | 7.79 (3.50) | 7.85 (3.62) | 7.69 (3.53)
SIFT |  5.04 (5.96) | 5.09 (6.20) | 4.96 (6.09) | 4.74 (5.28) | 4.72 (5.51) | 4.65 (5.58) | 5.43 (6.53) | 4.63 (5.16) | 5.58 (6.72) | 5.67 (6.72) | 5.06 (6.01)


## MP.8 Performance Evaluation 2

Average number of matching points on preceding car for all detector + descriptor combinations:

| BRISK| BRIEF| ORB| FREAK| AKAZE| SIFT
---|---|---|---|---|---|---
SHITOMASI | 85| 104| 100| 85| 0| 103
HARRIS | 15| 18| 17| 15| 0| 17
HARRIS_GFT | 43| 51| 50| 44| 0| 51
FAST | 99| **122**| **119**| 97| 0| 116
BRISK | 171| 186| 164| 166| 0| 179
ORB | 82| 60| 83| 46| 0| 84
AKAZE | 133| 139| 130| 131| 138| 140
SIFT | 65| 77| 0| 65| 0| 87

I've highlighted the 2 best options considering time as well.

## MP.9 Performance Evaluation 3

Total time for detection on 10 images and matching on consecutive pairs (9 total) in milliseconds:

| BRISK| BRIEF| ORB| FREAK| AKAZE| SIFT
---|---|---|---|---|---|---
SHITOMASI | 2.131| 0.128| 0.129| 0.399| 0.000| 0.261
HARRIS | 2.117| 0.118| 0.119| 0.387| 0.000| 0.265
HARRIS_GFT | 2.080| 0.118| 0.122| 0.400| 0.000| 0.250
FAST | 2.015| **0.020**| **0.021**| 0.289| 0.000| 0.169
BRISK | 4.365| 2.394| 2.390| 2.681| 0.000| 2.715
ORB | 2.309| **0.080**| 0.118| 0.344| 0.000| 0.426
AKAZE | 2.770| 0.725| 0.741| 0.995| 1.327| 0.910
SIFT | 2.826| 0.818| 0.000| 1.108| 0.000| 1.590

FAST detection is exceptionally fast compared to the others. Among the possible descriptors, BRIEF is the fastest and has the most matching point, but ORB is a close second.
The third option would be ORB detector with BRIEF descriptor. The match count is a bit low (60), but it's probably because the ORB detector is limited to 500 keypoints, and many of them are outside of the region of interest.

So my recommended 3 best detector + descriptor pairs are:
FAST+BRIEF
FAST+ORB
ORB+BRIEF
