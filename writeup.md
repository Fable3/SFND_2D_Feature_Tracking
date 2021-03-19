# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

## MP.1 Data Buffer Optimization

The ring buffer is implemented by using an STL container: deque<DataFrame> dataBuffer
The interface is similar to vector, but calling dataBuffer.pop_front() just advances the head pointer. After that the memory is reused when calling dataBuffer.push_back(frame) instead of growing the container.

## MP.2 Keypoint Detection

SHITOMASI and HARRIS have the same interface in OpenCV, so I created a detKeypointsShiTomasiOrHarris in which bUseHarris is a parameter passed to the cv::goodFeaturesToTrack.
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

## MP.5 Descriptor Matching

There are 3 parameters for matching:
- matcherType supports MAT_BF for Brute Force matching, and MAT_FLANN for FLANN based matching.
- descriptorType can be DES_BINARY for binary descriptors and DES_HOG for float based descriptors. This is only used for BF matching, because FLANN only supports float type descriptors, and thus binary descriptors are converted to float as a workaround.
- selectorType is either SEL_NN for nearest neighbor, or SEL_KNN (see next paragraph)

## MP.6 Descriptor Distance Ratio

If selectorType is SEL_KNN, the knnMatch will be used with k=2. This is for a more advanced selection where the best match is only accepted if the second best is much worse, using a 0.8 ratio.
