#ifndef VECTOR_MEDIAN_FILTER_HPP
#ifndef IMAGE_PROCESSOR_HPP
#define VECTOR_MEDIAN_FILTER_HPP
#define IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

class VectorMedianFilter {
public:
    Vec3b calculateMedian(const vector<Vec3b>& pixels);
};

class ImageProcessor {
public:
    Mat applyVectorMedianFilter(const Mat& src, int kSize);
};

#endif // VECTOR_MEDIAN_FILTER_HPP
#endif // IMAGE_PROCESSOR_HPP
