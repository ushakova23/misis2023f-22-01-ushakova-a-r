#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include "median_filter.hpp"

using namespace cv;
using namespace std;

class VectorMedianFilter {
public:
    Vec3b calculateMedian(const vector<Vec3b>& pixels) {
        auto distance = [](const Vec3b& a, const Vec3b& b) {
            return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2));
        };

        double minSumDistance = numeric_limits<double>::max();
        Vec3b median = pixels[0];

        for (const auto& p : pixels) {
            double sumDistance = 0.0;
            for (const auto& q : pixels) {
                sumDistance += distance(p, q);
            }

            if (sumDistance < minSumDistance) {
                minSumDistance = sumDistance;
                median = p;
            }
        }

        return median;
    }
};

class ImageProcessor {
public:
    Mat applyVectorMedianFilter(const Mat& src, int kSize) {
        CV_Assert(src.type() == CV_8UC3); // Проверяем, что изображение в формате BGR
        Mat dst = src.clone();
        int radius = kSize / 2;

        for (int y = radius; y < src.rows - radius; ++y) {
            for (int x = radius; x < src.cols - radius; ++x) {
                vector<Vec3b> neighborhood;
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        Vec3b pixel = src.at<Vec3b>(y + dy, x + dx);
                        neighborhood.push_back(pixel);
                    }
                }

                VectorMedianFilter vmf;
                dst.at<Vec3b>(y, x) = vmf.calculateMedian(neighborhood);
            }
        }

        return dst;
    }
};
