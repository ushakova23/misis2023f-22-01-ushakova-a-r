#pragma once
#ifndef MISIS2025S_3_SEMCV
#define MISIS2025S_3_SEMCV

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>

enum class FileFormats {PNG, TIF, JPG};
enum class FigTypes {Point, Line, Polygon, Ellipse, Text};

// Lab 01
std::string type_to_str(int type);
std::string strid_from_mat(const cv::Mat& img, const int n = 4);
/* Returns std::string like WWWWxHHHH.C.TYPE */
std::vector<std::filesystem::path> get_list_of_file_paths(const std::filesystem::path& path_lst);

// Lab 02
cv::Mat gen_tgtimg00(const int lev0, const int lev1, const int lev2);

cv::Mat add_noise_gau(const cv::Mat& img, const int std);
/* Adds Gauss noise */
cv::Mat add_noise_salt(const cv::Mat& img, const int pa, const int pb);
/* Adds Salt & Pepper Noise */

cv::Mat calcHistNaive(const cv::Mat& src);
cv::Mat calcHistLib(const cv::Mat& src);
cv::Mat drawHist(const cv::Mat& counter, cv::Scalar backgroundColor = cv::Scalar(0, 0, 0), const int height = 256, const int width = 256);
cv::Mat drawHistCurve(const cv::Mat& counter, cv::Scalar backgroundColor = cv::Scalar(0, 0, 0), const int height = 256, const int width = 256);

// Lab 03
cv::Mat autocontrast(const cv::Mat& img, const double q_black, const double q_white);
/* Linear contrast of grayscale image */
cv::Mat autocontrast_naive(const cv::Mat& img, const double q_black, const double q_white);
/* Naive contrast baised on autocontrast() for each channel */
cv::Mat autocontrast_rgb(const cv::Mat& img, const double q_black, const double q_white);
/* Contrast enhancement based on layered difference representation of 2D histograms */
cv::Mat autocontrast_agcie(const cv::Mat & img);
/* Adaptive Gamma Correction for Image Enhancement */

// Lab 04
struct Params {
    std::string output_path;
    int n;
    int bg_color;
    int elps_color;
    float noise_std;
    float blur_size;
    float min_elps_width;
    float max_elps_width;
    float min_elps_height;
    float max_elps_height;
};

using json = nlohmann::json;

cv::Mat genSynthIm(const Params& params, const std::string& genImgStd, int seed);
cv::Mat detectBinNaive(const cv::Mat& src);
cv::Mat detectBinRobust(const cv::Mat& src);
cv::Mat genGroundTruth(const json& jw);
double  compare(const cv::Mat& et, const cv::Mat& im);

// Lab 04 - EXTRA
cv::Mat genSynthRGB(const int width, const int height);
cv::Mat detectRoberts(const cv::Mat& img);
cv::Mat detectPrewitt(const cv::Mat& img);
cv::Mat detectSobel(const cv::Mat& img);

// Lab 05
/* All functions are in task05.cpp */

// Lab 06
cv::Mat detectBlobNaive(const cv::Mat& src, const bool isDoG = true);

#endif
