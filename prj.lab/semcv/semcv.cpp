#include <cstdio>
#include <semcv/semcv.hpp>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>

// Lab 01

std::string type_to_str(int type) {
    uchar depth = type & CV_MAT_DEPTH_MASK;
    switch (depth) {
        case CV_8U:  return "uint08";
        case CV_8S:  return "sint08";
        case CV_16U: return "uint16";
        case CV_16S: return "sint16";
        // case CV_32U: return "uint32"; My compiler does not support this type...
        case CV_32S: return "sint32";
        case CV_32F: return "real32";
        case CV_64F: return "real64";
        default:     return "user"  ;
    }
}

std::string strid_from_mat(const cv::Mat& img, const int n) {
	cv::Size size = img.size();
	auto h = std::string(n - std::min(n, (int) std::to_string(size.height).length()), '0') + std::to_string(size.height);
	auto w = std::string(n - std::min(n, (int) std::to_string(size.width).length()), '0') + std::to_string(size.width);

	return h + "X" + w + "." + std::to_string(img.channels()) + "." + type_to_str(img.type());
}

std::vector<std::filesystem::path> get_list_of_file_paths(const std::filesystem::path& path_lst) {
	std::vector<std::filesystem::path> ps;
	std::ifstream infile(path_lst);
	std::string line;

	while (std::getline(infile, line)) {
		std::filesystem::path temp = path_lst;
		temp.remove_filename();
		temp /= line;
		ps.push_back(temp);
	}

	return ps;
}

// Lab 02

const float WIDTH_OUT = 256;
const float WIDTH_IN  = 209;
const float RADIUS    =  83;

cv::Mat gen_tgtimg00(const int lev0, const int lev1, const int lev2) {
	// Level 0
	cv::Mat img(WIDTH_OUT, WIDTH_OUT, CV_8UC1, cv::Scalar(lev0));

	// Level 1
	cv::Rect rect(0.5 * (WIDTH_OUT - WIDTH_IN), 0.5 * (WIDTH_OUT - WIDTH_IN), WIDTH_IN, WIDTH_IN);
	cv::rectangle(img, rect, cv::Scalar(lev1), cv::FILLED);

	// Level 2
	cv::Point center(0.5 * WIDTH_OUT, 0.5 * WIDTH_OUT);
	cv::circle(img, center, RADIUS, cv::Scalar(lev2), cv::FILLED);

	return img;
}

cv::Mat add_noise_gau(const cv::Mat& img, const int std) {
    cv::Mat noise(img.size(), CV_16SC1), floatImg, noiseImg;
    cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(std));
    img.convertTo(floatImg, CV_16SC1);
    floatImg += noise;
    floatImg.convertTo(noiseImg, CV_8UC1);

    return noiseImg;
}

cv::Mat add_noise_salt(const cv::Mat& img, const int pa, const int pb) {
    cv::Mat noiseImg = img.clone();

    int x = 0, y = 0;
    int nSalt   = pa + (std::rand() % (pb - pa + 1));
    int nPepper = pa + (std::rand() % (pb - pa + 1));

    // Add some salt
    for (int i = 0; i < nSalt; i++) {
        x = std::rand() % img.cols;
        y = std::rand() % img.rows;
        noiseImg.at<uchar>(y, x) = 255;
    }

    // Add some pepper
    for (int i = 0; i < nPepper; i++) {
        x = std::rand() % img.cols;
        y = std::rand() % img.rows;
        noiseImg.at<uchar>(y, x) = 0;
    }

    cv::putText(noiseImg,
        "Salt: " + std::to_string(nSalt) + " Pepper: " + std::to_string(nPepper),
        cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1.f,
        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    return noiseImg;
}

cv::Mat calcHistLib(const cv::Mat& src) {
    cv::Mat hist;
    int channels[] = { 0 };
    int hist_size[] = { 256 };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };

    calcHist(&src, 1, channels, cv::Mat(), hist, 1, hist_size, ranges, true, false);
    // cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    return hist;
}

cv::Mat calcHistNaive(const cv::Mat& src) {
    if (src.channels() == 1) {
        cv::Mat counter = cv::Mat::zeros(256, 1, CV_32S);
        for (int i = 0; i < src.rows; i++)
            for (int j = 0; j < src.cols; j++)
                counter.at<int>(static_cast<uchar>(src.at<uchar>(i, j)), 0)++;
        return counter;
    }
    if (src.channels() == 3) {
        cv::Mat RGB_channels[3], counter = cv::Mat::zeros(256, 3, CV_32S);
        cv::split(src, RGB_channels);
        for (int k = 0; k < 3; k++)
            for (int i = 0; i < RGB_channels[k].rows; i++)
                for (int j = 0; j < RGB_channels[k].cols; j++)
                    counter.at<int>(static_cast<uchar>(RGB_channels[k].at<uchar>(i, j)), k)++;
        return counter;
    }
    return cv::Mat::zeros(256, 3, CV_32S);
}

// Round up to next higher power of 2 (return x if it's already a power of 2).
int nextPow2(int n) {
    if (n <= 1) return n;
    float d = n - 1;
    return 1 << ((((int*) & d)[1] >> 20) - 1022);
}

double getMaxValue(const cv::Mat& src) {
    double minValue, maxValue; cv::Point minLoc, maxLoc;
    cv::minMaxLoc(src, &minValue, &maxValue, &minLoc, &maxLoc);
    return maxValue;
}

cv::Point getMaxLoc(const cv::Mat& src) {
    double minValue, maxValue; cv::Point minLoc, maxLoc;
    cv::minMaxLoc(src, &minValue, &maxValue, &minLoc, &maxLoc);
    return maxLoc;
}

cv::Mat drawHist(const cv::Mat& counter, cv::Scalar backgroundColor, const int height, const int width) {
    cv::Mat histImage = cv::Mat::zeros(height, width, CV_8UC3), counterNorm;
    int binW = cvRound(static_cast<double>(width) / counter.rows), amount = 0;
    double maxValue = getMaxValue(counter);
    histImage.setTo(backgroundColor);
    counter.convertTo(counterNorm, CV_32S, height / maxValue, 0);

    if (counter.cols == 1) {
        for (int i = 0; i < counter.rows; i++) {
        	amount = counterNorm.at<int>(i, 0);
        	if (amount > 0)
        		cv::rectangle(histImage,
        			cv::Point(i * binW, height - amount),
        			cv::Point((i + 1) * binW, height), cv::Scalar(0, 0, 255), -1);
        }
    }
    else if (counter.cols == 3) {
        // Red
        for (int i = 0; i < counter.rows; i++) {
            amount = counterNorm.at<int>(i, 0);
            if (amount > 0)
                cv::rectangle(histImage,
                    cv::Point(i * binW, height - amount),
                    cv::Point((i + 1) * binW, height),
                    cv::Scalar(0, 0, 255), -1);
        }
        // Green
        for (int i = 0; i < counter.rows; i++) {
            amount = counterNorm.at<int>(i, 1);
            if (amount > 0)
                cv::rectangle(histImage,
                    cv::Point(i * binW, height - amount),
                    cv::Point((i + 1) * binW, height),
                    cv::Scalar(0, 255, 0), -1);
        }
        // Blue
        for (int i = 0; i < counter.rows; i++) {
            amount = counterNorm.at<int>(i, 2);
            if (amount > 0)
                cv::rectangle(histImage,
                    cv::Point(i * binW, height - amount),
                    cv::Point((i + 1) * binW, height),
                    cv::Scalar(255, 0, 0), -1);
        }
    }

    // Description is useful for researches!
    cv::putText(histImage, std::to_string(static_cast<int>(maxValue)) + "px_" + std::to_string(static_cast<int>(std::ceil(std::log2(counter.rows)))) + "u",
                cv::Point(5, 25), cv::FONT_HERSHEY_PLAIN, 1.f,
                cv::Scalar(std::abs(255 - backgroundColor[0]),
                        std::abs(255 - backgroundColor[1]),
                        std::abs(255 - backgroundColor[2])), 2, cv::LINE_AA);
    return histImage;
}

cv::Mat drawHistCurve(const cv::Mat& counter, cv::Scalar backgroundColor, const int height, const int width) {
    cv::Mat histImage = cv::Mat::zeros(height, width, CV_8UC3), counterNorm;
    int binW = cvRound(static_cast<double>(width) / counter.rows);
    double maxValue = getMaxValue(counter);
    histImage.setTo(backgroundColor);
    counter.convertTo(counterNorm, CV_32S, height / maxValue, 0);

    if (counter.cols == 1) {
        for (int i = 1; i < 256; i++) {
            cv::line(histImage,
                cv::Point(binW * (i - 1), height - counterNorm.at<int>(i - 1)),
                cv::Point(binW * i, height - counterNorm.at<int>(i)),
                cv::Scalar(0, 0, 255), 2, 8, 0);
        }
    }
    else if (counter.cols == 3) {
        // Red
        for (int i = 1; i < 256; i++) {
            cv::line(histImage,
                cv::Point(binW * (i - 1), height - counterNorm.at<int>(i - 1, 0)),
                cv::Point(binW * i, height - counterNorm.at<int>(i, 0)),
                cv::Scalar(0, 0, 255), 2, 8, 0);
        }
        // Green
        for (int i = 1; i < 256; i++) {
            cv::line(histImage,
                cv::Point(binW * (i - 1), height - counterNorm.at<int>(i - 1, 1)),
                cv::Point(binW * i, height - counterNorm.at<int>(i, 1)),
                cv::Scalar(0, 255, 0), 2, 8, 0);
        }
        // Blue
        for (int i = 1; i < 256; i++) {
            cv::line(histImage,
                cv::Point(binW * (i - 1), height - counterNorm.at<int>(i - 1, 2)),
                cv::Point(binW * i, height - counterNorm.at<int>(i, 2)),
                cv::Scalar(255, 0, 0), 2, 8, 0);
        }
    }

    // Description is useful for researches!
    cv::putText(histImage, std::to_string(static_cast<int>(maxValue)) + "px_" + std::to_string(static_cast<int>(std::ceil(std::log2(counter.rows)))) + "u",
                cv::Point(5, 25), cv::FONT_HERSHEY_PLAIN, 1.f,
                cv::Scalar(std::abs(255 - backgroundColor[0]),
                        std::abs(255 - backgroundColor[1]),
                        std::abs(255 - backgroundColor[2])), 2, cv::LINE_AA);
    return histImage;
}

// Lab 03

std::vector<float> getCDF(const cv::Mat& histogram, const int N, const int bpp = 8) {
    /** Calculate CDF (cumulative distribution function):
     CDF_X(x) = Prob{X <= x} **/

    int histSize = std::pow(2, bpp);

    std::vector<float> cdf(histSize, 0.f);
    for (int i = 0; i < histSize; i++) {
        cdf[i] = histogram.at<float>(i, 0) / static_cast<float>(N);
    }
    for (int i = 1; i < histSize; i++) {
        cdf[i] += cdf[i - 1];
    }

    return cdf;
}

std::vector<int> getThresholds(std::vector<float> cdf, const double q_black, const double q_white) {
    int blackThreshold = 0, whiteThreshold = 255;

    for (int i = 0; i < cdf.size(); i++) {
        if (cdf[i] >= q_black) {
            blackThreshold = i;
            break;
        }
    }

    for (int i = cdf.size() - 1; i >= 0; i--) {
        if (cdf[i] <= (1.0 - q_white)) {
            whiteThreshold = i;
            break;
        }
    }

    // Some logs
    printf("Black threshold: %d\n", blackThreshold);
    printf("White threshold: %d\n", whiteThreshold);

    std::vector<int> res = {blackThreshold, whiteThreshold};
    return res;
}

cv::Mat autocontrast(const cv::Mat& img, const double q_black, const double q_white) {
    int histSize = 256; // 8 bpp
    auto thresholds = getThresholds(getCDF(calcHistLib(img), img.rows * img.cols), q_black, q_white);
    int blackThreshold = thresholds[0], whiteThreshold = thresholds[1];
    cv::Mat res(img.size(), CV_8UC1);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int pixel = img.at<uchar>(i, j);

            // METHOD 1
            // Naive implementation: pixels to the left of whiteThreshold are reset to 0,
            // so pixels to the right of blackThreshold are reset to 255.
            // However this method is pretty dump (see report.md)
            /*
            if (pixel <= blackThreshold) {
                res.at<uchar>(i, j) = 0;
            }
            else if (pixel >= whiteThreshold) {
                res.at<uchar>(i, j) = 255;
            }
            */

            // METHOD 2
            // Linear transformation aX + b:
            // if pixel > wT then new_pixel = 255 (min func),
            // if pixel < bT then new_pixel = 0   (max func),
            // otherwise new_pixel approximately equals to pixel.
            // This linearity provides sufficient smoothness to other pixels!
            res.at<uchar>(i, j) = static_cast<uchar>(std::max(0.f, std::min(255.f, 255.f / static_cast<float>(std::abs(whiteThreshold - blackThreshold)) * (pixel - static_cast<float>(blackThreshold)))));
        }
    }

    /*cv::putText(res, "bT: " + std::to_string(blackThreshold) + " wT: " + std::to_string(whiteThreshold),
    cv::Point(15, 15), cv::FONT_HERSHEY_PLAIN, 1.f, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);*/

    return res;
}

cv::Mat autocontrast_naive(const cv::Mat& img, const double q_black, const double q_white) {
    if (img.channels() == 1) { // Grayscale image
        return autocontrast(img, q_black, q_white);
    }
    // RGB image, channels == 3
    cv::Mat RGB_channels[3], res; cv::split(img, RGB_channels);
    cv::Mat res_channels[3];
    res_channels[0] = autocontrast(RGB_channels[0], q_black, q_white);
    res_channels[1] = autocontrast(RGB_channels[1], q_black, q_white);
    res_channels[2] = autocontrast(RGB_channels[2], q_black, q_white);

    cv::merge(res_channels, 3, res);
    return res;
}

cv::Mat autocontrast_rgb(const cv::Mat& img, const double q_black, const double q_white) {
    cv::Mat HSV; std::vector<cv::Mat> HSV_channels;
    cv::cvtColor(img, HSV, cv::COLOR_BGR2HSV_FULL);
    cv::split(HSV, HSV_channels);
    cv::Mat L = HSV_channels[2];

    auto hist = calcHistLib(L);
    auto thresholds = getThresholds(getCDF(hist, img.rows * img.cols), q_black, q_white);
    int blackThreshold = thresholds[0], whiteThreshold = thresholds[1];

    std::vector<float> modifiedHist(256, 0);
    float sum = 0;
    for (int i = 0; i < 256; i++) {
        modifiedHist[i] = std::round(std::max(0.f, std::min(255.f, 255.f / static_cast<float>(std::abs(whiteThreshold - blackThreshold)) * (hist.at<float>(i, 0) - static_cast<float>(blackThreshold)))));
        sum += modifiedHist[i];
    }

    std::vector<float> cdf(256, 0);
    float culsum = 0;
    for (int i = 0; i < 256; i++) {
        culsum += modifiedHist[i] / sum;
        cdf[i] = culsum;
    }

    std::vector<uchar> table_uchar(256, 0);
    for (int i = 1; i < 256; i++)
        table_uchar[i] = cv::saturate_cast<uchar>(255.0 * cdf[i]);

    cv::LUT(L, table_uchar, L);

    cv::Mat res; cv::merge(HSV_channels, res);
    cv::cvtColor(res, res, cv::COLOR_HSV2BGR_FULL);

    return res;
}

cv::Mat autocontrast_agcie(const cv::Mat& img) {
    int rows = img.rows; int cols = img.cols;
    cv::Mat L, HSV;
    std::vector<cv::Mat> HSV_channels;
    cv::cvtColor(img, HSV, cv::COLOR_BGR2HSV_FULL);
    cv::split(HSV, HSV_channels);
    L = HSV_channels[2];

    cv::Mat L_norm, res;
    L.convertTo(L_norm, CV_64F, 1.f / 255.f);

    cv::Mat mean, stddev;
    cv::meanStdDev(L_norm, mean, stddev);
    double mu = mean.at<double>(0, 0);
    double sigma = stddev.at<double>(0, 0);

    double tau = 3.0;

    double gamma;
    if (4 * sigma <= 1.0 / tau) // low-contrast
        gamma = -std::log2(sigma);
    else // high-contrast
        gamma = std::exp((1.0 - mu - sigma) / 2.0);

    std::vector<double> table_double(256, 0);
    for (int i = 1; i < 256; i++)
        table_double[i] = i / 255.0;

    if (mu >= 0.5) { // bright image
        for (int i = 1; i < 256; i++)
            table_double[i] = std::pow(table_double[i], gamma);
    }
    else { // dark image
        double mu_gamma = std::pow(mu, gamma);
        for (int i = 1; i < 256; i++) {
            double in_gamma = std::pow(table_double[i], gamma);;
            table_double[i] = in_gamma / (in_gamma + (1.0 - in_gamma) * mu_gamma);
        }
    }

    std::vector<uchar> table_uchar(256, 0);
    for (int i = 1; i < 256; i++) {
        table_uchar[i] = cv::saturate_cast<uchar>(255.0 * table_double[i]);
    }

    cv::LUT(L, table_uchar, L);

    cv::merge(HSV_channels, res);
    cv::cvtColor(res, res, cv::COLOR_HSV2BGR_FULL);

    return res;
}

// Lab 04

void drawRandomEllipse(cv::Mat& src, const int w, const int h,
    const int iw, const int ih, const Params& params, json& jsonArray) {
    cv::RNG rng(std::rand());
    cv::Point center;
    cv::Size axes;

    center.x     = rng.uniform(w * iw + w / 4, w * (iw + 1) - w / 4);
    center.y     = rng.uniform(h * ih + h / 4, h * (ih + 1) - h / 4);
    axes.width   = rng.uniform(params.min_elps_width,  params.max_elps_width);
    axes.height  = rng.uniform(params.min_elps_height, params.max_elps_height);
    double angle = rng.uniform(0, 360);
    double start = 0;
    double end   = 360;

    json jw;
    jw["elps_parameters"]["elps_angle"]  = angle;
    jw["elps_parameters"]["elps_height"] = axes.height;
    jw["elps_parameters"]["elps_width"]  = axes.width;
    jw["elps_parameters"]["elps_x"]      = center.x;
    jw["elps_parameters"]["elps_y"]      = center.y;
    jw["pic_coordinates"]["col"]         = iw;
    jw["pic_coordinates"]["row"]         = ih;
    jsonArray.push_back(jw);

    cv::ellipse(src, center, axes, angle, start, end, cv::Scalar(params.elps_color), -1);
}

void addRandomBlur(cv::Mat& src, const int w, const int h,
    const int iw, const int ih, const Params& params) {
    cv::RNG rng(std::rand());
    cv::Point center;

    center.x = rng.uniform(w * iw / 2, w * (iw + 1) / 2);
    center.y = rng.uniform(h * ih / 2, h * (ih + 1) / 2);

    cv::Rect roi(center.x, center.y, w, h);
    cv::Mat roiIm = src(roi);
    cv::Mat blurRoi;
    cv::GaussianBlur(roiIm, blurRoi, cv::Size(params.blur_size, params.blur_size), params.noise_std);
    blurRoi.copyTo(src(roi));
}

cv::Mat genSynthIm(const Params& params, const std::string& genTruth, int seed) {
    int margin = 32, width = 255, height = 255; // Task 04
    cv::Mat img = cv::Mat::zeros(params.n * width, params.n * height, CV_8UC1);
    json jsonWriter;
    jsonWriter["blur_size"]            = params.blur_size;
    jsonWriter["colors"]["bg_color"]   = params.bg_color;
    jsonWriter["colors"]["elps_color"] = params.elps_color;
    jsonWriter["noise_std"]            = params.noise_std;
    jsonWriter["size_of_collage"]      = params.n;
    json objects = json::array();

    std::srand(seed);
    for (int i = 0; i < params.n; i++) {
        for (int j = 0; j < params.n; j++) {
            drawRandomEllipse(img, width, height, i, j, params, objects);
            addRandomBlur(img, width, height, i, j, params);
        }
    }

    jsonWriter["objects"] = objects;
    std::ofstream outfile(genTruth);
    outfile << std::setw(4) << jsonWriter << std::endl;
    outfile.close();

    return add_noise_gau(img, params.noise_std);
}

struct Colors {
    int bg_color;
    int elps_color;
};

struct ElpsParameters {
    float elps_angle;
    int elps_height;
    int elps_width;
    int elps_x;
    int elps_y;
};

struct PicCoordinates {
    int col;
    int row;
};

struct Object {
    ElpsParameters elps_parameters;
    PicCoordinates pic_coordinates;
};

struct Settings {
    float blur_size;
    float noise_std;
    int size_of_collage;
    Colors colors;
    std::vector<Object> objects;
};

Settings genSettings(const json& jw) {
    Settings s;
    s.blur_size         = jw["blur_size"].get<float>();
    s.noise_std         = jw["noise_std"].get<float>();
    s.size_of_collage   = jw["size_of_collage"].get<int>();
    s.colors.bg_color   = jw["colors"]["bg_color"].get<int>();
    s.colors.elps_color = jw["colors"]["elps_color"].get<int>();
    for (const auto& objson : jw["objects"]) {
        Object obj;
        obj.elps_parameters.elps_angle  = objson["elps_parameters"]["elps_angle"].get<float>();
        obj.elps_parameters.elps_height = objson["elps_parameters"]["elps_height"].get<int>();
        obj.elps_parameters.elps_width  = objson["elps_parameters"]["elps_width"].get<int>();
        obj.elps_parameters.elps_x      = objson["elps_parameters"]["elps_x"].get<int>();
        obj.elps_parameters.elps_y      = objson["elps_parameters"]["elps_y"].get<int>();
        obj.pic_coordinates.col         = objson["pic_coordinates"]["col"].get<int>();
        obj.pic_coordinates.row         = objson["pic_coordinates"]["row"].get<int>();
        s.objects.push_back(obj);
    }
    return s;
}

cv::Mat genGroundTruth(const json& jw) {
    Settings s = genSettings(jw);
    int margin = 32, width = 255, height = 255; // Task 04
    cv::Mat groundTruth = cv::Mat::zeros(s.size_of_collage * width, s.size_of_collage * height, CV_8UC1);
    for (Object obj : s.objects) {
        cv::Point center; cv::Size axes;
        center.x     = obj.elps_parameters.elps_x;
        center.y     = obj.elps_parameters.elps_y;
        axes.width   = obj.elps_parameters.elps_width;
        axes.height  = obj.elps_parameters.elps_height;
        double angle = obj.elps_parameters.elps_angle;
        double start = 0;
        double end   = 360;
        cv::ellipse(groundTruth, center, axes, angle, start, end, cv::Scalar(s.colors.elps_color), -1);
    }
    return groundTruth;
}

double compare(const cv::Mat& et, const cv::Mat& im) {
    double counterBAD = 0;
    for (int i = 0; i < et.rows; i++) {
        for (int j = 0; j < et.cols; j++) {
            if (et.at<uchar>(i, j) != im.at<uchar>(i, j)) {
                counterBAD += 1.;
            }
        }
    }
    return (1 - counterBAD / (static_cast<double>(et.rows) * static_cast<double>(et.cols))) * 100;
}

cv::Mat detectBinNaive(const cv::Mat& src) {
    /*
    Naive method of edges detection by substraction:
    - firstly, filter source image
    - secondly, image binarization to detect objects on background
    (we assume that background color is mode in histogram)
    - thirdly, if neighboring pixels are different then this is edge
    (neighborhood means 4-connected component)
    */
    cv::Mat fastIm;
    cv::fastNlMeansDenoising(src, fastIm, 35.0f, 7, 21);

    cv::Mat trIm, edgeIm = cv::Mat::zeros(fastIm.size(), CV_8UC1);
    int threshold = static_cast<int>(getMaxLoc(calcHistNaive(fastIm)).y);
    cv::threshold(fastIm, trIm, threshold, 255, cv::THRESH_BINARY);

    for (int i = 1; i < trIm.rows - 1; i++) {
        for (int j = 1; j < trIm.cols - 1; j++) {
            if ((trIm.at<uchar>(i, j) != trIm.at<uchar>(i + 1, j)) ||
                (trIm.at<uchar>(i, j) != trIm.at<uchar>(i, j + 1)) ||
                (trIm.at<uchar>(i, j) != trIm.at<uchar>(i, j - 1)) ||
                (trIm.at<uchar>(i, j) != trIm.at<uchar>(i - 1, j)))
                    edgeIm.at<uchar>(i, j) = 255;
        }
    }

    /*cv::putText(edgeIm, "th: " + std::to_string(threshold),
        cv::Point(5, 25), cv::FONT_HERSHEY_PLAIN, 1.f,
        cv::Scalar::all(200), 2, cv::LINE_AA);*/

    return edgeIm;
}

cv::Mat erodeEllipse(const cv::Mat& src) {
    cv::Mat res = src.clone();
    for (int i = 0; i < src.cols; i++) {
        bool startLeft  = false;
        bool startRight = false;
        bool interior   = false;
        for (int j = 0; j < src.rows; j++) {
            uchar value = src.at<uchar>(i, j);
            if (value == 255) {
                if (interior) {
                    interior = false;
                    startRight = true;
                    continue;
                }
                if (startLeft == false) {
                    if (startRight == false) {
                        startLeft = true;
                        res.at<uchar>(i, j) = 0;
                    }
                    else {
                        res.at<uchar>(i, j) = 0;
                    }
                }
                else {
                    if (startRight == false) {
                        res.at<uchar>(i, j) = 0;
                    }
                    else {

                    }
                }
            }
            else {
                if (interior) {
                    continue;
                }
                if (startLeft == false) {
                    if (startRight == false) {
                        continue;
                    }
                    else {
                        res.at<uchar>(i, j) = 0;
                        startRight = false; // End of ellipse
                    }
                }
                if (startLeft == true) {
                    res.at<uchar>(i, j - 1) = 255;
                    startLeft = false;
                    interior = true;
                }
            }
        }
    }
    return res;
}

cv::Mat detectBinRobust(const cv::Mat& src) {
    int blurSize  = 3;  // Gaussian kernel
    int blockSize = 35; // Block used for adaptive thresholding
    int threshold = 7;  // Constant subtracted from mean

    // Gaussian blur to reduce noise
    cv::Mat blurIm;
    cv::GaussianBlur(src, blurIm, cv::Size(blurSize, blurSize), 0);

    // Adaptive thresholding for binarization
    cv::Mat binIm;
    cv::adaptiveThreshold(blurIm, binIm, 255,
        cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, // Invert for edges to be white
        blockSize, threshold);

    // Some useful morphology
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binIm, binIm, cv::MORPH_OPEN, kernel); // Opening to remove small noise
    // Next 2 methods could be used but I am not satisfacted by results...
    // cv::erode(binIm, binIm, kernel);
    // cv::erode(binIm, binIm, kernel);
    // ...so we have this method
    // binIm = erodeEllipse(binIm);

    return binIm;
}

// Lab 04 - EXTRA

void drawRandomLine(cv::Mat& src) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::srand(std::time(0));
    cv::RNG rng(std::rand());
    cv::Point ptA, ptB;

    ptA.x = rng.uniform(0, src.cols);
    ptA.y = rng.uniform(0, src.rows);
    ptB.x = rng.uniform(0, src.cols);
    ptB.y = rng.uniform(0, src.rows);

    cv::line(src, ptA, ptB,
        cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
        rng.uniform(1, 10), cv::LINE_AA);
}

void drawRandomPolyline(cv::Mat& src) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::srand(std::time(0));
    cv::RNG rng(std::rand());
    std::vector<cv::Point> points(rng.uniform(3, 10));

    for (int i = 0; i < points.size(); i++) {
        points[i].x = rng.uniform(0, src.cols);
        points[i].y = rng.uniform(0, src.rows);
    }

    cv::polylines(src, points, false,
        cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
        rng.uniform(1, 10), cv::LINE_AA);
}

void drawRandomPolygon(cv::Mat& src) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::srand(std::time(0));
    cv::RNG rng(std::rand());
    std::vector<cv::Point> points(rng.uniform(3, 10));

    for (int i = 0; i < points.size(); i++) {
        points[i].x = rng.uniform(0, src.cols);
        points[i].y = rng.uniform(0, src.rows);
    }

    cv::fillConvexPoly(src, points,
        cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
        cv::LINE_AA);
    cv::polylines(src, points, true,
        cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
        1, cv::LINE_AA);
}

void drawRandomText(cv::Mat& src) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::srand(std::time(0));
    cv::RNG rng(std::rand());
    cv::Point org;
    org.x = rng.uniform(0, src.cols);
    org.y = rng.uniform(0, src.rows);

    cv::putText(src, "TestText",
            org, rng.uniform(0, 8),
            rng.uniform(0, 100) * 0.05 + 0.1,
            cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
            rng.uniform(1, 10), cv::LINE_AA);
}

cv::Mat genSynthRGB(const int width, const int height) {
    cv::Mat img = cv::Mat::zeros(width, height, CV_8UC3);
    cv::RNG rng(std::rand());
    const int minNumber = 3, maxNumber = 5;
    const int N = 3;

    img.setTo(cv::Scalar(rng.uniform(0, 255),
        rng.uniform(0, 255), rng.uniform(0, 255)));

    for (int i = 0; i < N; i++)
        drawRandomLine(img);
    for (int i = 0; i < N; i++)
        drawRandomPolyline(img);
    for (int i = 0; i < N; i++)
        drawRandomPolygon(img);
    //for (int i = 0; i < N; i++)
        //drawRandomEllipse(img);
    //for (int i = 0; i < N; i++)
        //addRandomBlur(img);
    /*for (int i = 0; i < N; i++)
        drawRandomText(img);*/
    cv::Mat noiseImg;
    //add_noise_gau(img, noiseImg);

    return noiseImg;
}

int roundAngle(const float& angle) {
    float phi = angle * 180.0 / 3.14;
    if (phi < 0)
        phi += 360;
    phi = std::fmod(phi, 360);
    if (phi > 180)
        phi -= 180;

    float dist0   = std::abs(phi - 0.0);
    float dist45  = std::abs(phi - 45.0);
    float dist90  = std::abs(phi - 90.0);
    float dist135 = std::abs(phi - 135.0);
    float minDist = std::min({dist0, dist45, dist90, dist135});

    if (minDist == dist45)
        return 45;
    if (minDist == dist90)
        return 90;
    if (minDist == dist135)
        return 135;
    return 0;
}

cv::Mat cannyAlgorithm(const cv::Mat& gradX, const cv::Mat& gradY) {
    // Magnitude and direction of gradient
    cv::Mat magnitude, angle;
    cv::cartToPolar(gradX, gradY, magnitude, angle, true);

    // Non-maximum suppression
    float norm, phi; int quant; float neighA, neighB;
    cv::Mat edges(angle.size(), angle.type());
    for (int i = 1; i < magnitude.rows - 1; i++) {
        for (int j = 1; j < magnitude.cols - 1; j++) {
            norm  = magnitude.at<float>(i, j);
            phi   = angle.at<float>(i, j);
            quant = roundAngle(phi);

            switch (quant) {
                case 0:
                    // East and west direction
                    neighA = magnitude.at<float>(i, j - 1);
                    neighB = magnitude.at<float>(i, j + 1);
                    break;
                case 45:
                    // North-east and south-west direction
                    neighA = magnitude.at<float>(i - 1, j + 1);
                    neighB = magnitude.at<float>(i + 1, j - 1);
                    break;
                case 90:
                    // North and south direction
                    neighA = magnitude.at<float>(i - 1, j);
                    neighB = magnitude.at<float>(i + 1, j);
                    break;
                case 135:
                    // North-west and south-east direction
                    neighA = magnitude.at<float>(i - 1, j - 1);
                    neighB = magnitude.at<float>(i + 1, j + 1);
                    break;
            }

            if ((norm >= neighA) && (norm >= neighB))
                edges.at<float>(i, j) = norm;
            else
                edges.at<float>(i, j) = 0;
        }
    }
    edges.convertTo(edges, CV_8UC1);
    return edges;
}

cv::Mat detectRoberts(const cv::Mat& img) {
    cv::Mat grayIm, denIm;
    cv::fastNlMeansDenoising(img, denIm, 15.);
    cv::cvtColor(denIm, grayIm, cv::COLOR_BGR2GRAY);

    float a[4] = {-1, 0, 0, 1};
    cv::Mat kernelA = cv::Mat(2, 2, CV_32F, a);
    float b[4] = {0, -1, 1, 0};
    cv::Mat kernelB = cv::Mat(2, 2, CV_32F, b);

    // Approximated gradient with Roberts kernel
    cv::Mat gradX, gradY;
    cv::filter2D(grayIm, gradX, -1, kernelA, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(grayIm, gradY, -1, kernelB, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    gradX.convertTo(gradX, CV_32F);
    gradY.convertTo(gradY, CV_32F);

    cv::Mat edges;
    cv::threshold(cannyAlgorithm(gradX, gradY), edges, 0, 255, cv::THRESH_BINARY);
    return edges;
}

cv::Mat detectPrewitt(const cv::Mat& img) {
    cv::Mat grayIm, denIm;
    cv::fastNlMeansDenoising(img, denIm, 15.);
    cv::cvtColor(denIm, grayIm, cv::COLOR_BGR2GRAY);

    float a[9] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    cv::Mat kernelA = cv::Mat(3, 3, CV_32F, a);
    float b[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1};
    cv::Mat kernelB = cv::Mat(3, 3, CV_32F, b);

    // Approximated gradient with Prewitt kernel
    cv::Mat gradX, gradY;
    cv::filter2D(grayIm, gradX, -1, kernelA, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(grayIm, gradY, -1, kernelB, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    gradX.convertTo(gradX, CV_32F);
    gradY.convertTo(gradY, CV_32F);

    cv::Mat edges;
    cv::threshold(cannyAlgorithm(gradX, gradY), edges, 0, 255, cv::THRESH_BINARY);
    return edges;
}

cv::Mat detectSobel(const cv::Mat& img) {
    cv::Mat grayIm, denIm;
    //cv::GaussianBlur(img, denIm, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::fastNlMeansDenoising(img, denIm, 15.);
    cv::cvtColor(denIm, grayIm, cv::COLOR_BGR2GRAY);

    /*
    float a[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    cv::Mat kernelA = cv::Mat(3, 3, CV_32F, a);
    float b[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    cv::Mat kernelB = cv::Mat(3, 3, CV_32F, b);

    // Approximated gradient with Sobel kernel
    cv::filter2D(grayIm, gradX, -1, kernelA, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(grayIm, gradY, -1, kernelB, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    */
    cv::Mat gradX, gradY, absGradX, absGradY, grad;
    cv::Sobel(grayIm, gradX, CV_16S, 1, 0, 3);
    cv::Sobel(grayIm, gradY, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);

    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, grad);
    return grad;
}

// Lab 06

cv::Mat DoG(const cv::Mat& src, double sigmaA, double sigmaB) {
    // Difference of Gaussians approach
    cv::Mat gauA, gauB;
    cv::GaussianBlur(src, gauA, cv::Size(0, 0), sigmaA);
    cv::GaussianBlur(src, gauB, cv::Size(0, 0), sigmaB);
    cv::Mat res = gauA - gauB;

    // Normalize to CV_8U
    cv::normalize(res, res, 0, 255, cv::NORM_MINMAX);
    res.convertTo(res, CV_8U);

    return res;
}

cv::Mat filterLaplace(const cv::Mat& src) {
    cv::Mat out = src.clone();

    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            int sum = src.at<uchar>(y - 1, x) + src.at<uchar>(y + 1, x) + src.at<uchar>(y, x - 1) +
                src.at<uchar>(y, x + 1) - 4 * src.at<uchar>(y, x);
            out.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum);
        }
    }
    return out;
}

cv::Mat LoG(const cv::Mat& src, double sigma) {
    // Laplacian of Gaussian
    cv::Mat res;
    cv::GaussianBlur(src, res, cv::Size(3, 3), sigma);
    res = filterLaplace(res);

    // Normalize to CV_8U
    cv::normalize(res, res, 0, 255, cv::NORM_MINMAX);
    res.convertTo(res, CV_8U);

    return res;
}

cv::Mat detectBlobNaive(const cv::Mat& src, const bool isDoG) {
    cv::Mat gauIm, binIm;
    if (isDoG)
        gauIm = DoG(src, 2.0, 3.0);
    else
        gauIm = LoG(src, 1.0);

    // Binarization
    cv::threshold(gauIm, binIm, 20, 255, cv::THRESH_BINARY);

    // Morphological operation (opening) to remove small noise
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    // cv::dilate(binIm, binIm, kernel);

    return binIm;
}
