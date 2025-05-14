#include <semcv/semcv.hpp>

const int WIDTH  = 127;
const int HEIGHT = 127;

cv::Mat gen_tgtimg01(const int lev0, const int lev1,
			const double length = 40, const double radius = 40) {
    cv::Mat img(WIDTH, HEIGHT, CV_8UC1);

    cv::rectangle(img, cv::Point(0, 0), cv::Point(WIDTH, HEIGHT),
    	cv::Scalar(lev0), -1);
    cv::circle(img, cv::Point(WIDTH / 2, HEIGHT / 2), radius,
    	cv::Scalar(lev1), -1);

    return img;
}

int main(int argc, char *argv[]) {
	std::vector<cv::Mat> imgArray;
	std::vector<int> colors = {0, 127, 255};
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i != j) {
				imgArray.push_back(gen_tgtimg01(colors[i], colors[j]));
			}
		}
	}

	float array1[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    cv::Mat ker1 = cv::Mat(3, 3, CV_32F, array1);
    float array2[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    cv::Mat ker2 = cv::Mat(3, 3, CV_32F, array2);

    for (int i = 0; i < imgArray.size(); i++) {
    	cv::Mat img = imgArray[i], img1, img2, temp1, temp2, res;
    	cv::filter2D(img, img1, -1, ker1, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    	cv::filter2D(img, img2, -1, ker2, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    	cv::Mat img3 = cv::Mat::zeros(img.size(), CV_8UC1);
    	for (int i = 0; i < img3.rows; i++) {
    		for (int j = 0; j < img3.cols; j++) {
    			img3.at<uchar>(i, j) = std::sqrt(std::pow(img1.at<uchar>(i, j), 2) +
    				std::pow(img2.at<uchar>(i, j), 2));
    		}
    	}
        cv::Mat channels[3] = {img1, img2, img3}, img4;
        cv::merge(channels, 3, img4);
        cv::Mat RGBimg1, RGBimg2, RGBimg3;
		cvtColor(img1, RGBimg1, cv::COLOR_GRAY2RGB);
		cvtColor(img2, RGBimg2, cv::COLOR_GRAY2RGB);
		cvtColor(img3, RGBimg3, cv::COLOR_GRAY2RGB);
        cv::hconcat(RGBimg1, RGBimg2, temp1);
        cv::hconcat(RGBimg3, img4, temp2);
        cv::vconcat(temp1, temp2, res);
        std::string path1 = argv[1], path2 = argv[2];
        cv::imwrite("0" + std::to_string(i) + "_" + path1, img);
        cv::imwrite("0" + std::to_string(i) + "_" + path2, res);
    }

	return 0;
}
