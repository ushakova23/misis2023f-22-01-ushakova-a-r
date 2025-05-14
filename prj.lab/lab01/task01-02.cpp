#include <semcv/semcv.hpp>
#include <stdio.h>
#include <fstream>

void gamma_correction(const cv::Mat &src, cv::Mat &dst, const float gamma) {
    cv::Mat table(1, 256, CV_8U);
    uchar *p = table.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = (uchar) (pow(i / 255.f, 1 / gamma) * 255.f);

    cv::LUT(src, table, dst);
}

int main(int argc, char *argv[]) {

	cv::Mat img(30, 768, CV_8UC1, cv::Scalar(0)), gamma_img, collage;
	int thickness = 3;

	for (int i = 1; i <= 255; i++) {
		cv::Point p1(3 * i, 0), p2(3 * (i + 1), 30);
		cv::line(img, p1, p2, cv::Scalar(i), thickness);
	}

	cv::Mat temp;
	gamma_correction(img, temp, 1.);
	for (float g = 1.8; g <= 2.8; g += 0.2) {
		gamma_correction(img, gamma_img, g);
		cv::vconcat(temp, gamma_img, collage);
		temp = collage.clone();
	}

	std::string save_path = "";
	if (argc == 1) {
		printf("Path to output image is not specified. Standard path is used: ./task01-02.png\n");
		save_path = "./task01-02.png";
	}
	else {
		save_path = argv[1];
	}

  	cv::imwrite(save_path, collage);
	return 0;
}