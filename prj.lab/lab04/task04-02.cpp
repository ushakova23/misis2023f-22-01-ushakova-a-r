#include <semcv/semcv.hpp>

int main(int argc, char *argv[]) {
    cv::Mat src = cv::imread(argv[1]), grayIm;
    cv::cvtColor(src, grayIm, cv::COLOR_BGR2GRAY);
	cv::imwrite(argv[2], detectBinRobust(grayIm));
	return 0;
}
