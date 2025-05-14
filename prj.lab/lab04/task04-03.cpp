#include <semcv/semcv.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
    //std::ifstream infile(argv[1]); json j; infile >> j; infile.close();
    //cv::imwrite(argv[2], genGroundTruth(j));
    cv::Mat et, im;
    cv::cvtColor(cv::imread(argv[1]), et, cv::COLOR_BGR2GRAY);
    cv::cvtColor(cv::imread(argv[2]), im, cv::COLOR_BGR2GRAY);
    std::cout << compare(et, im) << std::endl;
	return 0;
}
