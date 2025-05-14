#include <semcv/semcv.hpp>

int main(int argc, char *argv[]) {
    cv::Mat im; cv::cvtColor(cv::imread(argv[1]), im, cv::COLOR_BGR2GRAY);
    cv::imwrite(argv[2], detectBlobNaive(im));
    return 0;
}
