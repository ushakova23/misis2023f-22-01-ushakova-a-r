#include <opencv2/opencv.hpp>
#include "median_filter.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <InputImage> <OutputImage>" << endl;
        return -1;
    }

    Mat inputImage = imread(argv[1], IMREAD_COLOR);
    if (InputImage.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    ImageProcessor ip;
    Mat outputImage = ip.applyVectorMedianFilter(InputImage, 3); // Пример использования с размером ядра 3x3

    imwrite(argv[2], OutputImage);

    return 0;
}
