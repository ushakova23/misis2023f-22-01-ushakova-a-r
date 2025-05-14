#include <semcv/semcv.hpp>
#include <stdio.h>

int main(int argc, char *argv[]) {
    std::string typeContrast = "naive"; // [naive|rgb|agcie]
    std::string inputPath    = "../prj.lab/test/lab02/lena.png";
    std::string outputPath   = "./res.png";
    double q_black = 0.01, q_white = 0.005;

    /* Testing on some images
    cv::Mat testIm;
    cv::imwrite("./hist.png", drawHistCurve(calcHistNaive(cv::imread("./test.jpg")), cv::Scalar::all(0), 480, 480));
    cv::imwrite("./test-naive.png", autocontrast_naive(cv::imread("./test.jpg"), q_black, q_white));
    cv::imwrite("./hist-naive.png", drawHistCurve(calcHistNaive(autocontrast_naive(cv::imread("./test.jpg"), q_black, q_white)), cv::Scalar::all(0), 480, 480));
    cv::imwrite("./test-rgb.png", autocontrast_rgb(cv::imread("./test.jpg"), q_black, q_white));
    cv::imwrite("./hist-rgb.png", drawHistCurve(calcHistNaive(autocontrast_rgb(cv::imread("./test.jpg"), q_black, q_white)), cv::Scalar::all(0), 480, 480));
    cv::imwrite("./test-agcie.png", autocontrast_agcie(cv::imread("./test.jpg")));
    cv::imwrite("./hist-agcie.png", drawHistCurve(calcHistNaive(autocontrast_agcie(cv::imread("./test.jpg"))), cv::Scalar::all(0), 480, 480));
    return 0;*/

    if (argc < 6)
        printf("Path to input image is not specified. Local path (to lena.png) is used.\n");
    else {
        typeContrast = argv[1];
        inputPath    = argv[2];
        q_black      = atof(argv[3]);
        q_white      = atof(argv[4]);
        outputPath   = argv[5];
    }

    // Three channels image (8 bpp)
    cv::Mat inputIm = cv::imread(inputPath);
    if (type_to_str(inputIm.type()) != "uint08") {
        printf("Your image has depth %s, we need 8bpp!", type_to_str(inputIm.type()).c_str());
        return 1;
    }
    
    if (typeContrast == "naive") {
        cv::imwrite(outputPath, autocontrast_naive(cv::imread(inputPath), q_black, q_white));
    }
    else if (typeContrast == "rgb") {
        cv::imwrite(outputPath, autocontrast_rgb(cv::imread(inputPath), q_black, q_white));
    }
    else if (typeContrast == "agcie") {
        cv::imwrite(outputPath, autocontrast_agcie(cv::imread(inputPath)));
    }
    else {
        printf("Type of contrast can be only naive, rgb or agcie!");
        return 1;
    }

    return 0;
}
