#include <semcv/semcv.hpp>
#include <stdio.h>
#include <fstream>
#include <iostream>

// Colors of histograms
auto anthrocite   = cv::Scalar(50, 50, 50);
auto gainsborough = cv::Scalar(128, 128, 128);

cv::Mat createNoiseCollage() {
	cv::Mat temp1, temp2, res;
	auto img00 = gen_tgtimg00(0,  127, 255);
	auto img01 = gen_tgtimg00(20, 127, 235);
	auto img02 = gen_tgtimg00(55, 127, 200);
	auto img03 = gen_tgtimg00(90, 127, 165);

	cv::hconcat(img00, img01, temp1);
	cv::hconcat(img02, img03, temp2);
	cv::hconcat(temp1, temp2, res);

	auto res00 = add_noise_gau(res, 3);
	auto res01 = add_noise_gau(res, 7);
	auto res02 = add_noise_gau(res, 15);

	cv::vconcat(res, res00, temp1);
	cv::vconcat(temp1, res01, temp2);
	cv::vconcat(temp2, res02, res);

	return res;
}

cv::Mat noiseHistConcat(const int h, const bool flagOdd) {
	cv::Mat tempA, tempB, res, img00, img01, img02, img03;

	if (flagOdd == true) {
		img00 = drawHist(calcHistNaive(add_noise_gau(gen_tgtimg00(0,  127, 255), h)), anthrocite);
		img01 = drawHist(calcHistNaive(add_noise_gau(gen_tgtimg00(20, 127, 235), h)), gainsborough);
		img02 = drawHist(calcHistNaive(add_noise_gau(gen_tgtimg00(55, 127, 200), h)), anthrocite);
		img03 = drawHist(calcHistNaive(add_noise_gau(gen_tgtimg00(90, 127, 165), h)), gainsborough);
	}
	else {
		img00 = drawHist(calcHistNaive(add_noise_gau(gen_tgtimg00(0,  127, 255), h)), gainsborough);
		img01 = drawHist(calcHistNaive(add_noise_gau(gen_tgtimg00(20, 127, 235), h)), anthrocite);
		img02 = drawHist(calcHistNaive(add_noise_gau(gen_tgtimg00(55, 127, 200), h)), gainsborough);
		img03 = drawHist(calcHistNaive(add_noise_gau(gen_tgtimg00(90, 127, 165), h)), anthrocite);
	}

	cv::hconcat(img00, img01, tempA);
	cv::hconcat(img02, img03, tempB);
	cv::hconcat(tempA, tempB, res);

	return res;
}

cv::Mat createHistCollage() {
	cv::Mat tempA, tempB, res0, res;

	auto img00 = drawHist(calcHistNaive(gen_tgtimg00(0,  127, 255)), anthrocite);
	auto img01 = drawHist(calcHistNaive(gen_tgtimg00(20, 127, 235)), gainsborough);
	auto img02 = drawHist(calcHistNaive(gen_tgtimg00(55, 127, 200)), anthrocite);
	auto img03 = drawHist(calcHistNaive(gen_tgtimg00(90, 127, 165)), gainsborough);

	cv::hconcat(img00, img01, tempA);
	cv::hconcat(img02, img03, tempB);
	cv::hconcat(tempA, tempB, res0);

	auto res1 = noiseHistConcat(3, false);
	auto res2 = noiseHistConcat(7, true);
	auto res3 = noiseHistConcat(15, false);

	cv::vconcat(res0, res1, tempA);
	cv::vconcat(tempA, res2, tempB);
	cv::vconcat(tempB, res3, res);

	return res;
}

void generateNoises(const cv::Mat& input2Gray) {
    auto gauNoisedIm  = add_noise_gau(input2Gray, 20);
    auto saltNoisedIm = add_noise_salt(input2Gray, 1000, 5000);

    cv::imwrite("./gauss.png", gauNoisedIm);
    cv::imwrite("./salt.png",  saltNoisedIm);

    cv::imwrite("./gauss-hist.png", drawHist(calcHistNaive(gauNoisedIm)));
    cv::imwrite("./salt-hist.png",  drawHist(calcHistNaive(saltNoisedIm)));
}

int main(int argc, char *argv[]) {
	std::string savePathCollage = "";
	std::string savePathHist    = "";

	if ((argc == 1) || (argc == 2)) {
		printf("Path to output images is not specified. Local path is used.\n");
		savePathCollage = "./collage.png";
		savePathHist    = "./hist.png";
	}
	else {
		savePathCollage = argv[1];
		savePathHist    = argv[2];
	}

	cv::imwrite(savePathCollage, createNoiseCollage());
	cv::imwrite(savePathHist,    createHistCollage());
	
	// Testing different noises
	/*cv::Mat testIm;
	cv::cvtColor(cv::imread("../prj.lab/test/lab02/lena.png"), testIm, cv::COLOR_BGR2GRAY);
	generateNoises(testIm);*/

	// Testing if distribution is really normal
	/*std::ofstream out;
	out.open("dist-gauss.txt", std::ios::app);

	for (int std = 1; std <= 50; std++) {
		cv::Mat noiseMat = cv::Mat::zeros(256, 256, CV_8UC1);
		cv::randn(noiseMat, cv::Scalar(255 / 2), cv::Scalar(std));

		for (int i = 0; i < 256; i++)
			for (int j = 0; j < 256; j++)
				out << std::to_string(noiseMat.at<uchar>(i, j)) << " ";
		out << "\n";

		auto histImg = drawHistCurve(calcHistLib(noiseMat));
		cv::imwrite("ttt/hist" + std::to_string(std) + ".png", histImg);
	}

	out.close();*/
	
  	return 0;
}