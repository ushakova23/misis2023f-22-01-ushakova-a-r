#include <semcv/semcv.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>

std::map<FileFormats, std::string> FF{{FileFormats::PNG, ".png"}, {FileFormats::TIF, ".tif"}, {FileFormats::JPG, ".jpg"}, };

void generate_test_images(const std::string& im_path, FileFormats ff, const int scale = 5) {
	cv::Mat im = cv::imread(im_path, cv::IMREAD_UNCHANGED);
	cv::Mat im_new;
	cv::Size im_size = im.size();

	std::ofstream outfile;
	outfile.open("test/task01.lst", std::ios_base::app);

	for (int i = 2; i <= scale; i += 1) {
		cv::resize(im, im_new, cv::Size(im_size.width * i, im_size.height * i));
		cv::imwrite("test/increased_x" + std::to_string(i) + FF[ff], im_new);
		outfile << "increased_x" << std::to_string(i) << FF[ff] << "\n";

		cv::resize(im, im_new, cv::Size(im_size.width / i, im_size.height / i));
		cv::imwrite("test/decreased_x" + std::to_string(i) + FF[ff], im_new);
		outfile << "decreased_x" << std::to_string(i) << FF[ff] << "\n";
	}

	outfile.close();
}

int main(int argc, char *argv[]) {
	// At first, create simple test dataset of images
  /*generate_test_images(cv::samples::findFile("test/base-dog.png"), FileFormats::PNG);
  generate_test_images(cv::samples::findFile("test/base-brain.tif"), FileFormats::TIF);
  generate_test_images(cv::samples::findFile("test/base-night.jpg"), FileFormats::JPG);*/

  // At second, there is vector with paths to test images
  std::string path_input = "";
	if (argc == 1) {
		printf("Path to .lst file is not specified. Standard path is used: ./test/task01.lst\n");
		path_input = "../prj.lab/test/lab01/task01-01/task01.lst";
	}
	else {
		path_input = argv[1];
	}

	std::vector<std::filesystem::path> path_lst = get_list_of_file_paths(path_input);
	if (path_lst.size() == 0) {
		std::cout << "Incorrect path to .lst file.\n";
		return 1;
	}

	for (std::filesystem::path path : path_lst) {
		std::string strid = strid_from_mat(cv::imread(path.string(), cv::IMREAD_UNCHANGED), 5);
		if (strid == path.stem())
			printf("good\n");
		else
			printf("bad, should be %s\n", strid.c_str());
	}

	return 0;
}