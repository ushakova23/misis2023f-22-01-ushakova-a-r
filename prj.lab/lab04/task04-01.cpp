#include <semcv/semcv.hpp>

bool is_number(const std::string& s) {
	return !s.empty() && std::find_if(s.begin(), s.end(),
		[](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

Params parseConfig(const std::string& filepath) {
	std::ifstream file(filepath);
	json j; file >> j; Params params;

	params.output_path     = j.at("output_path").get<std::string>();
	params.n               = j.at("n").get<int>();
	params.bg_color        = j.at("bg_color").get<int>();
	params.elps_color      = j.at("elps_color").get<int>();
	params.noise_std       = j.at("noise_std").get<float>();
	params.blur_size       = j.at("blur_size").get<float>();
	params.min_elps_width  = j.at("min_elps_width").get<float>();
	params.max_elps_width  = j.at("max_elps_width").get<float>();
	params.min_elps_height = j.at("min_elps_height").get<float>();
	params.max_elps_height = j.at("max_elps_height").get<float>();

	return params;
}

void genTestIm(const std::string& configPath, const std::string& genIm,
	const std::string& genTruth, int seed) {
	Params params;
	if (configPath == "") {
		// autogenerate config parameters
		params.output_path     = genIm;
		params.n               = 4;
		params.bg_color        = 0;
		params.elps_color      = 255;
		params.noise_std       = 10;
		params.blur_size       = 15;
		params.min_elps_width  = 10;
		params.max_elps_width  = 50;
		params.min_elps_height = 10;
		params.max_elps_height = 50;
	}
	else
		params = parseConfig(configPath);

	cv::imwrite(genIm, genSynthIm(params, genTruth, seed));
}

int main(int argc, char *argv[]) {
	if (argc == 5) {
		genTestIm(argv[1], argv[2], argv[3], std::atoi(argv[3]));
	}
	else if (argc == 4) {
		std::string var3 = argv[3];
		if (is_number(var3)) {
			genTestIm("", argv[1], argv[2], std::atoi(argv[3]));
		}
		else {
			genTestIm(argv[1], argv[2], argv[3], 0);
		}
	}
	else if (argc == 3) {
	    genTestIm("", argv[1], argv[2], 0);
	}
	else
	    return 1;

	return 0;
}
