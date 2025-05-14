add_library(semcv semcv.hpp semcv.cpp)
target_link_libraries(semcv ${OpenCV_LIBS} nlohmann_json::nlohmann_json)