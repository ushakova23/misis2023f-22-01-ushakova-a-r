cmake_minimum_required(VERSION 3.21)
project(median_filter_example)

find_package(OpenCV CONFIG REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIR})

add_executable(${PROJECT_NAME} example_cmd_app.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ccn)

target_compile_definitions(${PROJECT_NAME} PRIVATE INSTALL_PATH="${CMAKE_INSTALL_PREFIX}/bin")
install(TARGETS ${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}
        RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
        POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
        DESTINATION bin
        )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_input DESTINATION ./bin)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_output DESTINATION ./bin)
