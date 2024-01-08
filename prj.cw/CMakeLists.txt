add_executable(main "main.cpp")

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIR})
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIR})
target_link_libraries(main PUBLIC ${OpenCV_LIBS})
target_link_libraries(main PUBLIC ${Boost_LIBRARIES})
