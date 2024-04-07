# Specify the minimum version.
cmake_minimum_required(VERSION 3.21)

# Specify the project info.
project(ccn VERSION 1.0.0 DESCRIPTION "median_filter library")

# For access to standard installation directory variables (CMAKE_INSTALL_xDIR).
include(GNUInstallDirs)

# Declare the library target.
add_library(
        ${PROJECT_NAME}
        source/median_filter.cpp
        source/median_filter.hpp
)

if(DEFINED MYLIB_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${MYLIB_SHARED_LIBS})
endif()

if (PROJECT_IS_TOP_LEVEL)
    set_target_properties(
            ${PROJECT_NAME}
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out
    )
endif ()

# Set the version property.
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER source/median_filter.hpp
)

# Configure the directories to search for header files.
target_include_directories(${PROJECT_NAME} PUBLIC source)

find_package(OpenCV CONFIG REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

# Set library shared object and API header file to install

add_subdirectory(example)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ccn_doc DESTINATION bin)
install(TARGETS ${PROJECT_NAME}
        RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
        POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
        DESTINATION lib
)
install(
        TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
