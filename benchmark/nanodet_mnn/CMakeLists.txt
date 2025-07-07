cmake_minimum_required(VERSION 3.9)
project(nanodet-mnn)

set(CMAKE_CXX_STANDARD 17)

# find_package(OpenCV REQUIRED PATHS "/work/dependence/opencv/opencv-3.4.3/build")

find_package(OpenCV REQUIRED)

set(mnn_HOME "${CMAKE_CURRENT_SOURCE_DIR}/mnn_3.2")
include_directories(
        ${mnn_HOME}/include
        ${mnn_HOME}/include/MNN
        CMAKE_CURRENT_SOURCE_DIR        
)

link_directories(${mnn_HOME}/lib)

add_executable(main_mnn test/main.cpp nanodet_mnn.cpp)
target_link_libraries(main_mnn MNN detector ${OpenCV_LIBS})
