cmake_minimum_required(VERSION 3.4.1)

project(yolox)

set(CMAKE_COLOR_DIAGNOSTICS ON)

# set(ncnn_DIR "/mnt/e/android/ncnn-20250503-android-shared/arm64-v8a/lib/cmake/ncnn")
set(ncnn_DIR "/mnt/e/android/ncnn-20250503-android-vulkan/arm64-v8a/lib/cmake/ncnn")
set(OpenCV_DIR /home/gy/dev/packages/install/opencv-mobile-3.4.20-android/sdk/native/jni/abi-arm64-v8a)

message(STATUS "OpenCV_DIR=${OpenCV_DIR}")
message(STATUS "ncnn_DIR=${ncnn_DIR}")

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

aux_source_directory(. SRCS)

add_library(yolox SHARED ${SRCS})

target_link_libraries(yolox
    ncnn
    ${OpenCV_LIBS}
)


add_executable(yolox_test main.cpp)
target_include_directories(yolox_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(yolox_test yolox ${OpenCV_LIBS} ncnn)