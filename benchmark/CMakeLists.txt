cmake_minimum_required(VERSION 3.10)

project(yolo_detect)

set(CMAKE_COLOR_DIAGNOSTICS ON)

add_compile_definitions(NCNN_VULKAN)

if(DEBUG_SAVE)
    add_compile_definitions(_DEBUG_SAVE)
    message(STATUS "Define DEBUG_SAVE ok")
endif()

add_compile_options(-O2)

# set(ncnn_DIR /mnt/e/dev/android/ncnn-20250503-android-vulkan-shared/${ANDROID_ABI}/lib/cmake/ncnn)
set(ncnn_DIR /mnt/e/dev/android/ncnn-20250503-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
# set(OpenCV_DIR "/mnt/e/dev/android/opencv-mobile-2.4.13.7-android/sdk/native/jni")
# set(OpenCV_DIR "/mnt/e/dev/android/opencv-mobile-4.11.0-android/sdk/native/jni")
set(OpenCV_DIR "/home/gy/dev/install/build_android/install/sdk/native/jni")
set(Eigen3_DIR "/usr/share/eigen3/cmake")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

set(proj_modules common nanodet nanodet_nihui nanodet_a_demo yolovx yolox picodet)

foreach(module IN LISTS proj_modules)
    file(GLOB module_srcs "${module}/*.cpp")
    list(APPEND srcs ${module_srcs})
endforeach()

# foreach(module IN LISTS modules)
#     list(APPEND inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/${module})
# endforeach()

add_subdirectory(common/fmt)

set(inc_dir ${proj_modules})

message(STATUS "srcs=${srcs}")
message(STATUS "proj_modules=${proj_modules}")

add_library(detector STATIC ${srcs})
target_include_directories(detector PUBLIC ${inc_dir})
target_link_libraries(detector ncnn ${OpenCV_LIBS} fmt::fmt)

set(exe_cpps main.cpp)

foreach(exe_cpp ${exe_cpps})
    get_filename_component(exe_name ${exe_cpp} NAME_WLE)
    message(STATUS "${exe_cpp} -> ${exe_name}")
    add_executable(${exe_name} ${exe_cpp})
    target_include_directories(${exe_name} PRIVATE ${proj_modules} ${CMAKE_SOURCE_DIR})
    target_link_libraries(${exe_name} ncnn ${OpenCV_LIBS} detector)
endforeach()




