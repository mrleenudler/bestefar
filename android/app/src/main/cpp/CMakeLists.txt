# JNI-bro + kjerne. Konsumerer samme core/CMakeLists.txt som desktop-bygget —
# kjernen bygges EN gang, verifiseres mot C-settet paa desktop, og brukes her.
cmake_minimum_required(VERSION 3.22)
project(bestefar_android CXX)

# OpenCV Android SDK: sett -DOpenCV_DIR=.../OpenCV-android-sdk/sdk/native/jni
# (via opencvDir i gradle.properties)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core
                 ${CMAKE_BINARY_DIR}/bestefar_core)

add_library(bestefar_jni SHARED jni_bridge.cpp)
target_link_libraries(bestefar_jni PRIVATE bestefar_core log)

# 16 kB-sidestoerrelse (Android 15+): NDK r28+ alignerer som standard, men vi
# setter flagget eksplisitt saa kravet ikke avhenger av NDK-versjonen.
target_link_options(bestefar_jni PRIVATE "-Wl,-z,max-page-size=16384")
