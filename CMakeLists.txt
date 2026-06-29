cmake_minimum_required(VERSION 3.12.0)
project(wave_equation LANGUAGES CXX C)

include(../nmpde-labs-aa-25-26-full/common/cmake-common.cmake)

add_executable(exercise-01 src/exercise-01.cpp src/Wave.cpp)
deal_ii_setup_target(exercise-01)
