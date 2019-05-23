# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2019 Intel Corporation. All Rights Reserved.
#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealsenseExamplesDrawRect)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

    add_executable(draw_3drect draw_3drect.cpp ../example.hpp)
    set_property(TARGET draw_3drect PROPERTY CXX_STANDARD 11)
    target_link_libraries(draw_3drect ${DEPENDENCIES} Threads::Threads)
    include_directories(draw_3drect ../)
    set_target_properties (draw_3drect PROPERTIES FOLDER "Examples")

    install(TARGETS draw_3drect RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
