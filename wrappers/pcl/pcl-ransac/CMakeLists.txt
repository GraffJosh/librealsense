cmake_minimum_required(VERSION 3.1.0)

project(RealSensePCLRansac)
set(CMAKE_BUILD_TYPE Debug)
add_executable (rs-pcl-ransac rs-pcl-ransac.cpp ../RS2_example.hpp)
set_property(TARGET rs-pcl-ransac PROPERTY CXX_STANDARD 11)
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(rs-pcl-ransac ${DEPENDENCIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})

find_package(OpenCV REQUIRED)
target_link_libraries(rs-pcl-ransac ${DEPENDENCIES} ${OpenCV_LIBS})

set_target_properties (rs-pcl-ransac PROPERTIES
    FOLDER "Examples/PCL"
)

foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MT")
        string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MT")
endforeach(flag_var)

install(
    TARGETS

    rs-pcl-ransac

    RUNTIME DESTINATION
    ${CMAKE_INSTALL_PREFIX}/bin
)
