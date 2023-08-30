cmake_minimum_required(VERSION 3.5.0)
project(GPlib VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CTest)
enable_testing()

set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 REQUIRED)

# add_subdirectory(src)
add_subdirectory(gplib)


# include_directories(include
#                     ${EIGEN_INCLUDE_DIRS})

# file(GLOB SOURCES src/*.cpp)
# file(GLOB HEADERS include/*.hpp)
# add_library(GPlib SHARED ${SOURCES} ${HEADERS})
# # add_library(gplib::gplib ALIAS GPlib)

# target_include_directories( GPlib PUBLIC
#                             $<INSTALL_INTERFACE:include>
#                             $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#                             PUBLIC
#                             ${CMAKE_CURRENT_SOURCE_DIR}/src)

# target_link_libraries(  GPlib
#                         Eigen3::Eigen)

# set_target_properties(GPlib PROPERTIES PUBLIC_HEADER include/gplib.hpp)

add_subdirectory(examples)

# add_executable(gplib_ex examples/gp_main.cpp)

# target_link_libraries(  gplib_ex
#                         Eigen3::Eigen
#                         ${CMAKE_CURRENT_BINARY_DIR}/libGPlib.so)    
                        
# message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
# include(autodiffInstallCMakeConfigFiles)
