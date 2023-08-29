include_directories(../gplib
                    ${EIGEN_INCLUDE_DIRS})

add_executable(gplib_ex gp_main.cpp)

# find_package(GPlib)
set(CMAKE_LIBRARY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../build/gplib)

target_link_libraries(  gplib_ex Eigen3::Eigen GPlib
                        # ${libpath}
                        )
# target_link_libraries(gplib_ex LINK_PUBLIC GPlib)