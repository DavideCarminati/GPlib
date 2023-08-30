include_directories(../gplib
                    ${EIGEN3_INCLUDE_DIRS})

message(STATUS "Eigen folder: ${EIGEN3_INCLUDE_DIRS}")

add_executable(gplib_ex gp_main.cpp)

# find_package(GPlib)
set(CMAKE_LIBRARY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../build/gplib)

target_link_libraries(  gplib_ex Eigen3::Eigen GPlib
                        # ${libpath}
                        )
install(TARGETS gplib_ex DESTINATION ${CMAKE_INSTALL_BINDIR})
# target_link_libraries(gplib_ex LINK_PUBLIC GPlib)