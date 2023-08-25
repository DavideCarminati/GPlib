include_directories(
                    # ../include
                    ${EIGEN_INCLUDE_DIRS})

file(GLOB SOURCES *.cpp)
file(GLOB HEADERS ../include/*.hpp)
add_library(GPlib INTERFACE)
# add_library(gplib::gplib ALIAS GPlib)

target_include_directories( GPlib INTERFACE
                            $<INSTALL_INTERFACE:include>
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            ${CMAKE_CURRENT_LIST_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(  GPlib INTERFACE
                        Eigen3::Eigen)