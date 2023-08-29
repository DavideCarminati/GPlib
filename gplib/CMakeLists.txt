set(GPLIB_USE_CUDA ON)
if(GPLIB_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "Found CUDA compiler at ${CMAKE_CUDA_COMPILER} version ${CMAKE_CUDA_COMPILER_VERSION}")
    else()
        message(FATAL_ERROR "No CUDA support")
    endif()
    
    # find_package(cublas)
    # find_package(cusolver)
    include_directories(
                    # ../include
                    ${EIGEN_INCLUDE_DIRS})

    file(GLOB SOURCES *.cpp)
    file(GLOB HEADERS ../include/*.hpp)
    add_library(GPlib INTERFACE)

    target_include_directories( GPlib INTERFACE
                                $<INSTALL_INTERFACE:include>
                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                ${CMAKE_CURRENT_LIST_DIR}/../include
                                ${CMAKE_CURRENT_SOURCE_DIR}/src)

    target_link_libraries(  GPlib INTERFACE
                            Eigen3::Eigen
                            # cublas
                            # cusolver
                            )
                            
else()

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

endif()