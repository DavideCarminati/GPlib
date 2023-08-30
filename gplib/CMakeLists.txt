set(GPLIB_USE_CUDA ON)
if(GPLIB_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "Found CUDA compiler at ${CMAKE_CUDA_COMPILER} (version ${CMAKE_CUDA_COMPILER_VERSION})")
    else()
        message(FATAL_ERROR "No CUDA support")
    endif()
endif()
    
include_directories(
                    # ../include
                    ${EIGEN_INCLUDE_DIRS})

file(GLOB SOURCES *.cpp)
file(GLOB HEADERS ../include/*.hpp)
add_library(GPlib INTERFACE)
add_library(GPlib::GPlib ALIAS GPlib)

target_include_directories( GPlib INTERFACE
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            # ${CMAKE_CURRENT_LIST_DIR}
                            )

target_link_libraries(  GPlib INTERFACE
                        Eigen3::Eigen
                        )

## Preparing CMake files to be exported
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )

include(CMakePackageConfigHelpers)
write_basic_package_version_file(   "${PROJECT_NAME}ConfigVersion.cmake"
                                    VERSION ${PROJECT_VERSION}
                                    COMPATIBILITY SameMajorVersion
                                    )

configure_package_config_file(  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
                                "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                                INSTALL_DESTINATION
                                ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
                                )

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
        )
                        
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${PROJECT_BINARY_DIR}/gplib/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
                        
install(DIRECTORY ${PROJECT_SOURCE_DIR}/gplib DESTINATION include)

export(PACKAGE GPlib)