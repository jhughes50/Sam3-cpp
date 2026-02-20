cmake_minimum_required(VERSION 3.17)
project(sam3 VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Torch_DIR  "/opt/libtorch/share/cmake/Torch")
set(TORCH_ROOT "/opt/libtorch")

find_package(Torch   REQUIRED PATHS "${TORCH_ROOT}")
find_package(OpenCV  REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(glog    REQUIRED)

find_package(jsoncpp QUIET)
if(NOT jsoncpp_FOUND)
    find_path(JSONCPP_INCLUDE_DIR json/json.h PATH_SUFFIXES jsoncpp)
    find_library(JSONCPP_LIBRARY NAMES jsoncpp)
    if(NOT JSONCPP_INCLUDE_DIR OR NOT JSONCPP_LIBRARY)
        message(FATAL_ERROR "jsoncpp not found. Install libjsoncpp-dev.")
    endif()
    add_library(jsoncpp_lib UNKNOWN IMPORTED)
    set_target_properties(jsoncpp_lib PROPERTIES
        IMPORTED_LOCATION "${JSONCPP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${JSONCPP_INCLUDE_DIR}"
    )
    set(JSONCPP_TARGET jsoncpp_lib)
else()
    set(JSONCPP_TARGET jsoncpp_lib)
endif()

add_library(${PROJECT_NAME}
    src/tokenizer.cpp
    src/processor_mixins.cpp
    src/processor.cpp
    src/sam3_model.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${JSONCPP_TARGET}
    glog::glog
    yaml-cpp
)

add_executable(sam3_inference_test test/test_inference.cpp)
target_link_libraries(sam3_inference_test PRIVATE ${PROJECT_NAME})

add_executable(sam3_road_test test/test_road.cpp)
target_link_libraries(sam3_road_test PRIVATE ${PROJECT_NAME})

include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)
