cmake_minimum_required(VERSION 3.10)
project(aclnn_custom_ops VERSION 1.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN 路径
set(CANN_HOME "/usr/local/Ascend/ascend-toolkit/8.0.0")
set(CANN_TARGET_HOME "${CANN_HOME}/aarch64-linux")

# 包含目录
include_directories(
    ${CANN_HOME}/include
    ${CANN_HOME}/include/aclnn
    ${CANN_HOME}/include/aclnn/opdev
    ${CANN_HOME}/include/graph
    ${CANN_TARGET_HOME}/include
)

# 库目录
link_directories(
    ${CANN_HOME}/lib64
    ${CANN_HOME}/aarch64-linux/lib64
    ${CANN_HOME}/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64
)

# 查找依赖库
find_library(ACLNN_LIB aclnn PATHS ${CANN_HOME}/lib64 NO_DEFAULT_PATH)
find_library(ACL_LIB acl PATHS ${CANN_HOME}/lib64 NO_DEFAULT_PATH)
find_library(GRAPH_LIB graph PATHS ${CANN_HOME}/lib64 NO_DEFAULT_PATH)

# 源文件
set(SOURCES
    custom_add.cpp
)

# 创建库
add_library(aclnn_custom_ops SHARED ${SOURCES})

# 链接库
target_link_libraries(aclnn_custom_ops
    ${ACLNN_LIB}
    ${ACL_LIB}
    ${GRAPH_LIB}
)

# 设置输出目录
set_target_properties(aclnn_custom_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# 安装规则
install(TARGETS aclnn_custom_ops
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY ${CANN_HOME}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 打印配置信息
message(STATUS "CANN_HOME: ${CANN_HOME}")
message(STATUS "Target architecture: aarch64-linux")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
