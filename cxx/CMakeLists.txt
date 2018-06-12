cmake_minimum_required (VERSION 2.6)
project (tf_vnet_inference)

# Tensorflow
find_package(Tensorflow REQUIRED)
include_directories(${TENSORFLOW_INCLUDE_DIRS})

# Protobuf
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# compiler setting required by tensorflow, to be tested on all compilers
# currently only tested on MSVC and GCC
if (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC) 
  add_definitions(-DCOMPILER_MSVC)
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
  if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "3")
    add_definitions(-DCOMPILER_GCC3)
  else()
    add_definitions(-D__GNUC__)
  endif()
else()
  message(ERROR " compiler ${CMAKE_CXX_COMPILER_ID} not supported by this CMakeList.txt, under development")
endif()

file(GLOB SRC *.c*)
file(GLOB HDR *.h)

add_executable(tf_vnet_inference ${SRC} ${HDR})
target_link_libraries(tf_vnet_inference ${TENSORFLOW_LIBRARIES} ${ITK_LIBRARIES} ${PROTOBUF_LIBRARIES})