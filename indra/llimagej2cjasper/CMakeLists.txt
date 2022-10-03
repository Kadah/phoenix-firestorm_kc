# -*- cmake -*-

project(llimagej2jasper)

include(00-Common)
include(LLCommon)
include(LLImage)
include(JasPer)

add_library (llimagej2cjasper
  llimagej2cjasper.cpp
  CMakeLists.txt )

target_include_directories( llimagej2cjasper PRIVATE
  ${LLIMAGE_INCLUDE_DIRS}
  ${LLCOMMON_INCLUDE_DIRS} )

target_link_libraries( llimagej2cjasper FV::jasper)

