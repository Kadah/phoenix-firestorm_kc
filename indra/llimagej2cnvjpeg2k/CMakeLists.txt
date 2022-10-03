# -*- cmake -*-

project(llimagej2cnvjpeg2k LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include(00-Common)
include(LLCommon)
include(LLFileSystem)
include(LLImage)
include(nvJPEG2k)

include_directories( llimagej2cnvjpeg2k
	${LLIMAGE_INCLUDE_DIRS}
	${LLCOMMON_INCLUDE_DIRS}
	SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${NVJPEG2K_INCLUDE_DIR}
)

set(llimagej2cnvjpeg2k_SOURCE_FILES
	llimagej2cnvjpeg2k.cpp
)

set(llimagej2cnvjpeg2k_HEADER_FILES
	CMakeLists.txt
	llimagej2cnvjpeg2k.h
)

set_source_files_properties(${llimagej2cnvjpeg2k_HEADER_FILES}
	PROPERTIES HEADER_FILE_ONLY TRUE)

list(APPEND llimagej2cnvjpeg2k_SOURCE_FILES ${llimagej2cnvjpeg2k_HEADER_FILES})

add_library (llimagej2cnvjpeg2k ${llimagej2cnvjpeg2k_SOURCE_FILES})


find_library(CUDART_LIB 
	NAMES cudart
	PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
)
	
target_link_libraries(
	llimagej2cnvjpeg2k
	${NVJPEG2K_LIBRARIES}
	${CUDART_LIB}
)
