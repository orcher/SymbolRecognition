cmake_minimum_required (VERSION 3.2)

set (PROJ_NAME "Symbol_Recognition")

project (${PROJ_NAME})

include_directories(
	${CMAKE_CURRENT_BINARY_DIR}
	include/
	)

set (HEADERS
	include/neuralnetwork.hpp
	)

set (SOURCES
	source/main.cpp
	source/neuralnetwork.cpp
	)

set (RESOURCES
	)

set (ALL_FILES
	 ${HEADERS}
	 ${SOURCES}
	 ${RESOURCES}
	 )

source_group (include FILES ${HEADERS})
source_group (source FILES ${SOURCES})
source_group (resource FILES ${RESOURCES})

add_executable (${PROJ_NAME} ${ALL_FILES})