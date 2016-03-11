cmake_minimum_required (VERSION 3.2)

set (PROJ_NAME "Symbol_Recognition")

project (${PROJ_NAME})

set(CMAKE_AUTOMOC ON)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Qml REQUIRED)
find_package(Qt5Quick REQUIRED)

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

set (QML
	resources/main.qml
	)

qt5_add_resources(RESOURCES ${CMAKE_CURRENT_SOURCE_DIR}/resources/recource.qrc)

set (ALL_FILES
	 ${HEADERS}
	 ${SOURCES}
	 ${RESOURCES}
	 ${QML}
	 )

add_executable (${PROJ_NAME} ${ALL_FILES})

set_source_files_properties(${RESOURCES} PROPERTIES GENERATED TRUE)

source_group (include FILES ${HEADERS})
source_group (source FILES ${SOURCES})
source_group (resource FILES ${RESOURCES})
source_group (qml FILES ${QML})

target_link_libraries(${PROJ_NAME} Qt5::Widgets Qt5::Qml Qt5::Quick)

if (WIN32)
	# this property is used to remove the prompt window when running the GUI from the explorer on WIN32
	# doesn't have effect on linux
	#set_target_properties(${PROJ_NAME} PROPERTIES WIN32_EXECUTABLE 1)
endif()

