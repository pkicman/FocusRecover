cmake_minimum_required(VERSION 3.1)
project( FocusRecover )

set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

set( "OpenCV_DIR" "F:\\workspace\\libraries\\opencv_3_4\\opencv_mybuild\\install\\" )

find_package( OpenCV REQUIRED )

include_directories( include ${OpenCV_INCLUDE_DIRS} )

file( GLOB SOURCES "src/*.cpp" )
			
add_executable( FocusRecover ${SOURCES} )
target_link_libraries( FocusRecover ${OpenCV_LIBS} )