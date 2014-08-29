cmake_minimum_required (VERSION 2.6)

project (RFcascade_supporting_code)

FILE(
GLOB
RFcascade_supporting_code_Headers
/Users/richmond/Repos/GIT/RFcascade_supporting_code/null.cxx
/Users/richmond/Repos/GIT/RFcascade_supporting_code/*.hxx)
add_library(RFcascade_supporting_code SHARED ${RFcascade_supporting_code_Headers})


# define includes ----------------->

#Very important: Link to header files in my_vigra project!
set(VIGRA_INCLUDE_DIR /Users/richmond/Repos/GIT/my_vigra/include)
message("Vigra include directories:" ${VIGRA_INCLUDE_DIR})


# define HDF5 --------------------->

find_package(HDF5)
find_library(HDF5_HL_LIB hdf5_hl)
message("HDF5 libraries:" ${HDF5_LIBRARIES} ${HDF5_HL_LIB})


# import includes and HDF5 -------------------->

target_include_directories(
RFcascade_supporting_code
PRIVATE
${VIGRA_INCLUDE_DIR}
${HDF5_INCLUDE_DIR}
)
message("HDF5 include directories: " ${HDF5_INCLUDE_DIR})


# define vigra impex lib -------------------->

# important: need to use the newest library, for consistency
set(VIGRA_LIB /Users/richmond/Repos/GIT/my_vigra-build/src/impex/libvigraimpex.dylib)
message("Vigra libraries:" ${VIGRA_LIB})


# import vigra impex --------------------->

target_link_libraries(
RFcascade_supporting_code
${HDF5_LIBRARIES}
${HDF5_HL_LIB}
${VIGRA_LIB}
)
