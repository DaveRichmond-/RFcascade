cmake_minimum_required (VERSION 2.6)

project (RF_Cascade_Learn)

add_executable(RF_Cascade_Learn RF_Cascade_Learn.cxx)


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
RF_Cascade_Learn
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
RF_Cascade_Learn
${HDF5_LIBRARIES}
${HDF5_HL_LIB}
${VIGRA_LIB}
)
