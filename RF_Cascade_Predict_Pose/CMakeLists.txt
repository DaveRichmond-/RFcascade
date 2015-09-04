cmake_minimum_required (VERSION 2.6)

project (RF_Cascade_Predict_Pose)

add_executable(RF_Cascade_Predict_Pose RF_Cascade_Predict_Pose.cxx)


# define includes ----------------->

#Very important: Link to header files in my_vigra project!
set(VIGRA_INCLUDE_DIR /Users/richmond/Repos/GIT/my_vigra/include)
message("Vigra include directories:" ${VIGRA_INCLUDE_DIR})

set(SUPPORTING_INCLUDE_DIR /Users/richmond/Repos/GIT/RFcascade/supporting_code)

# define HDF5 --------------------->

find_package(HDF5)
find_library(HDF5_HL_LIB hdf5_hl)
message("HDF5 libraries:" ${HDF5_LIBRARIES} ${HDF5_HL_LIB})

#set(HDF5_INCLUDE_DIR /Users/richmond/anaconda/pkgs/hdf5-1.8.15.1-1/include)
#set(HDF5_LIBRARIES /Users/richmond/anaconda/pkgs/hdf5-1.8.15.1-1/lib/libhdf5.dylib)
#set(HDF5_HL_LIB /Users/richmond/anaconda/pkgs/hdf5-1.8.15.1-1/lib/libhdf5_hl.dylib)

#set(HDF5_INCLUDE_DIR /opt/local/include)
#set(HDF5_LIBRARIES /opt/local/lib/libhdf5.dylib)
#set(HDF5_HL_LIB /opt/local/lib/libhdf5_hl.dylib)


# import includes and HDF5 -------------------->

target_include_directories(
RF_Cascade_Predict_Pose
PRIVATE
${VIGRA_INCLUDE_DIR}
${HDF5_INCLUDE_DIR}
${SUPPORTING_INCLUDE_DIR}
)
message("HDF5 include directories: " ${HDF5_INCLUDE_DIR})


# define vigra impex lib -------------------->

# important: need to use the newest library, for consistency
set(VIGRA_LIB /Users/richmond/Repos/GIT/my_vigra-build/src/impex/libvigraimpex.dylib)
message("Vigra libraries:" ${VIGRA_LIB})


#find_package(Boost)
#message("Is Boost found: " ${Boost_FOUND})
#message("Boost library: " ${Boost_LIBRARIES})

FILE(GLOB
Boost_LIBRARIES
/usr/local/Cellar/boost/1.57.0/lib/libboost_*)
#message("Boost libraries: " ${Boost_LIBRARIES})

# import vigra impex --------------------->

target_link_libraries(
RF_Cascade_Predict_Pose
${HDF5_LIBRARIES}
${HDF5_HL_LIB}
${VIGRA_LIB}
${Boost_LIBRARIES}
)
