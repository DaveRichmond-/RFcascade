#include <iostream>

#include <array>
#include <chrono>
#include <algorithm>
#include <math.h>
#include <ctime>

#include <vigra/impex.hxx>
#include <vigra/multi_impex.hxx>
#include <vigra/multi_array.hxx>

#include <vigra/random_forest.hxx>
#include <vigra/random_forest/rf_earlystopping.hxx>
#include <vigra/random_forest_hdf5_impex.hxx>
#include <vigra/hdf5impex.hxx>

#include <vigra/convolution.hxx>

#include <imagetools.hxx>
#include <smoothingtools.hxx>
#include <inferencetools.hxx>


using namespace vigra;

int main(int argc, const char **argv)
{

//    std::string baseOutputPath("/Users/richmond/Analysis/SomiteTracker/RFs/real_data/on_registered_data/Cascade_w_Smoothing");
    std::string rfPath;
    rfPath = argv[2];

    std::string rfName;
    rfName = rfPath + "/" + "rf_cascade";

    std::string rfName2;
    rfName2 = rfPath + "/" + "rf_cascade_Level0";

    std::cout << "rf loaded from: " << rfPath << std::endl;

    ArrayVector< RandomForest<float> > rf_cascade;

    HDF5File hdf5_file(rfName, HDF5File::Open);
    rf_import_HDF5(rf_cascade, hdf5_file);

    // check import parameters
    std::cout << "\n" << "check rf parameters after load" << std::endl;
    std::cout << "tree count: "  << rf_cascade[0].tree_count()  << std::endl;
    std::cout << "class count: " << rf_cascade[0].class_count() << std::endl;
    std::cout << "number of levels: " << rf_cascade.size() << std::endl;

    // save first level only as a cascade
    rf_cascade.resize(1);

    HDF5File hdf5_file2(rfName2, HDF5File::New);
    rf_export_HDF5(rf_cascade, hdf5_file2);

}
