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

    // save first level only as a cascade
    rf_cascade.resize(1);

    HDF5File hdf5_file2(rfName2, HDF5File::New);
    rf_export_HDF5(rf_cascade, hdf5_file2);



    /*
    typedef double  FactorType;
    typedef double  MarginalType;
    typedef int     MAPType;

    // load test data
    VolumeImportInfo fits_info(argv[1]);
    MultiArray<3, int> fits(Shape3(fits_info.shape()));
    importVolume(fits_info, fits);

    // set up
    int numVars = 2;
    int numLabels = 2;

    MultiArray<2, FactorType> unaryFactors(Shape2(numVars, numLabels));
    MultiArray<3, FactorType> pairwiseFactors(Shape3(numLabels, numLabels, numVars-1));

    unaryFactors.init(1.0);
    pairwiseFactors.init(1.0);

//    unaryFactors(0,0) = 2.0;
//    unaryFactors(1,0)

    MultiArray<2, MarginalType> marginals(Shape2(numVars, numLabels));
    MultiArray<1, MAPType> MAPChain;

    // probabilistic inference
    inferencetools::chainProbInference<FactorType, MarginalType>(unaryFactors, pairwiseFactors, marginals);

    // output
    for(int i = 0; i < numVars; ++i)
    {
        std::cout<< "Variable " << i << " has the following marginal distribution P(x_" << i << ") : ";
        for(int j = 0; j < numLabels; ++j)
            std::cout <<marginals(i,j) << " ";        
        std::cout<<std::endl;
    }

    // MAP inference
    inferencetools::chainMAPInference<FactorType, MAPType>(unaryFactors, pairwiseFactors, MAPChain);

    // output
    for(int i = 0; i < numVars; ++i)
        std::cout<< "Variable " << i << " has the following MAP label : " << MAPChain(i) << std::endl;

    // apply to images
    MultiArray<3,float> probs;
    smoothingtools::probFromFits<FactorType, float>(unaryFactors, pairwiseFactors, fits, probs);

    // output result
    VolumeExportInfo Export_info("/Users/richmond/Desktop/tests/smooth/smoothProbs",".tif");
    exportVolume(probs, Export_info);

    // apply to images
    MultiArray<2,int> MAPLabels;
    smoothingtools::MAPFromFits<FactorType>(unaryFactors, pairwiseFactors, fits, MAPLabels);

    //export labels
    exportImage(MAPLabels, "/Users/richmond/Desktop/tests/MAP/MAPLabels.tif");
    */
}
