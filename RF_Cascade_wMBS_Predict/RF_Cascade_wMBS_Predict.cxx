#include <iostream>
#include "libmodelBasedSmoothing2.h"

#include <array>
#include <chrono>
#include <algorithm>
#include <math.h>
#include <ctime>

#include <vigra/impex.hxx>
#include <vigra/multi_impex.hxx>
#include <vigra/multi_array.hxx>

#include <vigra/matrix.hxx>

#include <vigra/random_forest.hxx>
#include <vigra/random_forest/rf_earlystopping.hxx>
#include <vigra/random_forest_hdf5_impex.hxx>
#include <vigra/hdf5impex.hxx>

#include <vigra/convolution.hxx>

#include <imagetools.hxx>
#include <smoothingtools.hxx>

using namespace vigra;

int run_main(int argc, const char **argv)
{
    // USER DEFINED PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    std::string imgPath("/Users/richmond/Data/Somites/Processed/First set/registered/Features/simpleFeatures_wXY/Test");
    std::string labelPath("/Users/richmond/Data/Somites/Processed/First set/registered/Labels/Test");
    std::string outputPath("/Users/richmond/Analysis/SomiteTracker/RFs/real_data/on_registered_data/Cascade_w_Smoothing/MBS_rigid/2Levels_6ImagesPerLevel_20trees/1_0_0");

    // some user defined parameters
    double smoothing_scale = 3.0;
    int numFits = 100;
    int numCentroidsUsed = 21;

    // USER DEFINED PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    typedef float ImageType;
    typedef UInt8 LabelType;

    // import images --------------------->

    ArrayVector< MultiArray<2, float> > rfFeaturesArray;
    ArrayVector< MultiArray<2, UInt8> > rfLabelsArray;
    Shape2 xy_dim(0,0);

    int num_images = atoi(argv[4]);
    int sampling = atoi(argv[5]);

    imagetools::getArrayOfFeaturesAndLabels(imgPath, labelPath, rfFeaturesArray, rfLabelsArray, xy_dim, 1, num_images, sampling);

    int num_samples = rfFeaturesArray[0].size(0);
    num_images = num_samples / (xy_dim[0]*xy_dim[1]);
    int num_filt_features = rfFeaturesArray[0].size(1);

    std::cout << "\n" << "num test images: " << num_images << std::endl;
    std::cout << "num test samples: " << num_samples << std::endl;

    // Load RF --------------------------------->

    std::string rfName(argv[6]);

    std::cout << "rf loaded from: " << boost::filesystem::current_path() << std::endl;

    ArrayVector<RandomForest<float> > rf_cascade;
    HDF5File hdf5_file(rfName, HDF5File::Open);
    rf_import_HDF5(rf_cascade, hdf5_file);

    int num_classes = rf_cascade[0].class_count();

    // check import parameters
    std::cout << "\n" << "check rf parameters after load" << std::endl;
    std::cout << "tree count: " << rf_cascade[0].tree_count() << std::endl;
    std::cout << "class count: " << num_classes << std::endl;

    // initialize matlab compiler runtime --------------------------------->

    // Initialize the MATLAB Compiler Runtime global state
    if (!mclInitializeApplication(NULL,0))
    {
        std::cerr << "Could not initialize the application properly."
                  << std::endl;
        return -1;
    }

    // Initialize the modelBasedSmoothing library
    if( !libmodelBasedSmoothing2Initialize() )
    {
        std::cerr << "Could not initialize the library properly."
                  << std::endl;
        return -1;
    }

    std::cout << "initialization succeeded!" << std::endl;

    try
    {
        // Predict Labels and save -------------------->

        MultiArray<2, ImageType> rfFeatures_wProbs;

        // setup clock
        std::clock_t start;
        float duration;

        // run cascade
        for (int i=0; i<rf_cascade.size(); ++i)
        {

            std::cout << "\n" << "level: " << i << std::endl;

            // set image shape
            rf_cascade[i].set_options().image_shape(xy_dim);

            // set test scale
            rf_cascade[i].set_options().test_scale(sampling);
            std::cout << "test scale factor: " << rf_cascade[i].options().test_scale_ << std::endl;

            // setup rfFeatures_wProbs
            if (i==0)
            {
                rfFeatures_wProbs.reshape(Shape2(num_samples, num_filt_features + 2*num_classes));
                rfFeatures_wProbs.subarray(Shape2(0,0), Shape2(num_samples,num_filt_features)) = rfFeaturesArray[0];
            }

            // define probs to store output of predictProbabilities
            MultiArray<2, float> probs(Shape2(num_samples, num_classes));
            MultiArray<2, float> smoothProbs(Shape2(num_samples, num_classes));

            // tic
            start = std::clock();
            // generate new probability map
            if (i==0)
                rf_cascade[0].predictProbabilities(rfFeaturesArray[0], probs);
            else
                rf_cascade[i].predictProbabilities(rfFeatures_wProbs, probs);
            // toc
            duration = ((std::clock() - start) / (float) CLOCKS_PER_SEC) / 60.0;
            std::cout << "time to predict [min]: " << duration << std::endl;

            ArrayVector<MultiArray<3, ImageType> > probArray(num_images);
            imagetools::probsToImages<ImageType>(probs, probArray, xy_dim);

            // tic
            start = std::clock();
            // smooth the new probability maps
            ArrayVector<MultiArray<3, ImageType> > smoothProbArray(num_images);
            for (int k=0; k<num_images; ++k){
                smoothProbArray[k].reshape(Shape3(xy_dim[0], xy_dim[1], num_classes));
                smoothingtools::rigidMBS<ImageType>(probArray[k], smoothProbArray[k], numFits, numCentroidsUsed, sampling);
            }
            // toc
            duration = ((std::clock() - start) / (float) CLOCKS_PER_SEC) / 60.0;
            std::cout << "time to smooth [min]: " << duration << std::endl;

            /*
            // Gaussian Smoothing (old)
            ArrayVector<MultiArray<3, ImageType> > smoothProbArray(num_images);
            for (int k=0; k<num_images; ++k){
                smoothProbArray[k].reshape(Shape3(xy_dim[0], xy_dim[1], num_classes));
                for (int l=0; l<num_classes; ++l){
                    // smooth individual probability images (slices) by some method.  insert "model-based smoothing" here...
                    gaussianSmoothing(probArray[k].bind<2>(l), smoothProbArray[k].bind<2>(l), smoothing_scale);
                }
            }
            */

            imagetools::imagesToProbs<ImageType>(smoothProbArray, smoothProbs);

            // update train_features with current probability map, and smoothed probability map
            rfFeatures_wProbs.subarray(Shape2(0,num_filt_features), Shape2(num_samples,num_filt_features+num_classes)) = probs;
            rfFeatures_wProbs.subarray(Shape2(0,num_filt_features+num_classes), Shape2(num_samples,num_filt_features+2*num_classes)) = smoothProbs;

            // convert probs to labels
            MultiArray<2, UInt8> labels(Shape2(num_samples, 1));
            for (int k = 0; k < labels.size(0); ++k)
            {
                rf_cascade[i].ext_param_.to_classlabel(linalg::argMax(probs.subarray(Shape2(k,0), Shape2(k+1,num_classes))), labels[k]);
            }

            ArrayVector<MultiArray<3, LabelType> > labelArray(num_images);
            imagetools::probsToImages<LabelType>(labels, labelArray, xy_dim);

            std::string level_idx = std::to_string(i);
            for (int j=0; j<num_images; ++j)
            {
                std::string image_idx = std::to_string(j);
                std::string fname(outputPath + "/" + "image#" + image_idx + "_level#" + level_idx);
                VolumeExportInfo Export_info(fname.c_str(),".tif");
                exportVolume(labelArray[j], Export_info);
            }

            // save probability maps
            if ( 1 )
            {
                for (int img_indx=0; img_indx<num_images; ++img_indx)
                {
                    std::string fname(outputPath + "/" + "level#" + std::to_string(i) + "_image#" + std::to_string(img_indx) + "_probs");
                    VolumeExportInfo Export_info(fname.c_str(), ".tif");
                    exportVolume(probArray[img_indx], Export_info);

                    std::string fname2(outputPath + "/" + "level#" + std::to_string(i) + "_image#" + std::to_string(img_indx) + "_smoothProbs");
                    VolumeExportInfo Export_info2(fname2.c_str(), ".tif");
                    exportVolume(smoothProbArray[img_indx], Export_info2);
                }
            }
        }
    }

    catch (const mwException& e)
    {
        std::cerr << e.what() << std::endl;
        return -2;
    }
    catch (...)
    {
        std::cerr << "Unexpected error thrown" << std::endl;
        return -3;
    }

    // Shut down the library and the application global state.
    libmodelBasedSmoothing2Terminate();
    mclTerminateApplication();
    std::cout << "\n" << "libMBS terminated" << std::endl;

    return 0;
}

int main(int argc, const char **argv)
{
    mclmcrInitialize();
    return mclRunMain((mclMainFcnType)run_main, argc, argv);
}
