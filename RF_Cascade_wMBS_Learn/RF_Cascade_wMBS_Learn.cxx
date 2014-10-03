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

    std::string imgPath("/Users/richmond/Data/Somites/Processed/First set/registered/Features/simpleFeatures_wXY/Train");
    std::string labelPath("/Users/richmond/Data/Somites/Processed/First set/registered/Labels/Train");
    std::string outputPath("/Users/richmond/Analysis/SomiteTracker/RFs/real_data/on_registered_data/Cascade_w_Smoothing/MBS_rigid/test");

    // some user defined parameters
    double smoothing_scale = 3.0;
    int numFits = 1;
    int numCentroidsUsed = 21;

    // USER DEFINED PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    typedef float ImageType;
    typedef UInt8 LabelType;

    // import images --------------------->
    int num_images = atoi(argv[4]);
    int num_levels = atoi(argv[5]);
    int sampling = atoi(argv[6]);

    ArrayVector< MultiArray<2, float> > rfFeaturesArray;
    ArrayVector< MultiArray<2, UInt8> > rfLabelsArray;
    Shape2 xy_dim(0,0);

    imagetools::getArrayOfFeaturesAndLabels(imgPath, labelPath, rfFeaturesArray, rfLabelsArray, xy_dim, num_levels, num_images, sampling);

    std::cout << "\n" << "image import succeeded!" << std::endl;

    // set up rf --------------------------------->

    ArrayVector< RandomForest<float> > rf_cascade;

    int num_classes = atoi(argv[7]);
    int tree_count = atoi(argv[8]);

    ArrayVector<int> feature_mix(3);
    feature_mix[0] = atoi(argv[9]);
    feature_mix[1] = atoi(argv[10]);
    feature_mix[2] = atoi(argv[11]);

    int max_offset = atoi(argv[12]) / sampling;        // account for resampling!
    std::cout << "\n" << "scaled max offset = " << max_offset << std::endl;

    // set early stopping depth
    int depth = atoi(argv[13]);
    int min_split_node_size = atoi(argv[14]);
    EarlyStopDepthAndNodeSize stopping(depth, min_split_node_size);

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

        // learn cascade --------------------------------->

        // set up clock
        std::clock_t start;
        float duration;

        // run cascade
        for (int i=0; i<num_levels; ++i)
        {

            // some useful constants
            int num_samples = rfFeaturesArray[i].size(0);
            int num_filt_features = rfFeaturesArray[i].size(1);
            int num_images_per_level = num_samples / (xy_dim[0]*xy_dim[1]);

            std::cout << "\n" << "level: " << i << std::endl;
            std::cout << "num images used: " << num_images_per_level << std::endl;
            std::cout << "num_samples: " << num_samples << std::endl;
            std::cout << "num_filt_features: " << num_filt_features << std::endl;

            MultiArray<2, ImageType> rfFeatures_wProbs;

            // generate prob maps for ith forest (from previous (i-1) forests)
            for (int j=0; j<i; ++j)
            {
                // setup rfFeatures_wProbs
                if (j==0)
                {
                    rfFeatures_wProbs.reshape(Shape2(num_samples, num_filt_features + 2*num_classes));
                    rfFeatures_wProbs.subarray(Shape2(0,0), Shape2(num_samples,num_filt_features)) = rfFeaturesArray[i];
                }

                // define probs to store output of predictProbabilities
                MultiArray<2, float> probs(Shape2(num_samples, num_classes));
                MultiArray<2, float> smoothProbs(Shape2(num_samples, num_classes));

                // set test scale
                rf_cascade[j].set_options().test_scale(sampling);
                std::cout << "test scale factor: " << rf_cascade[j].options().test_scale_ << std::endl;

                // generate new probability map
                if (j==0)
                    rf_cascade[j].predictProbabilities(rfFeaturesArray[i], probs);
                else
                    rf_cascade[j].predictProbabilities(rfFeatures_wProbs, probs);

                ArrayVector<MultiArray<3, ImageType> > probArray(num_images_per_level);
                imagetools::probsToImages<ImageType>(probs, probArray, xy_dim);

                // smooth the new probability maps
                ArrayVector<MultiArray<3, ImageType> > smoothProbArray(num_images_per_level);
                for (int k=0; k<num_images_per_level; ++k){
                    smoothProbArray[k].reshape(Shape3(xy_dim[0], xy_dim[1], num_classes));
                    smoothingtools::rigidMBS<ImageType>(probArray[k], smoothProbArray[k], numFits, numCentroidsUsed, sampling);
                }

                // Gaussian Smoothing (old)
                /*
                ArrayVector<MultiArray<3, ImageType> > smoothProbArray(num_images_per_level);
                for (int k=0; k<num_images_per_level; ++k){
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

                // save probability maps
                if ( 1 ) //(rf_cascade.size()-1) )
                {
                    for (int img_indx=0; img_indx<num_images_per_level; ++img_indx)
                    {
                        std::string fname(outputPath + "/" + "level#" + std::to_string(j) + "_image#" + std::to_string(img_indx) + "_probs");
                        VolumeExportInfo Export_info(fname.c_str(), ".tif");
                        exportVolume(probArray[img_indx], Export_info);

                        std::string fname2(outputPath + "/" + "level#" + std::to_string(j) + "_image#" + std::to_string(img_indx) + "_smoothProbs");
                        VolumeExportInfo Export_info2(fname2.c_str(), ".tif");
                        exportVolume(smoothProbArray[img_indx], Export_info2);
                    }
                }
            }

            // create space for next forest in the cascade
            RandomForest<float> rf;
            rf_cascade.push_back(rf);

            // set options for new forest
            rf_cascade[i].set_options()
                    .train_scale(sampling)
                    .image_shape(xy_dim)
                    .tree_count(tree_count)
                    .use_stratification(RF_EQUAL)
                    .max_offset_x(max_offset)
                    .max_offset_y(max_offset)
                    .feature_mix(feature_mix);

            std::cout << "training scale factor: " << rf_cascade[i].options().train_scale_ << std::endl;

            // tic
            start = std::clock();
            // learn ith forest
            if (i==0)
                rf_cascade[i].learn(rfFeaturesArray[i], rfLabelsArray[i], rf_default(), rf_default(), stopping);
            else
                rf_cascade[i].learn(rfFeatures_wProbs, rfLabelsArray[i], rf_default(), rf_default(), stopping);
            // toc
            duration = ((std::clock() - start) / (float) CLOCKS_PER_SEC) / 60.0;
            std::cout << "time to learn level " << i << " [min]: " << duration << std::endl;

            // save RF cascade after each level (to be safe)
            HDF5File hdf5_file(outputPath + "/" + "rf_cascade", HDF5File::New);
            rf_export_HDF5(rf_cascade, hdf5_file);

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
