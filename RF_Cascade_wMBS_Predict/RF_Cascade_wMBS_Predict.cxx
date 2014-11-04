#include <iostream>

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

    std::string baseInputPath = argv[1];
    std::string baseOutputPath = argv[2];
    //
    std::string rawPath;
    rawPath = baseInputPath + argv[3];
    std::string featPath;
    featPath = baseInputPath + argv[4];
    std::string labelPath;
    labelPath = baseInputPath + argv[5];
    std::string rfPath;
    rfPath = baseOutputPath + argv[6];
    std::string outputPath;
    outputPath = rfPath;

    std::string rfName;
    rfName = rfPath + "/" + argv[7];

    // some user defined parameters
    double smoothing_scale = 3.0;
    int numGDsteps = 2;
    float lambdaU = 2;
    float lambdaPW = 1;
    int numFits = 1;
    int numCentroidsUsed = 21;

    // END USER DEFINED PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    typedef float ImageType;
    typedef UInt8 LabelType;

    // import images --------------------->

    int num_images = atoi(argv[8]);
    int sampling = atoi(argv[9]);

    int smooth_flag = atoi(argv[10]);

    ArrayVector< MultiArray<2, float> > rfFeaturesArray;
    ArrayVector< MultiArray<2, UInt8> > rfLabelsArray;
    Shape2 xy_dim(0,0);

    imagetools::getArrayOfFeaturesAndLabels(featPath, labelPath, rfFeaturesArray, rfLabelsArray, xy_dim, 1, num_images, sampling);

    int num_samples = rfFeaturesArray[0].size(0);
    num_images = num_samples / (xy_dim[0]*xy_dim[1]);
    int num_filt_features = rfFeaturesArray[0].size(1);

    // re-use above strategy to get grayscale images.  need some dummy variables.
    ArrayVector< MultiArray<2, ImageType> > rfRawImageArray;
    Shape2 raw_dim(0,0);

    imagetools::getArrayOfRawImages(rawPath, rfRawImageArray, raw_dim, 1, num_images);

    std::cout << "\n" << "image import succeeded!" << std::endl;

    std::cout << "\n" << "num test images: " << num_images << std::endl;
    std::cout << "num test samples: " << num_samples << std::endl;

    // Load RF --------------------------------->

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

            std::cout << "\n" << "check forest properties at each level..." << std::endl;
            std::cout << "tree count: " << rf_cascade[i].tree_count() << std::endl;
            std::cout << "class count: " << rf_cascade[i].class_count() << std::endl;
            std::cout << "feature count: " << rf_cascade[i].feature_count() << std::endl;


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

            // convert raw images to array of stacks
            ArrayVector<MultiArray<3, ImageType> > rawImageArray(num_images);
            imagetools::probsToImages<ImageType>(rfRawImageArray[0], rawImageArray, raw_dim);

            // tic
            start = std::clock();
            // smooth the new probability maps
            ArrayVector<MultiArray<3, ImageType> > smoothProbArray(num_images);
            for (int k=0; k<num_images; ++k){
                smoothProbArray[k].reshape(Shape3(xy_dim[0], xy_dim[1], num_classes));
//              smoothingtools::rigidMBS<ImageType>(probArray[k], smoothProbArray[k], numFits, numCentroidsUsed, sampling);
                if ( smooth_flag == 0 )
                    smoothProbArray[k].init(0.0);
                else if ( smooth_flag  == 1 )
                    smoothingtools::AAM_MBS<ImageType>(probArray[k], rawImageArray[k], smoothProbArray[k], sampling, numGDsteps, lambdaU);
                else if ( smooth_flag == 2 ){
                    MultiArray<2, int> MAPLabels;     // for now, just throw away the MAPLabels
                    smoothingtools::AAM_Inference<ImageType>(probArray[k], rawImageArray[k], smoothProbArray[k], MAPLabels, sampling, numGDsteps, lambdaU, lambdaPW);
                } else if ( smooth_flag == 3 ){
                    MultiArray<2, int> MAPLabels;     // for now, just throw away the MAPLabels
                    smoothingtools::AAM_Inference_2inits<ImageType>(probArray[k], rawImageArray[k], smoothProbArray[k], MAPLabels, sampling, numGDsteps, lambdaU, lambdaPW);
                }
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

            // partition labels into array of label images
            ArrayVector<MultiArray<3, LabelType> > labelArray(num_images);
            imagetools::probsToImages<LabelType>(labels, labelArray, xy_dim);

            // repeat for gt labels
//            ArrayVector<MultiArray<3, LabelType> > gtLabelArray(num_images);
//            imagetools::probsToImages<LabelType>(rfLabelsArray[0], gtLabelArray, xy_dim);

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

                    if ( smooth_flag )
                    {
                        std::string fname2(outputPath + "/" + "level#" + std::to_string(i) + "_image#" + std::to_string(img_indx) + "_smoothProbs");
                        VolumeExportInfo Export_info2(fname2.c_str(), ".tif");
                        exportVolume(smoothProbArray[img_indx], Export_info2);
                    }
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
