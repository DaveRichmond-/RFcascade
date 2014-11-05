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


using namespace vigra;

int run_main(int argc, const char **argv)
{
    if (argc<23) {

        std::cout << "RF_Cascade_wMBS_Learn needs 22 arguments. Usage: " << std::endl;
        std::cout << "RF_Cascade_wMBS_Learn  <baseInputPath> <baseOutputPath> <rawPath> <featurePath> <labelPath> <randomForestPath> ";
        std::cout << "useExistingForest numImages numLevels reSampleBy numClasses numTrees featureMix_features featureMix_offsetFeatures featureMix_offsetDifferenceFeatures ";
        std::cout << "maxOffset treeDepth splitNodeSize howToSmoothProbMaps sampleFraction numAAMsteps useAllImagesAtEveryLevel" << std::endl;
        std::cout << "Aborting. " << std::endl;

        return 0;

    }
    // USER DEFINED PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    std::string baseInputPath = argv[1];
    std::string baseOutputPath = argv[2];

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
    rfName = rfPath + "/" + "rf_cascade";

    int loadRF_flag = atoi(argv[7]);

    // some user defined parameters
    double smoothing_scale = 3.0;
    int numGDsteps = atoi(argv[21]);
    float lambdaU = 4;
    float lambdaPW = 4;
    int numFits = 1;
    int numCentroidsUsed = 21;

    // USER DEFINED PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    typedef float ImageType;
    typedef UInt8 LabelType;

    // import images --------------------->
    int num_images = atoi(argv[8]);
    int num_levels = atoi(argv[9]);
    int sampling = atoi(argv[10]);

    bool useAllImagesAtEveryLevel = (atoi(argv[22])>0);

    // build random order in which to use images:
    std::srand ( unsigned ( std::time(0) ) );
    std::vector<int> imgNumVector;
    for (int i=0; i<num_images; ++i) imgNumVector.push_back(i); // 0 1 2 3 4 5 6 7 8 9 ...
    std::random_shuffle ( imgNumVector.begin(), imgNumVector.end() );

    ArrayVector< MultiArray<2, ImageType> > rfFeaturesArray;
    ArrayVector< MultiArray<2, LabelType> > rfLabelsArray;
    Shape2 xy_dim(0,0);

    ArrayVector< MultiArray<2, ImageType> > rfRawImageArray;
    Shape2 raw_dim(0,0);

    bool rfFeaturesArraySize = useAllImagesAtEveryLevel?1:num_levels;
    if (useAllImagesAtEveryLevel) {
        imagetools::getArrayOfFeaturesAndLabels(featPath, labelPath, rfFeaturesArray, rfLabelsArray, xy_dim, imgNumVector, rfFeaturesArraySize, sampling);
        // re-use above strategy to get grayscale images.  need some dummy variables.
        imagetools::getArrayOfRawImages(rawPath, rfRawImageArray, raw_dim, imgNumVector, rfFeaturesArraySize);
    }

    std::cout << "\n" << "image import succeeded!" << std::endl;

    // set up rf --------------------------------->

    ArrayVector< RandomForest<float> > rf_cascade;

    int num_classes = atoi(argv[11]);
    int tree_count = atoi(argv[12]);

    ArrayVector<int> feature_mix(3);
    feature_mix[0] = atoi(argv[13]);
    feature_mix[1] = atoi(argv[14]);
    feature_mix[2] = atoi(argv[15]);

    int max_offset = atoi(argv[16]) / sampling;        // account for resampling!
    std::cout << "\n" << "scaled max offset = " << max_offset << std::endl;

    // set early stopping depth
    int depth = atoi(argv[17]);
    int min_split_node_size = atoi(argv[18]);
    EarlyStopDepthAndNodeSize stopping(depth, min_split_node_size);

    // even more params...
    int smooth_flag = atoi(argv[19]);
    double sample_fraction = atof(argv[20]);

    std::cout << "sample fraction is: " << sample_fraction << std::endl;

    if (loadRF_flag)
    {
        std::cout << "rf loaded from: " << rfPath << std::endl;

        HDF5File hdf5_file(rfName, HDF5File::Open);
        rf_import_HDF5(rf_cascade, hdf5_file);

        for (int i = 0; i<rf_cascade.size(); ++i)
        {
            rf_cascade[i].set_options().image_shape(xy_dim);
        }

        // check import parameters
        std::cout << "\n" << "check rf parameters after load" << std::endl;
        std::cout << "tree count: "  << rf_cascade[0].tree_count()  << std::endl;
        std::cout << "class count: " << rf_cascade[0].class_count() << std::endl;
    }
    else
    {
        rf_cascade.resize(0);
    }

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
        for (int i=rf_cascade.size(); i<num_levels; ++i)
        {

            int featureArrayIdx = 0; //useAllImagesAtEveryLevel?0:i;

            // load images on demand:
            if (!useAllImagesAtEveryLevel) {
                // pick the right entries from randVector:
                int a=i*num_images/num_levels;
                int b=(i+1)*num_images/num_levels;

                std::vector<int> imgNumVectorAtLevel(&imgNumVector[a],&imgNumVector[b]);

                imagetools::getArrayOfFeaturesAndLabels(featPath, labelPath, rfFeaturesArray, rfLabelsArray, xy_dim, imgNumVectorAtLevel, rfFeaturesArraySize, sampling);
                // re-use above strategy to get grayscale images.  need some dummy variables.
                imagetools::getArrayOfRawImages(rawPath, rfRawImageArray, raw_dim, imgNumVectorAtLevel, rfFeaturesArraySize);
            }

            // some useful constants
            int num_samples = rfFeaturesArray[featureArrayIdx].size(0);
            int num_filt_features = rfFeaturesArray[featureArrayIdx].size(1);
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
                    rfFeatures_wProbs.subarray(Shape2(0,0), Shape2(num_samples,num_filt_features)) = rfFeaturesArray[featureArrayIdx];
                }

                // define probs to store output of predictProbabilities
                MultiArray<2, float> probs(Shape2(num_samples, num_classes));
                MultiArray<2, float> smoothProbs(Shape2(num_samples, num_classes));

                // set test scale
                rf_cascade[j].set_options().test_scale(sampling);
                std::cout << "test scale factor: " << rf_cascade[j].options().test_scale_ << std::endl;

                // generate new probability map
                if (j==0)
                    rf_cascade[j].predictProbabilities(rfFeaturesArray[featureArrayIdx], probs);
                else
                    rf_cascade[j].predictProbabilities(rfFeatures_wProbs, probs);

                // convert to array of stacks
                ArrayVector<MultiArray<3, ImageType> > probArray(num_images_per_level);
                imagetools::probsToImages<ImageType>(probs, probArray, xy_dim);

                // convert raw images to array of stacks
                ArrayVector<MultiArray<3, ImageType> > rawImageArray(num_images_per_level);
                imagetools::probsToImages<ImageType>(rfRawImageArray[featureArrayIdx], rawImageArray, raw_dim);

                // smooth the new probability maps
                ArrayVector<MultiArray<3, ImageType> > smoothProbArray(num_images_per_level);
                for (int k=0; k<num_images_per_level; ++k){
                    smoothProbArray[k].reshape(Shape3(xy_dim[0], xy_dim[1], num_classes));
//                    smoothingtools::rigidMBS<ImageType>(probArray[k], smoothProbArray[k], numFits, numCentroidsUsed, sampling);
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

                        if ( smooth_flag )
                        {
                            std::string fname2(outputPath + "/" + "level#" + std::to_string(j) + "_image#" + std::to_string(img_indx) + "_smoothProbs");
                            VolumeExportInfo Export_info2(fname2.c_str(), ".tif");
                            exportVolume(smoothProbArray[img_indx], Export_info2);
                        }
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
                    .use_stratification(RF_PROPORTIONAL)
                    .max_offset_x(max_offset)
                    .max_offset_y(max_offset)
                    .feature_mix(feature_mix)
                    .samples_per_tree(sample_fraction);

            std::cout << "training scale factor: " << rf_cascade[i].options().train_scale_ << std::endl;

            // tic
            start = std::clock();
            // learn ith forest
            if (i==0)
                rf_cascade[i].learn(rfFeaturesArray[featureArrayIdx], rfLabelsArray[featureArrayIdx], rf_default(), rf_default(), stopping);
            else
                rf_cascade[i].learn(rfFeatures_wProbs, rfLabelsArray[featureArrayIdx], rf_default(), rf_default(), stopping);
            // toc
            duration = ((std::clock() - start) / (float) CLOCKS_PER_SEC) / 60.0;
            std::cout << "time to learn level " << i << " [min]: " << duration << std::endl;

            // save RF cascade after each level (to be safe)
            HDF5File hdf5_file(rfName, HDF5File::New);
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
