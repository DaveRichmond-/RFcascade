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

// #include <omp.h>
#include <vigra/convolution.hxx>
#include <imagetools.hxx>

using namespace vigra;

int main(int argc, char ** argv)
{

	// for now, manually define some useful constants
    int num_classes = 2; // could pass in as argument, or extract from array_labels;
    int num_levels = atoi(argv[3]);
    int num_samples_per_level = 2;

	// set up clock for timing
	std::clock_t start;
	float duration;
	
	// import images --------------------->
	
    /* CALL FUNCTION HERE THAT ACCEPTS "MODE" OF USING GT DATA, AND RETURNS:
     * (1) ARRAY OF FEATURE_ARRAYS,
     * (2) ARRAY OF LABEL_ARRAYS
     */
	std::string imgPath("C:\\data\\somites\\Features\\");
	std::string labelPath("C:\\data\\somites\\Labels\\");
	ArrayVector< MultiArray<2, float> > array_train_features;
	ArrayVector< MultiArray<2, UInt8> > array_train_labels;
	imagetools::getArrayOfFeaturesAndLabels(num_levels, imgPath, labelPath, array_train_features, array_train_labels);


    // set useful stuff
    Shape2 image_shape(atoi(argv[4]),atoi(argv[5]));                // return shape from initial call on directory.  all images will have the same shape...
    double smoothing_scale = 3.0;

    // import RF options ----------------------->

    // import feature_mix
    ArrayVector<int> feature_mix(3);
    feature_mix[0] = atoi(argv[7]);
    feature_mix[1] = atoi(argv[8]);
    feature_mix[2] = atoi(argv[9]);

    // set early stopping depth
    int depth = atoi(argv[11]);
    int min_split_node_size = atoi(argv[12]);
    EarlyStopDepthAndNodeSize stopping(depth, min_split_node_size);

    // calc some more useful constants ----------------------->

    int num_train_samples = array_train_features[0].size(0);
	int num_filt_features = array_train_features[0].size(1);
    int num_all_features = num_filt_features + 2*num_classes;

    std::cout << "num_train_samples: " << num_train_samples << std::endl;
    std::cout << "num_features: " << num_all_features << std::endl;

    // set up arrays of data ------------------------->

    // set up array of RFs --------------------->

    // learn cascade --------------------------------->

    //
    ArrayVector< RandomForest<float> > rf_cascade(num_levels);

    // tic
    std::clock_t start;
    float duration;
    start = std::clock();

    // run cascade
    for (int i=0; i<num_levels; ++i)
    {
        // some useful constants
        int num_samples = rfFeaturesArray[i].size(0);
        int num_filt_features = features1.size(2);                              // WILL NEED TO CHANGE
        int num_all_features = rfFeaturesArray[i].size(1);

        std::cout << "level " << i << std::endl;
        std::cout << "num_samples: " << num_samples << std::endl;
        if (i==0)
            std::cout << "num_features: " << num_filt_features << std::endl;
        else
            std::cout << "num_features: " << num_all_features << std::endl;

        // generate prob maps for ith forest (from previous (i-1) forests)
        for (int j=0; j<i; ++j)
        {
            // define probs to store output of predictProbabilities
            MultiArray<2, float> probs(Shape2(num_samples, num_classes));
            MultiArray<2, float> smoothProbs(Shape2(num_samples, num_classes));

            // generate new probability map
            if (j==0)
                rf_cascade[j].predictProbabilities(rfFeaturesArray[i].subarray(Shape2(0,0), Shape2(num_samples,num_filt_features)), probs);    // there are no probs from previous rf, therefore don't look into this part of feature array
            else
                rf_cascade[j].predictProbabilities(rfFeaturesArray[i], probs);

            ArrayVector<MultiArray<3, ImageType> > probArray(num_images_per_level);
            imagetools::probsToImages<ImageType>(probs, probArray, image_shape);

            // smooth the new probability maps
            ArrayVector<MultiArray<3, ImageType> > smoothProbArray(num_images_per_level);
            for (int k=0; k<num_images_per_level; ++k){
                for (int l=0; l<num_classes; ++l){
                    // smooth individual probability images (slices) by some method.  insert "model-based smoothing" here...
                    gaussianSmoothing(probArray[k].bind<3>(l), smoothProbArray[k].bind<3>(l), smoothing_scale);
                }
            }
            imagetools::imagesToProbs<ImageType>(smoothProbArray, smoothProbs);

            // update train_features with current probability map, and smoothed probability map
            rfFeaturesArray[i].subarray(Shape2(0,num_filt_features), Shape2(num_samples,num_filt_features+num_classes)) = probs;
            rfFeaturesArray[i].subarray(Shape2(0,num_filt_features+num_classes), Shape2(num_samples,num_all_features))  = smoothProbs;
        }

        // learn ith forest
        rf_cascade[i].set_options().image_shape(image_shape)
                             .tree_count(atoi(argv[6]))
                             .use_stratification(RF_EQUAL)
                             .max_offset_x(atoi(argv[10]))
                             .max_offset_y(atoi(argv[10]))
                             .feature_mix(feature_mix);

//        // some tests of the data
//        std::cout << "shape of rfFeaturesArray[i] is: " << rfFeaturesArray[i].shape() << std::endl;
//        int temp_count = 0;
//        for (int temp_idx = 0; temp_idx < num_samples*num_filt_features; ++temp_idx){
//            temp_count += rfFeaturesArray[i][temp_idx];
//        }
//        std::cout << "contents of rfFeaturesArray[i] are:" << temp_count << std::endl;

        if (i==0)
            rf_cascade[i].learn(rfFeaturesArray[i].subarray(Shape2(0,0), Shape2(num_samples,num_filt_features)), rfLabelsArray[i], rf_default(), rf_default(), stopping);
        else
            rf_cascade[i].learn(rfFeaturesArray[i], rfLabelsArray[i], rf_default(), rf_default(), stopping);

        // save RFs one at a time (for now)
        std::string rf_idx = static_cast<std::ostringstream*>( &(std::ostringstream() << i) )->str();
        HDF5File hdf5_file("rf" + rf_idx, HDF5File::New);
        rf_export_HDF5(rf_cascade[i], hdf5_file);

    }

    // toc
	duration = (std::clock() - start) / (float) CLOCKS_PER_SEC;
    std::cout << "time to learn cascade: " << duration << std::endl;
	
}
