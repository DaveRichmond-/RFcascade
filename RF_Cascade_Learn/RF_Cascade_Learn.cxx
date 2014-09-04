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
//#include <vigra/random_forest/rf_visitors.hxx>

#include <vigra/hdf5impex.hxx>
#include <vigra/random_forest_hdf5_impex.hxx>

#include <imagetools.hxx>

// #include <omp.h>

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

    ArrayVector< RandomForest<float> > array_rf(num_levels);

    // learn cascade --------------------------------->

    // define probs to store output of predictProbabilities
    MultiArray<2, float> probs(Shape2(num_train_samples, num_classes));
    MultiArray<2, float> smooth_probs(Shape2(num_train_samples, num_classes));

    // tic
    start = std::clock();

    // run cascade
    for (int i=0; i<num_levels; ++i)
    {
        // generate prob maps for ith forest (from previous (i-1) forests)
        for (int j=0; j<i; ++j)
        {
            // generate new probability map
            if (j==0)
                (array_rf[j]).predictProbabilities((array_train_features[i]).subarray(Shape2(0,0), Shape2(num_train_samples,num_filt_features)), probs);    // there are no probs from previous rf, therefore don't look into this part of feature array
            else
                (array_rf[j]).predictProbabilities(array_train_features[i], probs);

            // smooth the new probability map
            smooth_probs = probs;       // FOR NOW, NO SMOOTHING

            // update train_features with current probability map, and smoothed probability map
            (array_train_features[i]).subarray(Shape2(0,num_filt_features), Shape2(num_train_samples,num_filt_features+num_classes)) = probs;
            (array_train_features[i]).subarray(Shape2(0,num_filt_features+num_classes), Shape2(num_train_samples,num_all_features)) = smooth_probs;
        }

        // learn ith forest
        (array_rf[i]).set_options().image_shape(Shape2(atoi(argv[4]),atoi(argv[5])))
                             .tree_count(atoi(argv[6]))
                             .use_stratification(RF_EQUAL)
                             .max_offset_x(atoi(argv[10]))
                             .max_offset_y(atoi(argv[10]))
                             .feature_mix(feature_mix);

        if (i==0)
            (array_rf[i]).learn((array_train_features[i]).subarray(Shape2(0,0), Shape2(num_train_samples,num_filt_features)), array_train_labels[i], rf_default(), rf_default(), stopping);
        else
            (array_rf[i]).learn(array_train_features[i], array_train_labels[i], rf_default(), rf_default(), stopping);
    }

    // toc
	duration = (std::clock() - start) / (float) CLOCKS_PER_SEC;
    std::cout << "time to learn cascade: " << duration << std::endl;
	
//    // SAVE LEARNED RFs

//    HDF5File hdf5_file(argv[3], HDF5File::New);
//    rf_export_HDF5(rf, hdf5_file);

}
