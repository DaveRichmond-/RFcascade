#include <iostream>
#include <array>
#include <chrono>
#include <algorithm>
#include <math.h>
#include <ctime>

#include <vigra/impex.hxx>
#include <vigra/multi_impex.hxx>
#include <vigra/multi_array.hxx>

#include <vigra/hdf5impex.hxx>
#include <vigra/random_forest_hdf5_impex.hxx>

#include <vigra/random_forest.hxx>
#include <vigra/random_forest/rf_earlystopping.hxx>

#include <vigra/convolution.hxx>
#include <imagetools.hxx>


using namespace vigra;

int main(int argc, char ** argv)
{

    typedef float ImageType;
    typedef UInt8 LabelType;

    // load test data and set up useful constants --------------------->


    // Load RF --------------------------------->

    RandomForest<float> rf;
    HDF5File hdf5_file(argv[4], HDF5File::Open);
    rf_import_HDF5(rf, hdf5_file);

    // set image shape
    rf.set_options().image_shape(Shape2(im_width, im_height));

    // check import parameters
    std::cout << "min split node size: " << rf.options().min_split_node_size_ << std::endl;
    std::cout << "tree count: " << rf.options().tree_count_ << std::endl;
    std::cout << "image shape: " << rf.options().image_shape_ << std::endl;

    /* SPECIFY INPUT DATA / VARIABLES, THAT ARE NEEDED FOR THE REST OF THE PROGRAM

    1) ArrayVector of RFs
    2) a single array of rfFeatures.  Can do prediction on all input at the same time.
    3) xy-dimensions of one image (as image_shape)
    4) number of images that were input
    5) num_filt_features

    */

    // Predict Labels and save -------------------->

    // // array for labels
    MultiArray<2, UInt8> test_labels(Shape2(num_test_samples, 1));
    rf.predictLabels(test_features, test_labels);




    // tic
    std::clock_t start;
    float duration;
    start = std::clock();

    // run cascade
    for (int i=0; i<num_levels; ++i)
    {
        // some useful constants
        int num_samples = rfFeatures.size(0);
//        int num_filt_features = features1.size(2);                              // WILL NEED TO CHANGE
        int num_all_features = rfFeatures.size(1);

        // define probs to store output of predictProbabilities
        MultiArray<2, float> probs(Shape2(num_samples, num_classes));
        MultiArray<2, float> smoothProbs(Shape2(num_samples, num_classes));

        // generate new probability map
        if (j==0)
            rf_cascade[i].predictProbabilities(rfFeatures.subarray(Shape2(0,0), Shape2(num_samples,num_filt_features)), probs);    // there are no probs from previous rf, therefore don't look into this part of feature array
        else
            rf_cascade[i].predictProbabilities(rfFeatures, probs);

        ArrayVector<MultiArray<3, ImageType> > probArray(num_images);
        imagetools::probsToImages<ImageType>(probs, probArray, image_shape);

        // smooth the new probability maps
        ArrayVector<MultiArray<3, ImageType> > smoothProbArray(num_images);
        for (int k=0; k<num_images; ++k){
            for (int l=0; l<num_classes; ++l){
                // smooth individual probability images (slices) by some method.  insert "model-based smoothing" here...
                gaussianSmoothing(probArray[k].bind<3>(l), smoothProbArray[k].bind<3>(l), smoothing_scale);
            }
        }
        imagetools::imagesToProbs<ImageType>(smoothProbArray, smoothProbs);

        // update train_features with current probability map, and smoothed probability map
        rfFeatures.subarray(Shape2(0,num_filt_features), Shape2(num_samples,num_filt_features+num_classes)) = probs;
        rfFeatures.subarray(Shape2(0,num_filt_features+num_classes), Shape2(num_samples,num_all_features))  = smoothProbs;

        // save output
        std::string level_idx = static_cast<std::ostringstream*>( &(std::ostringstream() << i) )->str();
        for (int j=0; j<num_images; ++j)
        {
            std::string image_idx = static_cast<std::ostringstream*>( &(std::ostringstream() << j) )->str();
            VolumeExportInfo Export_info("level#" + level_idx + "_image#" + image_idx + "_probabilities", ".tif");
            exportVolume(probArray[j], Export_info);
        }
    }

    // toc
    duration = (std::clock() - start) / (float) CLOCKS_PER_SEC;
    std::cout << "time to learn cascade: " << duration << std::endl;

}
