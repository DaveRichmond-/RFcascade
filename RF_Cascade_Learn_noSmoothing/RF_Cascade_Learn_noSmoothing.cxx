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

using namespace vigra;

int main(int argc, char ** argv)
{

    typedef float ImageType;
    typedef UInt8 LabelType;

	// import images --------------------->
	
    std::string imgPath(argv[1]);
    std::string labelPath(argv[2]);

    int num_images = atoi(argv[3]);
    int num_levels = atoi(argv[4]);
    int sampling = atoi(argv[5]);

    // specify order to use data in cascade
    bool useAllImagesAtEveryLevel = (atoi(argv[14])>0);

    // build order in which to use images:
    // std::srand ( unsigned ( std::time(0) ) );
    std::vector<int> imgNumVector;
    if ( useAllImagesAtEveryLevel ) {
        for (int i=0; i<num_images; i++){
            imgNumVector.push_back(i); // 0 1 2 3 4 5 6 7 8 9 ...
        }
    } else {
        int numRotations=1;
        int numOrigImagesPerChunk=(num_images/num_levels)/numRotations;
        for (int l=0; l<num_levels; ++l) {
            for (int i=0; i<numOrigImagesPerChunk; i++){
                for (int r=0; r<numRotations; r++ ) {
                    imgNumVector.push_back((num_levels*i+l)*numRotations+r); // 0 1 2 3 4 5 6 7 8 9 ...
                }
            }
        }
        if (imgNumVector.size()<num_images) {
            for (int i=imgNumVector.size(); i<num_images; i++){
                imgNumVector.push_back(i);
            }
        }
    }
    // std::random_shuffle ( imgNumVector.begin(), imgNumVector.end() );
    std::cout << "image order: ";
    for(int x=0; x<imgNumVector.size(); x++) {
        std::cout << imgNumVector[x] << " ";
    }
    std::cout << "\n" << std::endl;

    // load features and labels
    ArrayVector< MultiArray<2, float> > rfFeaturesArray;
    ArrayVector< MultiArray<2, UInt8> > rfLabelsArray;
    Shape2 xy_dim(0,0);

    imagetools::getArrayOfFeaturesAndLabels(imgPath, labelPath, rfFeaturesArray, rfLabelsArray, xy_dim, imgNumVector, num_levels, sampling);

    // set up rf --------------------------------->

    ArrayVector< RandomForest<float> > rf_cascade;

    int num_classes = atoi(argv[6]);
    int tree_count = atoi(argv[7]);

    ArrayVector<int> feature_mix(3);
    feature_mix[0] = atoi(argv[8]);
    feature_mix[1] = atoi(argv[9]);
    feature_mix[2] = atoi(argv[10]);

    int max_offset = atoi(argv[11]) / sampling;        // account for resampling!
    std::cout << "\n" << "scaled max offset = " << max_offset << std::endl;

    // set early stopping depth
    int depth = atoi(argv[12]);
    int min_split_node_size = atoi(argv[13]);
    EarlyStopDepthAndNodeSize stopping(depth, min_split_node_size);

    // learn cascade --------------------------------->

    // set up clock
    std::clock_t start;
    float duration;

    // run cascade
    for (int i=0; i<num_levels; ++i)
    {
        // tic
        start = std::clock();

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
                rfFeatures_wProbs.reshape(Shape2(num_samples, num_filt_features + num_classes));
                rfFeatures_wProbs.subarray(Shape2(0,0), Shape2(num_samples,num_filt_features)) = rfFeaturesArray[i];
            }

            // define probs to store output of predictProbabilities
            MultiArray<2, float> probs(Shape2(num_samples, num_classes));

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

            // update train_features with current probability map, and smoothed probability map
            rfFeatures_wProbs.subarray(Shape2(0,num_filt_features), Shape2(num_samples,num_filt_features+num_classes)) = probs;

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

        // learn ith forest
        if (i==0)
            rf_cascade[i].learn(rfFeaturesArray[i], rfLabelsArray[i], rf_default(), rf_default(), stopping);
        else
            rf_cascade[i].learn(rfFeatures_wProbs, rfLabelsArray[i], rf_default(), rf_default(), stopping);

        // toc
        duration = ((std::clock() - start) / (float) CLOCKS_PER_SEC) / 60.0;
        std::cout << "time to learn level " << i << " [min]: " << duration << std::endl;

        // save RF cascade after each level (to be safe)
        HDF5File hdf5_file("rf_cascade", HDF5File::New);
        rf_export_HDF5(rf_cascade, hdf5_file);

    }

}
