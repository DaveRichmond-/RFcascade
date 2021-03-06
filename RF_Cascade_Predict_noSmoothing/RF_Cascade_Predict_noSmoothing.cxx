#include <iostream>
#include <array>
#include <chrono>
#include <algorithm>
#include <math.h>
#include <ctime>

#include <vigra/matrix.hxx>

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

    // import images --------------------->

    std::string imgPath(argv[1]);
    std::string labelPath(argv[2]);
    std::string outputPath(argv[3]);
    std::string rfPath = outputPath;

    ArrayVector< MultiArray<2, float> > rfFeaturesArray;
    ArrayVector< MultiArray<2, UInt8> > rfLabelsArray;
    Shape2 xy_dim(0,0);

    int num_images = atoi(argv[4]);
    int sampling = atoi(argv[5]);

    std::vector<int> imgNumVector;
    for (int i=0; i<num_images; ++i) imgNumVector.push_back(i); // 0 1 2 3 4 5 6 7 8 9

    int featDim = atoi(argv[7]);

    imagetools::getArrayOfFeaturesAndLabels(imgPath, labelPath, rfFeaturesArray, rfLabelsArray, xy_dim, imgNumVector, 1, sampling, featDim);

    int num_samples = rfFeaturesArray[0].size(0);
    num_images = num_samples / (xy_dim[0]*xy_dim[1]);
    int num_filt_features = rfFeaturesArray[0].size(1);

    std::cout << "\n" << "num test images: " << num_images << std::endl;
    std::cout << "num test samples: " << num_samples << std::endl;

    // Load RF --------------------------------->

    std::string rfName(argv[6]);
    rfName = rfPath + "/" + rfName;

    ArrayVector<RandomForest<float> > rf_cascade;
    HDF5File hdf5_file(rfName, HDF5File::Open);
    rf_import_HDF5(rf_cascade, hdf5_file);

    int num_classes = rf_cascade[0].class_count();

    // check import parameters
    std::cout << "\n" << "check rf parameters after load" << std::endl;
    std::cout << "tree count: " << rf_cascade[0].tree_count() << std::endl;
    std::cout << "class count: " << num_classes << std::endl;

    // Predict Labels and save -------------------->

    MultiArray<2, ImageType> rfFeatures_wProbs;

    // tic
    std::clock_t start;
    float duration;
    start = std::clock();

    // run cascade
    for (int i=0; i<rf_cascade.size(); ++i)
    {

        std::cout << "level: " << i << std::endl;

        // set image shape
        rf_cascade[i].set_options().image_shape(xy_dim);

        // set test scale
        rf_cascade[i].set_options().test_scale(sampling);
        std::cout << "test scale factor: " << rf_cascade[i].options().test_scale_ << std::endl;

        // setup rfFeatures_wProbs
        if (i==0)
        {
            rfFeatures_wProbs.reshape(Shape2(num_samples, num_filt_features + num_classes));
            rfFeatures_wProbs.subarray(Shape2(0,0), Shape2(num_samples,num_filt_features)) = rfFeaturesArray[0];
        }

        // define probs to store output of predictProbabilities
        MultiArray<2, float> probs(Shape2(num_samples, num_classes));

        // generate new probability map
        if (i==0)
            rf_cascade[0].predictProbabilities(rfFeaturesArray[0], probs);
        else
            rf_cascade[i].predictProbabilities(rfFeatures_wProbs, probs);

        ArrayVector<MultiArray<3, ImageType> > probArray(num_images);
        imagetools::probsToImages<ImageType>(probs, probArray, xy_dim);

        // update train_features with current probability map
        rfFeatures_wProbs.subarray(Shape2(0,num_filt_features), Shape2(num_samples,num_filt_features+num_classes)) = probs;

        // convert probs to labels
        MultiArray<2, UInt8> labels(Shape2(num_samples, 1));
        for (int k = 0; k < labels.size(0); ++k)
        {
            rf_cascade[i].ext_param_.to_classlabel(linalg::argMax(probs.subarray(Shape2(k,0), Shape2(k+1,num_classes))), labels[k]);
        }

        ArrayVector<MultiArray<3, LabelType> > labelArray(num_images);
        imagetools::probsToImages<LabelType>(labels, labelArray, xy_dim);

        std::string level_idx = static_cast<std::ostringstream*>( &(std::ostringstream() << i) )->str();
        for (int j=0; j<num_images; ++j)
        {
            std::string imageNameDummy;
            if (j<10) {
                imageNameDummy="image#0";
            } else {
                imageNameDummy="image#";
            }
            std::string image_idx = static_cast<std::ostringstream*>( &(std::ostringstream() << j) )->str();
            std::string fname(imageNameDummy + image_idx + "_level#" + level_idx);
            fname = outputPath + "/" + fname;
            VolumeExportInfo Export_info(fname.c_str(),".tif");
            exportVolume(labelArray[j], Export_info);
        }

        // save probability maps

        if ( 1 )
        {
            std::string level_idx = static_cast<std::ostringstream*>( &(std::ostringstream() << i) )->str();
            for (int j=0; j<num_images; ++j)
            {
                std::string image_idx = static_cast<std::ostringstream*>( &(std::ostringstream() << j) )->str();
                std::string fname("level#" + level_idx + "_" + image_idx + "_Probabilities");
                fname = outputPath + "/" + fname;
                VolumeExportInfo Export_info(fname.c_str(), ".tif");
                exportVolume(probArray[j], Export_info);
            }
        }

    }

    // toc
    duration = ((std::clock() - start) / (float) CLOCKS_PER_SEC) / 60.0;
    std::cout << "time to predict with cascade [min]: " << duration << std::endl;

}
