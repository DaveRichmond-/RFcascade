#include <iostream>
#include <array>
#include <chrono>
#include <algorithm>
#include <math.h>
#include <ctime>

#include <vigra/random.hxx>
#include <vigra/impex.hxx>
#include <vigra/multi_impex.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/multi_convolution.hxx>

#define BOOST_FILESYSTEM_DEPRECATED

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

namespace fs = boost::filesystem;

using namespace vigra;

class imagetools
{
public:

	static void getTrainingDataPaths(const char* basePath, ArrayVector<const char*> imagePaths, ArrayVector<const char*> labelPaths)
	{
		//
	}

	template <class T1, class T2> 
	static void imageToFeatures(const MultiArray<3, T1> & image, const MultiArray<2, T2> & labels, MultiArray<2, T1> & rfFeatures, MultiArray<2, T2> & rfLabels, int downSample = 0, float downSampleFraction = 0.05)
	{
		// calc some useful constants
        int num_samples = image.size(0)*image.size(1);
		int num_features = image.size(2);
        int count = 0;
        int num_used_samples = 0;

		// 
        if (downSample)
		{
			// initialize
            MultiArray<2, T1> rfFeatures_temp;
            MultiArray<2, T2> rfLabels_temp;
            rfFeatures_temp.reshape(Shape2(num_samples, num_features));
            rfLabels_temp.reshape(Shape2(num_samples, 1));

            MersenneTwister random;
			
			for (int j = 0; j<image.size(1); ++j) {
				for (int i = 0; i<image.size(0); ++i) {
					if (labels(i, j) == 0) {
						if (random.uniform() <= downSampleFraction) {
                            rfLabels_temp(count, 0) = labels(i, j);
							for (int k = 0; k<num_features; ++k) {
                                rfFeatures_temp(count, k) = image(i, j, k);
							}
                            ++count;
						}
					}
					else {
                        rfLabels_temp(count, 0) = labels(i, j);
						for (int k = 0; k<num_features; ++k) {
                            rfFeatures_temp(count, k) = image(i, j, k);
						}
                        ++count;
					}
				}
			}
            num_used_samples = count;

            // allocate memory and transfer from 'temp' arrays
            rfFeatures.reshape(Shape2(num_used_samples,num_features));
            rfFeatures = rfFeatures_temp.subarray(Shape2(0,0),Shape2(num_used_samples,num_features));

            rfLabels.reshape(Shape2(num_used_samples,1));
            rfLabels = rfLabels_temp.subarray(Shape2(0,0),Shape2(num_used_samples,1));
        }
        else {
            // all samples are used
            num_used_samples = num_samples;
            rfFeatures.reshape(Shape2(num_used_samples, num_features));
            rfLabels.reshape(Shape2(num_used_samples, 1));

            //
            for (int j = 0; j<image.size(1); ++j) {
                for (int i = 0; i<image.size(0); ++i) {
                    rfLabels(count, 0) = labels(i, j);
                    for (int k = 0; k<num_features; ++k) {
                        rfFeatures(count, k) = image(i, j, k);
                    }
                    ++count;
                }
            }
		}		
	}

	template <class T1, class T2>
	static void imagesToFeatures(const ArrayVector< MultiArray<3, T1> > & imageArray, const ArrayVector< MultiArray<2, T2> > & labelArray, MultiArray<2, T1> & rfFeatures, MultiArray<2, T2> & rfLabels, int downSample = 0, float downSampleFraction = 0.05)
	{
		// calc some useful constants
        int num_images = imageArray.size();
		if (!num_images) return;

        int num_samples_per_image = imageArray[0].size(0) * imageArray[0].size(1);
        int num_samples = num_images*num_samples_per_image;
        int num_features = imageArray[0].size(2);

        rfFeatures.reshape(Shape2(num_samples, num_features));
        rfLabels.reshape(Shape2(num_samples, 1));

        for (int imgIdx = 0; imgIdx < num_images; imgIdx++) {
            MultiArray<2, T1> rfFeaturesPerImage = rfFeatures.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, num_features));
            MultiArray<2, T2> rfLabelsPerImage = rfLabels.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, 1));
            imageToFeatures<T1, T2>(imageArray[imgIdx], labelArray[imgIdx], rfFeaturesPerImage, rfLabelsPerImage, downSample, downSampleFraction);
		}
	}


    template <class T1>
    static void probsToImage(const MultiArray<2, T1> & probs, MultiArray<3, T1> & image, const Shape2 image_shape)
    {
        image.reshape(Shape3(image_shape[0], image_shape[1], probs.size(1)));

        // consistency check
        if (image_shape[0]*image_shape[1] != probs.size(0)) return;

        int count = 0;
        for (int j = 0; j<image.size(1); ++j) {
            for (int i = 0; i<image.size(0); ++i) {
                for (int k = 0; k<image.size(2); ++k) {
                    image(i,j,k) = probs(count,k);
                }
                ++count;
            }
        }
    }

    template <class T1>
    static void probsToImages(const MultiArray<2, T1> & probs, ArrayVector<MultiArray<3, T1> > & imageArray, const Shape2 image_shape)
    {
        //
        int num_samples = probs.size(0);
        int num_classes = probs.size(1);
        int num_samples_per_image = image_shape[0] * image_shape[1];
        int num_images = floor(num_samples/num_samples_per_image);

        imageArray.resize(num_images);

        // consistency check
        if (num_images*num_samples_per_image != num_samples) return;

        for (int imgIdx = 0; imgIdx < num_images; ++imgIdx){
            MultiArray<2, T1> probsPerImage = probs.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, num_classes));
            probsToImage<T1>(probsPerImage, imageArray[imgIdx], image_shape);
        }
    }

    template <class T1>
    static void imageToProbs(const MultiArray<3, T1> & image, MultiArray<2, T1> & probs)
    {
        // calc some useful constants
        int num_samples = image.size(0)*image.size(1);
        int num_classes = image.size(2);
        int count = 0;

        // all samples are used
        probs.reshape(Shape2(num_samples, num_classes));

        //
        for (int j = 0; j<image.size(1); ++j) {
            for (int i = 0; i<image.size(0); ++i) {
                for (int k = 0; k<num_classes; ++k) {
                    probs(count, k) = image(i, j, k);
                }
                ++count;
            }
        }

    }

    template <class T1>
    static void imagesToProbs(const ArrayVector<MultiArray<3, T1> > & imageArray, MultiArray<2, T1> & probs)
    {
        // calc some useful constants
        int num_images = imageArray.size();
        if (!num_images) return;

        int num_samples_per_image = imageArray[0].size(0) * imageArray[0].size(1);
        int num_samples = num_images*num_samples_per_image;
        int num_classes = imageArray[0].size(2);

        probs.reshape(Shape2(num_samples, num_classes));

        for (int imgIdx = 0; imgIdx < num_images; imgIdx++) {
            MultiArray<2, T1> probsPerImage = probs.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, num_classes));
            imageToProbs<T1>(imageArray[imgIdx], probsPerImage);
        }

    }

    template <class T2>
    static ArrayVector<float> dice(const MultiArray<2, T2> & groundTruthLabels, const MultiArray<2, T2> & autoLabels, const int num_classes)
    {

        // store diceScore for every class in an ArrayVector
        ArrayVector<UInt32> volume_of_intersection(num_classes);
        ArrayVector<UInt32> volume_of_GT(num_classes);
        ArrayVector<UInt32> volume_of_AL(num_classes);
        ArrayVector<float> diceScore(num_classes);
        int c_GT, c_AL;

        //
        for (int j=0; j<groundTruthLabels.size(1); ++j){
            for (int i=0; i<groundTruthLabels.size(0); ++i){

                c_GT = groundTruthLabels(i,j);
                c_AL = autoLabels(i,j);

                volume_of_GT[c_GT] += 1;
                volume_of_AL[c_AL] += 1;

                if (c_GT == c_AL)
                    volume_of_intersection[c_GT] += 1;

            }
        }

        for (int c=0; c<num_classes; ++c){
            if (volume_of_GT[c] + volume_of_AL[c] != 0)
                diceScore[c] = 2 * static_cast<float>(volume_of_intersection[c]) / (volume_of_GT[c] + volume_of_AL[c]);
            else
                diceScore[c] = 0;
        }

        return diceScore;

    }

//

	static ArrayVector<std::string> getAllFilenames(std::string basePath)
	{
		fs::path full_path(fs::initial_path<fs::path>());

		full_path = fs::system_complete(fs::path(basePath.c_str()));

		unsigned long file_count = 0;
		unsigned long dir_count = 0;
		unsigned long other_count = 0;
		unsigned long err_count = 0;

		if (!fs::exists(full_path))
		{
			std::cout << "\nNot found: " << full_path.file_string() << std::endl;
			return ArrayVector<std::string>(0);
		}

		if (fs::is_directory(full_path))
		{
			std::cout << "\nIn directory: "
				<< full_path.directory_string() << "\n\n";
			fs::directory_iterator end_iter;
			for (fs::directory_iterator dir_itr(full_path);
				dir_itr != end_iter;
				++dir_itr)
			{
				try
				{
					if (fs::is_directory(dir_itr->status()))
					{
						++dir_count;
						std::cout << dir_itr->path().filename() << " [directory]\n";
					}
					else if (fs::is_regular_file(dir_itr->status()))
					{
						++file_count;
                        std::cout << dir_itr->path().filename() << "\n";
					}
					else
					{
						++other_count;
						std::cout << dir_itr->path().filename() << " [other]\n";
					}

				}
				catch (const std::exception & ex)
				{
					++err_count;
					std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
				}
			}
			std::cout << "\n" << file_count << " files\n"
				<< dir_count << " directories\n"
				<< other_count << " others\n"
				<< err_count << " errors\n";
		}
		else // must be a file
		{
			std::cout << "\nFound: " << full_path.file_string() << "\n";
		}
        return ArrayVector<std::string>(0);

    }

    static void getArrayOfFeaturesAndLabels(int num_levels, std::string imgPath, std::string labelPath, ArrayVector< MultiArray<2, float> > rfFeaturesArray, ArrayVector< MultiArray<2, UInt8> > rfLabelsArray)
	{
		// get all names:
		ArrayVector<std::string> allImageNames = imagetools::getAllFilenames(imgPath);
		ArrayVector<std::string> allLabelNames = imagetools::getAllFilenames(labelPath);

		int numNames = allImageNames.size();
		int chunkSize = numNames / num_levels;

		// load a chunk of them and put into feature array: 
		for (int chunkIdx = 0; chunkIdx < num_levels; chunkIdx++)
		{
			int fromIdx = chunkIdx*chunkSize;
			int toIdx = fromIdx+chunkSize; // toIdx is not included!
			ArrayVector< MultiArray<3, float> > imageArray(toIdx - fromIdx);
			for (int idx = fromIdx; idx < toIdx; idx++)
			{
				if (idx >= allImageNames.size()) break;
				MultiArray<3, float> volume;
				std::string name = allImageNames[idx];
				std::string path = imgPath + name;
				importVolume(imageArray[idx - fromIdx], name.c_str());
			}
			ArrayVector< MultiArray<2, UInt8> > labelArray(toIdx - fromIdx);
			for (int idx = fromIdx; idx < toIdx; idx++)
			{
				if (idx >= allLabelNames.size()) break;
				MultiArray<2, float> labelImg;
				std::string name = allLabelNames[idx];
				std::string path = labelPath + name;
				importImage(name.c_str(), labelArray[idx - fromIdx]);
			}
			MultiArray<2, float> rfFeatures;
			MultiArray<2, UInt8> rfLabels;
            imagetools::imagesToFeatures<float, UInt8>(imageArray, labelArray, rfFeaturesArray[chunkIdx], rfLabelsArray[chunkIdx]);
		}
	}



/*
    template <class T1, class T2>
    static void featuresToImage(const MultiArray<2, T1> & rfFeatures, const MultiArray<2, T2> & rfLabels, MultiArray<3, T1> & image, MultiArray<2, T2> & labels, const Shape2 image_shape)
    {
        image.reshape(Shape3(image_shape[0], image_shape[1], rfFeatures.size(1)));
        labels.reshape(image_shape);

        // consistency check
        if (image.shape(0)*image.shape(1) != rfFeatures.size(0)) return;

        int count = 0;
        for (int j = 0; j<image.size(1); ++j) {
            for (int i = 0; i<image.size(0); ++i) {
                labels(i,j) = rfLabels(count,0);
                for (int k = 0; k<image.size(2); ++k) {
                    image(i,j,k) = rfFeatures(count,k);
                }
                ++count;
            }
        }

    }

    template <class T1, class T2>
    static void featuresToImages(const MultiArray<2, T1> & rfFeatures, const MultiArray<2, T2> & rfLabels, ArrayVector<MultiArray<3, T1> > & imageArray, ArrayVector<MultiArray<2, T2> > & labelArray, const Shape2 image_shape)
    {
        //
        int num_samples = rfFeatures.size(0);
        int num_features = rfFeatures.size(1);
        int num_samples_per_image = image_shape[0] * image_shape[1];
        int num_images = floor(num_samples/num_samples_per_image);

        imageArray.resize(num_images);
        labelArray.resize(num_images);

        // consistency check
        if (num_images*num_samples_per_image != num_samples) return;

        for (int imgIdx = 0; imgIdx < num_images; ++imgIdx){
            MultiArray<2, T1> rfFeaturesPerImage = rfFeatures.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, num_features));
            MultiArray<2, T2> rfLabelsPerImage = rfLabels.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, 1));
            featuresToImage<T1,T2>(rfFeaturesPerImage, rfLabelsPerImage, imageArray[imgIdx], labelArray[imgIdx], image_shape);
        }

//        MultiArray<3, T1> tempImage(Shape3(image_shape[0],image_shape[1],rfLabels.size(1)));
//        MultiArray<2, T2> tempLabels(image_shape);
//        int count = 0;
//        for (int imgIdx=0; imgIdx<num_images; ++imgIdx){
//            for (int j=0; j<tempImage.size(1); ++j){
//                for (int i=0; i<tempImage.size(0); ++i){
//                    tempLabels(i,j) = rfLabels(count, 0);
//                    for (int k=0; k<tempImage.size(2); ++k){
//                        tempImage(i,j,k) = rfFeatures(count,k);
//                    }
//                    count++;
//                }
//                imageArray[imgIdx] = tempImage;
//                labelArray[imgIdx] = tempLabels;
//            }
//        }
    }
*/

};
