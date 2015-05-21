#ifndef IMAGETOOLS_HXX
#define IMAGETOOLS_HXX

#include <iostream>
#include <string>
#include <fstream>
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
    static void imageToFeatures(const MultiArray<3, T1> & image, const MultiArray<2, T2> & labels, MultiArray<2, T1> & rfFeatures, MultiArray<2, T2> & rfLabels, int sampling = 1)
    {
        // calc some useful constants
        int num_samples = static_cast<int>(ceil(static_cast<float>(image.size(0))/sampling)) * static_cast<int>(ceil(static_cast<float>(image.size(1))/sampling));
        int num_features = image.size(2);

        // initialize arrays
        rfFeatures.reshape(Shape2(num_samples, num_features));
        rfLabels.reshape(Shape2(num_samples, 1));

        int count = 0;
        for (int j = 0; j<image.size(1); j += sampling) {
            for (int i = 0; i<image.size(0); i += sampling) {
                rfLabels(count, 0) = labels(i, j);
                for (int k = 0; k<num_features; ++k) {
                    rfFeatures(count, k) = image(i, j, k);
                }
                ++count;
            }
        }
    }

    template <class T1, class T2>
    static void imageToFeatures(const MultiArray<2, T1> & image, const MultiArray<2, T2> & labels, MultiArray<2, T1> & rfFeatures, MultiArray<2, T2> & rfLabels, int sampling = 1)
    {
        // calc some useful constants
        int num_samples = static_cast<int>(ceil(static_cast<float>(image.size(0))/sampling)) * static_cast<int>(ceil(static_cast<float>(image.size(1))/sampling));
        int num_features = 1;

        // initialize arrays
        rfFeatures.reshape(Shape2(num_samples, num_features));
        rfLabels.reshape(Shape2(num_samples, 1));

        int count = 0;
        for (int j = 0; j<image.size(1); j += sampling) {
            for (int i = 0; i<image.size(0); i += sampling) {
                rfLabels(count, 0) = labels(i, j);
                rfFeatures(count, 0) = image(i, j);
                ++count;
            }
        }
    }

	template <class T1, class T2>
    static void imagesToFeatures(const ArrayVector< MultiArray<3, T1> > & imageArray, const ArrayVector< MultiArray<2, T2> > & labelArray, MultiArray<2, T1> & rfFeatures, MultiArray<2, T2> & rfLabels, int sampling = 1)
	{
		// calc some useful constants
        int num_images = imageArray.size();
		if (!num_images) return;

        int num_samples_per_image = static_cast<int>(ceil(static_cast<float>(imageArray[0].size(0))/sampling)) * static_cast<int>(ceil(static_cast<float>(imageArray[0].size(1))/sampling));
        int num_samples = num_images * num_samples_per_image;

        // below: works for images of unequal size.  remove 2 lines above.
        /*
        int num_samples = 0;
        std::vector<int> num_samples_cumsum;
        for (int imgIdx = 0; imgIdx < num_images; imgIdx++)
        {
            num_samples += static_cast<int>(ceil(static_cast<float>(imageArray[imgIdx].size(0))/sampling)) * static_cast<int>(ceil(static_cast<float>(imageArray[imgIdx].size(1))/sampling));
            num_samples_cumsum.push_back(num_samples);
        }
        */

        int num_features = imageArray[0].size(2);

        rfFeatures.reshape(Shape2(num_samples, num_features));
        rfLabels.reshape(Shape2(num_samples, 1));

        for (int imgIdx = 0; imgIdx < num_images; imgIdx++) {

            MultiArray<2, T1> rfFeaturesPerImage;
            MultiArray<2, T2> rfLabelsPerImage;
            imageToFeatures<T1, T2>(imageArray[imgIdx], labelArray[imgIdx], rfFeaturesPerImage, rfLabelsPerImage, sampling);

            rfFeatures.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, num_features)) = rfFeaturesPerImage;
            rfLabels.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, 1)) = rfLabelsPerImage;

            // below: works for images of unequal size.  remove 2 lines above.
            /*
            int indxL = imgIdx == 0 ? 0 : num_samples_cumsum[imgIdx-1];
            int indxR = num_samples_cumsum[imgIdx];

            rfFeatures.subarray(Shape2(indxL, 0), Shape2(indxR, num_features)) = rfFeaturesPerImage;
            rfLabels.subarray(Shape2(indxL, 0), Shape2(indxR, 1)) = rfLabelsPerImage;
            */
        }
	}

    template <class T1, class T2>
    static void imagesToFeatures(const ArrayVector< MultiArray<2, T1> > & imageArray, const ArrayVector< MultiArray<2, T2> > & labelArray, MultiArray<2, T1> & rfFeatures, MultiArray<2, T2> & rfLabels, int sampling = 1)
    {
        // calc some useful constants
        int num_images = imageArray.size();
        if (!num_images) return;

        int num_samples_per_image = static_cast<int>(ceil(static_cast<float>(imageArray[0].size(0))/sampling)) * static_cast<int>(ceil(static_cast<float>(imageArray[0].size(1))/sampling));
        int num_samples = num_images * num_samples_per_image;

        // below: works for images of unequal size.  remove 2 lines above.
        /*
        int num_samples = 0;
        std::vector<int> num_samples_cumsum;
        for (int imgIdx = 0; imgIdx < num_images; imgIdx++)
        {
            num_samples += static_cast<int>(ceil(static_cast<float>(imageArray[imgIdx].size(0))/sampling)) * static_cast<int>(ceil(static_cast<float>(imageArray[imgIdx].size(1))/sampling));
            num_samples_cumsum.push_back(num_samples);
        }
        */

        int num_features = 1;

        rfFeatures.reshape(Shape2(num_samples, num_features));
        rfLabels.reshape(Shape2(num_samples, 1));

        for (int imgIdx = 0; imgIdx < num_images; imgIdx++) {

            MultiArray<2, T1> rfFeaturesPerImage;
            MultiArray<2, T2> rfLabelsPerImage;
            imageToFeatures<T1, T2>(imageArray[imgIdx], labelArray[imgIdx], rfFeaturesPerImage, rfLabelsPerImage, sampling);

            rfFeatures.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, num_features)) = rfFeaturesPerImage;
            rfLabels.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, 1)) = rfLabelsPerImage;

            // below: works for images of unequal size.  remove 2 lines above.
            /*
            int indxL = imgIdx == 0 ? 0 : num_samples_cumsum[imgIdx-1];
            int indxR = num_samples_cumsum[imgIdx];

            rfFeatures.subarray(Shape2(indxL, 0), Shape2(indxR, num_features)) = rfFeaturesPerImage;
            rfLabels.subarray(Shape2(indxL, 0), Shape2(indxR, 1)) = rfLabelsPerImage;
            */
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

        // all samples are used
        probs.reshape(Shape2(num_samples, num_classes));

        //
        int count = 0;
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
            MultiArray<2, T1> probsPerImage;
            imageToProbs<T1>(imageArray[imgIdx], probsPerImage);

            probs.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, num_classes)) = probsPerImage;
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

        int dimAutoX=autoLabels.size(0);
        int dimAutoY=autoLabels.size(1);

        int dimGTX=groundTruthLabels.size(0);
        int dimGTY=groundTruthLabels.size(1);

        double autoFactorX=dimAutoX/(double)dimGTX;
        double autoFactorY=dimAutoY/(double)dimGTY;

        //
        for (int j=0; j<groundTruthLabels.size(1); ++j){
            for (int i=0; i<groundTruthLabels.size(0); ++i){

                c_GT = groundTruthLabels(i,j);

                int iAuto=(int)(i*autoFactorX);
                int jAuto=(int)(j*autoFactorY);
                c_AL = autoLabels(iAuto,jAuto);

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

    template <class T2>
    static void diceOnFolder(std::string gtPath, std::string resultsPath, int num_levels = 1, int num_images = 0, int num_classes = 22)
    {

        ArrayVector<std::string> allGtNames = imagetools::getAllFilenames(gtPath);
        ArrayVector<std::string> allResultsNames = imagetools::getAllFilenames(resultsPath);

        if (num_images)
        {
            allGtNames.resize(num_images);
            allResultsNames.resize(num_images*num_levels);
        } else
            num_images = allGtNames.size();

        MultiArray<3, float> diceArray(Shape3(num_classes, num_levels, num_images));

        // load gt image, and corresponding set of results images from cascade.  run dice() on each pair.
        for (int imgIdx = 0; imgIdx < num_images; imgIdx++)
        {
            MultiArray<2, T2> gtLabelImg;
            fs::path name(allGtNames[imgIdx]);
            fs::path path(gtPath);
            fs::path full_path = path / name;                           // OS specific?
            importImage(full_path.string(), gtLabelImg);

            for (int levelIdx = 0; levelIdx < num_levels; levelIdx++)
            {
                int cascIdx = imgIdx*num_levels + levelIdx;

                MultiArray<2, T2> resultsLabelImg;
                fs::path name(allResultsNames[cascIdx]);
                fs::path path(resultsPath);
                fs::path full_path = path / name;                           // OS specific?
                importImage(full_path.string(), resultsLabelImg);

                ArrayVector<float> temp_dice = dice<T2>(gtLabelImg, resultsLabelImg, num_classes);
                for (int classIdx = 0; classIdx < temp_dice.size(); ++classIdx)
                    diceArray(classIdx, levelIdx, imgIdx) = temp_dice[classIdx];
            }

        }
        // write diceArray to csv file
        std::ofstream myfile;
        std::string fname;
        fname = resultsPath + "/diceScores.txt";
        myfile.open(fname);
        // first write column headers
        myfile << "image" << "," << "level" << "," << "class" << "," << "diceScore" << std::endl;
        // now write data
        for (int k=0; k<diceArray.size(2); k++)
        {
            for (int j=0; j<diceArray.size(1); j++)
            {
                for (int i=0; i<diceArray.size(0); i++)
                {
                    myfile << k << "," << j << "," << i << "," << diceArray(i,j,k) << std::endl;
                }
            }
        }
        myfile.close();

        /*
        for (int j=0; j<diceArray.size(1); j++)
        {
            for (int i=0; i<diceArray.size(0); i++)
            {
                myfile << "level" << j << "_" << "class" << i << ",";
            }
        }
        myfile << std::endl;
        // now write data
        for (int k=0; k<diceArray.size(2); k++)
        {
            for (int j=0; j<diceArray.size(1); j++)
            {
                for (int i=0; i<diceArray.size(0); i++)
                {
                    myfile << diceArray(i,j,k) << ",";
                }
            }
            myfile << std::endl;
        }
        myfile.close();
        */
    }

    template <class T2>
    static void diceOnTwoFolds(std::string gtPath1, std::string gtPath2, std::string resultsPath1, std::string resultsPath2, std::string dicePath, int num_levels, int num_images1, int num_images2, int num_classes = 22)
    {

        ArrayVector<std::string> allGtNames1 = imagetools::getAllFilenames(gtPath1);
        ArrayVector<std::string> allGtNames2 = imagetools::getAllFilenames(gtPath2);
        ArrayVector<std::string> allResultsNames1 = imagetools::getAllFilenames(resultsPath1);
        ArrayVector<std::string> allResultsNames2 = imagetools::getAllFilenames(resultsPath2);

        allGtNames1.resize(num_images1);
        allGtNames2.resize(num_images2);
        allResultsNames1.resize(num_images1*num_levels);
        allResultsNames2.resize(num_images2*num_levels);

        MultiArray<3, float> diceArray(Shape3(num_classes, num_levels, num_images1 + num_images2));

        // FOLD1 - load gt image, and corresponding set of results images from cascade.  run dice() on each pair.
        for (int imgIdx = 0; imgIdx < num_images1; imgIdx++)
        {
            MultiArray<2, T2> gtLabelImg;
            fs::path name(allGtNames1[imgIdx]);
            fs::path path(gtPath1);
            fs::path full_path = path / name;                           // OS specific?
            importImage(full_path.string(), gtLabelImg);

            for (int levelIdx = 0; levelIdx < num_levels; levelIdx++)
            {
                int cascIdx = imgIdx*num_levels + levelIdx;

                MultiArray<2, T2> resultsLabelImg;
                fs::path name(allResultsNames1[cascIdx]);
                fs::path path(resultsPath1);
                fs::path full_path = path / name;                           // OS specific?
                importImage(full_path.string(), resultsLabelImg);

                ArrayVector<float> temp_dice = dice<T2>(gtLabelImg, resultsLabelImg, num_classes);
                for (int classIdx = 0; classIdx < temp_dice.size(); ++classIdx)
                    diceArray(classIdx, levelIdx, imgIdx) = temp_dice[classIdx];
            }
        }

        // FOLD2 - load gt image, and corresponding set of results images from cascade.  run dice() on each pair.
        for (int imgIdx = 0; imgIdx < num_images2; imgIdx++)
        {
            MultiArray<2, T2> gtLabelImg;
            fs::path name(allGtNames2[imgIdx]);
            fs::path path(gtPath2);
            fs::path full_path = path / name;                           // OS specific?
            importImage(full_path.string(), gtLabelImg);

            for (int levelIdx = 0; levelIdx < num_levels; levelIdx++)
            {
                int cascIdx = imgIdx*num_levels + levelIdx;

                MultiArray<2, T2> resultsLabelImg;
                fs::path name(allResultsNames2[cascIdx]);
                fs::path path(resultsPath2);
                fs::path full_path = path / name;                           // OS specific?
                importImage(full_path.string(), resultsLabelImg);

                ArrayVector<float> temp_dice = dice<T2>(gtLabelImg, resultsLabelImg, num_classes);
                for (int classIdx = 0; classIdx < temp_dice.size(); ++classIdx)
                    diceArray(classIdx, levelIdx, num_images1 + imgIdx) = temp_dice[classIdx];
            }
        }

        // write diceArray to csv file
        std::ofstream myfile;
        std::string fname;
        fname = dicePath + "/diceScores.txt";
        myfile.open(fname);
        // first write column headers
        myfile << "image" << "," << "level" << "," << "class" << "," << "diceScore" << std::endl;
        // now write data
        for (int k=0; k<diceArray.size(2); k++)
        {
            for (int j=0; j<diceArray.size(1); j++)
            {
                for (int i=0; i<diceArray.size(0); i++)
                {
                    myfile << k << "," << j << "," << i << "," << diceArray(i,j,k) << std::endl;
                }
            }
        }
        myfile.close();

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
            ArrayVector<std::string> allFilenames;
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
                        boost::filesystem::path ext = dir_itr->path().extension();
                        if (ext.string() == ".tif" || ext.string() == ".jpg" || ext.string() == ".png")
                        {
                            std::cout << dir_itr->path().filename() << "\n";
                            allFilenames.push_back(dir_itr->path().filename().string());
                            ++file_count;
                        }
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

            return allFilenames;
		}
		else // must be a file
		{
			std::cout << "\nFound: " << full_path.file_string() << "\n";
		}
        return ArrayVector<std::string>(0);

    }

    static void getArrayOfFeaturesAndLabels(std::string imgPath,
                                            std::string labelPath,
                                            ArrayVector< MultiArray<2, float> > & rfFeaturesArray,
                                            ArrayVector< MultiArray<2, UInt8> > & rfLabelsArray,
                                            Shape2 & xy_dim,
                                            const std::vector<int> & randVector,
                                            int num_levels = 1,
                                            int sampling = 1,
                                            int featDim = 3)
    {
        // get all names:
        ArrayVector<std::string> allImageNamesOrdered = imagetools::getAllFilenames(imgPath);
        ArrayVector<std::string> allLabelNamesOrdered = imagetools::getAllFilenames(labelPath);

        int num_images=randVector.size();

        // randomize order of names:
        ArrayVector<std::string> allImageNames(num_images);
        ArrayVector<std::string> allLabelNames(num_images);

        for (int i=0; i<num_images; i++) {
            allImageNames[i]=allImageNamesOrdered[randVector[i]];
            allLabelNames[i]=allLabelNamesOrdered[randVector[i]];
        }

		int numNames = allImageNames.size();
		int chunkSize = numNames / num_levels;

        //
        rfFeaturesArray.resize(num_levels);
        rfLabelsArray.resize(num_levels);

        // load a chunk of them and put into feature array:
        if (featDim == 3)
        {
            for (int chunkIdx = 0; chunkIdx < num_levels; chunkIdx++)
            {
                int fromIdx = chunkIdx*chunkSize;
                int toIdx = fromIdx+chunkSize; // toIdx is not included!
                ArrayVector< MultiArray<3, float> > imageArray(toIdx - fromIdx);
                for (int idx = fromIdx; idx < toIdx; idx++)
                {
                    if (idx >= allImageNames.size()) break;
                    MultiArray<3, float> volume;
                    fs::path name(allImageNames[idx]);
                    fs::path path(imgPath);
                    fs::path full_path = path / name;                           // OS specific?
                    importVolume(volume, full_path.string());
                    imageArray[idx - fromIdx] = volume;
                }
                ArrayVector< MultiArray<2, UInt8> > labelArray(toIdx - fromIdx);
                for (int idx = fromIdx; idx < toIdx; idx++)
                {
                    if (idx >= allLabelNames.size()) break;
                    MultiArray<2, float> labelImg;
                    fs::path name(allLabelNames[idx]);
                    fs::path path(labelPath);
                    fs::path full_path = path / name;                           // OS specific?
                    importImage(full_path.string(), labelImg);
                    labelArray[idx - fromIdx] = labelImg;
                }
                MultiArray<2, float> rfFeatures;
                MultiArray<2, UInt8> rfLabels;
                imagetools::imagesToFeatures<float, UInt8>(imageArray, labelArray, rfFeatures, rfLabels, sampling);

                rfFeaturesArray[chunkIdx] = rfFeatures;
                rfLabelsArray[chunkIdx] = rfLabels;

                // set xy_dim, accounting for resampling
                if (chunkIdx == 0)
                {
                    xy_dim[0] = static_cast<int>(ceil(static_cast<float>(imageArray[0].size(0))/sampling));
                    xy_dim[1] = static_cast<int>(ceil(static_cast<float>(imageArray[0].size(1))/sampling));
                }

            }
        }
        else if (featDim == 2)
        {
            for (int chunkIdx = 0; chunkIdx < num_levels; chunkIdx++)
            {
                int fromIdx = chunkIdx*chunkSize;
                int toIdx = fromIdx+chunkSize; // toIdx is not included!
                ArrayVector< MultiArray<2, float> > imageArray(toIdx - fromIdx);
                for (int idx = fromIdx; idx < toIdx; idx++)
                {
                    if (idx >= allImageNames.size()) break;
                    MultiArray<2, float> volume;
                    fs::path name(allImageNames[idx]);
                    fs::path path(imgPath);
                    fs::path full_path = path / name;                           // OS specific?
                    importImage(full_path.string(), volume);
                    imageArray[idx - fromIdx] = volume;
                }
                ArrayVector< MultiArray<2, UInt8> > labelArray(toIdx - fromIdx);
                for (int idx = fromIdx; idx < toIdx; idx++)
                {
                    if (idx >= allLabelNames.size()) break;
                    MultiArray<2, float> labelImg;
                    fs::path name(allLabelNames[idx]);
                    fs::path path(labelPath);
                    fs::path full_path = path / name;                           // OS specific?
                    importImage(full_path.string(), labelImg);
                    labelArray[idx - fromIdx] = labelImg;
                }
                MultiArray<2, float> rfFeatures;
                MultiArray<2, UInt8> rfLabels;
                imagetools::imagesToFeatures<float, UInt8>(imageArray, labelArray, rfFeatures, rfLabels, sampling);

                rfFeaturesArray[chunkIdx] = rfFeatures;
                rfLabelsArray[chunkIdx] = rfLabels;

                // set xy_dim, accounting for resampling
                if (chunkIdx == 0)
                {
                    xy_dim[0] = static_cast<int>(ceil(static_cast<float>(imageArray[0].size(0))/sampling));
                    xy_dim[1] = static_cast<int>(ceil(static_cast<float>(imageArray[0].size(1))/sampling));
                }

            }
        }
    }


    /*
    static void getArrayOfFeaturesAndLabels(std::string featList,
                                            std::string labelList,
                                            ArrayVector< MultiArray<2, float> > & rfFeaturesArray,
                                            ArrayVector< MultiArray<2, UInt8> > & rfLabelsArray,
                                            Shape2 & xy_dim,
                                            const std::vector<int> & randVector,
                                            int num_levels = 1,
                                            int sampling = 1)
    {
        // get all names:
        ArrayVector<std::string> allImageNamesOrdered = imagetools::getAllFilenames(featList);
        ArrayVector<std::string> allLabelNamesOrdered = imagetools::getAllFilenames(labelList);

        int num_images=randVector.size();

        // randomize order of names:
        ArrayVector<std::string> allImageNames(num_images);
        ArrayVector<std::string> allLabelNames(num_images);

        for (int i=0; i<num_images; i++) {
            allImageNames[i]=allImageNamesOrdered[randVector[i]];
            allLabelNames[i]=allLabelNamesOrdered[randVector[i]];
        }

        int numNames = allImageNames.size();
        int chunkSize = numNames / num_levels;

        //
        rfFeaturesArray.resize(num_levels);
        rfLabelsArray.resize(num_levels);

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
                fs::path full_path(allImageNames[idx]);
                importVolume(volume, full_path.string());
                imageArray[idx - fromIdx] = volume;
            }
            ArrayVector< MultiArray<2, UInt8> > labelArray(toIdx - fromIdx);
            for (int idx = fromIdx; idx < toIdx; idx++)
            {
                if (idx >= allLabelNames.size()) break;
                MultiArray<2, float> labelImg;
                fs::path full_path(allLabelNames[idx]);
                importImage(full_path.string(), labelImg);
                labelArray[idx - fromIdx] = labelImg;
            }
            MultiArray<2, float> rfFeatures;
            MultiArray<2, UInt8> rfLabels;
            imagetools::imagesToFeatures<float, UInt8>(imageArray, labelArray, rfFeatures, rfLabels, sampling);

            // set xy_dim, accounting for resampling
            if (chunkIdx == 0)
            {
                xy_dim[0] = static_cast<int>(ceil(static_cast<float>(imageArray[0].size(0))/sampling));
                xy_dim[1] = static_cast<int>(ceil(static_cast<float>(imageArray[0].size(1))/sampling));
            }

            rfFeaturesArray[chunkIdx] = rfFeatures;
            rfLabelsArray[chunkIdx] = rfLabels;
        }
    }
    */

    static void getArrayOfRawImages(std::string imgPath,
                                    ArrayVector< MultiArray<2, float> > & rfRawImagesArray,
                                    Shape2 & xy_dim,
                                    const std::vector<int> & randVector,
                                    int num_levels = 1)
    {
        // get all names:
        ArrayVector<std::string> allImageNamesOrdered = imagetools::getAllFilenames(imgPath);

        int num_images=randVector.size();

        // randomize order of names:
        ArrayVector<std::string> allImageNames(num_images);
        for (int i=0; i<num_images; i++) {
            allImageNames[i]=allImageNamesOrdered[randVector[i]];
        }

        int numNames = allImageNames.size();
        int chunkSize = numNames / num_levels;

        //
        rfRawImagesArray.resize(num_levels);

        // load a chunk of them and put into array:
        for (int chunkIdx = 0; chunkIdx < num_levels; chunkIdx++)
        {
            int fromIdx = chunkIdx*chunkSize;
            int toIdx = fromIdx+chunkSize; // toIdx is not included!
            ArrayVector< MultiArray<3, float> > imageArray(toIdx - fromIdx);
            for (int idx = fromIdx; idx < toIdx; idx++)
            {
                if (idx >= allImageNames.size()) break;
                MultiArray<2, float> image2;
                fs::path name(allImageNames[idx]);
                fs::path path(imgPath);
                fs::path full_path = path / name;                           // OS specific?
                importImage(full_path.string(), image2);
                MultiArray<3, float> image3;
                image3.reshape(Shape3(image2.size(0),image2.size(1),1));
                image3.bindOuter(0) = image2;
                imageArray[idx - fromIdx] = image3;
            }

            MultiArray<2, float> rfRawImages;
            imagetools::imagesToProbs<float>(imageArray, rfRawImages);

            // set xy_dim, accounting for resampling
            if (chunkIdx == 0)
            {
                xy_dim[0] = imageArray[0].size(0);
                xy_dim[1] = imageArray[0].size(1);
            }

            rfRawImagesArray[chunkIdx] = rfRawImages;
        }
    }

    static void getListOfTrainingData(std::string imgPath, ArrayVector<std::string> & allImageNames, const std::vector<int> & randVector)
    {
        // get all names:
        ArrayVector<std::string> allImageNamesOrdered = imagetools::getAllFilenames(imgPath);

        int num_images=randVector.size();

        // randomize order of names:
//        ArrayVector<std::string> allImageNames(num_images);
        allImageNames.resize(num_images);
        for (int i=0; i<num_images; i++) {
            allImageNames[i]=allImageNamesOrdered[randVector[i]];
        }

    }

    // weightedProbMap somehow got broken when making diceOnFolder.  comment out for now...
    /*

*/

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

    // old versions of imageToFeatures and imagesToFeatures.  changed above to a simpler (uniform) down-sampling strategy.  delete when happy with new versions.

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
//            MultiArray<2, T1> rfFeaturesPerImage = rfFeatures.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, num_features));
//            MultiArray<2, T2> rfLabelsPerImage = rfLabels.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, 1));
//            imageToFeatures<T1, T2>(imageArray[imgIdx], labelArray[imgIdx], rfFeaturesPerImage, rfLabelsPerImage, downSample, downSampleFraction);

            MultiArray<2, T1> rfFeaturesPerImage;
            MultiArray<2, T2> rfLabelsPerImage;
            imageToFeatures<T1, T2>(imageArray[imgIdx], labelArray[imgIdx], rfFeaturesPerImage, rfLabelsPerImage, downSample, downSampleFraction);

            rfFeatures.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, num_features)) = rfFeaturesPerImage;
            rfLabels.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, 1)) = rfLabelsPerImage;
        }
    }
*/

};

#endif // IMAGETOOLS_HXX
