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

using namespace vigra;

class imagetools
{
public:

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
    static void imagesToFeatures(const MultiArray<1, MultiArray<3, T1> > & imageArray, const MultiArray<1, MultiArray<2, T2> > & labelArray, MultiArray<2, T1> & rfFeatures, MultiArray<2, T2> & rfLabels, int downSample = 0, float downSampleFraction = 0.05)
	{
		// calc some useful constants
        int num_images = imageArray.size(0);
		if (!num_images) return;

        int num_samples_per_image = imageArray(0).size(0)*imageArray(0).size(1);
        int num_samples = num_images*num_samples_per_image;
        int num_features = imageArray(0).size(2);

        rfFeatures.reshape(Shape2(num_samples, num_features));
        rfLabels.reshape(Shape2(num_samples, 1));

        for (int imgIdx = 0; imgIdx < num_images; imgIdx++) {
            MultiArray<2, T1> rfFeaturesPerImage = rfFeatures.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, num_features));
            MultiArray<2, T2> rfLabelsPerImage = rfLabels.subarray(Shape2(imgIdx*num_samples_per_image, 0), Shape2((imgIdx + 1)*num_samples_per_image, 1));
            imageToFeatures<T1, T2>(imageArray(imgIdx), labelArray(imgIdx), rfFeaturesPerImage, rfLabelsPerImage, downSample, downSampleFraction);
		}
	}

	template <class T1, class T2>
    static void featuresToImage(const MultiArray<2, T1> & rfFeatures, const MultiArray<2, T2> & rfLabels, MultiArray<3, T1> & image, MultiArray<2, T2> & labels, Shape2 & image_shape)
	{
        image.reshape(Shape3(image_shape[0], image_shape[1], rfFeatures.size(1)));
        labels.reshape(image_shape);

        // consistency check
        if (image.shape(0)*image.shape(1) != rfFeatures.size(0))
        {
            std::cout << "image shape is inconsistent with dimensions of rfFeatures" << std::endl;
            exit(-1);
        }

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
    static void featuresToImages()
    {

    }

	template <class T2> 
	static float dice(MultiArray<2, T2> & groundTruthLabels, MultiArray<2, T2> & autoLabels)
	{
		// float diceScore = 2*volume_of_intersection/(volume_1+volume_2);
		// return diceScore;
	}

};
