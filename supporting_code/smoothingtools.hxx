#ifndef SMOOTHINGTOOLS_HXX
#define SMOOTHINGTOOLS_HXX

#include <iostream>
#include <array>
#include <chrono>
#include <algorithm>
#include <math.h>
#include <ctime>

#include "libmodelBasedSmoothing2.h"

#include "inferencetools.hxx"

using namespace vigra;

class smoothingtools
{
public:

    template <class T1>
    static void rigidMBS(const MultiArray<3, T1> & probStack, MultiArray<3, T1> & smoothProbStack, int numFits = 10, int numCentroidsUsed = 21, int sampling = 1)
    {

        // convert parameters to Matlab types
        mwArray mwNumFits;
        mwNumFits = mwArray(numFits);

        mwArray mwNumCentroidsUsed;
        mwNumCentroidsUsed = mwArray(numCentroidsUsed);

        mwArray mwShape(1, 3, mxDOUBLE_CLASS);
        mwShape(1) = mwArray(static_cast<double>(probStack.size(0)));
        mwShape(2) = mwArray(static_cast<double>(probStack.size(1)));
        mwShape(3) = mwArray(static_cast<double>(probStack.size(2)));

        mwArray mwSampling;
        mwSampling = mwArray(sampling);

        // convert multiArray to mwArray for Matlab
        mwArray mwProbs(1, probStack.size(), mxSINGLE_CLASS);
        vigraToMatlabArray<T1>(probStack, mwProbs);

        // call MBS routine
        mwArray mwSmoothProbs(1, probStack.size(), mxSINGLE_CLASS);
        modelBasedSmoothing_wARGS(1, mwSmoothProbs, mwProbs, mwShape, mwNumCentroidsUsed, mwNumFits, mwSampling);

        // get data out of mwArray
        smoothProbStack.reshape(Shape3(probStack.size(0), probStack.size(1), probStack.size(2)));
        matlabToVigraArray(mwSmoothProbs, smoothProbStack);
    }

    template <class T1>
    static void AAM_MBS(const MultiArray<3, T1> & probStack, const MultiArray<3, T1> & rawImage, MultiArray<3, T1> & smoothProbStack, int sampling = 1, int numGDsteps = 50, float lambda = 2)
    {

        // convert parameters to Matlab types
        mwArray mwNumGDsteps;
        mwNumGDsteps = mwArray(numGDsteps);

        mwArray mwLambda;
        mwLambda = mwArray(lambda);

        mwArray mwProbStackShape(1, 3, mxDOUBLE_CLASS);
        mwProbStackShape(1) = mwArray(static_cast<double>(probStack.size(0)));
        mwProbStackShape(2) = mwArray(static_cast<double>(probStack.size(1)));
        mwProbStackShape(3) = mwArray(static_cast<double>(probStack.size(2)));

        mwArray mwRawImageShape(1, 2, mxDOUBLE_CLASS);
        mwRawImageShape(1) = mwArray(static_cast<double>(rawImage.size(0)));
        mwRawImageShape(2) = mwArray(static_cast<double>(rawImage.size(1)));

        mwArray mwSampling;
        mwSampling = mwArray(sampling);

        // convert multiArray to mwArray for Matlab
        mwArray mwProbs(1, probStack.size(), mxSINGLE_CLASS);
        vigraToMatlabArray<T1>(probStack, mwProbs);

        mwArray mwRawImage(1, rawImage.size(), mxSINGLE_CLASS);
        vigraToMatlabArray<T1>(rawImage, mwRawImage);

        // call MBS routine
        mwArray mwSmoothProbs(1, probStack.size(), mxSINGLE_CLASS);
//        MBS_AAM_gS_DEBUG(mwSampling);
//        MBS_AAM_gS(1, mwSmoothProbs, mwSampling, mwNumGDsteps, mwLambda);
        MBS_AAM_gS(1, mwSmoothProbs, mwRawImage, mwRawImageShape, mwProbs, mwProbStackShape, mwSampling, mwNumGDsteps, mwLambda);

        // get data out of mwArray
        smoothProbStack.reshape(Shape3(probStack.size(0), probStack.size(1), probStack.size(2)));
        matlabToVigraArray(mwSmoothProbs, smoothProbStack);
    }

//    template <class T1>
//    static void AAM_Inference(const MultiArray<3, T1> & probStack, const MultiArray<3, T1> & rawImage, MultiArray<3, T1> & smoothProbStack, int sampling = 1, int numGDsteps = 50, float lambda = 2)
//    {

//        // copy from above...



//    }

    template <class T1, class T2>
    static void probFromFits(const MultiArray<2,T1> & unaryFactors, const MultiArray<3,T1> & pairwiseFactors, const MultiArray<3, int> & fits, MultiArray<3,T2> & probs)
    {

        // useful constants
        const int numClasses = unaryFactors.size(0) + 1;
        const int numFits    = unaryFactors.size(1);

        // allocate memory for smoothProbs
        probs.reshape(Shape3(fits.size(0), fits.size(1), numClasses));

        // do inference
        MultiArray<2, double> allMarginals;
        inferencetools::chainProbInference<T1,double>(unaryFactors, pairwiseFactors, allMarginals);

        // transpose marginals, so that first index is over states
        MultiArrayView<2, double> allMarginalsTranspose = allMarginals.transpose();

        // reshape fits and smoothProbs for convenience (using View to avoid copying data).  DIDN'T WORK!
        /*
        int count = 0;
        MultiArrayView<2, int> reshapedFits; //(Shape2(fits.size(0)*fits.size(1), fits.size(2)));       // NOT SURE IF I CAN ASSIGN A VIEW ONE PIXEL AT A TIME
        MultiArrayView<2, float> reshapedProbs; //(reshapedFits.shape());
        for (int k = 0; k < reshapedFits.size(1); ++k){
            for (int i = 0; i < reshapedFits.size(0); ++i){
                reshapedFits(i,k) = fits[count];
                reshapedProbs(i,k) = probs[count];
                count += 1;
            }
        }
        reshapedFits = reshapedFits.transpose();
        reshapedProbs = reshapedProbs.transpose();

        MultiArray<2, float> weightedFits(Shape2(reshapedFits.shape()));
        for (int i = 0; i < reshapedFits.size(0); ++i)
                for (int k = 0; k < reshapedFits.size(1); ++k)
                    weightedFits(i,k) = fits(i,k) * allMarginalsTranspose[i];           // check orientation!!!

        float test = 0;
        for (int i = 0; i < weightedFits.size(); ++i)
            test += weightedFits[i];
        std::cout << test << std::endl;

        // fill smoothProbs thru reshapedSmoothProbs (View).                                        // check that smoothProbs actually gets filled!!!
        reshapedProbs.init(0.0);
        for (int j = 0; j < reshapedFits.size(1); ++j){
            // store sum of probabilities, for normalization and to calculate background probability
            float sumProbs = 0.0;
            // aggregate all probs for each class independently
            for (int c = 1; c < numClasses; ++c){
                for (int i = 0; i < numFits; ++i)
                    reshapedProbs(c,j) += weightedFits((c-1)*numFits + i, j); //.subarray(Shape2((c-1)*numFits,j), Shape2(c*numFits, j+1));
                sumProbs += reshapedProbs(c,j);                                               // check that sumProbs gets proper value from View
            }
            // compute background prob, and renormalize smoothProbs if necessary
            if ( sumProbs == 0.0 )
                reshapedProbs(0,j) = 1.0;
            else if ( (0.0 < sumProbs) && (sumProbs <= 1.0))
                reshapedProbs(0,j) = 1.0 - sumProbs;
            else if ( 1.0 < sumProbs ){
                reshapedProbs(0,j) = 0.0;
                reshapedProbs.subarray(Shape2(1,j),Shape2(numClasses-1,j+1)) /= sumProbs;
            } else {
                std::cout << "Error with smoothProb map.  Negative probs? " << std::endl;
                exit(-1);
            }

        }
        */

        // compute weighted masks - EXPLICIT
        MultiArray<3, float> weightedFits(Shape3(fits.shape()));
        for (int k = 0; k < fits.size(2); ++k)
            for (int j = 0; j < fits.size(1); ++j)
                for (int i = 0; i < fits.size(0); ++i)
                    weightedFits(i,j,k) = static_cast<float>(fits(i,j,k)) * static_cast<float>(allMarginalsTranspose[k]);

        // test
        VolumeExportInfo Export_info("/Users/richmond/Desktop/tests/weightedFits/weightedFits",".tif");
        exportVolume(weightedFits, Export_info);

        // fill smoothProbs - EXPLICIT
        probs.init(0.0);
        for (int j = 0; j < weightedFits.size(1); ++j){
            for (int i = 0; i < weightedFits.size(0); ++i){
                // store sum of probabilities, for normalization and to calculate background probability
                float sumProbs = 0.0;
                // aggregate all probs for each class independently
                for (int c = 1; c < numClasses; ++c){
                    for (int f = 0; f < numFits; ++f)
                        probs(i,j,c) += weightedFits(i,j,(c-1)*numFits+f);
                    sumProbs += probs(i,j,c);
                }
                // compute background prob, and renormalize smoothProbs if necessary
                if ( sumProbs == 0.0 )
                    probs(i,j,0) = 1.0;
                else if ( (0.0 < sumProbs) && (sumProbs <= 1.0)){
                    probs(i,j,0) = 1.0 - sumProbs;
//                    std::cout << "probs(" << i << "," << j << ") = " << probs() << std::endl;
                }
                else if ( 1.0 < sumProbs ){
                    probs(i,j,0) = 0.0;
                    for (int c = 1; c < numClasses; ++c)
                        probs(i,j,c) /= sumProbs;
                } else {
                    std::cout << "Error with smoothProb map.  Negative probs? " << std::endl;
                    exit(-1);
                }
            }
        }
    }

    template <class T1>
    static void MAPFromFits(const MultiArray<2,T1> & unaryFactors, const MultiArray<3,T1> & pairwiseFactors, const MultiArray<3, int> & fits, MultiArray<2,int> & MAPLabels)
    {

        // useful constants
        const int numClasses = unaryFactors.size(0) + 1;
        const int numFits    = unaryFactors.size(1);

        // allocate memory for smoothProbs
        MAPLabels.reshape(Shape2(fits.size(0), fits.size(1)));

        // MAP inference
        MultiArray<1, int> MAPChain;
        inferencetools::chainMAPInference<T1, int>(unaryFactors, pairwiseFactors, MAPChain);

        // fill MAPLabels
        MAPLabels.init(0);
        for (int j = 0; j < MAPLabels.size(1); ++j){
            for (int i = 0; i < MAPLabels.size(0); ++i){
                // figure out which class to assign
                T1 maxUnary = static_cast<T1>(0);
                for (int c = 1; c < numClasses; ++c){
                    int MAPindx = (c-1)*numFits + MAPChain(c-1);

                    if ( (fits(i,j,MAPindx) == 1) && (unaryFactors(c-1, MAPChain(c-1)) > maxUnary) ){
                        MAPLabels(i,j) = c;
                        maxUnary = unaryFactors(c-1, MAPChain(c-1));
                    }
                }
            }
        }
    }


    template <class T1>
    static void vigraToMatlabArray(const MultiArray<3, T1> & probStack, mwArray & mwProbs)
    {
        const int num_px = probStack.size();

        // create std array and transfer data
        T1* probArray = new float[num_px];

        int count = 0;
        for (int k = 0; k < probStack.size(2); ++k){
            for (int j = 0; j < probStack.size(1); ++j){
                for (int i = 0; i < probStack.size(0); ++i){
                    probArray[count] = probStack(i,j,k);
                    count += 1;
                }
            }
        }

        // create Matlab Array and transfer data
        mwProbs.SetData(probArray, num_px);

        delete [] probArray;
    }

    template <class T1>
    static void matlabToVigraArray(const mwArray & mwSmoothProbs, MultiArray<3, T1> & smoothProbStack)
    {
        const int num_px = mwSmoothProbs.NumberOfElements();

        float* smoothProbArray = new float[num_px];
        mwSmoothProbs.GetData(smoothProbArray, num_px);

        // transfer back to MultiArray
        int count = 0;
        for (int k = 0; k < smoothProbStack.size(2); ++k){
            for (int j = 0; j < smoothProbStack.size(1); ++j){
                for (int i = 0; i < smoothProbStack.size(0); ++i){
                    smoothProbStack(j,i,k) = smoothProbArray[count];    // transpose b/c row-major order!
                    count += 1;
                }
            }
        }

        delete [] smoothProbArray;
    }
};

#endif // SMOOTHINGTOOLS_HXX
