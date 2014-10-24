#ifndef SMOOTHINGTOOLS_HXX
#define SMOOTHINGTOOLS_HXX

#include <iostream>
#include <array>
#include <chrono>
#include <algorithm>
#include <math.h>
#include <ctime>

#include "libmodelBasedSmoothing2.h"
//#include "libMBS_AAM_gS.h"

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
