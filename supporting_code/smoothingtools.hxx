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
using namespace vigra::multi_math;

#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

#include <vigra/convolution.hxx>
#include <vigra/multi_impex.hxx>
#include <vigra/basicgeometry.hxx>

using namespace boost;

class smoothingtools
{
public:

    /*
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
        vigraToMatlab_Array<T1>(probStack, mwProbs);

        // call MBS routine
        mwArray mwSmoothProbs(1, probStack.size(), mxSINGLE_CLASS);
        modelBasedSmoothing_wARGS(1, mwSmoothProbs, mwProbs, mwShape, mwNumCentroidsUsed, mwNumFits, mwSampling);

        // get data out of mwArray
        smoothProbStack.reshape(Shape3(probStack.size(0), probStack.size(1), probStack.size(2)));
        matlabToVigraArray(mwSmoothProbs, smoothProbStack);
    }
    */

    static void buildAAM(const char* dataPath, const char* outputPath, ArrayVector<std::string> fnameList, const int marginType, const int numP, const int numLambda, const int saveModelForTraining, const int saveModelForTest)
    {

        // convert to matlab types
        mwArray mwDataPath = mwArray(dataPath);
        mwArray mwOutputPath = mwArray(outputPath);

        mwArray mwFnameList(1, fnameList.size(), mxCELL_CLASS);
        for (int i = 0; i < fnameList.size(); ++i)
            mwFnameList(i+1) = fnameList[i].c_str();

        mwArray mwMarginType = mwArray(marginType);
        mwArray mwNumP = mwArray(numP);
        mwArray mwNumLambda = mwArray(numLambda);

        buildAllModels(mwDataPath, mwFnameList, mwMarginType, mwNumP, mwNumLambda, mwOutputPath, mwArray(saveModelForTraining), mwArray(saveModelForTest));

    }


    template <class T1>
    static void AAM_perSomite_Inference_2inits(const char* modelPath, const MultiArray<3, T1> & probStack, const MultiArray<3, T1> & rawImage, MultiArray<3, T1> & margProbStack, MultiArray<2, int> & MAPLabels, MultiArray<1, double> priorStrength, int numOffsets=5, double offsetScale=1.0, int sampling=1, int numGDsteps=50, float lambdaU=4, float lambdaPW=4)
    {

        // useful constants
        int num_classes = probStack.size(2);

        // convert to Matlab types ------------------------------------------------------------------------------>

        // convert parameters to Matlab types

        mwArray mwModelPath = modelPath;

        mwArray mwPriorStrength(1, priorStrength.size(), mxDOUBLE_CLASS);
        for (int i = 0; i < priorStrength.size(); ++i)
            mwPriorStrength(i+1) = mwArray(priorStrength(i));

        mwArray mwNumOffsets;
        mwNumOffsets = mwArray(numOffsets);

        mwArray mwOffsetScale(1, 1, mxDOUBLE_CLASS);
        mwOffsetScale = mwArray(offsetScale);

        mwArray mwSampling;
        mwSampling = mwArray(sampling);

        mwArray mwNumGDsteps;
        mwNumGDsteps = mwArray(numGDsteps);

        mwArray mwLambdaU;
        mwLambdaU = mwArray(lambdaU);

        mwArray mwLambdaPW;
        mwLambdaPW = mwArray(lambdaPW);

        mwArray mwProbStackShape(1, 3, mxDOUBLE_CLASS);
        mwProbStackShape(1) = mwArray(static_cast<double>(probStack.size(1)));      // flip xy for Matlab
        mwProbStackShape(2) = mwArray(static_cast<double>(probStack.size(0)));
        mwProbStackShape(3) = mwArray(static_cast<double>(probStack.size(2)));

        mwArray mwRawImageShape(1, 2, mxDOUBLE_CLASS);
        mwRawImageShape(1) = mwArray(static_cast<double>(rawImage.size(1)));        // these are flipped for Matlab
        mwRawImageShape(2) = mwArray(static_cast<double>(rawImage.size(0)));

        // convert multiArray to mwArray for Matlab
        mwArray mwProbs(1, probStack.size(), mxSINGLE_CLASS);
        vigraToMatlab_Array<T1>(probStack, mwProbs);

        mwArray mwRawImage(1, rawImage.size(), mxSINGLE_CLASS);
        vigraToMatlab_Array<T1>(rawImage, mwRawImage);

        // initialize arrays for output
        mwArray mwUnaryFactors(1, (num_classes-1)*2*numOffsets, mxDOUBLE_CLASS);
        mwArray mwPairwiseFactors(1, 4*numOffsets*numOffsets*(num_classes-1), mxDOUBLE_CLASS);
        mwArray mwFitMasks(1, probStack.size(0)*probStack.size(1)*(num_classes-1)*2*numOffsets, mxINT8_CLASS);

        // call MBS routine ----------------------------------------------------------------------------------------------->

        AAM_perSomite_Inf_2inits(3, mwUnaryFactors, mwPairwiseFactors, mwFitMasks, mwModelPath, mwRawImage, mwRawImageShape, mwProbs, mwProbStackShape, mwSampling, mwNumGDsteps, mwPriorStrength, mwNumOffsets, mwOffsetScale, mwLambdaU, mwLambdaPW);

        // convert from Matlab types ---------------------------------------------------------------------------->

        // get data out of mwArrays
        MultiArray<3, int> fitMasks(Shape3(probStack.size(0), probStack.size(1), (num_classes-1)*2*numOffsets));
        matlabToVigra_Array<int>(mwFitMasks, fitMasks);

        MultiArray<2, double> unaryFactors(Shape2(num_classes-1, 2*numOffsets));
        matlabToVigra_UnaryFactors(mwUnaryFactors, unaryFactors);

        MultiArray<3, double> pairwiseFactors(Shape3(2*numOffsets, 2*numOffsets, num_classes-1));
        matlabToVigra_PairwiseFactors(mwPairwiseFactors, pairwiseFactors);

        // calc marginal and MAP solutions, and return corresponding probability and label images ---------------------------------------------------------------------------->

        margProbStack.reshape(probStack.shape());
        probFromFits<double, T1>(unaryFactors, pairwiseFactors, fitMasks, margProbStack);

        MAPLabels.reshape(Shape2(probStack.size(0), probStack.size(1)));
        MAPFromFits<double>(unaryFactors, pairwiseFactors, fitMasks, MAPLabels);
    }

    template <class T1>
    static void AAM_MBS(const MultiArray<3, T1> & probStack, const MultiArray<3, T1> & rawImage, MultiArray<3, T1> & smoothProbStack, MultiArray<1, double> priorStrength, int numOffsets=5, double offsetScale=1.0, int sampling=1, int numGDsteps=50, float lambda=2)
    {

        // convert parameters to Matlab types
        mwArray mwPriorStrength(1, priorStrength.size(), mxDOUBLE_CLASS);
        for (int i = 0; i < priorStrength.size(); ++i)
            mwPriorStrength(i+1) = mwArray(priorStrength(i));

        mwArray mwNumOffsets;
        mwNumOffsets = mwArray(numOffsets);

        mwArray mwOffsetScale(1, 1, mxDOUBLE_CLASS);
        mwOffsetScale = mwArray(offsetScale);

        mwArray mwNumGDsteps;
        mwNumGDsteps = mwArray(numGDsteps);

        mwArray mwLambda;
        mwLambda = mwArray(lambda);

        mwArray mwProbStackShape(1, 3, mxDOUBLE_CLASS);
        mwProbStackShape(1) = mwArray(static_cast<double>(probStack.size(1)));
        mwProbStackShape(2) = mwArray(static_cast<double>(probStack.size(0)));
        mwProbStackShape(3) = mwArray(static_cast<double>(probStack.size(2)));

        mwArray mwRawImageShape(1, 2, mxDOUBLE_CLASS);
        mwRawImageShape(1) = mwArray(static_cast<double>(rawImage.size(1)));
        mwRawImageShape(2) = mwArray(static_cast<double>(rawImage.size(0)));

        mwArray mwSampling;
        mwSampling = mwArray(sampling);

        // convert multiArray to mwArray for Matlab
        mwArray mwProbs(1, probStack.size(), mxSINGLE_CLASS);
        vigraToMatlab_Array<T1>(probStack, mwProbs);

        mwArray mwRawImage(1, rawImage.size(), mxSINGLE_CLASS);
        vigraToMatlab_Array<T1>(rawImage, mwRawImage);

        // call MBS routine
        mwArray mwSmoothProbs(1, probStack.size(), mxSINGLE_CLASS);
        MBS_AAM_gS(1, mwSmoothProbs, mwRawImage, mwRawImageShape, mwProbs, mwProbStackShape, mwSampling, mwNumGDsteps, mwPriorStrength, mwNumOffsets, mwOffsetScale, mwLambda);

        // get data out of mwArray
        smoothProbStack.reshape(Shape3(probStack.size(0), probStack.size(1), probStack.size(2)));
        matlabToVigra_Array(mwSmoothProbs, smoothProbStack);
    }

    template <class T1>
    static void GeodesicSmoothing(const MultiArray<3, T1> & probStack, const MultiArray<3, T1> & rawImage, MultiArray<3, T1> & smoothProbStack, int num_images, Shape2 xy_dim, int sampling, float maxDist=10)
    {
        std::cout << "running geodesic smoothing..." << std::endl;

        int num_classes=probStack.size(2);

        smoothProbStack.copy(probStack);
        // for each pixel p

            // for each class

        float sigma2=0.2;//0.02;//2;//0.2;
        float nu=1;
        float sigmaGrad=3;

        T1 pixelDistFactor = 0.001;//0;//0.2;
        T1 gradMagWeight=0.1;//0.05;

        int num_pixels=xy_dim[0]*xy_dim[1];

        // always the same: pixel seed image (don't need it for graph, really...)
/*        MultiArray<2, T1> pixelSeed(0);
        pixelSeed.reshape(Shape2(xy_dim[0],xy_dim[1]),0);
        pixelSeed(p)=1;
*/
        // Map = nu*(1-prob(class,x))
        MultiArray<3, T1> MaskImgStack = nu*(1-probStack);

        int excerptRadius=maxDist;
        int excerptSize=2*excerptRadius+1;

        typedef adjacency_list < listS, vecS, directedS,
        no_property, property < edge_weight_t, float > > graph_t;
        typedef graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
        typedef graph_traits < graph_t >::edge_descriptor edge_descriptor;
        typedef std::pair<int, int> Edge;
        const int num_nodes = excerptSize*excerptSize;
        const int max_num_edges = num_nodes*8;
        Edge* edge_array = new Edge[max_num_edges];
        T1* weight = new T1[max_num_edges];

        MultiArray<2, T1> geoDistOfPixel(0);
        geoDistOfPixel.reshape(Shape2(excerptSize,excerptSize),0);

        MultiArray<2, T1> backgroundProb = probStack.bindOuter(0);
        MultiArray<2, T1> rawImage2d = rawImage.bindOuter(0);
        MultiArray<2, T1> gradMagImageFull;
        gradMagImageFull.reshape(rawImage2d.shape());

//        for (int k=0; k<num_images; ++k){

        // debug output:
        gaussianGradientMagnitude( rawImage2d,gradMagImageFull,sigmaGrad);
        MultiArray<2, T1> gradMagImage;
        gradMagImage.reshape(ceil(1./sampling*gradMagImageFull.shape()));
        resampleImage(gradMagImageFull,gradMagImage,1./sampling);
        exportImage(gradMagImage, "/Users/kainmull/testGradMag.tif");

        // iterate over pixels...
        for (int x=excerptRadius; x<xy_dim[0]-excerptRadius; x++) {
//            std::cout << "smoothing pixel (x,.)" << x << " " << std::endl;
            for (int y=excerptRadius; y<xy_dim[1]-excerptRadius; y++) {

                // check if we need it for that pixel --- prob threshold in background prob image
                bool doPixel = backgroundProb(x,y) < 0.9 ;
                if (!doPixel) continue;

                // geo dist transform in raw image (downsampled!! max distance!!)
                // get excerpt of gradMagImage:
                MultiArray<2, T1> gradMagExcerpt = gradMagWeight*gradMagImage.subarray(Shape2(x-excerptRadius,y-excerptRadius), Shape2(x+excerptRadius+1,y+excerptRadius+1));
                if (x==43 && y==150)
                   exportImage(gradMagExcerpt, "/Users/kainmull/testGradMagExcerpt.tif");

                // build graph:

                int edge_count=0;
                for (int excX=0; excX<excerptSize; excX++)
                for (int excY=0; excY<excerptSize; excY++) {
                    // edges and weights
                    // edges in x-dir: -1, +1
                    // edges in y dir: -1, +1
                    for (int dx=-1; dx<2; dx++)
                    for (int dy=-1; dy<2; dy++) {
                        if (dx==0 && dy==0) continue;
                        int nx=excX+dx;
                        int ny=excY+dy;
                        if (nx<0 || nx>=excerptSize) continue;
                        if (ny<0 || ny>=excerptSize) continue;

                        edge_array[edge_count]=Edge(excX*excerptSize+excY,nx*excerptSize+ny);
                        T1 pixelDist = (dx==0 || dy==0)?sampling:sampling*std::sqrt(2);
                        weight[edge_count]=(sq(gradMagExcerpt(nx,ny))+pixelDistFactor)*pixelDist;
                        if (weight[edge_count]<0) {
                            std::cout << " weight below 0!! " << std::endl;
                        }
                        edge_count++;

                    }
                }
// done build graph. optimize:
                    //optimize
                graph_t g(num_nodes);
                property_map<graph_t, edge_weight_t>::type weightmap = get(edge_weight, g);
                for (std::size_t j = 0; j < edge_count; ++j) {
                    edge_descriptor e; bool inserted;
                    tie(e, inserted) = add_edge(edge_array[j].first, edge_array[j].second, g);
                    weightmap[e] = weight[j];
                }

                std::vector<vertex_descriptor> p_vec(num_vertices(g));
                std::vector<T1> d_vec(num_vertices(g));
                vertex_descriptor s = vertex(num_nodes/2, g);

                property_map<graph_t, vertex_index_t>::type indexmap = get(vertex_index, g);
                dijkstra_shortest_paths(g, s, &p_vec[0], &d_vec[0], weightmap, indexmap,
                                  std::less<T1>(), closed_plus<T1>(),
                                  (std::numeric_limits<T1>::max)(), 0,
                                  default_dijkstra_visitor());

                //write result into geoDistOfPixel

                for (int excX=0; excX<excerptSize; excX++)
                for (int excY=0; excY<excerptSize; excY++) {
                    geoDistOfPixel(excX,excY)=d_vec[excX*excerptSize+excY];
                }

                if (x==43 && y==150)
                   exportImage(geoDistOfPixel, "/Users/kainmull/testGeoDist.tif");
 //               return;

                float ZnoBkg=0;
                for (int c=1; c<num_classes; ++c){

                    // Q= geoDist+MaskImgStack
                    MultiArray<2, T1> maskImgExcerpt = (MaskImgStack.bindOuter(c)).subarray(Shape2(x-excerptRadius,y-excerptRadius), Shape2(x+excerptRadius+1,y+excerptRadius+1));
                    MultiArray<2, T1> Qimg = geoDistOfPixel + maskImgExcerpt;
                    if (x==43 && y==150 && c==7) {
                       exportImage(Qimg, "/Users/kainmull/testQImg.tif");
                       exportImage(maskImgExcerpt, "/Users/kainmull/testMaskImg.tif");
                    }
                    T1 minQ[1];
                    T1 maxQ[1];
                    Qimg.minmax(minQ,maxQ);

                    T1 pNew=probStack(x,y,c)*exp(-1.0*sq(minQ[0])/sigma2 );
                    //T1 pNew=exp(-1.0*sq(minQ[0])/sigma2 );
                    // prob_new(class,p) = prob(class,p)*e-(Q/sigma)square
//                    std::cout << "c p pNew " << c << " " << probStack(x,y,c) << " " << pNew << std::endl;
                    smoothProbStack(x,y,c) = pNew;
                    ZnoBkg+=pNew;

                }
                //normalize classes...
                for (int c=1; c<num_classes; ++c){
                    smoothProbStack(x,y,c) /= ZnoBkg;
                    smoothProbStack(x,y,c) *= (1-smoothProbStack(x,y,0));
                }
            }
        }
    }

    template <class T1>
    static void AAM_Inference(const MultiArray<3, T1> & probStack, const MultiArray<3, T1> & rawImage, MultiArray<3, T1> & margProbStack, MultiArray<2, int> & MAPLabels,  MultiArray<1, double> priorStrength, int numOffsets=5, double offsetScale=1.0, int sampling=1, int numGDsteps=50, float lambdaU=4, float lambdaPW=4)
    {

        // useful constants
        int num_classes = probStack.size(2);

        // convert to Matlab types ------------------------------------------------------------------------------>

        // convert parameters to Matlab types
        mwArray mwPriorStrength(1, priorStrength.size(), mxDOUBLE_CLASS);
        for (int i = 0; i < priorStrength.size(); ++i)
            mwPriorStrength(i+1) = mwArray(priorStrength(i));

        mwArray mwNumOffsets;
        mwNumOffsets = mwArray(numOffsets);

        mwArray mwOffsetScale(1, 1, mxDOUBLE_CLASS);
        mwOffsetScale = mwArray(offsetScale);

        mwArray mwSampling;
        mwSampling = mwArray(sampling);

        mwArray mwNumGDsteps;
        mwNumGDsteps = mwArray(numGDsteps);

        mwArray mwLambdaU;
        mwLambdaU = mwArray(lambdaU);

        mwArray mwLambdaPW;
        mwLambdaPW = mwArray(lambdaPW);

        mwArray mwProbStackShape(1, 3, mxDOUBLE_CLASS);
        mwProbStackShape(1) = mwArray(static_cast<double>(probStack.size(1)));      // flip xy for Matlab
        mwProbStackShape(2) = mwArray(static_cast<double>(probStack.size(0)));
        mwProbStackShape(3) = mwArray(static_cast<double>(probStack.size(2)));

        mwArray mwRawImageShape(1, 2, mxDOUBLE_CLASS);
        mwRawImageShape(1) = mwArray(static_cast<double>(rawImage.size(1)));        // these are flipped for Matlab
        mwRawImageShape(2) = mwArray(static_cast<double>(rawImage.size(0)));

        // convert multiArray to mwArray for Matlab
        mwArray mwProbs(1, probStack.size(), mxSINGLE_CLASS);
        vigraToMatlab_Array<T1>(probStack, mwProbs);

        mwArray mwRawImage(1, rawImage.size(), mxSINGLE_CLASS);
        vigraToMatlab_Array<T1>(rawImage, mwRawImage);

        // initialize arrays for output
        mwArray mwUnaryFactors(1, (num_classes-1)*numOffsets, mxDOUBLE_CLASS);
        mwArray mwPairwiseFactors(1, numOffsets*numOffsets*(num_classes-1), mxDOUBLE_CLASS);
        mwArray mwFitMasks(1, probStack.size(0)*probStack.size(1)*(num_classes-1)*numOffsets, mxINT8_CLASS);

        // call MBS routine ----------------------------------------------------------------------------------------------->

        MBS_AAM_forINF(3, mwUnaryFactors, mwPairwiseFactors, mwFitMasks, mwRawImage, mwRawImageShape, mwProbs, mwProbStackShape, mwSampling, mwNumGDsteps, mwPriorStrength, mwNumOffsets, mwOffsetScale, mwLambdaU, mwLambdaPW);

        // convert from Matlab types ---------------------------------------------------------------------------->

        // get data out of mwArrays
        MultiArray<3, int> fitMasks(Shape3(probStack.size(0), probStack.size(1), (num_classes-1)*numOffsets));
        matlabToVigra_Array<int>(mwFitMasks, fitMasks);

        MultiArray<2, double> unaryFactors(Shape2(num_classes-1, numOffsets));
        matlabToVigra_UnaryFactors(mwUnaryFactors, unaryFactors);

        MultiArray<3, double> pairwiseFactors(Shape3(numOffsets, numOffsets, num_classes-1));
        matlabToVigra_PairwiseFactors(mwPairwiseFactors, pairwiseFactors);

        // calc marginal and MAP solutions, and return corresponding probability and label images ---------------------------------------------------------------------------->

        margProbStack.reshape(probStack.shape());
        probFromFits<double, T1>(unaryFactors, pairwiseFactors, fitMasks, margProbStack);

        MAPLabels.reshape(Shape2(probStack.size(0), probStack.size(1)));
        MAPFromFits<double>(unaryFactors, pairwiseFactors, fitMasks, MAPLabels);

    }

    template <class T1>
    static void AAM_Inference_2inits(const MultiArray<3, T1> & probStack, const MultiArray<3, T1> & rawImage, MultiArray<3, T1> & margProbStack, MultiArray<2, int> & MAPLabels, MultiArray<1, double> priorStrength, int numOffsets=5, double offsetScale=1.0, int sampling=1, int numGDsteps=50, float lambdaU=4, float lambdaPW=4)
    {

        // useful constants
        int num_classes = probStack.size(2);

        // convert to Matlab types ------------------------------------------------------------------------------>

        // convert parameters to Matlab types
        mwArray mwPriorStrength(1, priorStrength.size(), mxDOUBLE_CLASS);
        for (int i = 0; i < priorStrength.size(); ++i)
            mwPriorStrength(i+1) = mwArray(priorStrength(i));

        mwArray mwNumOffsets;
        mwNumOffsets = mwArray(numOffsets);

        mwArray mwOffsetScale(1, 1, mxDOUBLE_CLASS);
        mwOffsetScale = mwArray(offsetScale);

        mwArray mwSampling;
        mwSampling = mwArray(sampling);

        mwArray mwNumGDsteps;
        mwNumGDsteps = mwArray(numGDsteps);

        mwArray mwLambdaU;
        mwLambdaU = mwArray(lambdaU);

        mwArray mwLambdaPW;
        mwLambdaPW = mwArray(lambdaPW);

        mwArray mwProbStackShape(1, 3, mxDOUBLE_CLASS);
        mwProbStackShape(1) = mwArray(static_cast<double>(probStack.size(1)));      // flip xy for Matlab
        mwProbStackShape(2) = mwArray(static_cast<double>(probStack.size(0)));
        mwProbStackShape(3) = mwArray(static_cast<double>(probStack.size(2)));

        mwArray mwRawImageShape(1, 2, mxDOUBLE_CLASS);
        mwRawImageShape(1) = mwArray(static_cast<double>(rawImage.size(1)));        // these are flipped for Matlab
        mwRawImageShape(2) = mwArray(static_cast<double>(rawImage.size(0)));

        // convert multiArray to mwArray for Matlab
        mwArray mwProbs(1, probStack.size(), mxSINGLE_CLASS);
        vigraToMatlab_Array<T1>(probStack, mwProbs);

        mwArray mwRawImage(1, rawImage.size(), mxSINGLE_CLASS);
        vigraToMatlab_Array<T1>(rawImage, mwRawImage);

        // initialize arrays for output
        mwArray mwUnaryFactors(1, (num_classes-1)*2*numOffsets, mxDOUBLE_CLASS);
        mwArray mwPairwiseFactors(1, 4*numOffsets*numOffsets*(num_classes-1), mxDOUBLE_CLASS);
        mwArray mwFitMasks(1, probStack.size(0)*probStack.size(1)*(num_classes-1)*2*numOffsets, mxINT8_CLASS);

        // call MBS routine ----------------------------------------------------------------------------------------------->

        AAM_Inf_2inits(3, mwUnaryFactors, mwPairwiseFactors, mwFitMasks, mwRawImage, mwRawImageShape, mwProbs, mwProbStackShape, mwSampling, mwNumGDsteps, mwPriorStrength, mwNumOffsets, mwOffsetScale, mwLambdaU, mwLambdaPW);

        // convert from Matlab types ---------------------------------------------------------------------------->

        // get data out of mwArrays
        MultiArray<3, int> fitMasks(Shape3(probStack.size(0), probStack.size(1), (num_classes-1)*2*numOffsets));
        matlabToVigra_Array<int>(mwFitMasks, fitMasks);

        MultiArray<2, double> unaryFactors(Shape2(num_classes-1, 2*numOffsets));
        matlabToVigra_UnaryFactors(mwUnaryFactors, unaryFactors);

        MultiArray<3, double> pairwiseFactors(Shape3(2*numOffsets, 2*numOffsets, num_classes-1));
        matlabToVigra_PairwiseFactors(mwPairwiseFactors, pairwiseFactors);

        // calc marginal and MAP solutions, and return corresponding probability and label images ---------------------------------------------------------------------------->

        margProbStack.reshape(probStack.shape());
        probFromFits<double, T1>(unaryFactors, pairwiseFactors, fitMasks, margProbStack);

        MAPLabels.reshape(Shape2(probStack.size(0), probStack.size(1)));
        MAPFromFits<double>(unaryFactors, pairwiseFactors, fitMasks, MAPLabels);

    }

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

/*        for(int i = 0; i < allMarginals.size(0); ++i)
        {
            std::cout<< "Variable " << i << " has the following marginal distribution P(x_" << i << ") : ";
            for(int j = 0; j < allMarginals.size(1); ++j)
                std::cout <<allMarginals(i,j) << " ";
            std::cout<<std::endl;
        }
*/
        // renormalize marginals
        for(int i = 0; i < allMarginals.size(0); ++i)
        {
            double sum_Marginals = 0.0;
            for(int j = 0; j < allMarginals.size(1); ++j)
                sum_Marginals += allMarginals(i,j);

//            std::cout<< "Variable " << i << " has the following sum of marginals: " << sum_Marginals << " " << std::endl;

            for(int j = 0; j < allMarginals.size(1); ++j)
                allMarginals(i,j) /= sum_Marginals;
        }

        // output
//        for(int i = 0; i < allMarginals.size(0); ++i)
//        {
//            std::cout<< "Variable " << i << " has the following NORMALIZED marginal distribution P(x_" << i << ") : ";
//            for(int j = 0; j < allMarginals.size(1); ++j)
//                std::cout <<allMarginals(i,j) << " ";
//            std::cout<<std::endl;
//        }

/*        for(int i = 0; i < allMarginals.size(0); ++i)
        {
            std::cout<< "Variable " << i << " has the following SUM marginal distribution P(x_" << i << ") : ";
            double sum_marg = 0.0;
            for(int j = 0; j < allMarginals.size(1); ++j){
                sum_marg += allMarginals(i,j);
            }
            std::cout << sum_marg << std::endl;
        }
*/
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
    static void vigraToMatlab_Array(const MultiArray<3, T1> & vigraArray, mwArray & matlabArray)
    {
        const int num_px = vigraArray.size();

        // create std array and transfer data
        T1* stdArray = new T1[num_px];

        int count = 0;
        for (int k = 0; k < vigraArray.size(2); ++k){
            for (int i = 0; i < vigraArray.size(0); ++i){            // transpose b/c row-major order!
                for (int j = 0; j < vigraArray.size(1); ++j){
                    stdArray[count] = vigraArray(i,j,k);
                    count += 1;
                }
            }
        }

        // create Matlab Array and transfer data
        matlabArray.SetData(stdArray, num_px);

        delete [] stdArray;
    }

    template <class T1>
    static void matlabToVigra_Array(const mwArray & matlabArray, MultiArray<3, T1> & vigraArray)
    {
        const int num_px = matlabArray.NumberOfElements();

        T1* stdArray = new T1[num_px];                     // potentially wastefull to go to float, when final type is T1 (eg, may be int)
        matlabArray.GetData(stdArray, num_px);

        // transfer back to MultiArray
        int count = 0;
        for (int k = 0; k < vigraArray.size(2); ++k){
            for (int i = 0; i < vigraArray.size(0); ++i){           // transpose b/c row-major order!
                for (int j = 0; j < vigraArray.size(1); ++j){
                    vigraArray(i,j,k) = stdArray[count];
                    count += 1;
                }
            }
        }

        delete [] stdArray;
    }

    static void matlabToVigra_UnaryFactors(const mwArray & mwUnaryFactors, MultiArray<2, double> & unaryFactors)
    {
        const int num_px = mwUnaryFactors.NumberOfElements();

        double* stdArray = new double[num_px];
        mwUnaryFactors.GetData(stdArray, num_px);

        // transfer back to MultiArray
        int count = 0;
        for (int f = 0; f < unaryFactors.size(1); ++f){           // transpose b/c row-major order!
            for (int c = 0; c < unaryFactors.size(0); ++c){
                unaryFactors(c,f) = stdArray[count];
                count += 1;
            }
        }
        delete [] stdArray;
    }

    static void matlabToVigra_PairwiseFactors(const mwArray & mwPairwiseFactors, MultiArray<3, double> & pairwiseFactors)
    {
        const int num_px = mwPairwiseFactors.NumberOfElements();

        double* stdArray = new double[num_px];
        mwPairwiseFactors.GetData(stdArray, num_px);

        // transfer back to MultiArray
        int count = 0;
        for (int c = 0; c < pairwiseFactors.size(2); ++c){           // transpose b/c row-major order!
            for (int fR = 0; fR < pairwiseFactors.size(1); ++fR){
                for (int fL = 0; fL < pairwiseFactors.size(0); ++fL){
                    pairwiseFactors(fL,fR,c) = stdArray[count];
                    count += 1;
                }
            }
        }
        delete [] stdArray;
    }

//    template <class T1>
//    static void vigraToMatlabArray(const MultiArray<3, T1> & probStack, mwArray & mwProbs)
//    {
//        const int num_px = probStack.size();

//        // create std array and transfer data
//        T1* probArray = new float[num_px];

//        int count = 0;
//        for (int k = 0; k < probStack.size(2); ++k){
//            for (int j = 0; j < probStack.size(1); ++j){
//                for (int i = 0; i < probStack.size(0); ++i){
//                    probArray[count] = probStack(i,j,k);
//                    count += 1;
//                }
//            }
//        }

//        // create Matlab Array and transfer data
//        mwProbs.SetData(probArray, num_px);

//        delete [] probArray;
//    }

//    template <class T1>
//    static void matlabToVigraArray(const mwArray & mwSmoothProbs, MultiArray<3, T1> & smoothProbStack)
//    {
//        const int num_px = mwSmoothProbs.NumberOfElements();

//        float* smoothProbArray = new float[num_px];
//        mwSmoothProbs.GetData(smoothProbArray, num_px);

//        // transfer back to MultiArray
//        int count = 0;
//        for (int k = 0; k < smoothProbStack.size(2); ++k){
//            for (int j = 0; j < smoothProbStack.size(1); ++j){
//                for (int i = 0; i < smoothProbStack.size(0); ++i){
//                    smoothProbStack(j,i,k) = smoothProbArray[count];    // transpose b/c row-major order!
//                    count += 1;
//                }
//            }
//        }

//        delete [] smoothProbArray;
//    }

    template <class T1, class T2>
    static void weightedProbMap(const MultiArray<2, T1> & rfFeatures, const MultiArray<2, T2> & rfLabels, RandomForest<float> rf, Shape2 xy_dim, int lambda, int mode, MultiArray<2, float> & predProbs, MultiArray<2, T2> & predLabels)
    {
        //
        int num_samples = rfFeatures.size(0);
//        int num_features = rfFeatures.size(1);
        int num_classes = rf.class_count();
        int num_trees = rf.tree_count();

        //
        predProbs.reshape(Shape2(num_samples, num_classes));
        predLabels.reshape(Shape2(num_samples, 1));

        rf.set_options().image_shape(Shape2(xy_dim[0], xy_dim[1]));

        // do one sample at a time
        for (int i = 0; i < num_samples; ++i)
        {

            // initialize cumulative probs and weights over all trees
            MultiArray<2, float> cumProbs(Shape2(1, num_classes));
            cumProbs.init(0);
            float cumW = 0;

            // loop over trees
            for (int t = 0; t < num_trees; ++t)
            {
                //
                MultiArray<2, float> probs(Shape2(1, num_classes));

                double totalWeight = 0.0;
                ArrayVector<double>::const_iterator weights;
                rf.trees_[t].set_options() = rf.options_;

                weights = rf.trees_[t].predict(rfFeatures, i);

                //update votecount.
                int weighted = rf.trees_[t].options_.predict_weighted_;
                for(int l=0; l<num_classes; ++l)
                {
                    double cur_w = weights[l] * (weighted * (*(weights-1)) + (1-weighted));
                    probs(0, l) += static_cast<float>(cur_w);
                    totalWeight += cur_w;
                }

                // normalize
                for(int l=0; l< num_classes; ++l)
                {
                    probs(0, l) /= detail::RequiresExplicitCast<float>::cast(totalWeight);
                }

                // calculate some fancy weights
                float w;
                switch (mode)
                {
                case 0:
                {
                    w = 1.0;
                }   break;
                case 1:
                {
                    UInt8 treeLabel = 0;
                    rf.ext_param_.to_classlabel(linalg::argMax(probs), treeLabel);
                    w = (rfLabels[i] == treeLabel)? 1.0 + lambda : 1.0;
                }    break;
                case 2:
                {
                    w = 1.0 + lambda * probs(0,rfLabels[i]);
                }    break;
                }

                // accumulate probs
                {
                    using namespace multi_math;
                    cumProbs += w * probs;
                }
                cumW += w;
            }

            // normalize cumulative probabilities
            {
                using namespace multi_math;
                cumProbs /= cumW;
            }
            predProbs.subarray(Shape2(i,0), Shape2(i+1,num_classes)) = cumProbs;
            rf.ext_param_.to_classlabel(linalg::argMax(cumProbs), predLabels[i]);
        }
    }
};

#endif // SMOOTHINGTOOLS_HXX
