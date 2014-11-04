#ifndef INFERENCETOOLS_HXX
#define INFERENCETOOLS_HXX

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>

#include <math.h>
#include <vigra/multi_math.hxx>

using namespace vigra;
using namespace opengm;

class inferencetools
{

public:

    template <class T1, class T2>
    static void chainProbInference(const MultiArray<2, T1> & unaryFactors, const MultiArray<3, T1> & pairwiseFactors, MultiArray<2, T2> & allMarginals, int output_flag = 0)
    {
        // typedefs
        typedef T1                                                                      ValueType;
        typedef size_t                                                                  IndexType;
        typedef size_t                                                                  LabelType;
        typedef Multiplier                                                              OperationType;
        typedef ExplicitFunction<ValueType, IndexType, LabelType>                       ExplicitFunction;
        typedef typename meta::TypeListGenerator<ExplicitFunction>::type                FunctionTypeList;
        typedef SimpleDiscreteSpace<IndexType, LabelType>                               SpaceType;

        typedef GraphicalModel<ValueType, OperationType, FunctionTypeList, SpaceType>   Model;
        typedef typename Model::FunctionIdentifier                                      FunctionIdentifier;

        typedef BeliefPropagationUpdateRules<Model, Integrator> UpdateRules;
        typedef MessagePassing<Model, Integrator, UpdateRules, MaxDistance> BeliefPropagation;

        //
        const IndexType numVariables = unaryFactors.size(0);
        const LabelType numLabels = unaryFactors.size(1);

        // initialize label space and graphical model
        SpaceType space(numVariables, numLabels);
        Model gm(space);

        // for numerical stability: make all pairwise and unary sum to 1. If any sums to (almost) 0, something is wrong in the first place.
        // this is just a heuristic. the partition function might still be zero.

        // normalization factors for unaries:
        double epsUnary = numLabels*1E-5;
        double epsPairwise = numLabels*numLabels*1E-5;
        double globalFactor=sqrt(numLabels);
        MultiArray<1, T1> unaryFactorSums(numVariables);
        for (IndexType v = 0; v < numVariables; ++v){
            unaryFactorSums(v) = 0;
            for (LabelType s = 0; s < numLabels; ++s){
                unaryFactorSums(v) += unaryFactors(v,s);
            }
            if (unaryFactorSums(v)<epsUnary) {
                std::cout<< "Numerical instability alert: Variable " << v << ": unary factor sums to " << unaryFactorSums(v) << " ";
                std::cout<<std::endl;
            }
            unaryFactorSums(v)/=sqrt(numLabels);
         }

        // normalization factors for pair-wise:
        MultiArray<1, T1> pairwiseFactorSums(numVariables-1);
        for (IndexType v = 0; v < numVariables-1; ++v){
            pairwiseFactorSums(v)=0;
            for (LabelType sL = 0; sL < numLabels; ++sL)
                for (LabelType sR = 0; sR < numLabels; ++sR)
                    pairwiseFactorSums(v) += pairwiseFactors(sL,sR,v);
            if (pairwiseFactorSums(v)<epsPairwise) {
                std::cout<< "Numerical instability alert: Variables " << v << " and " << v+1 << " : pairwise factor sums to " << pairwiseFactorSums(v) << " ";
                std::cout<<std::endl;
            }
            pairwiseFactorSums(v)/=numLabels*sqrt(numLabels);
        }

        // load unary factors into gm
        for (IndexType v = 0; v < numVariables; ++v){
            const LabelType shape[] = {numLabels};
            ExplicitFunction f(shape, shape+1,1); // initialize to 1
            if (unaryFactorSums(v)>=epsUnary) {
                for (LabelType s = 0; s < numLabels; ++s){
                    f(s) = unaryFactors(v,s)/unaryFactorSums(v);
                }
            }
            FunctionIdentifier fid = gm.addFunction(f);
            IndexType variableIndices[] = {v};
            gm.addFactor(fid, variableIndices, variableIndices + 1);
        }

        // load pair-wise factors into gm
        for (IndexType v = 0; v < numVariables-1; ++v){
            const LabelType shape[] = {numLabels, numLabels};
            ExplicitFunction f(shape, shape+2, 1); // initialize to 1
            if (pairwiseFactorSums(v)>=epsPairwise) {
                for (LabelType sL = 0; sL < numLabels; ++sL){
                    for (LabelType sR = 0; sR < numLabels; ++sR){
                        f(sL,sR) = pairwiseFactors(sL,sR,v)/pairwiseFactorSums(v);
                    }
                }
            }
            FunctionIdentifier fid = gm.addFunction(f);
            IndexType variableIndices[] = {v,v+1};
            gm.addFactor(fid, variableIndices, variableIndices + 2);
        }

        // set up the optimizer (belief propagation) for probabilistic inference

        const size_t maxNumberOfIterations = numVariables * 2;
        const double convergenceBound = 0.0000001;
        const double damping = 0.0;
        typename BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
        bool normalization = true;
        parameter.useNormalization_ = normalization;
        BeliefPropagation bp(gm, parameter);

        // optimize (approximately) with output
        if ( output_flag ){
            typename BeliefPropagation::VerboseVisitorType visitor;
            bp.infer(visitor);
        } else
            bp.infer();

        // store marginals for output
        allMarginals.reshape(Shape2(numVariables, numLabels));
        typename Model::IndependentFactorType marginal;
        for(IndexType v = 0; v < gm.numberOfVariables(); ++v)
        {
            bp.marginal(v, marginal);
            for(LabelType s = 0; s < gm.numberOfLabels(v); ++s)
                allMarginals(v,s) = static_cast<T2>(marginal(&s));
        }

    }

    template <class T1, class T2>
    static void chainMAPInference(const MultiArray<2, T1> & unaryFactors, const MultiArray<3, T1> & pairwiseFactors, MultiArray<1, T2> & MAPLabels, int output_flag = 0)
    {
        // typedefs
        typedef T1                                                                      ValueType;
        typedef size_t                                                                  IndexType;
        typedef size_t                                                                  LabelType;
        typedef Adder                                                                   OperationType;                  // Adder is key parameter for MAP inference
        typedef ExplicitFunction<ValueType, IndexType, LabelType>                       ExplicitFunction;
        typedef typename meta::TypeListGenerator<ExplicitFunction>::type                FunctionTypeList;
        typedef SimpleDiscreteSpace<IndexType, LabelType>                               SpaceType;

        typedef GraphicalModel<ValueType, OperationType, FunctionTypeList, SpaceType>   Model;
        typedef typename Model::FunctionIdentifier                                      FunctionIdentifier;

        typedef BeliefPropagationUpdateRules<Model, Minimizer>                          UpdateRules;                   // Minimizer is key parameter for MAP inference
        typedef MessagePassing<Model, Minimizer, UpdateRules, MaxDistance>              BeliefPropagation;             // Minimizer is key parameter for MAP inference


        //
        const IndexType numVariables = unaryFactors.size(0);
        const LabelType numLabels = unaryFactors.size(1);

        // initialize label space and graphical model
        SpaceType space(numVariables, numLabels);
        Model gm(space);

        // CONVERT FACTORS INTO COSTS
        MultiArray<2, T1> unaryCosts(unaryFactors.shape());
        MultiArray<3, T1> pairwiseCosts(pairwiseFactors.shape());
        {
            using namespace multi_math;

            unaryCosts = -log(unaryFactors);
            pairwiseCosts = -log(pairwiseFactors);
        }

        // load unary costs into gm
        for (IndexType v = 0; v < numVariables; ++v){
            const LabelType shape[] = {numLabels};
            ExplicitFunction f(shape, shape+1);
            for (LabelType s = 0; s < numLabels; ++s)
                f(s) = unaryCosts(v,s);
            FunctionIdentifier fid = gm.addFunction(f);         // is it ok that each function has the same variable name (f)?  they should all get a different id, which is used for reference.  seems ok...
            IndexType variableIndices[] = {v};
            gm.addFactor(fid, variableIndices, variableIndices + 1);
        }

        // load pair-wise costs into gm
        for (IndexType v = 0; v < numVariables-1; ++v){
            const LabelType shape[] = {numLabels, numLabels};
            ExplicitFunction f(shape, shape+2);
            for (LabelType sL = 0; sL < numLabels; ++sL)
                for (LabelType sR = 0; sR < numLabels; ++sR)
                    f(sL,sR) = pairwiseCosts(sL,sR,v);
            FunctionIdentifier fid = gm.addFunction(f);
            IndexType variableIndices[] = {v,v+1};
            gm.addFactor(fid, variableIndices, variableIndices + 2);
        }

        // set up the optimizer (belief propagation) for probabilistic inference

        const size_t maxNumberOfIterations = numVariables * 2;
        const double convergenceBound = 0.0000001;
        const double damping = 0.0;
        typename BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
        BeliefPropagation bp(gm, parameter);

        // optimize (approximately) with output
        if ( output_flag ){
            typename BeliefPropagation::VerboseVisitorType visitor;
            bp.infer(visitor);
        } else
            bp.infer();

        // obtain the (approximate) argmin
        std::vector<LabelType> labeling(numVariables);
        bp.arg(labeling);

        // transfer to vigra array for output
        MAPLabels.reshape(Shape1(numVariables));
        for (IndexType v = 0; v < numVariables; ++v)
            MAPLabels(v) = static_cast<T2>(labeling[v]);

    }

};

#endif // INFERENCETOOLS_HXX
