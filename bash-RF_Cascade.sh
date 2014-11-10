#!/bin/sh

#  bash-RF_Cascade-Try.sh
#  
#
#  Created by kainmull on 05/11/14.
#
# RF_Cascade input parameters:

args=( "$*" )
numArgs=$(echo "$*" | wc -w)

if [ $numArgs -eq 0 ]
then
    experimentID=201411072252_oldDataCopy-testGeoSmooth
else
    experimentID=$1
fi
echo $experimentID

#############  Training: #############

numImagesTrain=12
numTrees=20
featureMix_features=1
featureMix_offsetFeatures=5
featureMix_offsetDifferenceFeatures=5
treeDepth=10
splitNodeSize=10
sampleFraction=0.33
numAAMsteps=50
useAllImagesAtEveryLevel=1

resultFolderName=Results
resultPathTrain=$resultFolderName/Train$experimentID
baseTrainInputPath=/Users/kainmull/Data/Somites_old/
baseTrainResultPath=/Users/kainmull/Data/Somites_old/
rawPathTrain=RawImages/Train
featurePathTrain=Features/Train
labelPathTrain=Labels/Train
AAMdataPath=/Users/richmond/Data/gtSomites/dataForModels/

useExistingForest=0
numLevels=1 ####!!!!
reSampleBy=3
numClasses=22
maxOffset=128
howToSmoothProbMaps=4
priorA1=0.001
priorA2=0.001
priorX=0.0001
priorY=0.0001
priorShape=1
priorAppearance=0
numOffsets=5
offsetScale=1.0

# model parameters
marginType=1
numP=1
numLambda=1

# tree weighting:
weightedProbsMode=2
weightedProbsLambda0=4
weightedProbsLambda1=4
weightedProbsLambda2=4
useWeightedRFsmoothing=0

############# Testing: #############

numImagesTest=10
resampleByTest=$reSampleBy ###!!!

resultPathTest=$resultFolderName/Test$experimentID/
baseTestInputPath=/Users/kainmull/Data/Somites_old/
baseTestResultPath=/Users/kainmull/Data/Somites_old/
rawPathTest=RawImages/Test
featurePathTest=Features/Test
labelPathTest=Labels/Test
rfName=rf_cascade

############# Eval: #############

gtPath=$baseTestInputPath$labelPathTest


#############  #############  #############

echo creating dir $baseTrainResultPath$resultPathTrain
mkdir $baseTrainResultPath$resultFolderName
mkdir $baseTrainResultPath$resultPathTrain

logfile=$baseTrainResultPath$resultPathTrain/log.txt
exec > $logfile 2>&1

echo " "
echo copying script to learn output folder...
echo " "
me=`basename $0`
cp ./$me $baseTrainResultPath$resultPathTrain

export DYLD_LIBRARY_PATH=./supporting_code/MBS_dylib

echo " "
echo running Learn
echo " "
echo ./RF_Cascade_wMBS_Learn-build/RF_Cascade_wMBS_Learn $baseTrainInputPath $baseTrainResultPath $rawPathTrain $featurePathTrain $labelPathTrain $resultPathTrain $useExistingForest $numImagesTrain $numLevels $reSampleBy $numClasses $numTrees $featureMix_features $featureMix_offsetFeatures $featureMix_offsetDifferenceFeatures $maxOffset $treeDepth $splitNodeSize $howToSmoothProbMaps $sampleFraction $numAAMsteps $useAllImagesAtEveryLevel $priorA1 $priorA2 $priorX $priorY $priorShape $priorAppearance $numOffsets $offsetScale $AAMdataPath $marginType $numP $numLambda $weightedProbsMode $weightedProbsLambda0 $weightedProbsLambda1 $weightedProbsLambda2 $useWeightedRFsmoothing
echo " "

./RF_Cascade_wMBS_Learn-build/RF_Cascade_wMBS_Learn $baseTrainInputPath $baseTrainResultPath $rawPathTrain $featurePathTrain $labelPathTrain $resultPathTrain $useExistingForest $numImagesTrain $numLevels $reSampleBy $numClasses $numTrees $featureMix_features $featureMix_offsetFeatures $featureMix_offsetDifferenceFeatures $maxOffset $treeDepth $splitNodeSize $howToSmoothProbMaps $sampleFraction $numAAMsteps $useAllImagesAtEveryLevel $priorA1 $priorA2 $priorX $priorY $priorShape $priorAppearance $numOffsets $offsetScale $AAMdataPath $marginType $numP $numLambda $weightedProbsMode $weightedProbsLambda0 $weightedProbsLambda1 $weightedProbsLambda2 $useWeightedRFsmoothing

echo creating dir $baseTestResultPath$resultPathTest
echo " "
mkdir $baseTestResultPath$resultFolderName
mkdir $baseTestResultPath$resultPathTest

logfile=$baseTestResultPath$resultPathTest/log.txt
exec > $logfile 2>&1

echo " "
echo copying script to predict output folder...
echo " "
cp ./$me $baseTestResultPath$resultPathTest

echo " "
echo cp $baseTrainResultPath$resultPathTrain/$rfName $baseTestResultPath/$resultPathTest$rfName
cp $baseTrainResultPath$resultPathTrain/$rfName $baseTestResultPath/$resultPathTest$rfName

echo " "
echo cp $baseTrainResultPath$resultPathTrain/$modelForSmoothing_forTest $baseTestResultPath/$resultPathTest$modelForSmoothing
cp $baseTrainResultPath$resultPathTrain/$modelForSmoothing_forTest $baseTestResultPath/$resultPathTest$modelForSmoothing

echo " "
echo running Predict
echo " "
echo ./RF_Cascade_wMBS_Predict-build/RF_Cascade_wMBS_Predict $baseTestInputPath $baseTestResultPath $rawPathTest $featurePathTest $labelPathTest $resultPathTest $rfName $numImagesTest $resampleByTest $howToSmoothProbMaps $numAAMsteps $priorA1 $priorA2 $priorX $priorY $priorShape $priorAppearance $numOffsets $offsetScale $weightedProbsMode $weightedProbsLambda0 $weightedProbsLambda1 $weightedProbsLambda2 $useWeightedRFsmoothing
echo " "

./RF_Cascade_wMBS_Predict-build/RF_Cascade_wMBS_Predict $baseTestInputPath $baseTestResultPath $rawPathTest $featurePathTest $labelPathTest $resultPathTest $rfName $numImagesTest $resampleByTest $howToSmoothProbMaps $numAAMsteps $priorA1 $priorA2 $priorX $priorY $priorShape $priorAppearance $numOffsets $offsetScale $weightedProbsMode $weightedProbsLambda0 $weightedProbsLambda1 $weightedProbsLambda2 $useWeightedRFsmoothing

evalRF=evalRF/
mkdir $baseTestResultPath$resultPathTest$evalRF
image="image#"
for name in $baseTestResultPath$resultPathTest$image*.tif
do
    cp $name $baseTestResultPath$resultPathTest$evalRF
done

evalSmooth=evalSmooth/
mkdir $baseTestResultPath$resultPathTest$evalSmooth
imageSmooth="image_smooth"
for name in $baseTestResultPath$resultPathTest$imageSmooth*.tif
do
cp $name $baseTestResultPath$resultPathTest$evalSmooth
done

echo " "
echo running Dice
echo " "
echo ./DiceScore-build/DiceScore $gtPath $baseTestResultPath $resultPathTest$evalRF $numLevels $numImagesTest $numClasses
echo " "
./DiceScore-build/DiceScore $gtPath $baseTestResultPath $resultPathTest$evalRF $numLevels $numImagesTest $numClasses

echo " "
echo running Rscript
echo " "
echo Rscript ./supporting_code/boxplotsInR.R $baseTestResultPath$resultPathTest$evalRF
echo " "
Rscript ./supporting_code/boxplotsInR.R $baseTestResultPath$resultPathTest$evalRF

echo " "
echo running Dice
echo " "
echo ./DiceScore-build/DiceScore $gtPath $baseTestResultPath $resultPathTest$evalSmooth $numLevels $numImagesTest $numClasses
echo " "
./DiceScore-build/DiceScore $gtPath $baseTestResultPath $resultPathTest$evalSmooth $numLevels $numImagesTest $numClasses

echo " "
echo running Rscript
echo " "
echo Rscript ./supporting_code/boxplotsInR.R $baseTestResultPath$resultPathTest$evalSmooth
echo " "
Rscript ./supporting_code/boxplotsInR.R $baseTestResultPath$resultPathTest$evalSmooth

