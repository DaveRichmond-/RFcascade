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
    experimentID=201411061101
else
    experimentID=$1
fi
echo $experimentID

#############  Training: #############

numImagesTrain=6
numTrees=20
featureMix_features=1
featureMix_offsetFeatures=5
featureMix_offsetDifferenceFeatures=5
treeDepth=15
splitNodeSize=10
sampleFraction=0.05
numAAMsteps=5
useAllImagesAtEveryLevel=1

resultFolderName=Results
resultPathTrain=$resultFolderName/Train$experimentID
baseTrainInputPath=/Users/kainmull/Data/Somites/
baseTrainResultPath=/Users/kainmull/Data/Somites/
rawPathTrain=RawImages/Train
featurePathTrain=Features/Train
labelPathTrain=Labels/Train
useExistingForest=0
numLevels=2
reSampleBy=3
numClasses=22
maxOffset=128
howToSmoothProbMaps=3
priorA1=0.001
priorA2=0.001
priorX=0.0001
priorY=0.0001
priorShape=1
priorAppearance=0
numOffsets=5
offsetScale=1.0

############# Testing: #############

numImagesTest=1
resampleByTest=$reSampleBy

resultPathTest=$resultFolderName/Test$experimentID/
baseTestInputPath=/Users/kainmull/Data/Somites/
baseTestResultPath=/Users/kainmull/Data/Somites/
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
echo ./RF_Cascade_wMBS_Learn-build/RF_Cascade_wMBS_Learn $baseTrainInputPath $baseTrainResultPath $rawPathTrain $featurePathTrain $labelPathTrain $resultPathTrain $useExistingForest $numImagesTrain $numLevels $reSampleBy $numClasses $numTrees $featureMix_features $featureMix_offsetFeatures $featureMix_offsetDifferenceFeatures $maxOffset $treeDepth $splitNodeSize $howToSmoothProbMaps $sampleFraction $numAAMsteps $useAllImagesAtEveryLevel $priorA1 $priorA2 $priorX $priorY $priorShape $priorAppearance $numOffsets $offsetScale
echo " "

./RF_Cascade_wMBS_Learn-build/RF_Cascade_wMBS_Learn $baseTrainInputPath $baseTrainResultPath $rawPathTrain $featurePathTrain $labelPathTrain $resultPathTrain $useExistingForest $numImagesTrain $numLevels $reSampleBy $numClasses $numTrees $featureMix_features $featureMix_offsetFeatures $featureMix_offsetDifferenceFeatures $maxOffset $treeDepth $splitNodeSize $howToSmoothProbMaps $sampleFraction $numAAMsteps $useAllImagesAtEveryLevel $priorA1 $priorA2 $priorX $priorY $priorShape $priorAppearance $numOffsets $offsetScale

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
echo running Predict
echo " "
echo ./RF_Cascade_wMBS_Predict-build/RF_Cascade_wMBS_Predict $baseTestInputPath $baseTestResultPath $rawPathTest $featurePathTest $labelPathTest $resultPathTest $rfName $numImagesTest $resampleByTest $howToSmoothProbMaps $numAAMsteps $priorA1 $priorA2 $priorX $priorY $priorShape $priorAppearance $numOffsets $offsetScale
echo " "

./RF_Cascade_wMBS_Predict-build/RF_Cascade_wMBS_Predict $baseTestInputPath $baseTestResultPath $rawPathTest $featurePathTest $labelPathTest $resultPathTest $rfName $numImagesTest $resampleByTest $howToSmoothProbMaps $numAAMsteps $priorA1 $priorA2 $priorX $priorY $priorShape $priorAppearance $numOffsets $offsetScale

echo " "
echo running Dice
echo " "
echo ./DiceScore-build/DiceScore $gtPath $baseTestResultPath $resultPathTest $numLevels $numImagesTest $numClasses
echo " "

./DiceScore-build/DiceScore $gtPath $baseTestResultPath $resultPathTest $numLevels $numImagesTest $numClasses

Rscript ./supporting_code/boxplotsInR.R $baseTestResultPath$resultPathTest

