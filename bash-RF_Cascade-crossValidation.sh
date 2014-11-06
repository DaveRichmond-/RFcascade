#!/bin/sh

#  bash-RF_Cascade-Try.sh
#  
#
#  Created by kainmull on 05/11/14.
#
# RF_Cascade input parameters:

# for name in "Focused "*; do dummy=($name); mv "Focused "${dummy[1]} Focused_${dummy[1]}; done
# for name in "feature-stack_Focused "*; do dummy=($name); mv "feature-stack_Focused "${dummy[1]} feature-stack_Focused_${dummy[1]}; done

experimentID=201411061810
batchScript=bash-RF_Cascade-my2.sh

# training data: "all"
# _1.tif (orig), and 5 random rotations

# test data: "only_originals"

# hide the ones that we do not want for a fold in "dummy"

# append fold to experimentID

numFolds=4
numRotations=2 # = orig plus num-1
maxRotations=6

baseAllTrainInputPath=/Users/kainmull/Data/Somites/all/
rawPathTrain=grayscale
featurePathTrain=features
labelPathTrain=labels

baseAllTestInputPath=/Users/kainmull/Data/Somites/only_originals/
rawPathTest=grayscale
featurePathTest=features
labelPathTest=labels

# recover all data from dummy folders:
dummyRot=dummyRot
dummy=dummy
us="_"
for somedir in $rawPathTrain $featurePathTrain $labelPathTrain
do
    somepath=$baseAllTrainInputPath$somedir
    # recover all training data from dummy folders
    for filename in $( ls $somepath/$dummy/*.tif | xargs -n1 basename )
    do
        mv $somepath/$dummy/$filename $somepath/$filename
    done
    for filename in $( ls $somepath/$dummyRot/*.tif | xargs -n1 basename )
    do
        mv $somepath/$dummyRot/$filename $somepath/$filename
    done
done
# test data:
for somedir in $rawPathTest $featurePathTest $labelPathTest
do
    somepath=$baseAllTestInputPath$somedir
    # recover all test data from dummy folders
    for filename in $( ls -p $somepath/$dummy/*.tif | xargs -n1 basename )
    do
        mv $somepath/$dummy/$filename $somepath/$filename
    done
done

# put away additional rotations from training data:
for somedir in $rawPathTrain $featurePathTrain $labelPathTrain
#for somedir in $rawPathTrain
do
    somepath=$baseAllTrainInputPath$somedir
    mkdir $somepath/$dummyRot
    mkdir $somepath/$dummy # for later
    let firstRot=$numRotations+1
    for i in $( seq $firstRot $maxRotations )
    do
        for filename in $( ls $somepath/*$us$i.tif | xargs -n1 basename )
        do
            mv $somepath/$filename $somepath/$dummyRot/$filename
        done
    done
done

#prepare test paths:
for somedir in $rawPathTest $featurePathTest $labelPathTest
do
    somepath=$baseAllTestInputPath$somedir
    mkdir $somepath/$dummy
done

somepath=$baseAllTrainInputPath$rawPathTrain
numTrainingImages=$(($(ls -p $somepath | grep -v / | wc -w )))

let trainingChunksize=$numTrainingImages/$numFolds

somepath=$baseAllTestInputPath$rawPathTest
numTestImages=$(($(ls -p $somepath | grep -v / | wc -w )))

let testChunksize=$(($numTestImages/$numFolds))

# run the folds:
for fold in $( seq 1 $numFolds )
do
    # set experiment id
    experimentIDfold=$experimentID$us$fold

    # prepare folder of training data

    let foldM1=-1+$fold
    let firstIdx=$foldM1*$trainingChunksize
    let lastIdxP1=$fold*$trainingChunksize
    for somedir in $rawPathTrain $featurePathTrain $labelPathTrain
    do
        somepath=$baseAllTrainInputPath$somedir

        # recover all training data from dummy folders
        for filename in $( ls -p $somepath/$dummy/*.tif | xargs -n1 basename )
        do
            mv $somepath/$dummy/$filename $somepath/$filename
        done

        # make complete list of training data
        trainingArr=($( ls -p $somepath/*.tif | xargs -n1 basename ))

        # move training data NOT to be used in this fold away
        let i=$firstIdx
        while [ $i -lt $lastIdxP1 ]
        do
            name=${trainingArr[$i]}
            echo moving away training data $i fold $fold
            mv $baseAllTrainInputPath$somedir/$name $baseAllTrainInputPath$somedir/$dummy/$name
            i=$[$i+1]
        done
    done

# prepare folder of test data

    let firstIdx=$foldM1*$testChunksize
    let lastIdxP1=$fold*$testChunksize
    for somedir in $rawPathTest $featurePathTest $labelPathTest
    do
        somepath=$baseAllTestInputPath$somedir

        # recover all test data from dummy folders
        for filename in $( ls -p $somepath/$dummy/*.tif | xargs -n1 basename )
        do
            mv $somepath/$dummy/$filename $somepath/$filename
        done

        # make complete list of test data
        testArr=($( ls -p $somepath/*.tif | xargs -n1 basename ))

        # move test data NOT to be used in this fold away
        let i=0
        while [ $i -lt $firstIdx ]
        do
            name=${testArr[$i]}
            echo moving away $i fold $fold
            mv $baseAllTestInputPath$somedir/$name $baseAllTestInputPath$somedir/$dummy/$name
            i=$[$i+1]
        done
        i=$lastIdxP1
        while [ $i -lt $numTestImages ]
        do
            echo moving away test data $i fold $fold
            name=${testArr[$i]}
            mv $baseAllTestInputPath$somedir/$name $baseAllTestInputPath$somedir/$dummy/$name
            i=$[$i+1]
        done
    done

# run batch for fold

sh ./$batchScript $experimentIDfold &

# merge dice etc

done


