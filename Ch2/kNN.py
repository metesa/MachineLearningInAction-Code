#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 2 - kNN Algorithm Module

from numpy import *
import operator

# Chapter 2.1.1 - Preparation: Import Data with Python
def createDataSet():
    # Define an 4x2 array(2-Dimensional)
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # Define the label list
    #     () means tuple
    #     [] means list
    #     {} means dict
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# Chapter 2.1.2 - Run kNN Algorithm (Program List 2-1)
def classify0(inX, dataSet, labels, k):
    # inX     - the unknown input data
    # dataSet - the known data
    # labels  - the real result of the known data
    # k       - pick the closest k records to determine the type of result

    # return  - classified result value

    # Get the row count
    #     NumPy.array.shape returns a tuple like (row count, column count)
    #         dataSet.shape = (4, 2)
    dataSetSize = dataSet.shape[0]

    # Calculate distance
    #     NumPy.tile(a, (r, c)) duplicates the data r rows and c columns
    #         inX = [0, 0]
    #         tile(inX, (4, 1)) = 
    #             [[0, 0], [0, 0], [0, 0], [0, 0]]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #     array**2 means square of each element
    sqDiffMat = diffMat**2
    #     array.sum means summing up specific elements
    #         for a N-dimentional array, axis can be in range(N)
    #         axis = None(default) means all elements
    #         axis = 0 means performing column actions and adding to one row
    #         axis = 1 means performing row actions and adding to one column
    #         Here means adding the dx^2 and dy^2 to the distance^2
    sqDistances = sqDiffMat.sum(axis=1)
    #     array**0.5 means square root of each element, which is the real distance
    distances = sqDistances**0.5

    # Sort
    #     NumPy.argsort() returns a sorted list of indices from small to big
    sortedDistIndicies = distances.argsort()
    
    # Get the best k result
    classCount={}
    for i in range(k):
        #         Get the label
        voteIlabel = labels[sortedDistIndicies[i]]
        #         Update the label count
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #         sort() means sort the original input
    #         sorted() means sort the copy of the input
    #         items() returns the original object
    #         iteritems() returns the iterator object
    #         operator.itemgetter(1) returns the 2nd(count) column of input(label, count)
    #         key can also be specified with lambda(key=lambda x : x[1])
    #         reverse = True means decent order
    sortedClassCount = sorted(classCount.iteritems(),
      key=operator.itemgetter(1), reverse=True)
    # Return the label of the item with the maximum count.
    return sortedClassCount[0][0]

# Chapter 2.2 - Project: Improve Recommendation for Dating Website with k-means algorithm.

# Chapter 2.2.1 - Preparation: Import Data from file
def file2matrix(filename):
    # open file
    fr = open(filename)
    #     read() gets the whole text of a file
    #     readline() gets the next line of a file
    #     readlines() gets the lines array of a file
    arrayOLines = fr.readlines()
    #     get the total line count
    numberOfLines = len(arrayOLines)
    #     create a NumPy Array
    returnMat = zeros((numberOfLines, 3))
    # import data
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        #     remove spaces (including '\n', '\r', '\t', ' ') on margins
        line = line.strip()
        #     split to list elements by '\t'
        #     '\t' means Tab
        listFromLine = line.split('\t')
        #     copy value from line to the specific line of the return Matrix
        returnMat[index,:] = listFromLine[0:3]
        #     convert the last value to integer and append it to the label
        classLabelVector.append(int(listFromLine[-1]))
        #     increase returnMat's line index by 1
        index += 1
    return returnMat, classLabelVector

# Chapter 2.2.3 - Processing: Auto normalization
def autoNorm(dataSet):
    # array.min(axis=0) means the minimum element for each column/feature(axis=1 for each row/record)
    minVals = dataSet.min(0)
    # array.max(axis=0) means the maximum element for each column/feature(axis=1 for each row/record)
    maxVals = dataSet.max(0)
    # range for each column/feature
    ranges = maxVals - minVals
    # create a new array that has the same shape as dataSet with 0
    normDataSet = zeros(shape(dataSet))
    # shape[0] returns the total row/record count
    m = dataSet.shape[0]
    # tile creates a new array by duplicating minVals by m times in row and keeping the original columns
    #     now the normDataSet has no negative values
    normDataSet = dataSet - tile(minVals, (m, 1))
    # tile creates a new array by duplication ranges by m times in row and keeping the original columns
    #     now the elements from norDataSet are in range 0 to 1
    normDataSet = normDataSet / tile(ranges, (m, 1))
    # normDataSet - the normalized dataset
    # ranges      - the range for each column/features
    # minVals     - the minimum values for each column/features
    return normDataSet, ranges, minVals

# Chapter 2.2.4 - Testing: Test the classifier
# Usually we use 90% of the data to train the model and the rest 10% of the data to test the model.
def datingClassTest():
    # ratio percentage for testing
    hoRatio = 0.10
    # load data from file
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # normalize data to range 0 ~ 1
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # get row/record count
    m = normMat.shape[0]
    # get count of the records for testing purpose
    numTestVecs = int(m*hoRatio)
    # get count of bad classification
    errorCount = 0.0
    # test the classifier
    for i in range(numTestVecs):
        # normMat[i,:] means the i'th row/record
        # normMat[numTestVecs:m,:] means all the data that is for training purpose
        # datingLabels[numTestVecs:m] means the real result of training data records
        # 3 means pick the closest 3 records to determine the type of result
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], \
                    datingLabels[numTestVecs:m], 3)
        # display the result for each record
        print "the classifier came back with: %d, the real answer is: %d" \
                    % (classifierResult, datingLabels[i])
        # update the count of bad classification
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    # display the final stats
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))

# Chapter 2.2.5 - Application: Build a complete system
def classifyPerson():
    # List for mapping result values from integers to strings
    resultList = ['not at all', 'in small doses', 'in large doses']
    # get the first value
    percentTats = float(raw_input("percentage of time spent playing video games?\n"))
    # get the second value
    ffMiles = float(raw_input("frequent flier miles earned per year?\n"))
    # get the third value
    iceCream = float(raw_input("liters of ice cream consumed per year?\n"))
    # load data from file
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # normalize data to range 0 ~ 1
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # convert a list to a array for storing input values
    inArr = array([ffMiles, percentTats, iceCream])
    # normalize input values and classify it with k=3
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    # display the result
    print "You will probably like this person: ", resultList[classifierResult - 1]