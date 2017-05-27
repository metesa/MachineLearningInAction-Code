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
