#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 2.2 - 4 Project: Improve Recommendation for Dating Website

print("*************************************\n  Machine Learning In Action - Code\n")
print("    Part    : 1   - Classification")
print("    Chapter : 2.2 - Project: Improve Recommendation for Dating Website")
print("    Index   : 4\n")
print("    Page    : 20\n")
print("    By : Troy Lewis\n*************************************\n\n")

print("=====================================\nStep 01: Import kNN.py and create dataset\n")
# "reload(kNN)" instead of "import kNN" if kNN.py changes
import kNN
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')

print("=====================================\nStep 02: Check value of imported dataset\n")
print("datingDataMat = ")
print(datingDataMat)
print("datingLabels = ")
print(datingLabels)

print("=====================================\nStep 03: Classify with kNN algorithm\n")
print("result = ")
print(kNN.classify0([0, 0], group, labels, 3))
