#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 2.1.2 - 3 Run kNN Algorithm

print("*************************************\n  Machine Learning In Action - Code\n")
print("    Part    : 1     - Classification")
print("    Chapter : 2.1.2 - Run kNN Algorithm")
print("    Index   : 3\n")
print("    Page    : 19\n")
print("    By : Troy Lewis\n*************************************\n\n")

print("=====================================\nStep 01: Import kNN.py and create dataset\n")
import kNN
group, labels = kNN.createDataSet()

print("=====================================\nStep 02: Check value of imported dataset\n")
print("group = ")
print(group)
print("labels = ")
print(labels)

print("=====================================\nStep 03: Classify with kNN algorithm\n")
print("result = ")
print(kNN.classify0([0, 0], group, labels, 3))
