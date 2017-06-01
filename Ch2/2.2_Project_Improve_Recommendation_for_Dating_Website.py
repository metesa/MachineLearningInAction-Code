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

dating_data_mat, dating_labels = kNN.file2matrix('datingTestSet2.txt')

print("=====================================\nStep 02: Check value of imported dataset\n")
print("dating_data_mat = ")
print(dating_data_mat)
print("\ndatingLabels = ")
print(dating_labels)

print("=====================================\nStep 03: Draw data plot with Matplotlib\n")
import matplotlib.pyplot as plt

# create a figure
fig = plt.figure()
# get the above part of the figure
#     211 means divided the figure to 2x1 pieces and ax stands for the 1st area
#     for 3,4,10, you can't use 3410, only (3, 4, 10) will do
#     add_subplot(111)
#         ax = fig.add_subplot(111)
#     subplot(111)
#         fig.subplot(111)
#         ax = fig.gca() # get current axis
ax = fig.add_subplot(211)
# draw scatter
#     scatter(x,y,c=T,s=25,alpha=0.4,marker='o')
#         x,y are Data
#             dating_data_mat[:, 1] means the amount of bought ice cream per week
#             dating_data_mat[:, 2] means the percentage of video game time
#         c is Color
#         s is Size
#         alpha is Transparency
#         marker is point shape
ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2])
# get the below part of the figure
ax = fig.add_subplot(212)
# draw the scatter with specific form
ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1], 15.0 * kNN.array(dating_labels),
           15.0 * kNN.array(dating_labels))
# show the figure and pause the program
plt.show()

print("=====================================\nStep 04: AutoNorm data\n")
norm_mat, ranges, min_vals = kNN.auto_normalization(dating_data_mat)
print("norm_mat = ")
print(norm_mat)
print("\nranges = ")
print(ranges)
print("\nminVals = ")
print(min_vals)

print("=====================================\nStep 05: Test the classifier\n")
kNN.dating_class_test()

print("=====================================\nStep 06: Use the classifier\n")
kNN.classify_person()
