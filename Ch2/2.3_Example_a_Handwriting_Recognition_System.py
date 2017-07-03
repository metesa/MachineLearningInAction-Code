#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 2.3 - 4 - Example: a Handwriting Recognition System


print("*************************************\n  Machine Learning In Action - Code\n")
print("    Part    : 1   - Classification")
print("    Chapter : 2.3 - Example: a Handwriting Recognition System")
print("    Index   : 4\n")
print("    Page    : 28\n")
print("    By : Troy Lewis\n*************************************\n\n")

print("=====================================\nStep 01: Import kNN.py and test img2vector\n")
import kNN
test_vector = kNN.img2vector('digits/testDigits/0_13.txt')
print "array1 = "
print test_vector[0, 0:32]
print "array2 = "
print test_vector[0, 32:64]

print("=====================================\nStep 02: Test the classifier\n")
print "run the classifier: "
kNN.handwriting_test()
