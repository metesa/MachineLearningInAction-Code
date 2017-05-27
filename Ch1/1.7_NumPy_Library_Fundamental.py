#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 1.7 - 1 NumPy Library Fundamental

print("*************************************\n  Machine Learning In Action - Code\n")
print("    Part    : 1   - Classification")
print("    Chapter : 1.7 - NumPy Library Fundamental")
print("    Index   : 1\n")
print("    Page    : 12\n")
print("    By : Troy Lewis\n*************************************\n\n")

print("=====================================\nStep 01: Import numpy library\n")
from numpy import *
print("from numpy import *")

print("=====================================\nStep 02: Generate a 4x4 array\n")
randArray = random.rand(4, 4)
print("randArray = random.rand(4, 4) = ")
print(randArray)

print("=====================================\nStep 03: Convert this array to matrix\n")
randMat = mat(randArray)
print("randMat = mat(randArray) = ")
print(randMat)

print("=====================================\nStep 04: Inverse this matrix\n")
invMat = randMat.I
print("invMat = randMat.I = ")
print(invMat)

print("=====================================\nStep 05: Multiply two matrix\n")
resultMat = randMat * invMat
print("randArray = randMat * invMat = ")
print(resultMat)

print("=====================================\nStep 06: Generate an 4x4 identity matrix\n")
myEye = eye(4)
print("myEye = eye(4) = ")
print(myEye)

print("=====================================\nStep 07: Calculate error between result matrix and identity matrix\n")
errorMat = resultMat - myEye
print("errorMat = resultMat - myEye = ")
print(errorMat)
