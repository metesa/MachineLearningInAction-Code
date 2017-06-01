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
rand_array = random.rand(4, 4)
print("rand_array = random.rand(4, 4) = ")
print(rand_array)

print("=====================================\nStep 03: Convert this array to matrix\n")
rand_mat = mat(rand_array)
print("rand_mat = mat(rand_array) = ")
print(rand_mat)

print("=====================================\nStep 04: Inverse this matrix\n")
inv_mat = rand_mat.I
print("inv_mat = rand_mat.I = ")
print(inv_mat)

print("=====================================\nStep 05: Multiply two matrix\n")
result_mat = rand_mat * inv_mat
print("rand_array = rand_mat * inv_mat = ")
print(result_mat)

print("=====================================\nStep 06: Generate an 4x4 identity matrix\n")
my_eye = eye(4)
print("my_eye = eye(4) = ")
print(my_eye)

print("=====================================\nStep 07: Calculate error between result matrix and identity matrix\n")
error_mat = result_mat - my_eye
print("error_mat = result_mat - my_eye = ")
print(error_mat)
