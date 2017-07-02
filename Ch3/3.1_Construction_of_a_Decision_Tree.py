#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 3.1 - Construction of a Decision Tree

print("*************************************\n  Machine Learning In Action - Code\n")
print("    Part    : 1     - Classification")
print("    Chapter : 3.1 - Construction of a Decision Tree")
print("    Index   : 6\n")
print("    Page    : 36\n")
print("    By : Troy Lewis\n*************************************\n\n")

print("=====================================\nStep 01: Import trees.py and create dataset\n")
import trees
my_data, labels = trees.create_dataset()
print("dataset = ")
print(my_data)

print("=====================================\nStep 02: Calculate Shannon Entropy of that dataset\n")
print("entropy = ")
print(trees.calculate_shannon_entropy(my_data))

print("=====================================\nStep 03: Modify the dataset and Calculate Shannon Entropy again\n")
print("Change the result of the first entry to 'maybe'")
my_data[0][-1] = 'maybe'
print("entropy = ")
print(trees.calculate_shannon_entropy(my_data))
