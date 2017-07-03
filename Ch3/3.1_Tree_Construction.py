#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 3.1 - 5 - Tree Construction

print("*************************************\n  Machine Learning In Action - Code\n")
print("    Part    : 1   - Classification")
print("    Chapter : 3.1 - Tree Construction")
print("    Index   : 5\n")
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

print("=====================================\nStep 04: Split the dataset\n")
print("split items whose 1st value equals to 1 to new dataset:")
print(trees.split_dataset(my_data, 0, 1))
print("split items whose 1st value equals to 0 to new dataset:")
print(trees.split_dataset(my_data, 0, 0))
print("split items whose 2nd value equals to 1 to new dataset:")
print(trees.split_dataset(my_data, 1, 1))
print("split items whose 2nd value equals to 0 to new dataset:")
print(trees.split_dataset(my_data, 1, 0))

print("=====================================\nStep 05: Calculate the best feature to split the dataset\n")
print("best feature to split the dataset is:")
print(trees.choose_best_feature_to_split(my_data))

print("=====================================\nStep 06: Create tree from dataset\n")
print("create tree:")
print(trees.create_tree(my_data, labels))
