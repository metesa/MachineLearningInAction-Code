#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 2.1.1 - 2 Preparation: Import Data with Python

print("*************************************\n  Machine Learning In Action - Code\n")
print("    Part    : 1     - Classification")
print("    Chapter : 2.1.1 - Preparation: Import Data with Python")
print("    Index   : 2\n")
print("    Page    : 17\n")
print("    By : Troy Lewis\n*************************************\n\n")

print("=====================================\nStep 01: Import kNN.py and create dataset\n")
import kNN
group, labels = kNN.create_dataset()

print("=====================================\nStep 02: Check value of imported dataset\n")
print("group = ")
print(group)
print("labels = ")
print(labels)
