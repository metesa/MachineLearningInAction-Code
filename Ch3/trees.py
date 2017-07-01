#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 3 - Decision Tree


from math import log


# Chapter 3.1.1 - Information gain (Program List 3-1)
def calculate_shannon_entropy(dataset):
    entry_count = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_entropy = 0.0
    for key in label_counts:
        probably = float(label_counts[key]) / entry_count
        shannon_entropy -= probably * log(probably, 2)
    return shannon_entropy
