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


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


# Chapter 3.1.2 - Split Dataset (Program List 3-2)
def split_dataset(dataset, axis, value):
    # create a new dataset to avoid modifying the original dataset
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            # get values in range 0 to (axis - 1)
            reduced_feat_vec = feat_vec[:axis]
            # a = [1, 2, 3]    b = [4, 5, 6]
            # a.append(b) ==> a = [1, 2, 3, [4, 5, 6]]
            # a.extend(b) ==> a = [1, 2, 3, 4, 5, 6]
            # extend values in range (axis + 1) to end
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


# Chapter 3.1.2 - Split Dataset (Program List 3-3)
def choose_best_feature_to_split(dataset):
    # one is result and others are features
    feature_count = len(dataset[0]) - 1
    # original entropy
    base_entropy = calculate_shannon_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(feature_count):
        # make a new list with the i th column data
        feature_list = [item[i] for item in dataset]
        # convert to set to remove duplicated item
        unique_values = set(feature_list)
        new_entropy = 0.0
        # calculate entropy under this split
        for value in unique_values:
            # split dataset based on selected feature and certain value
            sub_dataset = split_dataset(dataset, i, value)
            probably = len(sub_dataset) / float(len(dataset))
            new_entropy += probably * calculate_shannon_entropy(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature
