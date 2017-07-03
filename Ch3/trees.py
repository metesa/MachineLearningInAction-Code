#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 3 - Decision Tree

from math import log
import operator


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
            reduced_feat_vec.extend(feat_vec[axis + 1:])
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


# Chapter 3.1.3 - Build tree recursively
def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# Chapter 3.1.3 - Build tree recursively (Program List 3-4)
def create_tree(dataset, labels):
    # get the result of each data in dataset (with duplicated items)
    class_list = [data[-1] for data in dataset]
    # Recursive Exit 1: Finish split because all results are in same type
    # list.count(item) returns the count of certain item in the list
    # this means the result contains only one type, then return the result type (terminating block)
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # Recursive Exit 2:
    # used all features but still had more than one result type
    # then get the type who has the most votes
    if len(dataset[0]) == 1:
        return majority_count(class_list)
    # get the best feature with most information gain
    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    # make a new tree dictionary
    my_tree = {best_feature_label: {}}
    # remove the used feature from labels
    del(labels[best_feature])
    # get all value of certain feature
    feature_values = [data[best_feature] for data in dataset]
    # remove duplicated items
    unique_values = set(feature_values)
    for value in unique_values:
        # copy all items in labels to sub_labels
        sub_labels = labels[:]
        # create sub tree with new dataset and new labels and then append it to certain value of best_feature
        # best_feature has been removed in new dataset and new labels
        my_tree[best_feature_label][value] = create_tree(split_dataset(dataset, best_feature, value), sub_labels)
    return my_tree
