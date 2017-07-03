#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 2 - kNN Algorithm Module

from numpy import *
import operator
import os


# Chapter 2.1.1 - Preparation: Import Data with Python
def create_dataset():
    # Define an 4x2 array(2-Dimensional)
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # Define the label list
    #     () means tuple
    #     [] means list
    #     {} means dict
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# Chapter 2.1.2 - Run kNN Algorithm (Program List 2-1)
def classify0(in_x, dataset, labels, k_count):
    # in_x    - the unknown input data
    # dataset - the known data
    # labels  - the real result of the known data
    # k_count - pick the closest k_count records to determine the type of result

    # return  - classified result value

    # Get the row count
    #     NumPy.array.shape returns a tuple like (row count, column count)
    #         dataset.shape = (4, 2)
    dataset_size = dataset.shape[0]

    # Calculate distance
    #     NumPy.tile(a, (r, c)) duplicates the data r rows and c columns
    #         in_x = [0, 0]
    #         tile(in_x, (4, 1)) =
    #             [[0, 0], [0, 0], [0, 0], [0, 0]]
    diff_mat = tile(in_x, (dataset_size, 1)) - dataset
    #     array**2 means square of each element
    sq_diff_mat = diff_mat ** 2
    #     array.sum means summing up specific elements
    #         for a N-dimensional array, axis can be in range(N)
    #         axis = None(default) means all elements
    #         axis = 0 means performing column actions and adding to one row
    #         axis = 1 means performing row actions and adding to one column
    #         Here means adding the dx^2 and dy^2 to the distance^2
    sq_distances = sq_diff_mat.sum(axis=1)
    #     array**0.5 means square root of each element, which is the real distance
    distances = sq_distances ** 0.5

    # Sort
    #     NumPy.argsort() returns a sorted list of indices from small to big
    sorted_dist_indicies = distances.argsort()

    # Get the best k_count result
    class_count = {}
    for i in range(k_count):
        #         Get the label
        vote_i_label = labels[sorted_dist_indicies[i]]
        #         Update the label count
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    # sort() means sort the original input
    #         sorted() means sort the copy of the input
    #         items() returns the original object
    #         iteritems() returns the iterator object
    #         operator.itemgetter(1) returns the 2nd(count) column of input(label, count)
    #         key can also be specified with lambda(key=lambda x : x[1])
    #         reverse = True means decent order
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1), reverse=True)
    # Return the label of the item with the maximum count.
    return sorted_class_count[0][0]


# Chapter 2.2 - Project: Improve Recommendation for Dating Website with k-means algorithm.


# Chapter 2.2.1 - Preparation: Import Data from file
def file2matrix(filename):
    # open file
    fr = open(filename)
    #     read() gets the whole text of a file
    #     readline() gets the next line of a file
    #     readlines() gets the lines array of a file
    array_of_lines = fr.readlines()
    #     get the total line count
    number_of_lines = len(array_of_lines)
    #     create a NumPy Array
    return_mat = zeros((number_of_lines, 3))
    # import data
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        #     remove spaces (including '\n', '\r', '\t', ' ') on margins
        line = line.strip()
        #     split to list elements by '\t'
        #     '\t' means Tab
        list_from_line = line.split('\t')
        #     copy value from line to the specific line of the return Matrix
        return_mat[index, :] = list_from_line[0:3]
        #     convert the last value to integer and append it to the label
        class_label_vector.append(int(list_from_line[-1]))
        #     increase return_mat's line index by 1
        index += 1
    return return_mat, class_label_vector


# Chapter 2.2.3 - Processing: Auto normalization
def auto_normalization(dataset):
    # array.min(axis=0) means the minimum element for each column/feature(axis=1 for each row/record)
    min_vals = dataset.min(0)
    # array.max(axis=0) means the maximum element for each column/feature(axis=1 for each row/record)
    max_vals = dataset.max(0)
    # range for each column/feature
    ranges = max_vals - min_vals
    # shape[0] returns the total row/record count
    m = dataset.shape[0]
    # tile creates a new array by duplicating min_vals by m times in row and keeping the original columns
    #     now the norm_dataset has no negative values
    norm_dataset = dataset - tile(min_vals, (m, 1))
    # tile creates a new array by duplication ranges by m times in row and keeping the original columns
    #     now the elements from norDataSet are in range 0 to 1
    norm_dataset = norm_dataset / tile(ranges, (m, 1))
    # norm_dataset - the normalized dataset
    # ranges      - the range for each column/features
    # min_vals     - the minimum values for each column/features
    return norm_dataset, ranges, min_vals


# Chapter 2.2.4 - Testing: Test the classifier
# Usually we use 90% of the data to train the model and the rest 10% of the data to test the model.
def dating_class_test():
    # ratio percentage for testing
    ho_ratio = 0.10
    # load data from file
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    # normalize data to range 0 ~ 1
    norm_mat, ranges, min_vals = auto_normalization(dating_data_mat)
    # get row/record count
    m = norm_mat.shape[0]
    # get count of the records for testing purpose
    num_test_vecs = int(m * ho_ratio)
    # get count of bad classification
    error_count = 0.0
    # test the classifier
    for i in range(num_test_vecs):
        # norm_mat[i,:] means the i'th row/record
        # norm_mat[num_test_vecs:m,:] means all the data that is for training purpose
        # dating_labels[num_test_vecs:m] means the real result of training data records
        # 3 means pick the closest 3 records to determine the type of result
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                      dating_labels[num_test_vecs:m], 3)
        # display the result for each record
        print "the classifier came back with: %d, the real answer is: %d" \
              % (classifier_result, dating_labels[i])
        # update the count of bad classification
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    # display the final stats
    print "the total error rate is: %f" % (error_count / float(num_test_vecs))


# Chapter 2.2.5 - Application: Build a complete system
def classify_person():
    # List for mapping result values from integers to strings
    result_list = ['not at all', 'in small doses', 'in large doses']
    # get the first value
    percent_tats = float(raw_input("percentage of time spent playing video games?\n"))
    # get the second value
    ff_miles = float(raw_input("frequent flier miles earned per year?\n"))
    # get the third value
    ice_cream = float(raw_input("liters of ice cream consumed per year?\n"))
    # load data from file
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    # normalize data to range 0 ~ 1
    norm_mat, ranges, min_vals = auto_normalization(dating_data_mat)
    # convert a list to a array for storing input values
    in_arr = array([ff_miles, percent_tats, ice_cream])
    # normalize input values and classify it with k=3
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    # display the result
    print "You will probably like this person: ", result_list[classifier_result - 1]


# Chapter 2.3.1 - Preparation: Convert image to testing vector
def img2vector(filename):
    # get a new numpy array with like
    #     return_vec = [[0,0...0,0]]
    return_vec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vec[0, 32 * i + j] = int(line_str[j])
    return return_vec


# Chapter 2.3.2 - Testing: Commit Handwriting Recognition with k-means Algorithm
def handwriting_test():
    handwriting_labels = []
    training_file_list = os.listdir('digits/trainingDigits')
    m_training = len(training_file_list)
    training_mat = zeros((m_training, 1024))
    for i in range(m_training):
        filename_str = training_file_list[i]
        # remove extension
        file_str = filename_str.split('.')[0]
        # get the real digit label
        class_number_str = int(file_str.split('_')[0])
        handwriting_labels.append(class_number_str)
        training_mat[i, :] = img2vector('digits/trainingDigits/%s' % filename_str)
    test_file_list = os.listdir('digits/testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        filename_str = test_file_list[i]
        file_str = filename_str.split('.')[0]
        class_number_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('digits/testDigits/%s' % filename_str)
        classifier_result = classify0(vector_under_test,
                                      training_mat, handwriting_labels, 3)
        if classifier_result != class_number_str:
            error_count += 1
            print "the classifier came back with: %d, the real answer is %d. Wrong answer!" % \
                  (classifier_result, class_number_str)
        else:
            print "the classifier came back with: %d, the real answer is %d." % \
                  (classifier_result, class_number_str)

    print "\nThe total number of errors is: %d" % error_count
    print "\nThe total error rate is: %f" % (error_count / float(m_test))
