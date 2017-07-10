#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 3 - Decision Tree Plotter

import matplotlib.pyplot as plt

# Chapter 3.2.1 - Matplotlib Annotation (Program List 3-5)
#   Matplotlib Annotation tool
#     place a annotation at point(0.35, 0.3) and an arrow pointing at point(0.2, 0.1)

# Define global variables to describe tree node
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# create an annotation with specific font
def plot_node_with_font_v1(axis, node_text, center_point, parent_point, node_type, font):
    axis.annotate(node_text, xy=parent_point, xycoords='axes fraction',
                  xytext=center_point, textcoords='axes fraction',
                  va="center", ha="center", bbox=node_type,
                  arrowprops=arrow_args, fontproperties=font)


# create an annotation
def plot_node_v1(axis, node_text, center_point, parent_point, node_type):
    axis.annotate(node_text, xy=parent_point, xycoords='axes fraction',
                  xytext=center_point, textcoords='axes fraction',
                  va="center", ha="center", bbox=node_type,
                  arrowprops=arrow_args)


# create an plot which contains two annotations with specific font
def create_plot_with_font_v1():
    # get the font properties for specific font
    import matplotlib.font_manager as fm
    font = fm.FontProperties(fname="/home/metesa/.local/share/fonts/simhei.ttf")
    # create a new figure
    fig = plt.figure(1, facecolor="white")
    # clear the painting area
    fig.clf()
    # get the above part of the figure
    #     111 means divided the figure to 1x1 pieces and ax stands for the 1st area, aka the whole figure as a plot
    #     for 3,4,10, you can't use 3410, only (3, 4, 10) will do
    #     add_subplot(111)
    #         ax = fig.add_subplot(111)
    #     subplot(111)
    #         fig.subplot(111)
    #         ax = fig.gca() # get current axis
    plot_axis = plt.subplot(111, frameon=False)
    # create two annotation
    #     the first pair of coordinate shows the arrow and the label
    #     the second pair of coordinate shows the start point
    plot_node_with_font_v1(plot_axis, u'决策节点', (0.5, 0.1), (0.1, 0.5), decision_node, font)
    plot_node_with_font_v1(plot_axis, u'叶节点', (0.8, 0.1), (0.3, 0.8), leaf_node, font)
    # set title
    plt.title(u'标题', fontproperties=font)
    # set legend
    plt.legend(['图例'], prop=font)
    # show the figure
    plt.show()


# create an plot which contains two annotations
def create_plot_v1():
    # set font for displaying Chinese
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # create a new figure
    fig = plt.figure(1, facecolor="white")
    # clear the painting area
    fig.clf()
    # get the above part of the figure
    #     111 means divided the figure to 1x1 pieces and ax stands for the 1st area, aka the whole figure as a plot
    #     for 3,4,10, you can't use 3410, only (3, 4, 10) will do
    #     add_subplot(111)
    #         ax = fig.add_subplot(111)
    #     subplot(111)
    #         fig.subplot(111)
    #         ax = fig.gca() # get current axis
    plot_axis = plt.subplot(111, frameon=False)
    # create two annotation
    #     the first pair of coordinate shows the arrow and the label
    #     the second pair of coordinate shows the start point
    plot_node_v1(plot_axis, u'决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node_v1(plot_axis, u'叶节点', (0.8, 0.1), (0.3, 0.8), leaf_node)
    # set title
    plt.title(u'标题')
    # set legend
    plt.legend(['图例'])
    # show the figure
    plt.show()


# Chapter 3.2.2 - Build an Annotation Tree (Program List 3-6)
def get_number_of_leafs(tree):
    number_of_leafs = 0
    # root node of this tree or subtree
    first_str = tree.keys()[0]
    # sub nodes under this root node
    second_dict = tree[first_str]
    # iterate among all sub nodes
    for key in second_dict.keys():
        # accumulate the leaf nodes under this sub node (including this sub node)
        if type(second_dict[key]).__name__ == 'dict':
            # if there are more nodes under this sub node, then do a recursive calculation
            number_of_leafs += get_number_of_leafs(second_dict[key])
        else:
            # if not, then count this sub node as a leaf node.
            number_of_leafs += 1
    return number_of_leafs


def get_depth_of_tree(tree):
    max_depth = 0
    # root node of this tree or subtree
    first_str = tree.keys()[0]
    # sub nodes under this root node
    second_dict = tree[first_str]
    # iterate among all sub nodes
    for key in second_dict.keys():
        # calculate the depth under this sub node (including this sub node)
        if type(second_dict[key]).__name__ == 'dict':
            # if there are more nodes under this sub node, then do a recursive calculation
            this_depth = 1 + get_depth_of_tree(second_dict[key])
        else:
            # if not, then the depth is 1.
            this_depth = 1
        # update max depth
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


# retrieve the tree which defined before
def retrieve_tree(i):
    list_of_trees = [
        {
            'no surfacing': {
                0: 'no', 1: {
                    'flippers': {
                        0: 'no',
                        1: 'yes'
                    }
                }
            }
        },
        {
            'no surfacing': {
                0: 'no',
                1: {
                    'flippers': {
                        0: {
                            'head': {
                                0: 'no',
                                1: 'yes'
                            }
                        },
                        1: 'no'
                    }
                }
            }
        }]
    return list_of_trees[i]
