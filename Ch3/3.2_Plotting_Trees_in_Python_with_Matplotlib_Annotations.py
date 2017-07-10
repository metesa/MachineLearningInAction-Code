#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Part 1 - Chapter 3.2 - 6 - Plotting Trees in Python with Matplotlib Annotations

print("*************************************\n  Machine Learning In Action - Code\n")
print("    Part    : 1   - Classification")
print("    Chapter : 3.2 - Plotting Trees in Python with Matplotlib Annotations")
print("    Index   : 6\n")
print("    Page    : 36\n")
print("    By : Troy Lewis\n*************************************\n\n")

import tree_plotter


print("=====================================\nStep 01: Import trees.py and create plot\n")
tree_plotter.create_plot_v1()


print("=====================================\nStep 02: Plot a tree\n")
print("Retrieve tree 1:")
print(tree_plotter.retrieve_tree(1))
my_tree = tree_plotter.retrieve_tree(0)
print("Depth of tree 0:")
print(tree_plotter.get_depth_of_tree(my_tree))
print("Leaf count of tree 0:")
print(tree_plotter.get_number_of_leafs(my_tree))
