#!/usr/bin/env python
# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import sys
import re
import matplotlib.pyplot as plt
import logging


if __name__ == '__main__':
    num_file = len(sys.argv)-1
    filename = []
    window_size = 1
    for i in range(1, num_file+1):
        filename.append(sys.argv[i])
        print(sys.argv[i])
    print(num_file)

    fig = plt.figure()
    for i in range(num_file):
        x = []
        y = []

        r_sum = 0
        step = 0
        f = open(filename[i])

        for line in f:

            if line.split("\t")[1].split(" ")[0] == "test_result":
                step += 1
                try:
                    episode = line.split("\t")[3]
                    r_str = line.split("\t")[2]
                    result = float(r_str)

                    r_sum += result
                    if step % window_size == 0:
                        x.append(int(episode))
                        y.append(result)
                        r_sum = 0
                except:
                    print(line)

        plt.plot(x, y)

    plt.xlabel('Training episode', fontsize=16)
    plt.ylabel('Step to capture', fontsize=16)
    plt.grid(True)
    # plt.ylim(0, 40)
    # plt.show()

    # Save to pdf file
    plt.savefig("plot_test.pdf")
