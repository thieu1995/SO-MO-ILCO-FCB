#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:27, 14/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange
import platform

# if platform.system() == "Linux":  # Linux: "Linux", Mac: "Darwin", Windows: "Windows"
#     import matplotlib
    # matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.

def visualize_3D(list_points):
    x = list_points[0]
    y = list_points[1]
    z = list_points[2]
    
    # x = (x - min(x)) / (max(x) - min(x))
    # y = (y - min(y)) / (max(y) - min(y))
    # x = (z - min(z)) / (max(z) - min(z))
    
    # print(x)
    # plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(x, y, z)
    ax.set_title('3d Scatter plot')
    # ax.set_xlim(50, 75)
    # ax.set_ylim(460, 520)
    # ax.set_zlim(24, 28)
    
    plt.show()

def visualize_2D(list_points):
    x = list_points[0]
    y = list_points[1]
    # z = list_points[2]
    
    # x = (x - min(x)) / (max(x) - min(x))
    # y = (y - min(y)) / (max(y) - min(y))
    # x = (z - min(z)) / (max(z) - min(z))
    
    # print(x)
    # plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.scatter(x, y)
    ax.set_title('3d Scatter plot')
    ax.set_xlim(50, 80)
    ax.set_ylim(440, 490)
    # ax.set_zlim(24, 28)
    
    plt.show()


def visualize_front_3d(list_points: list, labels:list, names:list, list_color:list, list_marker:list,
                       filename:str, pathsave: list, exts=(".png", ".pdf"), inside=True):
    fig = plt.figure()
    if inside:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = Axes3D(fig)

    if len(list_points) == 1:
        points = list_points[0]
        x_vals = points[:, 0:1]
        y_vals = points[:, 1:2]
        z_vals = points[:, 2:3]

        ax.scatter(x_vals, y_vals, z_vals, c=list_color[0], marker=list_marker[0], label=names[0])
    else:
        for idx, points in enumerate(list_points):
            xs = points[:, 0:1]
            ys = points[:, 1:2]
            zs = points[:, 2:3]

            ax.scatter(xs, ys, zs, c=list_color[idx], marker=list_marker[idx], label=names[idx])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.legend()
    for idx, ext in enumerate(exts):
        plt.savefig(pathsave[idx] + filename + ext, bbox_inches='tight')
    if platform.system() != "Linux":
        plt.show()
    plt.close()


def visualize_front_2d(list_points: list, labels: list, names:list, list_color: list, list_marker: list,
                       filename: str, pathsave:list, exts:list):
    if len(list_points) == 1:
        points = list_points[0]
        x_vals = points[:, 0:1]
        y_vals = points[:, 1:2]
        plt.scatter(x_vals, y_vals, c=list_color[0], marker=list_marker[0], label=names[0])
    else:
        for idx, points in enumerate(list_points):
            xs = points[:, 0:1]
            ys = points[:, 1:2]
            plt.scatter(xs, ys, c=list_color[idx], marker=list_marker[idx], label=names[idx])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(loc='upper left')
    for idx, ext in enumerate(exts):
        plt.savefig(pathsave[idx] + filename + ext, bbox_inches='tight')
    if platform.system() != "Linux":
        plt.show()
    plt.close()


def visualize_front_1d(list_points: list, labels: list, names: list, list_color: list, list_marker: list,
                       filename: str, pathsave:list, exts:list):
    if len(list_points) == 1:
        points = list_points[0]
        xs = arange(len(points))
        ys = points
        plt.plot(xs, ys, c=list_color[0], marker=list_marker[0], label=names[0])
    else:
        for idx, points in enumerate(list_points):
            xs = arange(len(points))
            ys = points
            plt.plot(xs, ys, c=list_color[idx], marker=list_marker[idx], label=names[idx])
    plt.xlabel("Solution")
    plt.ylabel(labels[0])
    plt.legend(loc='upper left')
    for idx, ext in enumerate(exts):
        plt.savefig(pathsave[idx] + filename + ext, bbox_inches='tight')
    if platform.system() != "Linux":
        plt.show()
    plt.close()

