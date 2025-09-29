import argparse
from scipy.optimize import fsolve
from math import log2, log, floor
import matplotlib.pyplot as plt


def H(q, x):
    """
    q-ary entropy function
    """
    if x <= 0 or x >= 1:
        return 0
    return x * log(q - 1, q) - x * log(x, q) - (1 - x) * log(1 - x, q)


def Hqi(v, q):
    """
    inverse of q-ary entropy function
    """
    if v == 1:
        return 1 - 1 / q
    return \
    fsolve(lambda x: v - (x[0] * log(q - 1, q) - x[0] * log(x[0], q) - (1 - x[0]) * log(1 - x[0], q)), 0.0000001)[0]


# def plot(L, label_list):
#     """
#     function to plot list of lists of the form L=[L1,L2,...] with Li=[[x1,y1],[x2,y2],...]
#     """
#     c = 0
#     for Li in L:
#         x, y = zip(*Li)
#         plt.scatter(x, y, label=label_list[c], s=1)
#         c += 1
#     # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

if __name__ == '__main__':

    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Plot the behavior of the solver when solving the LCE with hints.')
    parser.add_argument('--n', type=int, default=1000, help='Code length.')

    # Parse the command-line arguments
    args = parser.parse_args()

    vals = []

    max_q = 100000
    for i in range(1, floor(args.n/2)+1):
        R = i / args.n
        for q in range(2, max_q, 1):
            w = Hqi(1 - R, q)
            val = w * (2 - w * (1 + 1 / q)) - H(q, w)
            if val < 0:
                continue
            vals.append([R, round(log2(q),ndigits=2)])
            break
    # plot([vals], [""])
    print(vals)