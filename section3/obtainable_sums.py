import argparse
import numpy as np
from collections import defaultdict
from itertools import product, combinations


def modinv(a, q):
    """Modular inverse in F_q."""
    a = int(a)
    if a == 0:
        raise ValueError("No inverse for zero")
    try:
        return pow(a, -1, q)
    except:
        print(a, q)


def classify_columns_by_type(A, q):
    """Classify columns in A by their types."""
    types = {'00': [], '10': [], '01': [], '11': defaultdict(list)}
    for i in range(A.shape[1]):
        a0, a1 = int(A[0, i]), int(A[1, i])
        key = ('0' if a0 == 0 else '1') + ('0' if a1 == 0 else '1')
        if key == '11':
            ratio = (a0 * modinv(a1, q)) % q
            types['11'][ratio].append(i)
        else:
            types[key].append(i)
    return types


def do(B, k, q):
    types_B = classify_columns_by_type(B, q)
    z = [len(types_B["11"][i]) for i in types_B["11"].keys()]
    z += [len(types_B["00"]), len(types_B["01"]), len(types_B["10"])]
    # print(z)
    return subset_sum_dp(z, k)


import random


def subset_sum_dp(nums, target):
    n = len(nums)
    total = sum(nums)
    dp = [[False] * (total + 1) for _ in range(n + 1)]
    dp[0][0] = True

    for i in range(1, n + 1):
        num = nums[i - 1]
        for t in range(total + 1):
            if t >= num:
                dp[i][t] = dp[i - 1][t] or dp[i - 1][t - num]
            else:
                dp[i][t] = dp[i - 1][t]
    return dp[n]


import matplotlib.pyplot as plt


def plot(L, label_list):
    """
    function to plot list of lists of the form L=[L1,L2,...] with Li=[[x1,y1],[x2,y2],...]
    """
    c = 0
    for Li in L:
        x, y = zip(*Li)
        if c % 2 == 0:
            color = "blue"
        else:
            color = "red"
        plt.scatter(x, y, label=label_list[c], s=1, color=color)
        c += 1
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


import random
from typing import List, Tuple


def generate_two_vectors(n: int, q: int, w: int) -> Tuple[List[int], List[int]]:
    """Generate two random vectors of length n, each of Hamming weight w, over F_q."""
    v1 = [0] * n
    v2 = [0] * n
    indices1 = random.sample(range(n), w)
    indices2 = random.sample(range(n), w)
    for i in indices1:
        v1[i] = random.randint(1, q - 1)
    for i in indices2:
        v2[i] = random.randint(1, q - 1)
    return v1, v2


def construct_matrix(v1: List[int], v2: List[int]) -> np.ndarray:
    """Construct a 2 x n matrix with v1 and v2 as rows."""
    return np.array([v1, v2])


def distribution_obtainable_sums(n, q, w):
    """Compute the distribution of obtainable sums for given parameters."""
    counts = defaultdict(int)
    N = 10
    for _ in range(N):
        v1, v2 = generate_two_vectors(n, q, w)
        A = construct_matrix(v1, v2)
        for k in range(1, n + 1):
            reachable = do(A, k, q)
            s = sum(reachable)
            counts[k] += s
    return counts

if __name__ == '__main__':

    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Plot the behavior of the solver when solving the LCE with hints.')
    parser.add_argument('--n', type=int, default=252, help='Length of the code.')
    parser.add_argument('--q', type=int, default=127, help='Field modulo')
    parser.add_argument('--w', type=int, default=107, help='Codewords weight')
    # Parse the command-line arguments
    args = parser.parse_args()

    counts = distribution_obtainable_sums(args.n, args.q, args.w)
    print(counts)
