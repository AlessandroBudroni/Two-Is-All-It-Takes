#!/usr/local/bin/sage -python

import time
from tabnanny import verbose

from sage.all import *
from utils_lce import *
import argparse
from tqdm import tqdm

if __name__ == "__main__":

    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Plot how much the rank of the system decreases when applying Rouché-Capelli row-column guessing.')
    parser.add_argument('--n', type=int, default=20, help='Length of the linear code.')
    parser.add_argument('--k', type=int, default=10, help='Dimension of the code.')
    parser.add_argument('--q', type=int, default=127, help='A prime number or a power of a prime.')
    parser.add_argument('--w', type=int, default=0, help='The hamming weight of the low weight pairs. If not given, the minimum weight is used by default.')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity.')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel computations.')

    # Parse the command-line arguments
    args = parser.parse_args()

    n = args.n
    q = args.q
    k = args.k

    if args.w == 0:
        w, _ = minimal_w(int(args.n), int(args.k), int(args.q))
    else:
        w = args.w

    if (k < 2):
        print("I am unable to generate this code. Try with a larger dimension k.")
        exit(0)

    print(f"Start experiment for parameters: n: {n}, k: {k}, q: {q}, w: {w}")

    # Sample a LCE instance
    G, G_, Q, pairs = sample_instance(n, k, q, w, 2, verbose=args.verbose)

    start = time.time()
    Q_found, _ = solve_or_guess(n, k, q, G, G_, pairs, verbose=args.verbose, parallel=args.parallel)
    end = time.time()

    # print(Q_found)
    # print("\n")
    # print(Q)

    if check_solution(G, G_, Q_found, args.verbose):
        print("Success")
    else:
        print("Failure")

    print("--- %s seconds ---" %  round(end-start, ndigits=2))










