#!/usr/local/bin/sage -python
from random import randint
import time

from sage.all import *

from utils_lce import *
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def process_task(task):
    i, j, k, n, q = task

    if k == 1: # cannot generate instance
        return i, j, -1

    if k == 2: # cannot solve instance
        return i, j, -1

    w, _ = minimal_w(n, k, q)

    # print(f"Running n: {n}, k: {k}, w: {w}")

    # Sample a LCE instance
    G, G_, Q, pairs = sample_instance(n, k, q, w, 2, verbose=False)
    Q_found, determined = solve_or_guess(n, k, q, G, G_, pairs, verbose=False, parallel=False)
    result = check_solution(G, G_, Q_found)

    # print(f"Done n: {n}, k: {k}, w: {w}")

    if determined == 1 and result == 1:
        return i, j, 2

    return i, j, result

def test_solver_all_n(min_n, max_n, step, q, parallel=False):
    # Precompute rates and dimensions
    all_n = [n for n in range(min_n, max_n + 1, step)]
    all_k = [[k for k in range(1,floor(n/2)+1)] for n in all_n ] # we use the symmetry and consider only rates (0,1/2]
    all_responses = [[-1 for _ in range(len(all_k[i]))] for i in range(len(all_n))]

    # Flatten the loops into a single list of tasks
    tasks = [
        (i, j, all_k[i][j], all_n[i], q)
        for i in range(len(all_n))
        for j in range(len(all_k[i]))
    ]

    if parallel:
        # Parallelize the tasks
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_task, tasks),total=len(tasks)))
    else:
        results = list(tqdm(map(process_task, tasks),total=len(tasks)))

    # Collect results back into the all_responses structure
    for i, j, result in results:
        all_responses[i][j] = result

    return all_responses, all_k, all_n


def get_data(min_n, max_n, step, q, parallel=True):
    all_responses, all_k, all_n = test_solver_all_n(min_n, max_n, step, q, parallel=True)

    print("all_n\n", all_n)
    print("all_k\n", all_k)
    print("all_responses\n", all_responses)

if __name__ == "__main__":
    ### PLOT ROUCHÉ-CAPELLI ...
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Plot the behavior of the solver when solving the LCE with hints.')
    parser.add_argument('--min_n', type=int, default=20, help='Minimum of the linear code.')
    parser.add_argument('--max_n', type=int, default=30, help='Maximum of the linear code.')
    parser.add_argument('--step', type=int, default=5, help='Step among the ns.')
    parser.add_argument('--q', type=int, default=127, help='A prime number or a power of a prime.')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity.')

    # Parse the command-line arguments
    args = parser.parse_args()

    start = time.time()
    # Loop over each provided n value
    get_data(min_n=args.min_n, max_n=args.max_n, step=args.step, q=args.q)
    end = time.time()
    print("--- %s seconds ---" % round(end - start, ndigits=2))

