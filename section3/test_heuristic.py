#!/usr/local/bin/sage -python
from random import randint
import time
from sage.all import *
from utils_lce import *
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor

def process_task(task):
    i, k, n, q, w = task

    if k == 1:
        return i, 0

    if k == 2:
        return i, 0

    set_random_seed(time.time() + i)

    # Sample a LCE instance
    G, G_, Q, pairs = sample_instance(n, k, q, w, 2, verbose=False)
    Q_found, _ = solve_or_guess(n, k, q, G, G_, pairs, verbose=False, parallel=False)
    result = check_solution(G, G_, Q_found)

    return i, result

def test_heuristic(n, k, q, trials, parallel=True):

    w, _ = minimal_w(n, k, q)
    # Flatten the loops into a single list of tasks
    tasks = [
        (i, k, n, q, w)
        for i in range(trials)
    ]

    if parallel:
        # Parallelize the tasks
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_task, tasks),total=len(tasks)))
    else:
        results = list(tqdm(map(process_task, tasks),total=len(tasks)))

    # Collect results back into the all_responses structure
    sum = 0
    for i, result in results:
        sum += result
    print(f"n: {n}, k: {k}, q: {q}, w: {w} - Result: {sum}/{trials}: {round(sum/trials,5)}")

    return



if __name__ == "__main__":
    ### PLOT ROUCHÉ-CAPELLI ...
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Plot the behavior of the solver when solving the LCE with hints.')
    parser.add_argument('--n', type=int, default=30, help='Linear code length.')
    parser.add_argument('--k', type=int, default=15, help='Linear code dimension.')
    parser.add_argument('--q', type=int, default=127, help='A prime number or a power of a prime.')
    parser.add_argument('--trials', type=int, default=10, help='Number of random trials.')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel computations.')

    # Parse the command-line arguments
    args = parser.parse_args()

    start = time.time()
    # Loop over each provided n value
    test_heuristic(args.n, args.k, args.q, args.trials, args.parallel)
    end = time.time()
    print("--- %s seconds ---" % round(end - start, ndigits=2))

