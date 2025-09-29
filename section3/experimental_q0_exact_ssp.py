
import time
from sage.all import line, floor, log, round
from utils_lce import determine_four_set_elements, map_supports_11, sample_pair_low_weight, minimal_w
from ssp import *
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor

primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019]

def get_information_set(n, k, q, pairs):

    elms = determine_four_set_elements(n, pairs[0][0], pairs[1][0])
    elms_ = determine_four_set_elements(n, pairs[0][1], pairs[1][1])

    D = map_supports_11(pairs, elms["s11"], elms_["s11"])

    z = [len(elms["s00"]), len(elms["s01"]), len(elms["s10"])]
    z += [len(x['left']) for x in D]

    t, subset = subset_sum_dp_longest(z, k)
    # print(f"[SSPinfo] (n,k,q): ({n},{k},{q}) trial 1 => z {z}, sub {subset}, k {k}, t {t}")

    if t != k:
        return None, None, abs(t-k)
    elif len(subset) <= 2:
        return None, None, 1

    return None, None, 0

def process_task(task):
    N_trials = 10
    j, k, n = task

    if k == 1 or k == 2:
        return j, 1

    i_q = 0
    while True:

        q = primes[i_q]
        w = minimal_w(n, k, q)

        # print(f"Running q: {q}, k: {k}, w: {w}")
        count = 0
        for i in range(N_trials):
            # Sample a LCE instance
            ell = int(w * w // n)
            lw = sample_pair_low_weight(n, q, w, ell)
            pairs = [[lw[i], lw[i]] for i in range(2)]

            _, _, bf = get_information_set(n, k, q, pairs)

            if bf == 0:
                count += 1
        if count == N_trials:
            break
        else:
            i_q += 1

    return j, primes[i_q]

def test_solver_all_rates(n, step, parallel=True):
    # Precompute rates and dimensions
    all_k = [k for k in range(floor(n / 2), 3, -step)]
    all_q = [1 for _ in range(floor(n / 2), 3, -step)]
    all_responses = [-1 for _ in all_k]

    # Flatten the loops into a single list of tasks
    tasks = [
        (j, all_k[j], n)
        for j in range(len(all_k))
    ]

    if parallel:
        # Parallelize the tasks
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_task, tasks),total=len(tasks)))
    else:
        results = list(tqdm(map(process_task, tasks),total=len(tasks)))

    # Collect results back into the all_responses structure
    for i, q in results:
        all_q[i] = q

    return all_k, all_q

if __name__ == '__main__':

    # Set up the argument parser
    parser = argparse.ArgumentParser(description='For each rate, find minimum q0 such that 10/10 SSP instances are solved exactly.')
    parser.add_argument('--n', type=int, default=127, help='Length of the linear code.')
    parser.add_argument('--step', type=int, default=2, help='Skip step when trying all dimensions.')
    args = parser.parse_args()

    start = time.time()
    print("Start")
    all_k, all_q = test_solver_all_rates(args.n, args.step)
    end = time.time()
    print("--- %s seconds ---" % round(end - start, ndigits=2))

    all_log2q = [round(log(q, 2),ndigits=2) for q in all_q]

    print("all_q\n", all_q)
    print("all_q in log2\n", all_log2q)
    print("all_k\n", all_k)
