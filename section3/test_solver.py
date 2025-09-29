import os


import time, argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sage.all import set_random_seed, rank, prod, line, floor
from nowakowski.lep_solver import recoverMon
from utils_lce import *

# FLAGS
NO_INFORMATION_SET = 0
SOLUTION_NOT_PRECISE = -2
CANONICAL_FORM_FAILED = -1
SOLUTION_RECOVERED = 1
UNEXPECTED_ERROR = -100

def solve_instance(n, k, G1, G2, pairs):
    '''
    Args:
        n: code length
        k: code dimension
        G1: first code generator matrix
        G2: second code generator matrix
        pairs: pairs of equivalent low weight codewords

    Returns: 1 if the solution was recovered, an error otherwise, together with the measured timings.

    '''

    start = time.time()
    J1, J2, ret = get_information_set(n, k, pairs, G1, shuffle=False)
    end = time.time()
    ssp_time = end - start

    if ret > 0:
        return SOLUTION_NOT_PRECISE, {"ssp": 0, "cf": 0}
    elif ret < 0:
        return NO_INFORMATION_SET, {"ssp": 0, "cf": 0}
    else:
        start = time.time()
        Qrec = recoverMon(G1, G2, J1, J2)
        end = time.time()
        cf_time = end - start

        if Qrec == None:
            return CANONICAL_FORM_FAILED, {"ssp": 0, "cf": 0}
        else:
            if check_solution(Qrec, G1, G2):
                return SOLUTION_RECOVERED, {"ssp": ssp_time, "cf": cf_time}
            else:
                return UNEXPECTED_ERROR, {"ssp": 0, "cf": 0}

def full_rank_probability(n, q):

    qf = float(q)
    return prod(1.0 - qf**(-j) for j in range(1, n + 1))

def process_task(task):

    (j, n, k, q, w) = task
    set_random_seed(time.time() + j)
    G1, G2, Q, pairs = sample_instance(n, k, q, w, 2)
    ret, total_time = solve_instance(n, k, G1, G2, pairs)

    if ret == SOLUTION_RECOVERED:
        return j, 0, total_time
    elif ret == CANONICAL_FORM_FAILED:
        return j, -1, total_time
    elif ret == SOLUTION_NOT_PRECISE:
        return j, 1, total_time
    elif ret == NO_INFORMATION_SET:
        return j, -2, total_time
    else:
        print("Something went wrong")
        print(G1, G2, Q)
        SystemExit()

    return j, -100, total_time


def main(n, k, q, w, N_TESTS, parallel=False):

    if w == 0:
        w = minimal_w(n, k, q)

    tasks = [(i, n, k, q, w) for i in range(N_TESTS)]

    if parallel:
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_task, tasks), total=len(tasks)))
    else:
        results = list(tqdm(map(process_task, tasks), total=len(tasks)))

    success = 0
    cf_failed = 0
    not_is = 0
    ssp_not_exact = 0
    cf_time = 0
    ssp_time = 0

    for result in results:
        j, res, t = result
        if res == 0:
            cf_time += t["cf"]
            ssp_time += t["ssp"]
            success += 1
        elif res == -2:
            not_is += 1
        elif res == -1:
            cf_failed += 1
        elif res > 0:
            ssp_not_exact += 1

    print(f"For $(n,k,q,w) = ({n},{k},{q},{w}) N_TESTS {N_TESTS} - Success: = {success}, CF failed: = {cf_failed}, No precise Solution: = {ssp_not_exact}, No IS = {not_is}")
    print(f"Average time SSP: {round(ssp_time / success,ndigits=3)} seconds")
    print(f"Average time  CF: {round(cf_time / success, ndigits=3)} seconds")
    print(f"Average time tot: {round((ssp_time+cf_time) / success, ndigits=3)} seconds")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate and solve random LCE instances when 2 pairs of equivalent codewords are given.')
    parser.add_argument('--n', type=int, default=64, help='Length of the code.')
    parser.add_argument('--k', type=int, default=32, help='Dimension of the code.')
    parser.add_argument('--q', type=int, default=127, help='Modulo of the field.')
    parser.add_argument('--w', type=int, default=0, help='Weight of the words. If not give, w_GV will be used by default.')
    parser.add_argument('--parallel', action='store_true', help='Use tp enable parallel mode.')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials.')

    # Parse the command-line arguments
    args = parser.parse_args()

    main(args.n, args.k, args.q, args.w, args.n_trials, args.parallel)





