#!/usr/local/bin/sage -python
from os import error

from sage.all import *
from utils_lce import *
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor

# Theoretical distribution of jars containing from exactly 0 up to exactly n balls.
def avg_expect(n,d):
    return [round(float(d*binomial(n,k)*(d-1)**(n-k)/d**n),3) for k in range(n+1)]

# Variable I can fix given that distribution
def compute_exp_cols(n,w,ell,q):
    v = avg_expect(ell,q-1)
    indeterminate = 0
    for i in range(2,ell):
        indeterminate += i**2 * v[i]

    return 2*(w-ell)**2 + (n-2*w+ell)**2 + indeterminate

def compute_line_exp_cols(n, q, w_in, all_k):
    exp_cols = [0 for _ in all_k]
    for i in range(len(all_k)):
        if w_in == 0:
            w = minimal_w_only_weight(n, all_k[i], q)
        elif w_in == 1:
            w = n - all_k[i]
        else:
            error(1)
        exp_cols[i] = round(compute_exp_cols(n, w, floor(w*w/n), q), 5)
    return exp_cols

def test_rank_cols(n, k, q, w, N):

    lw = [sample_low_weight(n, q, w) for _ in range(N)]

    G = lw[0]
    for i in range(1, N):
        G = G.stack(lw[i])
    G = G.stack(random_matrix(GF(q), k - N, n))

    Q = sample_monomial_matrix(n, q)
    G_ = (G * Q).rref()
    pairs = [[lw[i], lw[i] * Q] for i in range(N)]

    S = get_system_lce(G, G_)

    L_zero, L_nonzero, _ = get_zero_nonzero_entries(n, q, pairs)
    A, b = reduce_system_to_non_homogeneous(n, k, q, S, L_zero, L_nonzero)

    Brank = A.rank()
    Bcols = A.ncols()

    return Brank, Bcols


def compute_for_k_rank(args):
    k, n, q, w, N, trials = args
    rate = round(k / n, ndigits=4)
    row = k * (n - k)

    if w == 0:
        w = minimal_w_only_weight(n, k, q)
    elif w == 1:
        w = n-k
    else:
        error(1)

    if N > k:
        return k, rate, 0, 0, row

    average_ranks = 0.
    average_cols = 0.
    for t in range(trials):
        rank, col = test_rank_cols(n, k, q, w, N)
        average_ranks += rank
        average_cols += col
        # print(average_ranks, average_cols)

    rank_avg = round(average_ranks / trials, 4)
    col_avg = round(average_cols / trials, 4)

    return k, rate, rank_avg, col_avg, row


def test_rank_all_rates(n, q, w, trials):
    all_k = [k for k in  range(1, floor(n/2) + 1)]

    rates = [0.] * len(all_k)
    ranks = [0.] * len(all_k)
    cols = [0.] * len(all_k)
    rows = [0] * len(all_k)

    # Prepare arguments for each `k`
    args_list = [(k, n, q, w, 2, trials) for k in all_k]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(tqdm(executor.map(compute_for_k_rank, args_list), total=len(all_k)))

    # Update rates, ranks, cols, and rows based on parallel results
    for k, rate, rank_avg, col_avg, row in results:
        rates[k - 1] = rate
        ranks[k - 1] = rank_avg
        cols[k - 1] = col_avg
        rows[k - 1] = row

    return rates, ranks, cols, rows, all_k



def compute_data(n, q, w, trials):

    rates, ranks, cols, rows, all_k = test_rank_all_rates(n, q, w, trials)

    print("For rates\n", rates)
    print("Ranks\n", ranks)
    print("Columns\n", cols)
    print("Rows\n", rows)

    # Add expected columns as a line
    print(all_k)
    all_exp_cols = compute_line_exp_cols(n, q, w, all_k)
    print("Expected columns")
    print(all_exp_cols)

    return True


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Generate two plots to compare rank vs. n_rows vs. n_cols')
    parser.add_argument('--n', type=int, default=20, help='Length of the linear code.')
    parser.add_argument('--q', type=int, default=127, help='The modulus q.')
    parser.add_argument('--w', type=int, default=0, help='use 0 for w=minimum weight (default), 1 for w=n-k')
    parser.add_argument('--trials', type=int, default=5, help='The number of trials.')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity.')

    # Parse the command-line arguments
    args = parser.parse_args()

    compute_data(n=args.n, q=args.q, w=args.w, trials=args.trials)

















