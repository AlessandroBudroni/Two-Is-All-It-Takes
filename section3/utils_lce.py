
from sage.all import *
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def check_solution_extact_matching(Q_found, Q_original, n):
    if Q_found == None:
        return 0

    # scale monomial
    factor = 0
    for i in range(n):
        if Q_original[0, i] != 0:
            factor = Q_original[0, i]
            break

    factor2 = 0
    for i in range(n):
        if Q_found[0, i] != 0:
            factor2 = Q_found[0, i]
            break
    if factor2 == 0:
        return 0

    if Q_original == (Q_found/factor2) * factor:
        return 1
    else:
        return 0

def check_solution(G, G_, Q_found, verbose=False):
    if Q_found == None:
        if verbose:
            print("Failed - Monomial not found.")
        return 0

    if verbose:
        print("Monomial computed")
        print(Q_found)

    if(not(is_monomial(Q_found))):
        if verbose:
            print("Failed - Not a Monomial.")
        return 0

    if (G*Q_found).rref() == G_.rref():
        if verbose:
            print("Success - Monomial found")
        return 1
    else:
        if verbose:
            print("Failed = Bad solution")
        return 0

def solve_or_guess(n, k, q, G, G_, pairs, verbose=False, parallel=False):

    S = get_system_lce(G, G_)
    # if verbose:
    #     print("rank initial system    ", S.rank())

    # variables to zero in Q, variables determined in Q from the intersection
    L_zero, L_nonzero, ell = get_zero_nonzero_entries(n, q, pairs)
    if verbose:
        print("Intesection of the support (ell): ", ell)

    A, b = reduce_system_to_non_homogeneous(n, k, q, S, L_zero, L_nonzero)

    # keep track of the zeroed variables in Q
    Q_companion = matrix(ZZ, n, n, [-100000000] * n * n)  # set to a value that means "undetermined"

    determined = 0

    if (rank(A) == A.ncols()):
        if verbose:
            print("The system is determined, no need for any guessing!")

        for x in L_zero:
            Q_companion[x[0], x[1]] = 0
        for x in L_nonzero:
            Q_companion[x[0], x[1]] = ZZ(x[2])

        Qred = A.solve_right(b) # Retrieve the remaining entries of Q via gaussian elimination
        determined = 1

    else:
        if verbose:
            print("The system is undetermined, need for guessing!")

        for x in L_zero:
            Q_companion[x[0], x[1]] = 0
        for x in L_nonzero:
            Q_companion[x[0], x[1]] = ZZ(x[2])

        # skip L_nonzero rows when doing Rouché-Capelli
        skip_rows = [x[0] for x in L_nonzero]
        loop = 0
        while loop < 10: # set a maximum loop to not increase the complexity of the algorithm asymptotically - only to handle very small q cases.
            len_hints = len(L_zero)
            for row in range(n):
                if row in skip_rows:
                    continue
                if verbose:
                    print("Guessing row ", row)
                Q_companion, L_zero = rouche_capelli_guess_row(n, S, b, L_zero, L_nonzero, Q_companion, row, verbose=verbose, parallel=parallel)
                if Q_companion == None:
                    return None, determined
                Ared, b = reduce_system_lce(n, q, S, L_zero, L_nonzero)
                rankAred = Ared.rank()
                if (rankAred == Ared.ncols()):
                    break
                else:
                    if verbose:
                        print("Loop", loop, "System still underdetermined", rankAred, "!=", Ared.ncols())
            if (rankAred == Ared.ncols()):
                break
            loop += 1
            if len_hints == len(L_zero):
                if (rankAred >= Ared.ncols() - 10): # cover cases for 2 or 3 multiple  solutions
                    if verbose:
                        print("There are multiple solutions - try selecting one")
                    max_iter = 0
                    while (rankAred < Ared.ncols() and max_iter < 10): # set iterations to a max
                        # if there are 2 solutions, try selecting one
                        L_zero, Q_companion = select_one_solution(L_zero, Q_companion, n, verbose)
                        Ared, b = reduce_system_lce(n, q, S, L_zero, L_nonzero)
                        rankAred = Ared.rank()
                        max_iter = max_iter + 1
                break

        try:
            Qred = Ared.solve_right(b)
        except:
            if verbose:
                print("The system does not accept solutions")
            return None, determined

    # Fill Q_companion with the content of Qred
    count = 0
    for i in range(n):
        for j in range(n):
            if Q_companion[i, j] == -100000000:
                Q_companion[i, j] = Qred[count, 0]
                count = count + 1

    return Q_companion.change_ring(GF(q)), determined

# select one possible monomial solution if there are 2 or 3. Hopefully, the selected one is correct.
def select_one_solution(L_zero, Q_companion, n, verbose):

    for i in range(n):
        n_nonzero = 0
        last_j = 0
        for j in range(n):
            if Q_companion[i, j] != 0:
                n_nonzero += 1
                if n_nonzero == 2:
                    L_zero = L_zero + [[i, last_j]]
                    Q_companion[i, last_j] = 0
                    if verbose:
                        print(f"putting to zero {[i, last_j]}")
                    for i_ in range(n):
                        if Q_companion[i_, j] != 0 and i != i_:
                            Q_companion[i_, j] = 0
                            L_zero = L_zero + [[i_, j]]
                            if verbose:
                                print(f"putting to zero {[i_, j]}")
                            return L_zero, Q_companion
                last_j = j
    return L_zero, Q_companion

def is_monomial(Q):
    for i in range(Q.nrows()):
        count = 0
        for j in range(Q.ncols()):
            if Q[i,j] != 0:
                count += 1
        if count != 1:
            return False
    for j in range(Q.ncols()):
        count = 0
        for i in range(Q.nrows()):
            if Q[i,j] != 0:
                count += 1
        if count != 1:
            return False
    return True

def sample_low_weight(n, q, w):
    '''
    Sample a codeword of length n and weight w
    :param n: length of the codeword
    :param q: modulo
    :param w: Hamming weight
    :return: low weight codeword
    '''
    a = [i for i in range(0, n)]
    for i in range( n -1, n- 1 - w, -1):
        j = randint(0, i)
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp

    a = a[n - w:n]

    lw = zero_matrix(GF(q), 1, n)
    for x in range(w):
        lw[:, a[x]] = randint(1, q - 1)

    return lw


# sample a pair with a fixed overlap ell
def sample_pair_low_weight(n, q, w, ell):
    '''
    Sample a pair of codewords of length n, weight w, and overlap ell
    :param n: length
    :param q: modulo
    :param w: hamming weight
    :param ell: overlap
    :return: list containing the two codewords
    '''
    values = [i for i in range(0, n)]
    a = [0]*w
    b = [0]*w
    for i in range(ell):
        tmp = values.pop(randint(0,len(values)-1))
        a[i] = tmp
        b[i] = tmp
    for i in range(ell, w):
        tmp = values.pop(randint(0,len(values)-1))
        a[i] = tmp
        tmp = values.pop(randint(0,len(values)-1))
        b[i] = tmp

    lw1 = zero_matrix(GF(q), 1, n)
    lw2 = zero_matrix(GF(q), 1, n)
    for x in range(w):
        lw1[:, a[x]] = randint(1, q - 1)
        lw2[:, b[x]] = randint(1, q - 1)

    return [lw1, lw2]

def sample_monomial_matrix(n, q):
    '''
    Sample random monomial matrix
    :param n: length
    :param q: modulo
    :return: monomial matrix
    '''
    P = zero_matrix(FiniteField(q), n, n)

    a = [i for i in range(0, n)]
    for i in range(n-1, 0, -1):
        j = randint(0, i)
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp

    for i in range(0, n):
        P[i,a[i]] = randint(1,q-1)

    return P

def minimal_w(n,k,q):
    '''
    Return the minimum weight of a (n,k) code over Fq
    :param n: length
    :param k: dimension
    :param q: modulo
    :return: w, minimum Hamming weight
    '''
    w = 1
    s = 0
    while True:
        s += binomial(n,w)*(q-1)**(w-1)
        if s > q**(n-k):
            return w, ceil(s/q**(n-k))
        w = w+1

def minimal_w_only_weight(n,k,q):
    '''
    Return the minimum weight of a (n,k) code over Fq
    :param n: length
    :param k: dimension
    :param q: modulo
    :return: w, minimum Hamming weight
    '''
    w = 1
    s = 0
    while True:
        s += binomial(n,w)*(q-1)**(w-1)
        if s > q**(n-k):
            return w
        w = w+1


def support_intersection(n, lw1, lw2):
    '''
    Compute the intersection of the support of two codewords, i.e.,
    the number of entries that are non-zero for both codewords
    Args:
        n: length of the codewords
        lw1: first codeword
        lw2: second codeword

    Returns: ell, the number of entries that are non-zero for both codewords
    '''
    ell = 0
    for i in range(n):
        if lw1[0][i] != 0 and lw2[0][i] != 0:
            ell += 1
    return ell

def optimal_ell(n, w):
    '''
    Value of ell s.t. it gives the minimum of the rank - obtained by computing the derivative of ell
    Args:
        n: length of the code
        w: weight of the words
    '''
    return round((4 * w - n) / 4, 4)

def sample_instance(n, k, q, w, N, verbose=False):
    """
    Sample a LCE instance together with N low-weight codewords pairs in the codes

    Parameters:
        n: code length
        k: code dimension
        q: modulo
        w: hamming weight of low-weight pairs
        N: number of low-weight codewords
        verbose: True or False for printing details or not
    Returns:
        G: generator of first code
        G_: generator of second code
        Q: monomial matrix
        pairs: low-weight codewords pairs
    """

    lw = [sample_low_weight(n, q, w) for _ in range(N)]

    while True:
        G = lw[0]
        for i in range(1, N):
            G = G.stack(lw[i])
        G = G.stack(random_matrix(GF(q), k - N, n))
        if rank(G) == k:
            break
    Q = sample_monomial_matrix(n, q)
    G_ = (G * Q).rref()
    pairs = [[lw[i], lw[i] * Q] for i in range(N)]

    if verbose:
        print("Generator matrix 1")
        print(G)
        print("Generator matrix 2")
        print(G_)
        print("Monomial matrix")
        print(Q)
        print("Low-weight pairs")
        for pair in pairs:
            print(pair)

    return G, G_, Q, pairs


def remove_duplicates(L):
    res = []
    [res.append(x) for x in L if x not in res]
    return res

def get_system_lce(G, G_):
    H_ = G_.right_kernel_matrix()
    System = G.tensor_product(H_)

    return System


def get_zero_entries(n, w1, w2):
    zero_entries = []
    for i in range(n):
        if (w1[:, i] != 0):
            for j in range(n):
                if not (w1[:, i] != 0 and w2[:, j] != 0):
                    zero_entries = zero_entries + [[i, j]]

    for i in range(n):
        if w2[:, i] != 0:
            for j in range(n):
                if (w1[:, j] == 0):
                    zero_entries = zero_entries + [[j, i]]

    res = remove_duplicates(zero_entries)

    return res


def get_zero_nonzero_entries(n, q, pairs):
    L_zero = []  # Here I save it as [i,j]
    L_nonzero = []  # Here I save it as [i,j, coefficient]

    c1 = pairs[0][0][0]
    c2 = pairs[1][0][0]
    d1 = pairs[0][1][0]
    d2 = pairs[1][1][0]

    ell1 = []
    for i1 in range(n):
        if c1[i1] != 0 and c2[i1] != 0:
            ell1 += [i1]

    ell2 = []
    for i1 in range(n):
        if d1[i1] != 0 and d2[i1] != 0:
            ell2 += [i1]

    ell = len(ell1)
    # get zero non-zero variables from the interception

    ell_ = 0
    d2_ = d2
    L_zero = []
    L_nonzero = []
    for i in range(ell):
        i1 = ell1[i]
        L_nonzero_tmp = []
        for j in range(ell):
            i2 = ell2[j]
            if c2[i1] / d2_[i2] == c1[i1] / d1[i2]:  # the entry might be non-zero
                coeff = d1[i2] / c1[i1]
                L_nonzero_tmp += [[i1, i2, coeff]]
                ell_ += 1
            else:  # Set entry to zero
                L_zero += [[i1, i2]]

        if len(L_nonzero_tmp) == 1:  # save only in case one value was found
            L_nonzero += L_nonzero_tmp

        if ell_ == 0 and ell != 0:  # if in the first loop nothing was found, is for sure the wrong value of a
            continue
    if ell_ < ell:  # it means we found enough value and most likely it was the correct value of a
        return None, None

    # get all other zero variables
    L_zero = L_zero + get_zero_entries(n, pairs[0][0], pairs[0][1]) + get_zero_entries(n, pairs[1][0], pairs[1][1])
    L_zero = remove_duplicates(L_zero)

    return L_zero, L_nonzero, ell

def reduce_system_to_non_homogeneous(n, k, q, system_, L_zero, L_nonzero):
    to_delete = []

    for l in range(len(L_zero)):
        row = L_zero[l][0]
        column = L_zero[l][1]
        to_delete = to_delete + [(row * n + column)]

    b = zero_matrix(GF(q), k * (n - k), 1)
    for l in range(len(L_nonzero)):
        row = L_nonzero[l][0]
        column = L_nonzero[l][1]
        entry = L_nonzero[l][2]
        to_delete = to_delete + [(row * n + column)]
        b -= entry*system_[:, (row * n + column)]

    A = system_.delete_columns(to_delete)

    return A, b

def reduce_system(n, system_, L):
    to_delete = []
    for l in range(len(L)):
        row = L[l][0]
        column = L[l][1]
        to_delete = to_delete + [(row * n + column)]

    reduced_system = system_.delete_columns(to_delete)

    return reduced_system

def reduce_system_lce(n, q, A, L_zero, L_nonzero):

    to_delete = []
    for l in L_zero:
        row = l[0]
        column = l[1]
        to_delete = to_delete + [(row * n + column)]
    b = zero_matrix(GF(q), A.nrows(), 1)
    for l in L_nonzero:
        row = l[0]
        column = l[1]
        value = l[2]
        to_delete = to_delete + [(row * n + column)]
        try:
            b -= value*A[:, (row * n + column)]
        except:
            print(f"A.nrows(): {A.nrows()}, b.nrows(): {b.nrows()}")
            exit()

    reduced_system = A.delete_columns(to_delete)

    return reduced_system, b

def rouche_capelli_guess_column(row, col, n, A, b, L_zero, L_nonzero, verbose=False):

    L_guess = L_zero + [[x[0], x[1]] for x in L_nonzero] + [[ii, col] for ii in range(n)] + [[row, jj] for jj in range(n)]
    L_guess = remove_duplicates(L_guess)
    c = -A[:, row*n + col]
    Aguess = reduce_system(n, A, L_guess).augment(c) # put back the column

    # Rouche-Capelli test
    Aguess_rank = Aguess.rank()
    Aguess_augmented_rank = Aguess.augment(b).rank()

    # if Sguess_rank == Sbase_rank:
    # 	print("ERROR! rank does not decrease")
    # 	continue

    if Aguess_rank != Aguess_augmented_rank:
        if verbose:
            print("Test entry: (",row, col,") - failed - ranks", Aguess_rank, Aguess_augmented_rank, ", cols", Aguess.ncols())
        return 0
    else:
        if verbose:
            print("Test entry: (",row, col,") - passed - rank", Aguess_rank, Aguess_augmented_rank, ", cols", Aguess.ncols())
        return 1

def rouche_capelli_guess_column_task(args):
    """
    Helper function to process a single column task for parallelization.
    """
    row, col, n, A, b, L_zero, L_nonzero, verbose = args
    response = rouche_capelli_guess_column(row, col, n, A, b, L_zero, L_nonzero, verbose)
    return col, response


def rouche_capelli_guess_row(n, A, b, L_zero, L_nonzero, Q_companion, row, verbose=False, parallel=False):

    columns = [col for col in range(n) if [row, col] not in L_zero]

    if len(columns) == 1:
        if verbose:
            "Only one to guess - skipping"
        return Q_companion, L_zero

    count = 0
    if parallel:
        # Create tasks for parallel execution
        tasks = [(row, col, n, A, b, L_zero, L_nonzero, verbose) for col in columns]

        # Use ProcessPoolExecutor to parallelize
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(rouche_capelli_guess_column_task, tasks))

        # Process results
        for col, response in results:
            if response == 0:
                L_zero = L_zero + [[row, col]]
                Q_companion[row, col] = 0
            else:
                count += 1
                nonzero_col = col
    else:
        # Sequential execution
        for col in columns:
            response = rouche_capelli_guess_column(row, col, n, A, b, L_zero, L_nonzero, verbose)
            if response == 0:
                L_zero = L_zero + [[row, col]]
                Q_companion[row, col] = 0
            else:
                count += 1
                nonzero_col = col


    if count == 0:
        return None, None

    if count == 1:
        for i in range(n):
            if i == row:
                continue
            L_zero = L_zero + [[i, nonzero_col]]
            Q_companion[i, nonzero_col] = 0
        L_zero = remove_duplicates(L_zero)

    if verbose:
        print("Total passed row ", row, ":", count)


    return Q_companion, L_zero

