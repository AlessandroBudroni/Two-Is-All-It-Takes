
from collections import defaultdict

# Modular inverse in F_q
def modinv(a, q):
    a = int(a)
    if a == 0:
        raise ValueError("No inverse for zero")
    return pow(a, -1, q)

# Classify columns of a 2xN matrix based on the zero/nonzero pattern
def classify_columns_by_type(A, q):
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

# Subset sum using dynamic programming - modified so that it returns a solution set with maximised number of elements.
def subset_sum_dp_longest(nums, target, no_single_solution=False):

    if no_single_solution:
        nums = [n for n in nums if n < target]

    n = len(nums)
    max_target = target + 20

    # dp[i][t] = max number of elements to reach t using first i items, or -inf if impossible
    dp = [[-10**9] * (max_target) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        for t in range(max_target):
            # option 1: don't take nums[i-1]
            dp[i][t] = max(dp[i][t], dp[i-1][t])
            # option 2: take nums[i-1]
            if t >= nums[i-1] and dp[i-1][t - nums[i-1]] >= 0:
                dp[i][t] = max(dp[i][t], dp[i-1][t - nums[i-1]] + 1)

    # find the closest achievable sum
    for offset in range(max_target):
        lower = target - offset
        upper = target + offset
        if lower >= 0 and dp[n][lower] >= 0:
            closest = lower
            break
        if upper < max_target and dp[n][upper] >= 0:
            closest = upper
            break
    else:
        return None, []

    # backtrack to recover subset
    subset = []
    i, t = n, closest
    while i > 0 and t >= 0:
        if t >= nums[i-1] and dp[i][t] == dp[i-1][t - nums[i-1]] + 1:
            subset.append(nums[i-1])
            t -= nums[i-1]
        i -= 1

    return closest, subset[::-1]

