## Scripts relative to Section 3
_________
### Plot rank vs n. columns vs n. rows vs expected columns
The script `get_rank_vs_cols_vs_rows.py` gives the data to produce the plot comparing, for each code rate, the average of the ranks, number of columns, rows, and expected columns of the system before the guessing phase.

**Usage example**
```
sage -python get_rank_vs_cols_vs_rows.py --n 40 --q 31 --trials 10
```
_________
### Solve a random LCE instance with two pair of equivalent codewords
The script `solve_random_lce_instance.py` samples a random LCE instance along with two pair of equivalent codewords, and tried to solve it.

**Usage example**
````
sage -python solve_random_lce_instance.py --n 40 --q 31 --k 20 
>> Success
````
Use `--verbose` to activate the verbosity in stdout and `--parallel` to make guesses in paralell (useful only for large dimensions >= 80).
One can specify the Hamming weight of the equivalent codewords with the command `--w`, otherwise is set by default to the minimum weight of the code.
_________
### Test Heuristic 1
The script `test_heuristic.py` tries to solve a number of LCE instances + two pairs of equivalent codewords, and returns the success rate.
Parallelization is enabled by default. 

**Usage example**
```
sage -python test_heuristic.py --n 40 --k 10 --q 31 --trials 10 --parallel
>> n: 40, k: 20, q: 31, w: 15 - Result: 10/10: 1.0
```
The Hamming weight of the equivalent codewords is set by default to the minimum weight of the code.
_________
### Generate success region plot
The script `get_solver_regions.py` solves LCE instances for a range of `n` and rates in `(0,0.5]` and a specific `q`, and prints the results as:
- -1 : setting not analyzed
- 0 : failure
- 1 : solved
- 2 : solved without guessing

**Usage example**
```
sage -python get_solver_regions.py --min_n 20 --max_n 40 --q 17 --step 2
```

Parallelization is enabled by default. The Hamming weight of the equivalent codewords is set by default to the minimum weight of the code.
_________
These scripts do not address the code rates `R \in (0.5,1)`. In these cases, one must compute the dual of the codes with rates `1-R` and solve the LCE instance on these.