## Scripts relative to Section 3
The directory `/nowaskowski` contains two modified python files from [https://github.com/juliannowakowski/lep-cf](https://github.com/juliannowakowski/lep-cf) by Julian Nowakowski.
_________

### 1. Solve a random LCE instances with two pair of equivalent codewords
The script `test_solver.py` samples a random LCE instance along with two pair of equivalent codewords, and tries to solve them. The script was used for filling Table 2.

**Usage example**
```
sage -python test_solver.py --n 128 --q 127 --k 64 --n_trials 10 
```
Use `--parallel` to solve `n_trials` instances in parallel. NOTE: parallelizing usually affects the average speed of each single instance.
One can specify the Hamming weight of the equivalent codewords with the command `--w`, otherwise is set by default to the minimum weight of the code.
Tested for code rates `R \in (0,0.5]`.
_________

### 2. Theoretical `q0` for approximate SSP solution for all rates
The script `theoretical_q0_approximate_ssp.py` computes the constant `q0` from Lemma 2 for all code rates between 0 and 0.5.
The script was used to generate part of Fig. 1.

**Usage example**
```
python3 theoretical_q0_approximate_ssp.py --n 1000
```
---------

### 3. Experimental minimal `q` for exact SSP solution for all rates
The script `experimental_q0_exact_ssp.py` computes the minimal modulo `q` such that 10/10 random SSP instances as per Algorithm 1 could be solved exactly.
This script was used to generate part of Fig. 1.

**Usage example**
```
sage -python experimental_q0_exact_ssp.py --n 256 --step 1
```
For each rate and each `q`, the script considers `w` as the estimated minimum weight of the code.
_________

### 4. Experimental minimal q for exact SSP solution for n as power of q
The script `estimate_exact_q0_for_2pow.py` computes the minimal modulo `q` such that 10/10 random SSP instances as per Algorithm 1 could be solved exactly.
This script was used to generate Fig. 2. It is designed to work for power of `n` and the rate is fixed.
**Usage example**
```
sage -python estimate_exact_q0_for_2pow.py --min_n 6 --max_n 10 --rate 0.5
```
The above tries for all `n=2^x` for `x=6,7,8,9,10`. Use `--parallel` to enable parallization. 

NOTE: if a `q` is found for `x`, then `x+1` is tested starting from the same `q`. Start from `min_n 6` as for very small values of `n` the behavior is not indicative.
For large `x` (e.g. `x=18`), it becomes quite slow.
_________

### 5. Obtainable sums
The notebook `obtainable_sums.ipynb` computes all possible obtainable sums from see input set S.
This script was used to generate Fig. 3.

For a detailed description of the dependencies to run jupyter notebooks please see the readme in the `section4 & section5` folder.



