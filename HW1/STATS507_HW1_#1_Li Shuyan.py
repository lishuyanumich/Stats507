# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Question 1 - Fibonacci Sequence [30]

# The Fibonacci seuqence $F_n$ begins with the numbers $0,1,1,...$ and continues so that each entry is the sum of adding its two immediate predecessors. The sequence can be defined as follows,

# $F_n = F_{n-2} + F_{n-1}$, with $F_0 =0$, $F_1 = 1$

# A common question in a code interview asks the interviewee to sketch a program generating Fibonacci numbers at a board. In this quesiton, you will write and compare several such programs. After each function definition, write a test to ensure your function returns the following vlaues:

# $F_7 =13$, $F_{11} =89$, and $F_{13} =233$

# a. Write a recursive function `fib_rec()` that takes a single input n and returns the value of $F_n$.

import numpy as np
import pandas as pd
from math import floor 
from timeit import Timer
from collections import defaultdict
from IPython.core.display import display, HTML
from scipy.stats import norm, binom, beta
from warnings import warn


def fib_rec(n, a, b):
    
    """
    Compute the Fibonacci number $F_n$, when $F_0 = a$ and $F_1 = b$.
    
    This function computes $F_n$ using a linear-complexity recursion.

    Parameters
    ----------
    n : int
        The desired Fibonacci number $F_n$.
    a, b : int, optional.
        Values to initialize the sequence $F_0 = a$, $F_1 = b$.

    Returns
    -------
    The Fibonacci number $F_n$.

    """
    
    F_0 = a
    F_1 = b
    
    if n == 0:
        return a
    elif n == 1:
        return b
    else:
        F_n = fib_rec(n-2, a, b)+fib_rec(n-1, a, b)
        return F_n


print(fib_rec(7, 0, 1))
print(fib_rec(11, 0, 1))
print(fib_rec(13, 0, 1))


# b. Write a function `fib_for()` with the same signature that computes $F_n$ by summation using a `for` loop.

def fib_for(n, a, b):
    
    """
    Compute the Fibonacci number $F_n$, when $F_0 = a$ and $F_1 = b$.

    This function computes $F_n$ directly by iterating using a for loop.

    Parameters
    ----------
    n : int
        The desired Fibonacci number $F_n$. 
    a, b : int, optional.
        Values to initialize the sequence $F_0 = a$, $F_1 = b$. 

    Returns
    -------
    The Fibonacci number $F_n$. 

    """
    
    F_0 = a
    F_1 = b
    
    if n == 0:
        return F_0
    elif n == 1:
        return F_1
    else:
        x = F_0
        y = F_1
        for i in range(1, n):
            tmp = x + y
            x,y = y,x+y
        return tmp


print(fib_for(7, 0, 1))
print(fib_for(11, 0, 1))
print(fib_for(13, 0, 1))


# c. Write a function `fib_whl()` with the same signature that computes $F_n$ by summation using a `while` loop.

def fib_whl(n, a, b):
    
    """
    Compute the Fibonacci number $F_n$, when $F_0 = a$ and $F_1 = b$.

    This function computes $F_n$ by direct summation, iterating using a
    while loop.

    Parameters
    ----------
    n : int
        The desired Fibonacci number $F_n$.
    a, b : int, optional.
        Values to initialize the sequence $F_0 = a$, $F_1 = b$.

    Returns
    -------
    The Fibonacci number $F_n$.

    """
    
    F_0 = a
    F_1 = b
    
    if n == 0:
        return F_0
    elif n == 1:
        return F_1
    else:
        x = F_0
        y = F_1
        i = 1
        while(i < n):
            tmp = x + y
            x,y = y,x+y
            i = i + 1
        return tmp


print(fib_whl(7, 0, 1))
print(fib_whl(11, 0, 1))
print(fib_whl(13, 0, 1))


# d. Write a function `fib_rnd()` with the same signature that computes $F_n$ using the rounding method described on the Wikipedia page linked above.

def fib_rnd(n, a, b):
    
    """
    Directly compute the Fibonacci number $F_n$, when $F_0 = a$ and $F_1 = b$.

    This function computes $F_n$ by rounding $\phi^n / sqrt(5)$.
    The formula is used directly for n < 250, and is applied on the log scale
    for 250 <= n < 1478. A ValueError is raised for larger n to avoid
    overflow errors.


    Parameters
    ----------
    n : int
        The desired Fibonacci number $F_n$, must be less than 1478.
    a, b : int, optional.
        Values to initialize the sequence $F_0 = a$, $F_1 = b$.

    Returns
    -------
    The Fibonacci number $F_n$ if n < 1478, else a ValueError.
    """
    
    U_0 = a
    U_1 = b
    phi = (1+np.sqrt(5))/2
    F_n = int((U_1 - U_0)*round(phi**n/np.sqrt(5)))
    return F_n


print(fib_rnd(7, 0, 1))
print(fib_rnd(11, 0, 1))
print(fib_rnd(13, 0, 1))


# e. Write a function `fib_flr()` with the same signature that computes $F_n$ using the truncation method described on the Wikipedia page linked above.

def fib_flr(n, a, b):
    
    """
    Directly compute the Fibonacci number $F_n$, when $F_0 = a$ and $F_1 = b$.

    This function computes $F_n$ by finding the smallest integer less than
    $\phi^n / sqrt(5) + 0.5$. The formula is used directly for n < 250, and is
    applied on the log scale for 250 <= n < 1477. A ValueError is raised for
    larger n to avoid integer overflow.


    Parameters
    ----------
    n : int
        The desired Fibonacci number $F_n$, must be less than 1478.
    a, b : int, optional.
        Values to initialize the sequence $F_0 = a$, $F_1 = b$.

    Returns
    -------
    The Fibonacci number $F_n$ if n < 1477, else a ValueError.
    """
    
    U_0 = a
    U_1 = b
    phi = (1+np.sqrt(5))/2
    F_n = (U_1 - U_0) * math.floor((phi**n/np.sqrt(5)+1/2))
    return F_n


print(fib_flr(7, 0, 1))
print(fib_flr(11, 0, 1))
print(fib_flr(13, 0, 1))

# f. For a sequence of increasingly large values of `n` compare the median computation time of each of the functions above. Present your results in a nicely formatted table. (Point estimates are sufficient).

# +
#Initialize a and b
a = 0
b = 1


# timing comparisons: ---------------------------------------------------------
n_mc = 10000
res = defaultdict(list)
n_seq = [21, 42, 233, 1001]
res['n'] = n_seq
for f in (fib_rec, fib_for, fib_whl, fib_rnd, fib_flr):
    for n in n_seq:
        t = Timer('f(n, a, b)', globals={'f': f, 'n': n, 'a': a, 'b': b})
        m = np.median([t.timeit(1) for i in range(n_mc)]) 
        res[f.__name__].append(round(m * 1e6, 1))
# -

# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 1.</b> <em> Timing comparisons for Fibonacci functions.</em>
Median computation times, in micro seconds, from 10,000 trial runs at
each n.  While the direct computation methods are faster, they become 
inaccurate for n > 71 due to finite floating point precision. 
"""
res = pd.DataFrame(res)
t1 = res.to_html(index=False)
t1 = t1.rsplit('\n')
t1.insert(1, cap)
tab1 = ''
for i, line in enumerate(t1):
    tab1 += line
    if i < (len(t1) - 1):
        tab1 += '\n'

display(HTML(tab1))

# Obviously, `recursive function` is the slowest way to calculate Fibonacci Sequence.
# With small n, we can find that `for loop` and `while loop` are the most efficient way and recursive function is the slowest way to calculate Fibonacci Sequence.

# Since recursive method is so time-consuming, we want to compare the other 4 methods with larger n.

# However, when n grows larger, `fib_flr` function runs fastest.
