# -*- coding: utf-8 -*-
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

# # Question 2 - Pascal’s Triangle [20]

# Pascal’s triangle is a way to organize, compute, and present the Binomial coefficients (${n \atop k}$). The triangle can be constructed from the top starting with rows 0 [(${0 \atop 0}$)=1], and 1 [(${1 \atop 0}$)=1, (${1 \atop 1}$)=1]. From there, subsequent rows can be computed by adding adjacent entries from the previous row (implicitly appending zeros on the left and right).

# a. Write a function to compute a specified row of Pascal’s triangle. An arbitrary row of Pascal’s triangle can be computed efficiently by starting with (${n \atop 0}$)=1 and then applying the following recurrence relation among binomial coefficients,

# (${n \atop k}$) = (${n \atop k-1}$)$\times$${n+1-k}\over{k}$

def getRow(n):
    
    """
    Compute an arbitrary row of Pacal's triangle.

    Row 0 is "1", row 1 is "1, 1".  

    Parameters
    ----------
    n : int
        The desired row.
 
    Returns
    -------
    A list with values for the desired row. 
    """
    
    row_lst = []
    def getElement(n, k):
        if (k == 0):
            return 1
        else:
            return (getElement(n, k-1)*((n+1-k)/k))
    for i in range(0,n+1):
        row_lst.append(int(getElement(n, i)))
    return row_lst


getRow(4)


# b. Write a function for printing the first n rows of Pascal’s triangle using the conventional spacing with the numbers in each row staggered relative to adjacent rows. Use your function to display a minimum of 10 rows in your notebook.

def Pascal_func(rownum):
    
    """
    Compute an arbitrary row of Pacal's triangle.

    Row 0 is "1", row 1 is "1, 1".  

    Parameters
    ----------
    rownum : int
        The desired number of rows.
    
    Returns
    -------
    A string which, when printed, displays the first n rows. 
    """
    
    Pascal_tri = []
    for i in range(0, rownum):
        Pascal_tri.append(getRow(i))                       
    return(Pascal_tri)
Pascal_func(10)


def triangle_print(n):

    """
    Print an Pacal's triangle.

    Parameters
    ----------
    n : int
        The desired number of rows.
    
    Returns
    -------
    There is no return.
    """    
    
    
    Tri = Pascal_func(n)
    for i in range(n):
        for j in range(n-i):
            print(format(" ","<2"), end = "")
        for j in range(i+1):
            print(format(Tri[i][j],"<4"), end = "")
        print()

triangle_print(10)
