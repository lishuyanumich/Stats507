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

# # Question 3 - Statistics 101 [40]

# In this question you will write functions to compute statistics providing point and interval estimates for common population parameters based on data. Parts a and below each ask you to write a function; each of those functions should default to returning a string of the form,
# $$\hat{\theta}\textrm{ }[XX\% CI: (\hat{\theta}_L, \hat{\theta}_U)$$

# The format of this string should also be configurable using an input parameter. Define the function to return a dictionary with keys `est`, `lwr`, `upr`, and `level` when the function is called with the parameter controlling the format of the input string set to `None`.

# Your functions should accept the data as a 1d Numpy array or any object coercable to such an array using `np.array()` and raise an informative exception if not.

# In this question you may use any function from Numpy, but may only use Scipy for the distribution functions found in the `stats` library. Your functions should not rely on any other modules.

# **a.** The standard point and interval estimate for the populaiton mean based on Normal theory takes the form $$\overline{x} \pm z \times se(x)$$ where $\bar x$ is the mean, se(x) is the standard error, and z is a Gaussian multiplier that depends on the desired confidence level. 
# Write a function to return a point and interval estimate for the mean based on these statistics.

import scipy.stats as st
import numpy as np
from scipy.stats import binom
import math
import random


def Call_CIDict():
    """
    Construct a ConfidenceInterval Format
        If `None` a dictionary with entries `mean`, `level`, `lwr`, and
        `upr` whose values give the point estimate, confidence level (as a %),
        lower and upper confidence bounds, respectively. If a string, it's the
        result of calling the `.format_map()` method using this dictionary.
        The default is "{mean:.1f} [{level:0.f}%: ({lwr:.1f}, {upr:.1f})]".
        
    Returns
    -------
    the function will return a desired ConfidenceInterval format.
        
    """
    
    ConfidenceInterval_Format = {'est':None, 'lwr': None, 'upr': None, 'level': None}
    return ConfidenceInterval_Format


#input an array
a = np.random.randint(2, size = 40)
#input an Confidence Level in range[0,1]
CL = 0.95


def Normal_Theory(numarray, CL, CI_Format):
    
    """
    Construct an estimate and confidence interval for the mean of `numarray`.

    Parameters
    ----------
    numarray : A 1-dimensional NumPy array or compatible sequence type (list, tuple).
        A data vector from which to form the estimates.
    CL : float, optional.
        The desired confidence level, converted to a percent in the output.
        The default is 0.95.
    CI_Format: str or None, optional.
        If `None` a dictionary with entries `mean`, `level`, `lwr`, and
        `upr` whose values give the point estimate, confidence level (as a %),
        lower and upper confidence bounds, respectively. If a string, it's the
        result of calling the `.format_map()` method using this dictionary.
        The default is "{mean:.2f} [{level:.0f}%: ({lwr:.2f}, {upr:.2f})]".

    Returns
    -------
    By default, the function returns a string with a 95% confidence interval
    in the form "mean [95% CI: (lwr, upr)]". A dictionary containing the mean,
    confidence level, lower, bound, and upper bound can also be returned.

    """
    
    CI_Format['level'] = str(CL*100)+"%"
    
    mean, sigma = np.mean(numarray), np.std(numarray)
    #Note:Standard error=Standard deviation / sqrt(n)
    se = sigma/ math.sqrt(len(numarray))
    CI_Format['est'] = format(mean,'.5f')
    
    z = st.norm.ppf((1+CL)/2)
    lwr = mean - z*se
    upr = mean + z*se
    CI_Format['lwr'] = format((lwr),'.5f')
    CI_Format['upr'] = format((upr),'.5f')
    
    
    String_Format = str(CI_Format['est'])+"["+CI_Format['level']+"CI:"+"("+str(CI_Format['lwr'])+","+str(CI_Format['upr'])+")]"
    return String_Format, CI_Format

CI_Format = Call_CIDict()
print(Normal_Theory(a, CL, CI_Format))


# **b.** There are a number of methods for computing a confidence interval for a population proportion arising from a Binomial experiment consisting of n independent and identically distributed (iid) Bernoulli trials. Let x be the number of successes in thes trials. In this question you will write a function to return point and interval estimates based on several of these methods. Your function should have a parameter `method` that controls the method used. Include functionality for each of the following methods.

# 1)The standard point and interval estimates for a population parameter based on the Normal approximation to the Binomial distribution takes the form $\hat{p} \pm z \times \sqrt{\hat{p} (1−\hat{p})/n}$ with $\hat{p}$ the sample proportion and z as in part a. The approximation is conventionally considered adequate when $n\hat{p} ∧n(1−\hat{p} )>12$. When this method is selected, your function should raise an informative warning if this condition is not satisfied.

def Normal_approximation(numarray, CL, CI_Format):
    
    """
    Construct point and interval estimates for a population proportion using Normal approximation method.

    Parameters
    ----------
    numarray : A 1-dimensional NumPy array or compatible sequence type (list, tuple).
        A data vector of 0/1 or False/True from which to form the estimates.
    CL : float, optional.
        The desired confidence level, converted to a percent in the output.
        The default is 0.95.
    CI_Format: str or None, optional.
        If `None` a dictionary with entries `mean`, `level`, `lwr`, and
        `upr` whose values give the point estimate, confidence level (as a %),
        lower and upper confidence bounds, respectively. If a string, it's the
        result of calling the `.format_map()` method using this dictionary.
        The default is "{mean:.1f} [{level:0.f}%: ({lwr:.1f}, {upr:.1f})]".

    Returns
    -------
    A string with a (100 * level)% confidence interval in the form
    "mean [(100 * level)% CI: (lwr, upr)]" or a dictionary containing the
    keywords shown in the string.

    """
    
    n = len(numarray)
    CI_Format['level'] = str(CL*100)+"%"
    x = np.sum(a)
    p_hat = x/n
    if (min(n*p_hat,n*(1-p_hat)) <= 12):
        return "Nomal approximation to the Binomial distribution is not adequate!"
    else:
        sigma = math.sqrt(p_hat*(1 - p_hat)/n)
        CI_Format['est'] = format(p_hat,'.5f')
        
        z = st.norm.ppf((1+CL)/2)
        lwr = p_hat - z*sigma
        upr = p_hat + z*sigma
        CI_Format['lwr'] = format(lwr ,'.5f')
        CI_Format['upr'] = format(upr,'.5f')
        
        String_Format = str(CI_Format['est'])+"["+CI_Format['level']+"CI:"+"("+str(CI_Format['lwr'])+","+str(CI_Format['upr'])+")]"
        return String_Format, CI_Format   


# 2)The Clopper-Pearson interval for a population proportion can be expressed using quantiles from Beta distributions. Specifically, for a sample of size n with x successes and $$\alpha =1 - ConfidenceLevel$$ the interval is,
#  $$(\hat {\theta}_{L}, \hat {\theta}_{U}) = ( B(\frac{\alpha}{2}, x, n - x + 1),B(1 - \frac{\alpha}{2}, x + 1, n - x)).$$

def Clopper_Pearson(numarray, CL, CI_Format):
    
    """
    Construct point and interval estimates for a population proportion using Clopper Pearson method.

    Parameters
    ----------
    numarray : A 1-dimensional NumPy array or compatible sequence type (list, tuple).
        A data vector of 0/1 or False/True from which to form the estimates.
    CL : float, optional.
        The desired confidence level, converted to a percent in the output.
        The default is 0.95.
    CI_Format: str or None, optional.
        If `None` a dictionary with entries `mean`, `level`, `lwr`, and
        `upr` whose values give the point estimate, confidence level (as a %),
        lower and upper confidence bounds, respectively. If a string, it's the
        result of calling the `.format_map()` method using this dictionary.
        The default is "{mean:.1f} [{level:0.f}%: ({lwr:.1f}, {upr:.1f})]".

    Returns
    -------
    A string with a (100 * level)% confidence interval in the form
    "mean [(100 * level)% CI: (lwr, upr)]" or a dictionary containing the
    keywords shown in the string.

    """
    
    alpha = 1 - CL
    CI_Format['level'] = str(CL*100)+"%"
    
    n = len(numarray)
    x = np.sum(a)
    theta = x/n
    CI_Format['est'] = format(theta, '.5f')
    
    lwr = st.beta.ppf(alpha/2, x, n-x+1)
    upr = st.beta.ppf(1-alpha/2, x+1, n-x)
    CI_Format['lwr'] = format(lwr ,'.5f')
    CI_Format['upr'] = format(upr,'.5f')
    
    String_Format = str(CI_Format['est'])+"["+CI_Format['level']+"CI:"+"("+str(CI_Format['lwr'])+","+str(CI_Format['upr'])+")]"
    return String_Format, CI_Format


# 3)The Jeffrey’s interval is a Bayesian credible interval with good frequentist properties. It is similar to the Clopper-Pearson interval in that it utilizes Beta quantiles, but is based on a so-called Jeffrey’s prior of $B(p,0.5,0.5)$. Specifically, the Jeffrey’s interval is $(0∨B(α/2,x+0.5,n−x+0.5),B(1−α/2,x+0.5,n−x+0.5)∧1)$. (Use the sample proportion as the point estimate).

def Jeffrey(numarray, CL, CI_Format):
    
        """
    Construct point and interval estimates for a population proportion using Jeffrey method.

    Parameters
    ----------
    numarray : A 1-dimensional NumPy array or compatible sequence type (list, tuple).
        A data vector of 0/1 or False/True from which to form the estimates.
    CL : float, optional.
        The desired confidence level, converted to a percent in the output.
        The default is 0.95.
    CI_Format: str or None, optional.
        If `None` a dictionary with entries `mean`, `level`, `lwr`, and
        `upr` whose values give the point estimate, confidence level (as a %),
        lower and upper confidence bounds, respectively. If a string, it's the
        result of calling the `.format_map()` method using this dictionary.
        The default is "{mean:.1f} [{level:0.f}%: ({lwr:.1f}, {upr:.1f})]".

    Returns
    -------
    A string with a (100 * level)% confidence interval in the form
    "mean [(100 * level)% CI: (lwr, upr)]" or a dictionary containing the
    keywords shown in the string.

    """
    
    alpha = 1 - CL
    CI_Format['level'] = str(CL*100)+"%"
    n = len(numarray)
    x = np.sum(a)
    theta = x/n
    CI_Format['est'] = format(theta, '.5f')
    
    lwr = max(0, st.beta.ppf(alpha/2, x+0.5, n-x+0.5))
    upr = min(st.beta.ppf(1-alpha/2, x+0.5, n-x+0.5), 1)
    CI_Format['lwr'] = format(lwr ,'.5f')
    CI_Format['upr'] = format(upr,'.5f')
    
    String_Format = str(CI_Format['est'])+"["+CI_Format['level']+"CI:"+"("+str(CI_Format['lwr'])+","+str(CI_Format['upr'])+")]"
    return String_Format, CI_Format   


# 4)Finally, the Agresti-Coull interval arises from a notion “add 2 failures and 2 successes” as a means of regularization. More specifically, define $\tilde{n} =n+z^2$ and $\tilde{p} =(x+z^2/2)/\tilde{n}$ . The Agresti-Coull interval is Normal approximation interval using $\tilde{p}$ in place of $\hat {p}$.

# > According to wikipedia "Agresti- Coull interval" definition, we can see the confidence interval for $p$ is give by
# $$\tilde{p}\pm z\sqrt{\tilde{p} (1−\tilde{p})/\tilde{n}}$$

def Agresti_Coull(numarray, CL, CI_Format):
    
    """
    Construct point and interval estimates for a population proportion using Agresti Coull method.

    Parameters
    ----------
    numarray : A 1-dimensional NumPy array or compatible sequence type (list, tuple).
        A data vector of 0/1 or False/True from which to form the estimates.
    CL : float, optional.
        The desired confidence level, converted to a percent in the output.
        The default is 0.95.
    CI_Format: str or None, optional.
        If `None` a dictionary with entries `mean`, `level`, `lwr`, and
        `upr` whose values give the point estimate, confidence level (as a %),
        lower and upper confidence bounds, respectively. If a string, it's the
        result of calling the `.format_map()` method using this dictionary.
        The default is "{mean:.1f} [{level:0.f}%: ({lwr:.1f}, {upr:.1f})]".

    Returns
    -------
    A string with a (100 * level)% confidence interval in the form
    "mean [(100 * level)% CI: (lwr, upr)]" or a dictionary containing the
    keywords shown in the string.

    """
    
    CI_Format['level'] = str(CL*100)+"%"
    z = st.norm.ppf((1+CL)/2)
    x = np.sum(a)
    n_tilde = len(numarray) + z**2
    p_tilde = (x + z**2/2)/n_tilde
    CI_Format['est'] = format(p_tilde,'.5f')
    
    sigma = math.sqrt(p_tilde*(1 - p_tilde)/n_tilde)
    
    lwr = p_tilde - z*sigma
    upr = p_tilde + z*sigma
    CI_Format['lwr'] = format(lwr ,'.5f')
    CI_Format['upr'] = format(upr,'.5f')
    
    String_Format = str(CI_Format['est'])+"["+CI_Format['level']+"CI:"+"("+str(CI_Format['lwr'])+","+str(CI_Format['upr'])+")]"
    return String_Format, CI_Format


# ### The following is a function to return point and interval estimates based on these 4 methods shown above.

def Interval_method(method, a, CL):
    
    """
    Construct point and interval estimates for a population proportion.
    The "method" argument controls the estimates returned. Available methods
    are "Normal", to use the normal approximation to the Binomial, "CP" to
    use the Clopper-Pearson method, "Jeffrey" to use Jeffery's method, and
    "AC" for the Agresti-Coull method.

    Parameters
    ----------
    
    method : str, optional
        The type of confidence interval and point estimate desired.  Allowed
        values are "Normal" for the normal approximation to the Binomial,
        "CP" for a Clopper-Pearson interval, "Jeffrey" for Jeffrey's method,
        or "AC" for the Agresti-Coull estimates.
    a : A 1-dimensional NumPy array or compatible sequence type (list, tuple).
        A data vector of 0/1 or False/True from which to form the estimates.
    CL : float, optional.
        The desired confidence level, converted to a percent in the output.
        The default is 0.95.

    Returns
    -------
    A string with a (100 * level)% confidence interval in the form
    "mean [(100 * level)% CI: (lwr, upr)]" or a dictionary containing the
    keywords shown in the string.

    """
    
    if method == "Normal_approximation":
        return Normal_approximation(a, CL, Call_CIDict())
    elif method == "Clopper_Pearson":
        return Clopper_Pearson(a, CL, Call_CIDict())
    elif method == "Jeffrey":
        return Jeffrey(a, CL, Call_CIDict())
    elif method == "Agresti_Coull":
        return Agresti_Coull(a, CL, Call_CIDict())
    else:
        return "No such method!"


print(PPCInterval_method("Normal_approximation", a, CL))
print(PPCInterval_method("Clopper_Pearson", a, CL))
print(PPCInterval_method("Jeffrey", a, CL))
print(PPCInterval_method("Agresti_Coull", a, CL))

# **c.** Create a 1d Numpy array with 42 ones and 48 zeros. Construct a nicely formatted table comparing 90, 95, and 99% confidence intervals using each of the methods above (including part a) on this data. Choose the number of decimals to display carefully to emphasize differences. For each confidence level, which method produces the interval with the smallest width?

# +
#Create a 1d Numpy Array of 90 zeros
oneD_Array = np.zeros((90,), dtype = int)
#in range(0, 90), generate 42 positions for 1

one_pos = range(0,90)
one_positions = random.sample(one_pos,42)
#Successfully create a 1d Numpy array with 42 ones and 48 zeros
for i in range(42):
    temp = one_positions[i]
    oneD_Array[temp] = 1
# -

CL_array = np.array([0.90, 0.95, 0.99])

width = np.zeros((3,5))
import pandas as pd
dataname = []
df = pd.DataFrame()
df

for i in range(3):
    
    CI_Format0, dict0 = Normal_Theory(oneD_Array, CL_array[i], CI_Format)
    width[i][0] = format(float(dict0['upr']) - float(dict0['lwr']),'.5f')
    Intl0 = CI_Format0[13:33] + "||width:" + str(width[i][0])
    
    
    CI_Format1, dict1 = Normal_approximation(oneD_Array, CL_array[i], CI_Format)
    width[i][1] = format(float(dict1['upr']) - float(dict1['lwr']),'.5f')
    Intl1 = CI_Format1[13:33] + "||width:" + str(width[i][1])
    
    CI_Format2, dict2 = Clopper_Pearson(oneD_Array, CL_array[i], CI_Format)
    width[i][2] = format(float(dict2['upr']) - float(dict2['lwr']),'.5f')
    Intl2 = CI_Format2[13:33] + "||width:" + str(width[i][2])
    
    
    CI_Format3, dict3 = Jeffrey(oneD_Array, CL_array[i], CI_Format)
    width[i][3] = format(float(dict3['upr']) - float(dict3['lwr']),'.5f')
    Intl3 = CI_Format3[13:33] + "||width:" + str(width[i][3])
        
    CI_Format4, dict4 = Agresti_Coull(oneD_Array, CL_array[i], CI_Format)
    width[i][4] = format(float(dict4['upr']) - float(dict4['lwr']),'.5f')
    Intl4 = CI_Format4[13:33] + "||width:" + str(width[i][4])
       
    data = {"Normal_Theory": Intl0,
            "Normal_approximation":Intl1,
            "Clopper_Pearson":Intl2,
            "Jeffrey":Intl3,
            "Agresti_Coull":Intl4}
    indexname = "CL="+str(CL_array[i]*100)+"%"
    df1 = pd.Series(data, name = indexname)
    df = df.append(df1)

width

df

# **From the above table, we can conclude that Jeffrey method produces the interval with the smallest width.**
