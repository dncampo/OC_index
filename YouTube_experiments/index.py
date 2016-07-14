"""
Created on Fri Oct  3 18:53:19 2014
@author: dncampo
"""
from math import sqrt, factorial
from numpy import power, sum


def nCr(n, r):
    f = factorial
    return f(n) / f(r) / f(n - r)


# Overlapped Clusters index
def oc(M, C1, C2):
    """
    Implementation of OC index defined in the paper
    "A new index for clustering validation with overlapped clusters"
    """
    k1 = C1.shape[0]
    k2 = C2.shape[0]

    n1 = C1.sum()
    n2 = C2.sum()  # patterns that can be counted

    n = C1.shape[1]  # the original number of patterns

    ks1 = C1.sum(1)  # row sum
    ks2 = C2.sum(1)

    t_tilde = 0
    for i in range(k1):
        for j in range(k2):
            if (M[i, j] > 1):
                t_tilde = t_tilde + nCr(M[i, j], 2)

    t_tilde = t_tilde / ((n * (n - 1) / 2.0) * (max(n1, n2) / float(n)) * min(k1, k2))  # max

    p_tilde = 0
    for i in range(k1):
        if (ks1[i] > 1):
            p_tilde = p_tilde + nCr(ks1[i], 2)

    p_tilde = p_tilde / (k1 * n * (n - 1) / 2.0)  # PAPER

    q_tilde = 0
    for j in range(k2):
        if (ks2[j] > 1):
            q_tilde = q_tilde + nCr(ks2[j], 2)

    q_tilde = q_tilde / (k2 * n * (n - 1) / 2.0)  # PAPER

    oc = 0
    if t_tilde > 0:
        oc = t_tilde / max(p_tilde, q_tilde)

    return oc


# Fowlkes-Mallows index
def fm(M, N):
    """
    Implementation of FM index defined in the paper:
    E. B. Fowkles and C. L. Mallows, 1983.
    "A method for comparing two hierarchical clusterings".
    Journal of the American Statistical Association
    """
    tk = power(M, 2).sum() - N
    pk = power(M.sum(1), 2).sum() - N
    qk = power(M.sum(0), 2).sum() - N

    bk = 0.0

    if (tk > 0.0):
        bk = float(tk) / sqrt(pk * qk)

    return bk


# Jaccard index
def jac(M, N):
    tk = power(M, 2).sum() - N
    pk = power(M.sum(1), 2).sum() - N
    qk = power(M.sum(0), 2).sum() - N

    bk = float(tk) / (pk + qk - tk)

    return bk
