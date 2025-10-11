# -*- coding: utf-8 -*-
"""
Minimal PageRank implementation in the exact worksheet style you expect.

Includes:
  1) The # GRADED FUNCTION `pageRank(linkMatrix, d)` using the power method.
  2) A small "alternative" using eigen-decomposition (for comparison only).
  3) A test harness that calls `generate_internet(50)` and runs `pageRank(L, 1)`.
     - If your environment provides `generate_internet`, you'll see the real output.
     - If not, we print the expected array from the worksheet so you can compare.

Why this converges (short sketch):
  - If M is diagonalizable, write M = C D C^{-1} with eigenvalues on D's diagonal (C - eigenbasis matrix, D - diagonal matrix of eigenvalues, C^{-1} - inverse of the eigenvector matrix)
  - For PageRank, the relevant matrix is the Google matrix M = d L + (1-d)/n 11^T.
  - The top eigenvalue is λ₁ = 1 (Perron); all others satisfy |λ_i| ≤ d < 1 when 0<d<1.
  - Then M^k = C D^k C^{-1} → C * diag(1,0,...,0) * C^{-1}: a projector onto the Perron eigenvector v (as λ_i where i!=1 lim -> 0 as k lim->inf)
  - Power iteration r_{k+1} = M r_k therefore converges to v (up to scaling). For d=1, convergence
    depends on L (it still works in the supplied worksheet setup).
"""

import numpy as np
import numpy.linalg as la


# ---------------------------------------------------------------
def pageRank(linkMatrix, d) :
    n = linkMatrix.shape[0]
    # Start from a uniform ranking that sums to 100 
    r = 100 * np.ones(n) / n

    M = d * linkMatrix + (1-d)/n * np.ones([n,n])

    lastR = r
    r = M @ r

    # Iterate until change is small in 2-norm 
    while la.norm(lastR - r) > 0.01:
        lastR = r
        r = M @ r

    return r
# ---------------------------------------------------------------

#Alternative (reference only): principal eigenvector approach on L
de pagerank_via_eig(L):
    eVals, eVecs = la.eig(L)  # Gets the eigenvalues and vectors
    order = np.absolute(eVals).argsort()[::-1]  # Orders them by their eigenvalues (magnitude)
    eVals = eVals[order]
    eVecs = eVecs[:,order]

    r = eVecs[:, 0]  # Sets r to be the principal eigenvector
    r = 100 * np.real(r / np.sum(r))  # Scale so components sum to 100
    return r
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# TEST  
# L = generate_internet(50)
# pageRank(L, 1)
# ---------------------------------------------------------------
if __name__ == "__main__":
    try:
        L = generate_internet(50) #  returns an n×n column-stochastic link matrix L where L[i,j] is the probability of moving from page j to page i. (at the bottom more info)
        print("Power method, d=1:")
        print(pageRank(L, 1))
        print("
Eigen (reference on L), scaled to sum to 100:")
        print(pagerank_via_eig(L))
    except NameError:
        # The expected array for your reference.
        print("Note: 'generate_internet' was not found. In your worksheet, it will be provided.")
        print("Expected output for pageRank(L, 1) with L = generate_internet(50):")
        expected = np.array([
            13.181893  ,   0.00311092,   0.00810856,   0.01277125,
             0.00598534,   0.00812891,   0.00172962,   0.01052407,
            60.05367711,   0.0066632 ,   0.00172962,   0.00593847,
             0.00533566,  13.181893  ,   0.0211951 ,   0.0047505 ,
             0.01381387,   0.01188144,   0.00172962,   0.01687346,
             0.01086087,   0.00172962,   0.00264135,   0.00458939,
             0.00483148,   0.00172962,   0.00847205,   0.00794855,
             0.00847331,   0.01077736,   0.02491321,   0.02058242,
             0.01246536,   0.0080947 ,   0.00528189,   0.01149876,
             0.01343239,   0.01069586,   0.01343239,  13.181893  ,
             0.00953283,   0.00172962,   0.00943397,   0.01427835,
             0.00449858,   0.00358666,   0.01439685,   0.01091736,
             0.00596288,   0.00358666
        ])
        print(expected)

        # Also show why power iteration converges: if M=C D C^{-1}, then M^k r0 → const * v (Perron)
        print("
Sketch: If M=C D C^{-1} with λ₁=1, |λ_i|<1 for i>1, then M^k=C D^k C^{-1} → projector onto v.")
**
# ---------------------------------------------------------------------
# About `generate_internet` (provided by the worksheet):
#
# - Purpose: returns a synthetic “internet” of size n as a link matrix L.
# - Shape: L is n x n (square).
# - Meaning: columns are source pages; L[i, j] is the probability of moving
#   from page j to page i (i.e., each column lists outgoing-link probabilities).
# - Stochasticity: each column of L sums to 1 (within numerical tolerance).
# - Dangling pages: any page with no out-links is handled in the worksheet
#   (typically by assigning a uniform 1/n distribution for that column),
#   so L remains column-stochastic.
# - Usage here: when d = 1 we iterate with M = L; for 0 < d < 1 we form the
#   damped Google matrix M = d*L + (1-d)/n * 11^T before power iteration.
# - Display convention: we scale the final rank vector so its entries sum to
#   ~100, matching the worksheet’s expected output format.
# ---------------------------------------------------------------------
