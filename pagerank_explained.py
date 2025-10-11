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
  - If M is diagonalizable, write M = C D C^{-1} with eigenvalues on D's diagonal.
  - For PageRank, the relevant matrix is the Google matrix M = d L + (1-d)/n 11^T.
  - The top eigenvalue is λ₁ = 1 (Perron); all others satisfy |λ_i| ≤ d < 1 when 0<d<1.
  - Then M^k = C D^k C^{-1} → C diag(1,0,...,0) C^{-1}: a projector onto the Perron eigenvector v.
  - Power iteration r_{k+1} = M r_k therefore converges to v (up to scaling). For d=1, convergence
    depends on L (it still works in the supplied worksheet setup).
"""

import numpy as np
import numpy.linalg as la


# ---------------------------------------------------------------
# GRADED FUNCTION
# Complete this function to provide the PageRank for an arbitrarily sized internet.
# I.e. the principal eigenvector of the damped system, using the power iteration method.
# (Normalisation doesn't matter here)
# The functions inputs are the linkMatrix, and d the damping parameter - as defined in this worksheet.
# (The damping parameter, d, will be set by the function - no need to set this yourself.)
def pageRank(linkMatrix, d) :
    n = linkMatrix.shape[0]
    # Start from a uniform ranking that sums to 100 (worksheet convention)
    r = 100 * np.ones(n) / n

    # Google/transition matrix for the damped random surfer
    # (For d=1, this reduces to the provided linkMatrix.)
    M = d * linkMatrix + (1-d)/n * np.ones([n,n])

    # First step
    lastR = r
    r = M @ r

    # Iterate until change is small in 2-norm (worksheet uses 0.01 threshold)
    while la.norm(lastR - r) > 0.01:
        lastR = r
        r = M @ r

    return r
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Alternative (reference only): principal eigenvector approach on L
# NOTE: This is dense and returns a vector that we then normalise to sum to 100
# to match the worksheet's display convention.

def pagerank_via_eig(L):
    eVals, eVecs = la.eig(L)  # Gets the eigenvalues and vectors
    order = np.absolute(eVals).argsort()[::-1]  # Orders them by their eigenvalues (magnitude)
    eVals = eVals[order]
    eVecs = eVecs[:,order]

    r = eVecs[:, 0]  # Sets r to be the principal eigenvector
    r = 100 * np.real(r / np.sum(r))  # Scale so components sum to 100
    return r
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# TEST HARNESS (exactly as you requested)
# L = generate_internet(50)
# pageRank(L, 1)
# ---------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Your worksheet should provide this function.
        L = generate_internet(50)  # noqa: F821 (provided in the worksheet environment)
        print("Power method, d=1:")
        print(pageRank(L, 1))
        print("
Eigen (reference on L), scaled to sum to 100:")
        print(pagerank_via_eig(L))
    except NameError:
        # If run outside the worksheet, show the expected array for your reference.
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
