# PageRank Algorithm — Power Iteration and Eigenvector Analysis in Python

This project implements **Google’s PageRank algorithm from scratch**, using both the **Power Iteration method** and an **Eigenvector decomposition** approach.  
It demonstrates how linear algebra (eigenvalues and eigenvectors) can model the ranking of web pages based on their hyperlink structure.

---

🧠 Overview (Simplified)

The PageRank algorithm gives each webpage an importance score based on how other pages link to it.
It imagines a “random surfer” — someone who keeps clicking links from one page to another on the internet.

Over time, some pages get visited more often than others. Those pages are considered more important.

Mathematically, we can describe this behavior using a matrix equation:

𝑟 = 𝑀𝑟

Here’s what each symbol means:
r → the vector of page ranks (how important each page is)
M → the “Google matrix” that represents all the link connections between pages
L → the basic link matrix (each column shows where a page links to)
d → the “damping factor” (usually 0.85), which means the surfer follows a link 85% of the time and randomly jumps to a new page 15% of the time

In simple terms:

The PageRank vector r is the steady-state result of repeatedly clicking through links (following M) until the system settles.
The page with the highest value in r is the one the surfer visits most often — that’s the most important page.

---

## ⚙️ Features

- ✅ **Implements the Power Method** (`pageRank(linkMatrix, d)`)  
  Iteratively computes the principal eigenvector of the damped Google matrix.  
- ✅ **Alternative Eigen-Decomposition Approach** for comparison (`pagerank_via_eig(L)`).
- ✅ Works with an **arbitrarily sized “internet”** (e.g., 50 pages using `generate_internet(50)`).
- ✅ Supports the **damping factor** `d` to simulate teleportation (stability + realism).
- ✅ Includes clear **comments and mathematical explanations** of convergence and eigen theory.

---
