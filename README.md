# Recommender Systems with Frank-Wolfe Methods

This project implements and compares different optimization algorithms for the **matrix completion problem**, which is a core task in recommender systems.

The goal is to predict missing values in a user–item interaction matrix (e.g., movie ratings), allowing systems to recommend items to users based on inferred preferences.

The project was developed as part of the **Optimization for Data Science (ODS) course** at the University of Padua.

---

# Problem Description

Modern recommender systems (such as Netflix, Spotify, or Amazon) rely on predicting missing user preferences from partially observed data.

This problem can be formulated as a **low-rank matrix completion problem**, where we aim to estimate a matrix (X) that approximates an observed sparse matrix (R).

The optimization objective is:

$$
\min_X \sum_{(i,j) \in \Omega} (X_{ij} - R_{ij})^2
$$

subject to a **nuclear norm constraint** that promotes low-rank solutions.

---

# Implemented Algorithms

The project implements three optimization algorithms:

### 1. Frank-Wolfe (FW)

A projection-free optimization method that iteratively improves the solution by computing a **rank-1 atom using singular value decomposition (SVD)** and combining it with the current estimate.

Advantages:

* avoids expensive projection steps
* naturally produces low-rank solutions
* efficient for high-dimensional problems

---

### 2. Pairwise Frank-Wolfe (PFW)

An extension of Frank-Wolfe that improves convergence by performing **pairwise updates**.

Instead of only moving toward a new atom, it also moves **away from poorly contributing atoms**, redistributing weight inside the active set.

Advantages:

* faster convergence
* better control of the active atom set

---

### 3. Projected Gradient Descent (PGD)

A classical gradient descent method where each step is followed by a **projection onto the nuclear norm ball**.

Steps:

1. Compute gradient
2. Perform gradient update
3. Project onto nuclear norm constraint using SVD

This method is included as a **baseline for comparison**.

---

# Project Structure

The notebook contains:

* implementation of all algorithms
* auxiliary functions
* data preprocessing
* training and evaluation pipeline
* visualizations of convergence

---

# Experimental Pipeline

The same evaluation pipeline is used for all datasets:

1. Load dataset
2. Normalize ratings
3. Split observed entries into **training and test sets**
4. Run matrix completion algorithms
5. Predict missing entries
6. Rescale predictions to original rating scale
7. Compute evaluation metrics
8. Visualize loss convergence

---

# Datasets

### Lens Movies (short version)

* 300 users
* 30 movies
* ~43% missing entries
* Used for parameter tuning and quick experiments

---

### Jester Dataset

* 50,692 users
* 150 jokes
* ~78% missing entries
* Continuous ratings in range **[-10, 10]**

---

### Amazon Products

* 622 users
* 827 products
* **Extremely sparse** (only ~0.6% observed entries)

---

# Evaluation Metrics

The following metrics are used depending on the dataset:

### Mean Absolute Error (MAE)

Measures prediction error:

$$
MAE = \frac{1}{N}\sum |y_{true} - y_{pred}|
$$

---

### Exact Accuracy

Percentage of predictions that exactly match the true rating.

---

### Rounded Accuracy

Accuracy after rounding predictions to the nearest integer rating.

---

# Results Summary

Key observations:

* **Frank-Wolfe and Pairwise Frank-Wolfe converge faster** than Projected Gradient Descent.
* **PFW slightly improves performance** by redistributing weight among atoms.
* **PGD performs better on the Jester dataset**, despite slower convergence.
* Performance decreases significantly for **extremely sparse datasets (Amazon)**.

---

# Technologies Used

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn
* Matplotlib
* tqdm

---

# How to Run

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib tqdm
```

Run the notebook:

```
jupyter notebook recommender_system.ipynb
```

---

# Authors

* Diego A. Brule Galleguillos
* Diana C. Andrade Damian
* Evgeni Markin
* **Marina Lima Braga**

University of Padova — 2025.
