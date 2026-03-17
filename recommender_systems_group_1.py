# -*- coding: utf-8 -*-
"""
***Importing libraries***
"""

import pandas as pd
import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

"""***Defining auxiliar functions for algorithms***

"""

def loss(X, U, mask):
    return 0.5 * np.sum((X[mask] - U[mask]) ** 2)

#Gradient for projected gradient descent
def gradient(X, U, mask):
    """Gradient of loss with respect to X over observed entries."""
    grad = np.zeros_like(U)
    grad[mask] = X[mask] - U[mask] #the exponent gets cancelled by the 1/2
    return grad

def is_duplicate(atom_list, atom, tol=1e-8):
    """Check if atom (matrix) is already in atom_list within tolerance."""
    for i, S in enumerate(atom_list):
        if np.allclose(S, atom, atol=tol):
            return i
    return -1

# Project onto the nuclear norm ball
def project_onto_nuclear_norm_ball(X, delta):
    """Decompose matrix and call auxiliar fuction to project, returns composed matrix with
    singular values clipped or bounded if necessary"""
    U, S, Vt = svds(X, k=1) #U (left-matrix), S (singular values), Vt (transpose right-matrix)
    # Soft-threshold the singular values, creates an upper bound with the value of delta
    S_clipped = project_onto_l1_ball(S, delta)
    # Reconstruct projected matrix
    return U @ np.diag(S_clipped) @ Vt

def project_onto_l1_ball(s, delta):
  """This function projects a vector s (which represents the singular values of a matrix)
  onto the l1-norm ball or radius delta"""
  #Condition to check if projection is needed
  if np.sum(s) <= delta:
      return s
  u = np.sort(s)[::-1] #Sort the vector in descending order
  cssv = np.cumsum(u)  #Compute cumulative sums of sorted values
  #Finds the right threshold,  gets the last element, which corresponds to the largest index satisfying the condition.
  rho = np.where(u > (cssv - delta) / (np.arange(len(s)) + 1))[0][-1]
  theta = (cssv[rho] - delta) / (rho + 1)
  return np.maximum(s - theta, 0)

""" ***Defining FW algorithm***"""

def frank_wolfe_matrix_completion(U, mask, tau=10, max_iter=30):
    X = np.zeros_like(U)
    norm_factor = np.sum(mask)
    losses = [loss(X, U, mask) / norm_factor]
    for k in tqdm(range(max_iter)):
        grad = gradient(X, U, mask)
        # LMO: top singular vectors of -grad
        u, s, vt = svds(-grad, k=1)
        S = tau * np.outer(u[:, 0], vt[0, :])  # rank-1 matrix scaled by tau
        # line search: minimize loss((1 - gamma)X + gamma S)
        numerator = np.sum((X[mask] - U[mask]) * (X[mask] - S[mask]))
        denominator = np.sum((X[mask] - S[mask]) ** 2)
        gamma = np.clip(numerator / denominator, 0, 1) if denominator > 0 else 0
        # gamma = 2 / (k + 2) # abandoned since line search yilded better results
        # update
        X = (1 - gamma) * X + gamma * S
        losses.append(loss(X, U, mask) / norm_factor)
    return X, losses

""" ***Defining PFW algorithm***"""

def pairwise_frank_wolfe_matrix_completion(U, mask, tau=10, max_iter=30):
    # Start by finding the first Frank-Wolfe atom using the gradient at zero
    grad = gradient(np.zeros_like(U), U, mask)
    u, s, vt = svds(-grad, k=1)
    S_FW = tau * np.outer(u[:, 0], vt[0, :])
    X = S_FW.copy()
    atoms = [S_FW]
    coefs = [1.0]

    norm_factor = np.sum(mask)
    losses = [loss(X, U, mask) / norm_factor]

    for k in tqdm(range(max_iter)):
        grad = gradient(X, U, mask)

        # Find the next FW atom (the best rank-1 direction to move towards)
        u, s, vt = svds(-grad, k=1)
        S_FW = tau * np.outer(u[:, 0], vt[0, :])

        # Find which current atom is "worst" to move away from (AWAY step)
        prods = [np.sum(grad * S) for S in atoms]
        away_idx = np.argmax(prods)
        S_away = atoms[away_idx]

        # Build the direction to move in (pairwise: towards FW, away from S_away)
        d = S_FW - S_away

        # The most we can move away is limited by the weight of the away atom
        gamma_max = coefs[away_idx]

        # Figure out the best step size (gamma), stay in [0, gamma_max]
        # Use the analytic solution for quadratic loss (simple line search)
        X_new = X + d
        numerator = np.sum((X[mask] - U[mask]) * (X[mask] - X_new[mask]))
        denominator = np.sum((X[mask] - X_new[mask]) ** 2)
        gamma = np.clip(numerator / denominator, 0, gamma_max) if denominator > 0 else 0

        # Actually update X using the new direction and step size
        X = X + gamma * d

        # Update coefficients for the atoms
        coefs[away_idx] -= gamma
        if coefs[away_idx] < 1e-8:
            # If the coefficient is almost zero, just remove that atom
            del atoms[away_idx]
            del coefs[away_idx]

        # If we've already seen this FW atom, just add its weight; else add it as new
        fw_idx = is_duplicate(atoms, S_FW)
        if fw_idx >= 0:
            coefs[fw_idx] += gamma
        else:
            atoms.append(S_FW)
            coefs.append(gamma)

        # Keep track of the loss at each iteration (normalized)
        losses.append(loss(X, U, mask) / norm_factor)

    return X, losses

""" ***Defining PGD algorithm***

 NOTE: Review convergence condition, lr and regularization parameters
"""

def projected_gradient_descent(U, mask, tau, max_iter, lr):
    """
    Perform matrix completion using Projected Gradient Descent.
    U: input matrix with 0 for missing entries
    tau: nuclear norm constraint
    lr: learning rate
    max_iters: number of iterations
    """
    X = np.zeros_like(U)
    norm_factor = np.sum(mask)
    losses_pgd = [loss(X, U, mask)/norm_factor]

    # Ensure R is float64 for numerical stability
    U = U.astype(np.float64)

    for t in tqdm(range(max_iter)):
        grad = gradient(X, U, mask)                  # Step 1: Compute the gradient
        X_temp = X - lr * grad                            # Step 2: Gradient descent update
        X = project_onto_nuclear_norm_ball(X_temp, tau)   # Step 3: Projection step

        losses_pgd.append(loss(X, U, mask)/norm_factor) #computes train loss #Step 4: compute the loss
    return X, losses_pgd

"""***Other functions to split and plot data***"""

#Defining functions to split and plot data

def split_data(matrix, test_size=0.1, random_state=42):
  observed_indices = np.argwhere(~np.isnan(matrix)) #Indices where there are not NaNs
  train_idx, test_idx = train_test_split(observed_indices, test_size=0.1, random_state=42) #Split indices of dataset into train and test
  train_mask = np.full_like(matrix, False, dtype=bool) #Creates a mask with the same dimensions of the original matrix but full with False
  for i, j in train_idx:
      train_mask[i, j] = True #Sets as "True" the positions in train_idx
  test_mask = ~train_mask & ~np.isnan(matrix) #Sets as "True" anything else that doesn't belong to the train mask and it's not NaN
  U_train = np.where(train_mask, matrix, 0) #Matrix with original values only in the positions of the train mask, and zeros anywhere else
  return U_train, train_mask, test_mask

def plottingData(losses_fw, losses_pfw, losses_pgd, matrix_name, step=10):
    custom_legend = [Line2D([0], [0], marker='o', color='w', label=f'Every {step} iterations',
           markerfacecolor='blue', markersize=10)]

    #Plot for FW
    indices_with_marker_fw = list(range(0, len(losses_fw), step))
    losses_with_marker_fw = [losses_fw[i] for i in indices_with_marker_fw]
    plt.figure(figsize=(8, 4), dpi=100)
    plt.title("Frank-Wolfe Matrix Completion " + matrix_name)
    plt.plot(range(len(losses_fw)), losses_fw, color='blue', ms=5)
    plt.plot(indices_with_marker_fw, losses_with_marker_fw, marker='o', linestyle='None', color='blue', ms=5, label=f'Marker every {step} iterations')
    plt.legend(handles=custom_legend)
    plt.xlabel("Iteration")
    plt.ylabel("Observed Loss")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    #Plot for PFW
    indices_with_marker_pfw = list(range(0, len(losses_pfw), step))
    losses_with_marker_pfw = [losses_pfw[i] for i in indices_with_marker_pfw]
    plt.figure(figsize=(8, 4), dpi=100)
    plt.title("Pairwise Frank-Wolfe Matrix Completion " + matrix_name)
    plt.plot(range(len(losses_pfw)), losses_pfw, color='blue', ms=5)
    plt.plot(indices_with_marker_pfw, losses_with_marker_pfw, marker='o', linestyle='None', color='blue', ms=5)
    plt.legend(handles=custom_legend)
    plt.xlabel("Iteration")
    plt.ylabel("Observed Loss")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    #Plot for PGD
    indices_with_marker_pgd = list(range(0, len(losses_pgd), step))
    losses_with_marker_pgd = [losses_pgd[i] for i in indices_with_marker_pgd]
    plt.figure(figsize=(8, 4), dpi=100)
    plt.title("Projected Gradient Descent Matrix Completion " + matrix_name)
    plt.plot(range(len(losses_pgd)), losses_pgd, color='blue', ms=5)
    plt.plot(indices_with_marker_pgd, losses_with_marker_pgd, marker='o', linestyle='None', color='blue', ms=5)
    plt.legend(handles=custom_legend)
    plt.xlabel("Iteration")
    plt.ylabel("Observed Loss")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    return

"""
# 1. Lens Movies Short

"""

df = pd.read_csv("lens_movies_short.csv", header=None)
df.head()

#Normalizing matrix
movies_matrix = df.to_numpy() / 5

"""***Running algorithms***"""

#Calling function to split the data
U_train, train_mask, test_mask = split_data(movies_matrix)

#Calling FW algo
X_hat_fw, losses_fw = frank_wolfe_matrix_completion(U_train, train_mask, tau=100, max_iter=100)

#Calling PFW algo
X_hat_pfw, losses_pfw = pairwise_frank_wolfe_matrix_completion(U_train, train_mask, tau=100, max_iter=100)

#Calling PGD algo
X_hat_pgd, losses_pgd = projected_gradient_descent(U_train, train_mask, tau=100, max_iter=100, lr=0.1)

"""***Plotting losses of each algorithm***"""

plottingData(losses_fw, losses_pfw, losses_pgd, matrix_name="Lens Movies Short", step=5)

"""***Computing metrics***"""

# back to the 0.5-5.0 system for FW, and PGD
X_rescaled_fw = np.round(X_hat_fw * 5 * 2) / 2
X_rescaled_fw = np.clip(X_rescaled_fw, 0.5, 5.0) #ensures all values are in the range (0.5-5.0)

X_rescaled_pfw = np.round(X_hat_pfw * 5 * 2) / 2
X_rescaled_pfw = np.clip(X_rescaled_pfw, 0.5, 5.0)

X_rescaled_pgd = np.round(X_hat_pgd * 5 * 2) / 2
X_rescaled_pgd = np.clip(X_rescaled_pgd, 0.5, 5.0)

true_ratings= movies_matrix[test_mask] * 5  # back to the 0.5-5.0 system
predicted_ratings_fw = X_rescaled_fw[test_mask]
predicted_ratings_pfw = X_rescaled_pfw[test_mask]
predicted_ratings_pgd = X_rescaled_pgd[test_mask]

mae_fw = mean_absolute_error(true_ratings, predicted_ratings_fw)
mae_pfw = mean_absolute_error(true_ratings, predicted_ratings_pfw)
mae_pgd = mean_absolute_error(true_ratings, predicted_ratings_pgd)

acc_05_fw = np.sum(true_ratings == predicted_ratings_fw) / len(predicted_ratings_fw) # exact accuracy
acc_05_pfw = np.sum(true_ratings == predicted_ratings_pfw) / len(predicted_ratings_pfw) # exact accuracy
acc_05_pgd = np.sum(true_ratings == predicted_ratings_pgd) / len(predicted_ratings_pgd) # exact accuracy
# accuracy within the whole point
acc_10_fw = np.sum(true_ratings.round(0) == predicted_ratings_fw.round(0)) / len(predicted_ratings_fw)
acc_10_pfw = np.sum(true_ratings.round(0) == predicted_ratings_pfw.round(0)) / len(predicted_ratings_pfw)
acc_10_pgd = np.sum(true_ratings.round(0) == predicted_ratings_pgd.round(0)) / len(predicted_ratings_pgd)

print(f"Frank Wolfe metrics - Mean Absolute Error: {mae_fw:4f}, Exact Accuracy: {acc_05_fw:.4f}, Round Accuracy: {acc_10_fw:.4f}")
print(f"Pairwise Frank Wolfe metrics - Mean Absolute Error: {mae_pfw:4f}, Exact Accuracy: {acc_05_pfw:.4f}, Round Accuracy: {acc_10_pfw:.4f}")
print(f"Projected GD metrics - Mean Absolute Error: {mae_pgd:4f}, Exact Accuracy: {acc_05_pgd:.4f}, Round Accuracy: {acc_10_pgd:.4f}")

"""***Conclusions***

# 2. Jester dataset
"""

file_path = "jester_dataset.csv"
df = pd.read_csv(file_path, header=None)
df.head()

"""***Running algorithms***"""

jester_matrix = df.to_numpy() / 10

#Calling function to split the data
U_train, train_mask, test_mask = split_data(jester_matrix)

#Calling FW algo
X_hat_fw, losses_fw = frank_wolfe_matrix_completion(U_train, train_mask, tau=250, max_iter=100)

#Calling PFW algo
X_hat_pfw, losses_pfw = pairwise_frank_wolfe_matrix_completion(U_train, train_mask, tau=250, max_iter=100)

#Calling PGD algo
X_hat_pgd, losses_pgd = projected_gradient_descent(U_train, train_mask, tau=50, max_iter=100, lr=0.1)

"""***Plotting losses of each algorithm***"""

plottingData(losses_fw, losses_pfw, losses_pgd, matrix_name="Jester Matrix", step=5)

"""***Computing metrics***"""

X_rescaled_fw = X_hat_fw * 10  # go back to the -10 - +10 system
X_rescaled_pfw = X_hat_pfw * 10
X_rescaled_pgd = X_hat_pgd * 10

true_ratings = jester_matrix[test_mask]
predicted_ratings_fw = X_rescaled_fw[test_mask]
predicted_ratings_pfw = X_rescaled_pfw[test_mask]
predicted_ratings_pgd = X_rescaled_pgd[test_mask]

mae_fw = mean_absolute_error(true_ratings, predicted_ratings_fw)
mae_pfw = mean_absolute_error(true_ratings, predicted_ratings_pfw)
mae_pgd = mean_absolute_error(true_ratings, predicted_ratings_pgd)

print(f"Frank Wolfe metrics - Mean Absolute Error: {mae_fw:4f}")
print(f"Pairwise Frank Wolfe metrics - Mean Absolute Error: {mae_pfw:4f}")
print(f"Projected GD metrics - Mean Absolute Error: {mae_pgd:4f}")

"""
# 3. Amazon products

"""

file_path = "ratings_matrix.csv"
df = pd.read_csv(file_path,)
df.head()

df = df.drop(['user_id'], axis=1)
df.columns = range(df.shape[1])
df.head()

"""***Running algorithms***"""

#Normalize matrix
amazon_matrix = df.to_numpy() / 5

#Calling function to split the data
U_train, train_mask, test_mask = split_data(amazon_matrix)

#Calling FW algo
X_hat_fw, losses_fw = frank_wolfe_matrix_completion(U_train, train_mask, tau=1000, max_iter=300)

#Calling PFW algo
X_hat_pfw, losses_pfw = pairwise_frank_wolfe_matrix_completion(U_train, train_mask, tau=1000, max_iter=300)

#Calling PGD algo
X_hat_pgd, losses_pgd = projected_gradient_descent(U_train, train_mask, tau=100, max_iter=300, lr=1)

"""***Plotting losses of each algorithm***"""

plottingData(losses_fw, losses_pfw, losses_pgd, matrix_name="Amazon books")

"""***Computing metrics***"""

# back to the 0.5-5.0 system for FW, and PGD
X_rescaled_fw = (X_hat_fw * 5).round(0)
X_rescaled_fw = np.clip(X_rescaled_fw, 1.0, 5.0)

X_rescaled_pfw = (X_hat_pfw * 5).round(0)
X_rescaled_pfw = np.clip(X_rescaled_pfw, 1.0, 5.0)

X_rescaled_pgd = (X_hat_pgd * 5).round(0)
X_rescaled_pgd = np.clip(X_rescaled_pgd, 1.0, 5.0)

true_ratings= amazon_matrix[test_mask] * 5  # back to the 0.5-5.0 system
predicted_ratings_fw = X_rescaled_fw[test_mask]
predicted_ratings_pfw = X_rescaled_pfw[test_mask]
predicted_ratings_pgd = X_rescaled_pgd[test_mask]

mae_fw = mean_absolute_error(true_ratings, predicted_ratings_fw)
mae_pfw = mean_absolute_error(true_ratings, predicted_ratings_pfw)
mae_pgd = mean_absolute_error(true_ratings, predicted_ratings_pgd)

acc_05_fw = np.sum(true_ratings == predicted_ratings_fw) / len(predicted_ratings_fw) # exact accuracy
acc_05_pfw = np.sum(true_ratings == predicted_ratings_pfw) / len(predicted_ratings_pfw) # exact accuracy
acc_05_pgd = np.sum(true_ratings == predicted_ratings_pgd) / len(predicted_ratings_pgd) # exact accuracy

print(f"Frank Wolfe metrics - Mean Absolute Error: {mae_fw:4f}, Exact Accuracy: {acc_05_fw:.4f}")
print(f"Pairwise Frank Wolfe metrics - Mean Absolute Error: {mae_pfw:4f}, Exact Accuracy: {acc_05_pfw:.4f}")
print(f"Projected GD metrics - Mean Absolute Error: {mae_pgd:4f}, Exact Accuracy: {acc_05_pgd:.4f}")

