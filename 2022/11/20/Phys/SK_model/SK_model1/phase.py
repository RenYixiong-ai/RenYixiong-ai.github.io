import numpy as np
from scipy.integrate import quad
import multiprocessing as mp
from sko.GA import GA
import pickle

def heff(beta, p, q, m, j, s, z):
    return -beta * j * (p - q) * s**2 - (m + j * np.sqrt(q) * z) * s

def phi(beta, p, q, m, j, z, k):
    heff_pos = heff(beta, p, q, m, j, 1, z)
    heff_neg = heff(beta, p, q, m, j, -1, z)
    denominator = heff_pos + heff_neg + 1e-10
    return (heff_pos + (-1)**k * heff_neg) / denominator

def funcM(beta, p, q, m, j):
    result, _ = quad(lambda z:1 / np.sqrt(2 * np.pi) * np.exp(-z**2 / 2) * phi(beta, p, q, m, j, z, 1), -np.inf, np.inf)
    return result

def funcP(beta, p, q, m, j):
    result, _ = quad(lambda z:1 / np.sqrt(2 * np.pi) * np.exp(-z**2 / 2) * phi(beta, p, q, m, j, z, 2), -np.inf, np.inf)
    return result

def funcQ(beta, p, q, m, j):
    result, _ = quad(lambda z:1 / np.sqrt(2 * np.pi) * np.exp(-z**2 / 2) * (phi(beta, p, q, m, j, z, 1))**2, -np.inf, np.inf)
    return result

def loss(p, q, m, beta, j):
    return (funcQ(beta, p, q, m, j) - q)**2+(funcP(beta, p, q, m, j)-p)**2+(funcM(beta, p, q, m, j)-m)**2

def optimize(beta, j):
    ga = GA(func=lambda p,q,m: loss(p,q,m,beta,j), n_dim=3, size_pop=50, max_iter=800, prob_mut=0.001, lb=[-5, 1e-5, -5], ub=[5, 5, 5], precision=1e-3)
    best_x, best_y = ga.run()
    return beta, j, best_x, best_y

if __name__ == '__main__':
    beta_values = np.linspace(1e-5, 2.0, 200)
    j_values = np.linspace(1e-5, 2.0, 200)

    # Create a list of tuples containing all combinations of beta and j
    params = [(beta, j) for beta in beta_values for j in j_values]

    # Use multiprocessing to run optimizations in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        print(mp.cpu_count())
        results = pool.starmap(optimize, params)

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)




