import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from exo_4 import PerceptronSimple

# Exercice 7
def generer_donnees_separables(n_points=100, noise=0.1):
    """
    Génère deux classes de points linéairement séparables
    """
    np.random.seed(42)  

    X1 = np.random.randn(n_points // 2, 2) * noise + np.array([2, 2])
    X2 = np.random.randn(n_points // 2, 2) * noise + np.array([-2, -2])
    X = np.vstack((X1, X2))  # Fusion des classes 1 et 2
    y1 = np.ones(n_points // 2)
    y2 = np.zeros(n_points // 2)
    y = np.hstack((y1, y2))  # Fusion des étiquettes 1 et 2
    # Mélange aléatoire des valeurs 
    indices = np.random.permutation(n_points)
    X = X[indices]
    y = y[indices]
    # Normalisation des données
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X, y


def visualiser_donnees(X, y, w=None, b=None, title="Données"):
    """
    Visualise les données et optionnellement la droite de séparation
    """
    plt.figure(figsize=(8, 6))
    
    # Afficher les points
    mask_pos = (y == 1)
    plt.scatter(X[mask_pos, 0], X[mask_pos, 1], c='blue', marker='+', s=100, label='Classe +1')
    plt.scatter(X[~mask_pos, 0], X[~mask_pos, 1], c='red', marker='*', s=100, label='Classe -1')
    
    # Afficher la droite de séparation si fournie
    if w is not None and b is not None:

        x_vals = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, 100)
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, 'k--', label='Séparateur')


    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":

    X, y = generer_donnees_separables(200,2.0)
    perceptron = PerceptronSimple(X, y)
    perceptron.fit()
    visualiser_donnees(X, y, w=perceptron.weights, b=perceptron.bias)

