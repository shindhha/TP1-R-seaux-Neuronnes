import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from exo_4 import PerceptronSimple
from exo_7 import generer_donnees_separables
# [0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0,100]
def analyser_convergence(X, y, learning_rates=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0,100]):

    """
    Analyse la convergence pour différents taux d'apprentissage
    """
    plt.figure(figsize=(12, 8))
    for i, lr in enumerate(learning_rates):
        # TODO: Entraîner le perceptron avec ce taux d'apprentissage
        # TODO: Enregistrer l'évolution de l'erreur à chaque époque
        perceptron = PerceptronSimple(X, y, lr)
        perceptron.fit()  
        # TODO: Tracer les courbes de convergence
        plt.plot(perceptron.errors_per_epoch, label=f"η = {lr}")
        pass
    plt.xlabel('Époque')
    plt.ylabel("Nombre d'erreurs")
    plt.title("Convergence pour différents taux d'apprentissage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
if __name__ == "__main__":
    X, y = generer_donnees_separables(n_points=200, noise=2.0)
    analyser_convergence(X, y)