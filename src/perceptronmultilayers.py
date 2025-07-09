# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 13:59:00 2025

@author: Guillaume
"""

# Exercice 1.1 - Questions d'analyse théorique :

# Que signifie concrètement le théorème d'approximation universelle ? 
# Il signifie que l'on peut approximer une fonction sur R^n par une autre fonction sur R
# Ce théorème garantit-il qu'on peut toujours trouver les bons poids ?
# Non, il faut de bonne données
# Quelle est la différence entre "pouvoir approximer" et "pouvoir apprendre" ?
# "pouvoir approximer" dit que l'on peut donné un résultat plausible à une probabilité
# correcte tandit que "pouvoir apprendre" signifie augmenter la probabilité, avoir plus de certitude.
# Pourquoi utilise-t-on souvent beaucoup plus de couches cachées en pratique ?
# 
# En principe, vous avez déjà vu au lycée un autre type d’approximateur de fonctions, donner leurs noms ?
# En I.U.T. on a utiliser des méthodes des trapèze pour approximer des intégrales

# Exercice 1.2 - Expliquer la phrase suivante

# Le théorème d’approximation universelle affirme qu’un réseau profond peut 
# exactement retrouver les données d’entraînement.

# Exercice 2.2.1

class CoucheNeurones:
    def __init__(self, n_input, n_neurons, activation='sigmoid', learning_rate=0.01):
        """
        Initialise une couche de neurones

        Parameters:
        - n_input: nombre d'entrées
        - n_neurons: nombre de neurones dans cette couche
        - activation: fonction d'activation ('sigmoid', 'tanh', 'relu')
        - learning_rate: taux d'apprentissage
        """
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.activation_name = activation
        self.learning_rate = learning_rate

        # Initialisation Xavier/Glorot pour éviter l'explosion/disparition des gradients
        limit = np.sqrt(6 / (n_input + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_neurons, n_input))
        self.bias = np.zeros((n_neurons, 1))

        # Variables pour stocker les valeurs lors de la propagation
        self.last_input = None
        self.last_z = None
        self.last_activation = None

        # Import de la fonction d'activation du TP précédent
        from activation_functions import ActivationFunction
        self.activation_func = ActivationFunction(activation)

    def forward(self, X):
        """
        Propagation avant
        X: matrice d'entrée (n_features, n_samples)
        """
        # TODO: Implémenter la propagation avant
        # Stocker les valeurs intermédiaires pour la rétropropagation
        self.last_input = X
        self.last_z = 0  # Combinaison linéaire
        self.last_activation = 0  # Après fonction d'activation

        return self.last_activation

    def backward(self, gradient_from_next_layer):
        """
        Rétropropagation
        gradient_from_next_layer: gradient venant de la couche suivante
        """
        # TODO: Calculer les gradients par rapport aux poids et biais
        # TODO: Calculer le gradient à propager vers la couche précédente

        # Gradient par rapport à la fonction d'activation
        grad_activation = 0

        # Gradient par rapport aux poids
        grad_weights = 0

        # Gradient par rapport aux biais  
        grad_bias = 0

        # Gradient à propager vers la couche précédente
        grad_input = 0

        # Mise à jour des paramètres
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias

        return grad_input