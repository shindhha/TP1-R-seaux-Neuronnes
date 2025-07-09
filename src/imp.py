import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
    def __init__(self, name, alpha=0.01):
        self.name = name.lower()
        self.alpha = alpha  # Pour Leaky ReLU

    def apply(self, z):
        if self.name == "heaviside":
            return self.heavy_side(z)
        elif self.name == "sigmoid":
            return self.sigmoid(z)
        elif self.name == "tanh":
            return self.tanh(z)
        elif self.name == "relu":
            return self.relu(z)
        elif self.name == "leaky_relu":
            return self.leaky_relu(z)
        else:
            raise ValueError(f"Activation '{self.name}' non reconnue.")

    def heavy_side(self, z):
        return np.where(z >= 0, 1, 0)

    def relu(self, z):
        return np.maximum(0, z)

    def leaky_relu(self, z):
        return np.where(z >= 0, z, self.alpha * z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)


class DerActivationFunction:
    def __init__(self, name, alpha=0.01):
        self.name = name.lower()
        self.alpha = alpha

    def apply(self, z):
        if self.name == "heaviside":
            return self.heavy_side(z)
        elif self.name == "sigmoid":
            return self.sigmoid(z)
        elif self.name == "tanh":
            return self.tanh(z)
        elif self.name == "relu":
            return self.relu(z)
        elif self.name == "leaky_relu":
            return self.leaky_relu(z)
        else:
            raise ValueError(f"Dérivée de '{self.name}' non définie.")

    def heavy_side(self, z):
        # Dirac delta (approximation visuelle : une grande valeur autour de zéro)
        return np.zeros_like(z)

    def relu(self, z):
        return np.where(z > 0, 1, 0)

    def leaky_relu(self, z):
        return np.where(z > 0, 1, self.alpha)

    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)

    def tanh(self, z):
        return 1 - np.tanh(z) ** 2


# Visualisation
def plot_activation_and_derivative(name, alpha=0.01):
    z = np.linspace(-10, 10, 500)
    act = ActivationFunction(name, alpha)
    der = DerActivationFunction(name, alpha)

    a_values = act.apply(z)
    d_values = der.apply(z)

    plt.figure(figsize=(10, 5))
    plt.plot(z, a_values, label=f'{name} activation')
    plt.plot(z, d_values, label=f'{name} dérivée', linestyle='--')
    plt.title(f"Fonction d'activation et dérivée : {name}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Exemples de visualisation
for func in ['heaviside', 'sigmoid', 'tanh', 'relu', 'leaky_relu']:
    plot_activation_and_derivative(func)
