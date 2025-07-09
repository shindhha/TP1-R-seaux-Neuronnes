import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PerceptronSimple:
    def __init__(self, X, y, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.epoch = None
        self.X = X
        self.y = y
    


    def fit(self, X=None, y=None, max_epochs=100):
        """
        Entraîne le perceptron
        X: matrice des entrées (n_samples, n_features)
        y: vecteur des sorties désirées (n_samples,...)
        """
        if (X == None):
            X = self.X
        if (y == None):
            y = self.y
        # Initialisation les poids et le biais
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0.0
        self.errors_per_epoch = []
        for epoch in tqdm(range(max_epochs)):
            errors = 0
            for i in range(X.shape[0]):
                x = X[i]
                y_true = y[i]
                result = np.dot(self.weights, x) + self.bias
                y_pred = 1 if result >= 0 else 0
                error = y_true - y_pred
                if error != 0:
                    errors += 1
                    self.weights += self.learning_rate * error * x
                    self.bias += self.learning_rate * error
            self.errors_per_epoch.append(errors)
            if errors == 0:
                break
        self.epoch = epoch



    def predict(self, X=None):
        """Prédit les sorties pour les entrées X"""
        if (X == None):
            X = self.X
        y_pred = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            # TODO: Calculer les prédictions
            x = X[i] # (n_features,)
            result = np.dot(self.weights, x) + self.bias
            y_pred[i] = 1 if result >= 0 else 0

        return y_pred

    def score(self, X=None, y=None):
        """Calcule l'accuracy"""
        if (X == None):
            X = self.X
        if (y == None):
            y = self.y
        predictions = self.predict()   
        return np.mean(predictions == y)
    
# Données pour la fonction AND
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # 0 pour False, 1 pour True

# Données pour la fonction OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

# Données pour la fonction XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

perceptronAND = PerceptronSimple(X=X_and, y=y_and)

perceptronAND.fit()

perceptronOR = PerceptronSimple(X=X_or, y=y_or)
perceptronOR.fit()

perceptronXOR = PerceptronSimple(X=X_xor, y=y_xor)
perceptronXOR.fit()


def run(perceptron, title):

    perceptron.fit()
    acc = perceptron.score()
    print(f"Précision du perceptron ({title}) :", acc)
    print(f"Nombre d'époques :", perceptron.epoch)
    print(f"Poids :", perceptron.weights)
    plt.figure(figsize=(6, 6))
    for i, (x, target) in enumerate(zip(perceptron.X, perceptron.y)):
        if target == 1:
            plt.scatter(x[0], x[1], color='blue', marker='o', label='1 (True)' if i == 0 else "")
        else:
            plt.scatter(x[0], x[1], color='red', marker='x', label='0 (False)' if i == 0 else "")

    # Frontière de décision
    w, b = perceptron.weights, perceptron.bias
    x_vals = np.linspace(-0.2, 1.2, 100)
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, 'k--', label='décision')
    else:
        plt.axvline(-b / w[0], color='k', linestyle='--', label='décision')

    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Perceptron avec le : {title}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    run(perceptronAND, "AND")
    run(perceptronOR, "OR")
    run(perceptronXOR, "XOR")
