import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from exo_4 import PerceptronSimple

class PerceptronMultiClasse:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.perceptrons = {}
        self.classes = None

    def fit(self, X, y, max_epochs=100):
        """
        Entraîne un perceptron par classe (stratégie un-contre-tous)
        """
        self.classes = np.unique(y)

        for classe in tqdm(self.classes, desc="Entraînement des perceptrons"):
            # Création des étiquettes binaires : 1 pour la classe courante, 0 sinon
            y_binaire = (y == classe).astype(int)

            # Entraînement du perceptron binaire
            perceptron = PerceptronSimple(X, y_binaire, learning_rate=self.learning_rate)
            perceptron.fit(max_epochs=max_epochs)

            # Stockage
            self.perceptrons[classe] = perceptron

    def predict(self, X):
        """Prédit la classe en utilisant les scores de chaque perceptron"""
        if not self.perceptrons:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")

        scores = np.zeros((X.shape[0], len(self.classes)))

        for i, classe in enumerate(self.classes):
            perceptron = self.perceptrons[classe]
            raw_scores = X.dot(perceptron.weights) + perceptron.bias
            scores[:, i] = raw_scores

        # Retourne l'indice du score maximal pour chaque exemple
        predicted_indices = np.argmax(scores, axis=1)
        return self.classes[predicted_indices]

    def predict_proba(self, X):
        """Retourne les scores de chaque perceptron (non normalisés)"""
        if not self.perceptrons:
            raise ValueError("Le modèle n'a pas été entraîné.")

        scores = np.zeros((X.shape[0], len(self.classes)))

        for i, classe in enumerate(self.classes):
            perceptron = self.perceptrons[classe]
            raw_scores = X.dot(perceptron.weights) + perceptron.bias
            scores[:, i] = raw_scores

        return scores
    


def charger_donnees_iris_binaire():
    iris = load_iris()
    # longueur sépales, longueur pétales
    X = iris.data[:, [0, 2]]
    y = iris.target
    # on garde seulement les classes 0 et 1
    mask = y < 2
    # y reste 0/1 — surtout pas -1/+1
    return X[mask], y[mask]


def charger_donnees_iris_complete():
    iris = load_iris()
    # mêmes 2 features pour la visu
    X = iris.data[:, [0, 2]]
    y = iris.target
    return X, y, iris.target_names


def visualiser_iris(X, y, target_names=None, title="Dataset Iris"):
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green']
    markers = ['*', '+', 'o']

    for i in range(len(np.unique(y))):
        mask = (y == i)
        label = target_names[i] if target_names is not None else f'Classe {i}'
        plt.scatter(X[mask, 0], X[mask, 1],c=colors[i], marker=markers[i], s=90,label=label, alpha=0.7)

    plt.xlabel('Longueur des sépales (cm)')
    plt.ylabel('Longueur des pétales (cm)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # --- version multi-classe sur Iris complet -------------------
    X_full, y_full, nom_classes = charger_donnees_iris_complete()
    # normalisation (recommandée pour les perceptrons)
    scaler = StandardScaler().fit(X_full)
    X_full = scaler.transform(X_full)
    # split train / test
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.5, stratify=y_full, random_state=0)
    # entraînement
    clf = PerceptronMultiClasse(learning_rate=0.1)
    clf.fit(X_tr, y_tr, max_epochs=100)
    y_pred = clf.predict(X_te)
    print("Accuracy test :", accuracy_score(y_te, y_pred))
    print(classification_report(y_te, y_pred, target_names=nom_classes))
    visualiser_iris(X_full, y_full, nom_classes,title="Iris (2 features) – points + zones ambiguës")

