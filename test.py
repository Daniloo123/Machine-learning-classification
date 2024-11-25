import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Vervang STUDENTNUMMER met je eigen studentnummer
STUDENTNUMMER = "0955071"

# URL's voor het ophalen van data
clustering_training_url = "https://programmeren9.cmgt.hr.nl:9000/{STUDENTNUMMER}/clustering/training"
classification_training_url = "https://programmeren9.cmgt.hr.nl:9000/{STUDENTNUMMER}/classification/training"
classification_test_url = "https://programmeren9.cmgt.hr.nl:9000/{STUDENTNUMMER}/classification/test"

# Functie om data op te halen van de opgegeven URL
def fetch_data(url):
    response = requests.get(url)
    data = response.json()
    return data
# Haal clustering training data op
clustering_training_data = fetch_data(clustering_training_url)
if clustering_training_data:
    X = np.array(clustering_training_data['data'])
    print("Clustering training data opgehaald:", X)
else:
    print("Fout bij het ophalen van clustering training data.")

# Haal classificatie trainingsdata op
classification_training_data = fetch_data(classification_training_url)
if classification_training_data:
    X_train = np.array(classification_training_data['data'])
    y_train = np.array(classification_training_data['labels'])
    print("Classificatie trainingsdata opgehaald:", X_train, y_train)
else:
    print("Fout bij het ophalen van classificatie trainingsdata.")

# Haal classificatie testdata op
classification_test_data = fetch_data(classification_test_url)
if classification_test_data:
    X_test = np.array(classification_test_data['data'])
    y_test = np.array(classification_test_data['labels'])
    print("Classificatie testdata opgehaald:", X_test, y_test)
else:
    print("Fout bij het ophalen van classificatie testdata.")

# Plot de data
plt.scatter(X[:, 0], X[:, 1])
plt.title('Clustering Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Gok voor het aantal clusters
k_guess = 3

# Voer KMeans uit voor verschillende waarden van k
for k in [k_guess - 1, k_guess, k_guess + 1]:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    centroids = kmeans.cluster_centers_

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='.')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5, label='Centroids')
    plt.title(f'KMeans Clustering (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Haal classificatie trainingsdata op
classification_training_data = fetch_data(classification_training_url)
X_train = np.array(classification_training_data['data'])
y_train = np.array(classification_training_data['labels'])

# Logistische regressie classifier
logreg_classifier = LogisticRegression()
logreg_classifier.fit(X_train, y_train)
y_train_predict_logreg = logreg_classifier.predict(X_train)
accuracy_logreg = accuracy_score(y_train, y_train_predict_logreg)
print("Nauwkeurigheid van logistische regressie op trainingsdata:", accuracy_logreg)

# Beslisboom classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_train_predict_dt = dt_classifier.predict(X_train)
accuracy_dt = accuracy_score(y_train, y_train_predict_dt)
print("Nauwkeurigheid van beslisboom op trainingsdata:", accuracy_dt)

# Plot resultaten van logistische regressie classifier
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_predict_logreg, cmap='viridis', marker='.')
plt.title('Logistic Regression Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plot resultaten van beslisboom classifier
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_predict_dt, cmap='viridis', marker='.')
plt.title('Decision Tree Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Haal classificatie testdata op
classification_test_data = fetch_data(classification_test_url)
X_test = np.array(classification_test_data['data'])
y_test = np.array(classification_test_data['labels'])

# Nauwkeurigheid op testdata voor logistische regressie
y_test_predict_logreg = logreg_classifier.predict(X_test)
accuracy_logreg_test = accuracy_score(y_test, y_test_predict_logreg)
print("Nauwkeurigheid van logistische regressie op testdata:", accuracy_logreg_test)

# Nauwkeurigheid op testdata voor beslisboom
y_test_predict_dt = dt_classifier.predict(X_test)
accuracy_dt_test = accuracy_score(y_test, y_test_predict_dt)
print("Nauwkeurigheid van beslisboom op testdata:", accuracy_dt_test)

# Vergelijk de nauwkeurigheid op trainings- en testdata
print("Nauwkeurigheid op trainingsdata (Logistic Regression):", accuracy_logreg)
print("Nauwkeurigheid op testdata (Logistic Regression):", accuracy_logreg_test)
print("Nauwkeurigheid op trainingsdata (Decision Tree):", accuracy_dt)
print("Nauwkeurigheid op testdata (Decision Tree):", accuracy_dt_test)
