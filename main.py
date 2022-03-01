import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import confusion_matrix

def analyze(data):
    print("Nombre d'exemple dans le dataset:", data[data.columns.values[0]].count())
    print("Nombre de caractéristiques:", len(data.columns))
    print("Shape: ", data.shape)
    # print(data.info())
    print(data.describe())
    print(data.corr())

def label_encode(data):
    # Transforme un type catégorie en entier
    columns = data.columns.values
    for col in columns:
        le = LabelEncoder()
        # On récupère chaque nom de catégories possibles
        unique_values = list(data[col].unique())
        le_fitted = le.fit(unique_values)
        # On liste l'ensemble des valeurs
        values = list(data[col].values)
        # On transforme les catégories en entier
        values_transformed = le.transform(values)
        # On fait le remplacement de la colonne dans le dataframe d'origine
        data[col] = values_transformed

def splitData(data, test_ratio, pred):
    train, test = train_test_split(data, test_size=test_ratio)
    x_train = train
    y_train = train[pred]
    del(train[pred])
    x_test = test
    y_test = test[pred]
    del(test[pred])
    return x_train, y_train, x_test, y_test

def display_score(classifier, x_train, y_train, x_test, y_test):
    print("Train score: {}, Test score {}".format(classifier.score(x_train, y_train), classifier.score(x_test, y_test)))
    
def neural_network(data, pred):
    # Create model
    mlp = MLPClassifier()
    # Split data
    x_train, y_train, x_test, y_test = splitData(data, 0.3, pred)
    # Train model
    mlp.fit(x_train, y_train)
    # Display score
    display_score(mlp, x_train, y_train, x_test, y_test)

def neural_network_regression(data, pred):
    # Create model
    mlp = MLPRegressor()
    # Split data
    x_train, y_train, x_test, y_test = splitData(data, 0.3, pred)
    # Train model
    mlp.fit(x_train, y_train)
    # Display score
    display_score(mlp, x_train, y_train, x_test, y_test)

def k_nearest_neighbors(data, pred):
    # Create model
    rf = KNeighborsClassifier(n_neighbors=3)
    # Split data
    x_train, y_train, x_test, y_test = splitData(data, 0.3, pred)
    # Train model
    rf.fit(x_train, y_train)
    # Display score
    display_score(rf, x_train, y_train, x_test, y_test)
    
def k_nearest_neighbors_regression(data, pred):
    # Create model
    knn = KNeighborsRegressor(n_neighbors=3)
    # Split data
    x_train, y_train, x_test, y_test = splitData(data, 0.3, pred)
    # Train model
    knn.fit(x_train, y_train)
    # Display score
    display_score(knn, x_train, y_train, x_test, y_test)

def decision_tree(data, pred):
    # Create model
    dt = DecisionTreeClassifier()
    # Split data
    x_train, y_train, x_test, y_test = splitData(data, 0.3, pred)
    # Train model
    dt.fit(x_train, y_train)
    # Display score
    display_score(dt, x_train, y_train, x_test, y_test)

def decision_tree_regression(data, pred):
    # Create model
    dt = DecisionTreeRegressor()
    # Split data
    x_train, y_train, x_test, y_test = splitData(data, 0.3, pred)
    # Train model
    dt.fit(x_train, y_train)
    # Display score
    display_score(dt, x_train, y_train, x_test, y_test)

def main():
    data1 = pd.read_csv('dataCCfinal_1.csv')
    data2 = pd.read_csv('dataCCfinal_2.csv')
    print("Analyse du dataset 1:")
    analyze(data1)
    print("\n--------------------\nAnalyse du dataset 2:\n")
    analyze(data2)
    label_encode(data1)
    label_encode(data2)
    # Neural network
    print('\n--------------------\nNeural network:\n')
    neural_network(data1, 'Z')
    neural_network_regression(data2, 'Z')
    # K-nearest neighbors
    print('\n--------------------\nK-nearest neighbors:\n')
    k_nearest_neighbors(data1, 'Z')
    k_nearest_neighbors_regression(data2, 'Z')
    # Decision tree
    print('\n--------------------\nDecision tree:\n')
    decision_tree(data1, 'Z')
    decision_tree_regression(data2, 'Z')

if __name__ == '__main__':
    main()