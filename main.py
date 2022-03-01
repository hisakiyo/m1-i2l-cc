# Remove SKLearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import time
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

def display_score(classifier, x_train, y_train, x_test, y_test, reg=False):
    print("Train score: {}, Test score {}".format(classifier.score(x_train, y_train), classifier.score(x_test, y_test)))
    if not reg:
        print("Confusion matrix: {}".format(confusion_matrix(y_test, classifier.predict(x_test))))
    else:
        print('MSE: {}, MAE: {}, R2: {}'.format(mean_squared_error(y_test, classifier.predict(x_test)), mean_absolute_error(y_test, classifier.predict(x_test)), r2_score(y_test, classifier.predict(x_test))))

    
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
    display_score(mlp, x_train, y_train, x_test, y_test, True)

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
    display_score(knn, x_train, y_train, x_test, y_test, True)

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
    display_score(dt, x_train, y_train, x_test, y_test, True)

def main():
    data1 = pd.read_csv('dataCCfinal_1.csv')
    data2 = pd.read_csv('dataCCfinal_2.csv')
    print("Analyse du dataset 1:")
    analyze(data1)
    print("\n--------------------\nAnalyse du dataset 2:\n")
    analyze(data2)

    # On transforme les catégories en entiers
    label_encode(data1)
    label_encode(data2)
    
    # Neural network
    print('\n--------------------\nNeural network:\n')
    start_time = time.time()
    neural_network(data1, 'Z')
    end_time = time.time()
    print("Temps d'exécution: {}s\n".format(end_time - start_time))
    start_time = time.time()
    neural_network_regression(data2, 'Z')
    end_time = time.time()
    print("Temps d'exécution: {}s".format(end_time - start_time))

    # K-nearest neighbors
    print('\n--------------------\nK-nearest neighbors:\n')
    start_time = time.time()
    k_nearest_neighbors(data1, 'Z')
    end_time = time.time()
    print("Temps d'exécution: {}s\n".format(end_time - start_time))
    start_time = time.time()
    k_nearest_neighbors_regression(data2, 'Z')
    end_time = time.time()
    print("Temps d'exécution: {}s\n".format(end_time - start_time))

    # Decision tree
    print('\n--------------------\nDecision tree:\n')
    start_time = time.time()
    decision_tree(data1, 'Z')
    end_time = time.time()
    print("Temps d'exécution: {}s\n".format(end_time - start_time))
    start_time = time.time()
    decision_tree_regression(data2, 'Z')
    end_time = time.time()
    print("Temps d'exécution: {}s\n".format(end_time - start_time))


if __name__ == '__main__':
    main()