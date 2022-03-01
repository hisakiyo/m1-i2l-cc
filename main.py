import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

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
    
def neural_network(data, pred):
    # Create model
    mlp = MLPClassifier()
    # Split data
    x_train, y_train, x_test, y_test = splitData(data, 0.3, pred)
    # Train model
    mlp.fit(x_train, y_train)
    # Predict
    y_pred = mlp.predict(x_test)
    # Evaluate
    print("Train score: {}, Test score: {}".format(mlp.score(x_train, y_train), mlp.score(x_test, y_test)))

def main():
    data1 = pd.read_csv('dataCCfinal_1.csv')
    data2 = pd.read_csv('dataCCfinal_2.csv')
    print("Analyse du dataset 1:")
    analyze(data1)
    print("\n--------------------\nAnalyse du dataset 2:\n")
    analyze(data2)
    label_encode(data1)
    neural_network(data1, 'Z')
    label_encode(data2)
    neural_network(data2, 'Z')

if __name__ == '__main__':
    main()