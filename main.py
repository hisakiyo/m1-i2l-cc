import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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

def main():
    data1 = pd.read_csv('dataCCfinal_1.csv')
    data2 = pd.read_csv('dataCCfinal_2.csv')
    print("Analyse du dataset 1:")
    analyze(data1)
    print("\n--------------------\nAnalyse du dataset 2:\n")
    analyze(data2)

if __name__ == '__main__':
    main()