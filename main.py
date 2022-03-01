import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def analyze(data):
    print("Nombre d'exemple dans le dataset:", data['A'].count())
    print("Nombre de caractéristiques:", len(data.columns))
    print("Shape: ", data.shape)
    # print(data.info())
    print(data.describe())
    print(data.corr())

def label_encode(data, cols):
    for col in cols:
        le = LabelEncoder()
        unique_values = list(data[col].unique())
        le_fitted = le.fit(unique_values)
        values = list(data[col].values)
        values_transformed = le.transform(values)
        data[col] = values_transformed

def main():
    data = pd.read_csv('dataCCfinal_1.csv')
    cols_to_labelize = ['C']
    label_encode(data, cols_to_labelize)
    analyze(data)

if __name__ == '__main__':
    main()