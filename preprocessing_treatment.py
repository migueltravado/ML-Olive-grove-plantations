import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.covariance import EllipticEnvelope


df = pd.read_csv("csv/csv_result-sds_PICA_H1.csv", 
                 decimal=".", delimiter=",", na_values="?", on_bad_lines='skip')


# ** TRATAMIENTO VALORES NULOS. SUSTITUCIÓN IN-PLACE POR MEDIA ARITMÉTICA
n_instances = df.shape[0]
headers = [*pd.read_csv('csv/csv_result-sds_PICA_H1.csv', nrows=1)]
for col in headers:
    if df[col].dtype == 'float64' and df[col].count() < n_instances:
        mean = df[col].mean()
        df[col].fillna(mean, inplace=True)
    else:
        df.drop(col, inplace=True, axis=1)
        headers.remove(col)

df.drop('domain', inplace=True, axis=1)
print(df.dtypes)

df.dropna()


# ** ERROR. INPUT CONTAINS NAN, INFINITY OR A VALUE TOO LARGE FOR DTYPE('float64') 
# ** EL TRATAMIENTO DE DATOS DEBERÍA HABER RELLENADO NULOS Y ELIMINADO COLUMNAS != FLOAT64
outlier_method = EllipticEnvelope().fit(df)

        





        







#outlier_method = EllipticEnvelope().fit(df)
#scores_pred = outlier_method.decision_function(df)
#threshold = stats.scoreatpercentile(scores_pred, 25)


