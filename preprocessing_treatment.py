import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope

# ** LECTURA DEL ARCHIVO ORIGINAL CSV
df = pd.read_csv("csv/csv_result-sds_PICA_H1.csv", 
                 decimal=".", delimiter=",", na_values=['','?', ' '], on_bad_lines='skip') # na_values=['','?']


# ** TRATAMIENTO VALORES NULOS. SUSTITUCIÓN IN-PLACE POR MEDIA ARITMÉTICA
n_instances = df.shape[0]
for col in df.columns:
    if df[col].dtype == 'float64' or df[col].isna():
        if df[col].count() < n_instances:
            mean = df[col].mean()
            df[col].fillna(mean, inplace=True)
    else:
        df.drop(col, inplace=True, axis=1)

# ** ELIMINACIÓN DE INSTANCIAS CON VALORES NULOS. MOTIVOS DE SEGURIDAD
df.drop('TRAT_LAST_bk_w-1_s21', inplace=True, axis=1)


# ** NORMALIZACIÓN DE VALORES
normalizer = preprocessing.MinMaxScaler()
df_norm = normalizer.fit_transform(df)


# ** CONVERSIÓN DE NP.ARRAY A PD.DATAFRAME
df_norm = pd.DataFrame(df_norm, columns=list(df.columns))  

# *+ REDUCCIÓN DECIMALES (MEDIDAS PARA ARREGLAR EL ERROR 0.1)
df_norm = df_norm.round(decimals=6)

# ** GUARDADO DE CSV CON LOS DATOS NORMALIZADOS Y VALORES NULOS REEMPLAZADOS
df_norm.to_csv('csv_processed/PICA_H1/csv_result-sds_PICA_H1_processed_normalized.csv', sep=',')


# ** ERROR 0.1. INPUT CONTAINS NAN, INFINITY OR A VALUE TOO LARGE FOR DTYPE('float64') 
# ** EL TRATAMIENTO DE DATOS DEBERÍA HABER RELLENADO NULOS Y ELIMINADO COLUMNAS != FLOAT64
outlier_method = EllipticEnvelope().fit(df_norm)
scores_pred = outlier_method.decision_function(df_norm)
threshold = stats.scoreatpercentile(scores_pred, 25)



        




