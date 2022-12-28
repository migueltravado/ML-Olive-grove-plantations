import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope

# ** LECTURA DEL ARCHIVO ORIGINAL CSV
df = pd.read_csv("ML - Data analysis - Olive grove plantations/csv/csv_result-sds_PICA_H1.csv", 
                 decimal=".", delimiter=",", na_values=['','?', ' '], on_bad_lines='skip') # na_values=['','?']


# ** TRATAMIENTO VALORES NULOS. SUSTITUCIÓN IN-PLACE POR MEDIA ARITMÉTICA
n_instances = df.shape[0]
for col in df.columns:
    if df[col].dtype == 'float64':
        if df[col].count() < n_instances:
            mean = df[col].mean()
            df[col].fillna(mean, inplace=True)
    else:
        df.drop(col, inplace=True, axis=1)

# ** ELIMINACIÓN DE INSTANCIAS CON VALORES NULOS. MOTIVOS DE SEGURIDAD
# En este caso, el atributo TRAT_LAST_bk_w-1_s21 no tiene ningún valor contable, todos son NULOS
df.drop('TRAT_LAST_bk_w-1_s21', inplace=True, axis=1)


# ** NORMALIZACIÓN DE VALORES
normalizer = preprocessing.MinMaxScaler()
df_norm = normalizer.fit_transform(df)


# ** CONVERSIÓN DE NP.ARRAY A PD.DATAFRAME
df_norm = pd.DataFrame(df_norm, columns=list(df.columns))  

# ** REDUCCIÓN DECIMALES (MEDIDAS PARA ARREGLAR EL ERROR 0.1)
df_norm = df_norm.round(decimals=6)

# ** GUARDADO DE CSV CON LOS DATOS NORMALIZADOS Y VALORES NULOS REEMPLAZADOS
#df_norm.to_csv('csv_processed/PICA_H1/csv_result-sds_PICA_H2_processed_normalized.csv', sep=',')

# ** APLICACIÓN DE DETECCIÓN DE OUTLIERS MEDIANTE ENVOLVENTE ELÍPTICA
outlier_method = EllipticEnvelope().fit(df_norm)
scores_pred = outlier_method.decision_function(df_norm)
threshold = stats.scoreatpercentile(scores_pred, 25)
# Eliminación de las instancias consideradas outliers
indexes = list()
for i in range(len(scores_pred)):
    if scores_pred[i] < threshold:
        indexes.append(i)
df_norm.drop(index=indexes, inplace=True, axis=0)


        




