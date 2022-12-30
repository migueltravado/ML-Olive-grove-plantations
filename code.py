import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from scipy import stats
from sklearn import preprocessing

from sklearn.covariance import EllipticEnvelope

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn import model_selection

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.feature_selection import RFE

from sklearn import metrics

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def plot_graph(target, y_pred, title, name, metricas):
        # ** CÁLCULO DE LAS MÉTRICAS DE EVALUACIÓN
        MAE = metricas['MAE'](target, y_pred)
        RMSE = metricas['RMSE'](target, y_pred)
        MAPE = mape(target, y_pred)
        R2 = metricas['R2'](target, y_pred)
        
        # ** PLOT GRAPH
        fig, ax = plt.subplots()
        ax.scatter(target, y_pred, edgecolors=(0,0,0), s=1)
        coefs = np.polyfit(target, y_pred, 1)
        recta = np.poly1d(coefs)
        x_min, x_max = ax.get_xlim()
        ax.plot([x_min, x_max],
                recta((x_min, x_max)), 'k--', lw=1)
        x = [x_min, x_max]
        ax.plot(x, x, linestyle="dashed", marker="none", lw=1, color="#085979")
        ax.set_xlabel('Valor real de la clase')
        ax.set_ylabel('Predicción')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        title_ = "MAE: %.3f   RMSE: %.3f  MAPE: %.2f  R2: %.3f --- " % (MAE, RMSE, MAPE, R2)
        plt.title(title_ + str(title))
        plt.savefig("ML - Data analysis - Olive grove plantations/graphs/" + str(name), dpi=300)

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


# ** MODELADO SUPERVISADO - REGRESIÓN LINEAL MÚLTIPLE
reg = LinearRegression()
# ** DEFINICIÓN DE MÉTRICAS
metricas = {
    'MAE' : lambda y, y_pred:
            metrics.mean_absolute_error(y, y_pred),
    'RMSE' : lambda y, y_pred:
            sqrt(metrics.mean_squared_error(y, y_pred)),
    'MAPE' : lambda y, y_pred:
            np.mean(np.abs((y - y_pred) / y) * 100),
    'R2' : lambda y, y_pred:
            metrics.r2_score(y, y_pred)
}
df_np = df_norm.to_numpy()
target = df_np[:, -1]
data = df_np[:, :-1]
# ** PREDICCIÓN DE LA VARIABLE OBJETIVO - CLASE (REGRESIÓN)
seed = 1
y_pred = model_selection.cross_val_predict(reg, data, target, 
                           cv = KFold(n_splits=10, random_state=seed, shuffle=True))
# ** GRÁFICA MATPLOTLIB
plot_graph(target, y_pred, "REG", "RLM.png", metricas)


# ** OPTMIZACIÓN DEL ALGORITMO CON ELIMINACIÓN RECURSIVA - SELECCIÓN DE ATRIBUTOS
reg_opt = LinearRegression()
# Aplicamos la eliminación recursiva de atributos
n_features = int(0.9 * len(df_norm.columns))
rfe = RFE(estimator=reg_opt, n_features_to_select=n_features)
fit = rfe.fit(data, target)
# Seleccionamos los atributos más importantes
selected_features = fit.support_
seed = 1
y_pred = model_selection.cross_val_predict(reg, data[:, selected_features], target, 
                           cv = KFold(n_splits=10, random_state=seed, shuffle=True))
# ** GRÁFICA MATPLOTLIB
plot_graph(target, y_pred, "REG OPT", "RLM_Optimized.png", metricas)


# ** MODELADO SUPERVISADO - KNN
k = 10
knn = KNeighborsRegressor(n_neighbors=k)
# ** PREDICCIÓN DE LA VARIABLE OBJETIVO - CLASE (KNN)
seed = 1
y_pred = model_selection.cross_val_predict(knn, data, target, 
                           cv = KFold(n_splits=10, random_state=seed, shuffle=True))
# ** GRÁFICA MATPLOTLIB
plot_graph(target, y_pred, "KNN", "KNN.png", metricas)







        




