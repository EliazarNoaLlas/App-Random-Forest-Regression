# ================================== Importando Librerias =============================
from Inicio import df
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import plotly as pl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import warnings
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer
warnings.filterwarnings('ignore')

# Título de la página
st.title(" :date: Modelo RandomForest Regression")
## espacio en la parte superior de la pagina

st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

# Leer el conjunto de datos por defecto
df = pd.read_excel("data/Melsol-test.xlsx", engine="openpyxl")

# ====================================================== MANEJO DE DATOS ======================================================
# Función para manejar outliers en una columna
def handle_outliers(column, cap_value=None):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    if cap_value is not None:
        column = column.clip(lower=lower_limit, upper=cap_value)
    else:
        column = column.clip(lower=lower_limit, upper=upper_limit)
    
    return column

# Función para analizar y eliminar columnas con un solo valor único
def analizar_y_eliminar_ruido(df):
    columnas_a_eliminar = []
    
    for columna in df.columns:
        if df[columna].nunique() == 1:
            columnas_a_eliminar.append(columna)
            print(f'Columna "{columna}" tiene un solo valor único. Se considera ruido.')
    
    df_sin_ruido = df.drop(columnas_a_eliminar, axis=1)
    
    print(f'\nColumnas eliminadas: {columnas_a_eliminar}')
    
    return df_sin_ruido

# Aplicar funciones para manejar outliers y eliminar ruido
df['PRODUCTOS ALMACENADOS'] = handle_outliers(df['PRODUCTOS ALMACENADOS'])
df['DEMANDA DEL PRODUCTO'] = handle_outliers(df['DEMANDA DEL PRODUCTO'])
df['PRODUCTOS VENDIDOS'] = handle_outliers(df['PRODUCTOS VENDIDOS'])
df_sin_ruido = analizar_y_eliminar_ruido(df)
# ====================================================== MANEJO DE DATOS ======================================================

# ====================================================== ENTRENAMIENTO ======================================================
# Definir las características (X) y la variable objetivo (y)
X = df_sin_ruido.drop('PRODUCTOS VENDIDOS', axis=1)
y = df_sin_ruido['PRODUCTOS VENDIDOS']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Random Forest Regression
bosque = RandomForestRegressor(
    n_estimators=100,
    criterion="squared_error",  # Utilizar "squared_error" en lugar de "mse"
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    random_state=42
)

# Entrenar el modelo en todo el conjunto de datos
bosque.fit(X_train, y_train)

# Evaluación del modelo
# Métrica a utilizar (en este caso, negativo del Error Cuadrático Medio para que sea coherente con la validación cruzada)
metrica = make_scorer(mean_squared_error, greater_is_better=False)

# Validación cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Puedes ajustar el número de divisiones (folds)
resultados_cross_val = cross_val_score(bosque, X_train, y_train, cv=kf, scoring=metrica)

# Imprimir los resultados de la validación cruzada
print("Resultados de la validación cruzada:")
print("MSE por fold:", resultados_cross_val)
print("Promedio MSE:", np.mean(resultados_cross_val))

# Métricas adicionales
# Puntuación R^2 en el conjunto completo
r2_full_set = bosque.score(X, y)
print("Puntuación R^2 en el conjunto completo:", r2_full_set)

# Puntuación "out-of-bag" (OOB)
oob_score = bosque.oob_score_
print("Puntuación OOB:", oob_score)

# Predicciones en el conjunto de prueba
y_pred = bosque.predict(X_test)

# Puntuación R^2 en el conjunto de prueba
r2_test_set = r2_score(y_test, y_pred)
print("Puntuación R^2 en el conjunto de prueba:", r2_test_set)

# ====================================================== GRAFICANDO ======================================================

# Gráfico de dispersión (Scatter Plot) con Plotly
scatter_fig = px.scatter(x=y_test, y=y_pred, title="Gráfico de Dispersión", labels={"x": "Data Actual", "y": "Predicciones"})

# Curva de regresión
x_range = np.linspace(min(y_test), max(y_test), 100)
y_range = x_range  # Línea de regresión ideal (y = x)
scatter_fig.add_scatter(x=x_range, y=y_range, mode='lines', name='Línea de Regresión Ideal', line=dict(color='red', dash='dash'))

def plot_metric(metrics_dict):
    fig = go.Figure()

    for label, value in metrics_dict.items():
        fig.add_trace(
            go.Indicator(
                value=value,
                gauge={"axis": {"visible": False}},
                number={
                    "prefix": "",
                    "suffix": "",
                    "font.size": 28,
                },
                title={
                    "text": label,
                    "font": {"size": 24},
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)
def plot_gauge(
    indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound
):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": indicator_suffix,
                "font.size": 26,
            },
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1},
                "bar": {"color": indicator_color},
            },
            title={
                "text": indicator_title,
                "font": {"size": 28},
            },
        )
    )
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)

# Crear dos columnas
col1, col2 = st.columns(2)

# Columna izquierda: Gráfico de dispersión
with col1:
    col1.header("Gráfico de Dispersión")
    col1.plotly_chart(scatter_fig)

# Columna derecha: Métricas de rendimiento
with col2:
    col2.header("Métricas de Rendimiento")
    # Supongamos que ya calculaste tus métricas
    mae_value = 10.5
    mse_value = 50.2
    rmse_value = 7.1
    r2_value = 0.85
    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    # Llama a la función plot_gauge para cada métrica
    plot_gauge(mae_value, "blue", "", "MAE", 20)
    plot_gauge(mse_value, "green", "", "MSE", 100)
    plot_gauge(rmse_value, "orange", "", "RMSE", 15)
    plot_gauge(r2_value, "red", "", "R²", 1)

# ============================================== CARACTERISTICAS DEL MODELO ==========================================
with col1:
    col1.header("Caracteristicas del Modelo")
    # Obtener características importantes del modelo (importances)
    importances = bosque.feature_importances_

    # Crear un DataFrame para mostrar características importantes
    importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    importances_df = importances_df.sort_values(by='Importance', ascending=False)

    # Gráfico de barras para mostrar características importantes con Plotly Express
    bar_fig = px.bar(
        importances_df,
        x='Importance',
        y='Feature',
        orientation='h',  # Orientación horizontal
        title="Importancia de las Características",
        labels={'Importance': 'Importancia', 'Feature': 'Característica'},
        color='Importance',  # Colorea las barras según la importancia
        color_continuous_scale='Viridis',  # Puedes cambiar el esquema de color aquí
    )

    # Ajustes adicionales para mejorar la visualización
    bar_fig.update_layout(
        xaxis_title='Importancia',
        yaxis_title='Característica',
        showlegend=False,  # Oculta la leyenda, ya que el color indica la importancia
        margin=dict(l=10, r=10, b=10, t=30),  # Ajusta los márgenes para una mejor visualización
    )

    # Métricas de rendimiento
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    metrics_text = f"Mean Absolute Error: {mae:.2f}\nMean Squared Error: {mse:.2f}\nRoot Mean Squared Error: {rmse:.2f}\nR²: {r2:.2f}"

    # Impresión del modelo y resultados
    st.plotly_chart(bar_fig)

    # Mostrar detalles del modelo RandomForestRegressor
    st.write("Detalles del modelo RandomForestRegressor:")
    st.write(f"Número de árboles: {bosque.n_estimators}")



    


