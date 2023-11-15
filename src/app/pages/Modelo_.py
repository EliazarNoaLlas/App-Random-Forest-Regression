# ================================== Importando Librerias =============================
from Inicio import df
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import plotly as pl
import plotly.express as px
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


# ------------------------- MANEJO DE DATOS ---------------------------
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

# ------------------------- ENTRENAMIENTO ---------------------------
# Definir las características (X) y la variable objetivo (y)
X = df_sin_ruido.drop('PRODUCTOS VENDIDOS', axis=1)
y = df_sin_ruido['PRODUCTOS VENDIDOS']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Random Forest Regression
bosque = RandomForestRegressor(n_estimators=100,
                                criterion="squared_error",  # Utilizar "squared_error" en lugar de "mse"
                                max_features="sqrt",
                                bootstrap=True,
                                oob_score=True,
                                random_state=42)

# Definir la métrica a utilizar (en este caso, negativo del Error Cuadrático Medio para que sea coherente con la validación cruzada)
metrica = make_scorer(mean_squared_error, greater_is_better=False)

# Realizar validación cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Puedes ajustar el número de divisiones (folds)
resultados_cross_val = -cross_val_score(bosque, X.values, y.values, cv=kf, scoring=metrica)

# Imprimir los resultados de la validación cruzada
print("Resultados de la validación cruzada:")
print("MSE por fold:", resultados_cross_val)
print("Promedio MSE:", np.mean(resultados_cross_val))

# Entrenar el modelo en todo el conjunto de datos
regressor = bosque.fit(X.values, y.values)

# Imprimir la puntuación R^2 en el conjunto completo
print("Puntuación R^2 en el conjunto completo:", bosque.score(X.values, y.values))

# Imprimir la puntuación "out-of-bag" (OOB)
print("Puntuación OOB:", bosque.oob_score_)

y_pred = regressor.predict(X_test)

r2 = r2_score(y_test, y_pred)

# Gráfico de dispersión (Scatter Plot) con Plotly
fig = px.scatter(x=y_test, y=y_pred, title="Gráfico de Dispersión", labels={"x": "Data Actual", "y": "Predicciones"})

# Curva de regresión
x_range = np.linspace(min(y_test), max(y_test), 100)
y_range = x_range  # Línea de regresión ideal (y = x)
fig.add_scatter(x=x_range, y=y_range, mode='lines', name='Línea de Regresión Ideal', line=dict(color='red', dash='dash'))

# Métricas de rendimiento
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
metrics_text = f"Mean Absolute Error: {mae:.2f}\nMean Squared Error: {mse:.2f}\nRoot Mean Squared Error: {rmse:.2f}\nR²: {r2:.2f}"

st.plotly_chart(fig)
st.text(metrics_text)

## ----------------------------- MODELO ------------------------------
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
st.write(f"Profundidad máxima de los árboles: {bosque.max_depth}")

# ------------------------ Realizar Predicciones ---------------------------
# Dividir la página en dos columnas
left_column, right_column = st.columns(2)

# Columna izquierda: Gráfico de dispersión
with left_column:
    left_column.header("Gráfico de Dispersión")
    scatter_fig = px.scatter(df, x="PRODUCTOS ALMACENADOS", y="PRODUCTOS VENDIDOS", title="Productos Almacenados vs Productos Vendidos")
    left_column.plotly_chart(scatter_fig)

# Columna derecha: Formulario para ingresar nuevos datos y botón de predicción
with right_column:
    st.header("Ingresar Datos para Predicción")
    # Filtrar las variables que no son consideradas ruido
    variables_no_ruido = df_sin_ruido.columns
    new_data = {
        "MES": right_column.number_input("MES", min_value=1, max_value=100),
        "PRODUCTOS ALMACENADOS": right_column.number_input("PRODUCTOS ALMACENADOS", min_value=0) if "PRODUCTOS ALMACENADOS" in variables_no_ruido else None,
        "GASTO DE MARKETING": right_column.number_input("GASTO DE MARKETING", min_value=0) if "GASTO DE MARKETING" in variables_no_ruido else None,
        "GASTO DE ALMACENAMIENTO": right_column.number_input("GASTO DE ALMACENAMIENTO", min_value=0) if "GASTO DE ALMACENAMIENTO" in variables_no_ruido else None,
        "DEMANDA DEL PRODUCTO": right_column.number_input("DEMANDA DEL PRODUCTO", min_value=1, max_value=10) if "DEMANDA DEL PRODUCTO" in variables_no_ruido else None,
        "FESTIVIDAD": right_column.number_input("FESTIVIDAD", min_value=0, max_value=1) if "FESTIVIDAD" in variables_no_ruido else None,
        "PRECIO DE VENTA": right_column.number_input("PRECIO DE VENTA", min_value=0) if "PRECIO DE VENTA" in variables_no_ruido else None
    }

    # Botón para realizar predicción
    if right_column.button("Realizar Predicción"):
        # Filtrar las variables que no son None (aquellas que no son consideradas ruido)
        new_data_filtered = {key: value for key, value in new_data.items() if value is not None}
        new_data_df = pd.DataFrame(new_data_filtered, index=[0])
        new_predictions = bosque.predict(new_data_df)
        # Estilo del cuadro de texto
        style = """
            padding: 10px;
            background-color: #001F3F; /* Color plateado o gris claro */
            border: 2px solid #C0C0C0; /* Borde del cuadro */
            border-radius: 5px; /* Esquinas redondeadas */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Sombra */
            color: #001F3F; /* Texto en azul oscuro */
            font-size: 18px; /* Tamaño de fuente más grande */
        """

        # Imprimir predicción en el cuadro de texto con estilo
        st.markdown(f'<div style="{style}">Predicción de Productos Vendidos: {new_predictions[0]:.2f}</div>', unsafe_allow_html=True)

    


