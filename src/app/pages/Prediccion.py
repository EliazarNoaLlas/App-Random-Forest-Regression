# ================================== Importando Librerias =============================
from pages.Modelo_ import regressor
from pages.Modelo_ import handle_outliers
from pages.Modelo_ import analizar_y_eliminar_ruido

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Título de la página
st.title(" :date: Predicción")
## espacio en la parte superior de la pagina
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Columna izquierda: Componente para cargar un archivo Excel
df = pd.read_excel("./data/Diclofenaco-prediccion.xlsx", engine="openpyxl")
# Columna izquierda: Componente para cargar un archivo Excel
with col1:
    uploaded_file = st.file_uploader(":file_folder: Cargar archivo Excel", type=["xlsx", "xls"])

    # Carga el archivo y crea un DataFrame si se ha cargado un archivo
    if uploaded_file is not None:
        filename = uploaded_file.name
        st.write(filename)

        try:
            new_df = pd.read_excel(uploaded_file, engine="openpyxl")

            # Verificar la consistencia de las columnas y tipos de datos
            if not df.columns.equals(new_df.columns) or not df.dtypes.equals(new_df.dtypes):
                st.error("Error: El archivo cargado no tiene el mismo formato que el archivo por defecto.")
                st.error("El archivo debe tener las mismas columnas y tipos de datos para cada columna.")
            else:
                df = new_df
                st.success("El archivo se ha cargado con éxito.")
                # Mostrar las 5 primeras filas del conjunto de datos actualizado
                st.write("Conjunto de Datos Actualizado:")
                st.write(df.head())
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
    else:
        st.warning("Ningún archivo se ha cargado, se utilizará un archivo por defecto.")

# Aplicar funciones para manejar outliers y eliminar ruido
df['PRODUCTOS ALMACENADOS'] = handle_outliers(df['PRODUCTOS ALMACENADOS'])
df['DEMANDA DEL PRODUCTO'] = handle_outliers(df['DEMANDA DEL PRODUCTO'])
df_nuevo = analizar_y_eliminar_ruido(df)


# Hacer predicciones con el modelo entrenado
nuevos_datos = df_nuevo  # Aquí debes reemplazarlo con tus nuevos datos
predicciones = regressor.predict(nuevos_datos)

# Columna derecha: Conjunto de datos con las primeras 5 filas
with col2:
    st.write("##  Conjunto de Datos de Prueba:")

    # Crear una tabla con Plotly solo con las primeras 5 filas
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color='darkslategray', line_color='white', align='center'),
        cells=dict(values=[df[col] for col in df.columns], fill_color='dimgray', line_color='white', align='center')
    )])

    # Estilo adicional para la tabla
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
    )

    st.plotly_chart(fig)

# Crear un nuevo DataFrame con las predicciones
df_predicciones = pd.DataFrame({"Productos Vendidos Predicho": predicciones})

# Agregar la columna de predicciones al DataFrame original
df_con_predicciones = pd.concat([df, df_predicciones], axis=1)

# Crear una tabla con Plotly que incluya la columna de predicciones con formato de 2 decimales
fig_predicciones = go.Figure(data=[go.Table(
    header=dict(values=list(df_con_predicciones.columns), fill_color='darkslategray', line_color='white', align='center'),
    cells=dict(values=[df_con_predicciones[col] if col != 'Productos Vendidos Predicho' else df_con_predicciones[col].apply(lambda x: f'{x:.2f}') for col in df_con_predicciones.columns], 
               fill_color=['dimgray' if col != 'Productos Vendidos Predicho' else 'steelblue' for col in df_con_predicciones.columns],
               line_color='white', align='center',
               format=[None if col != 'Productos Vendidos Predicho' else ',.2f' for col in df_con_predicciones.columns])
)])

# Estilo adicional para la tabla
fig_predicciones.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
)

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

with col1:
    # Mostrar la tabla con predicciones debajo de los gráficos
    st.plotly_chart(fig_predicciones)
with col2:

    # Crear gráfico de dispersión entre Demanda del Producto y Productos Vendidos Predichos
    scatter_predicciones = px.scatter(df_con_predicciones, x='DEMANDA DEL PRODUCTO', y='Productos Vendidos Predicho',
                                    title='Diagrama de Dispersión: Demanda del Producto vs Productos Vendidos Predicho',
                                    labels={'DEMANDA DEL PRODUCTO': 'DEMANDA DEL PRODUCTO', 'Productos Vendidos Predicho': 'Productos Vendidos Predicho'})

    # Mostrar el gráfico de dispersión
    st.plotly_chart(scatter_predicciones)