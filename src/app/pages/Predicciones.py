# ================================== Importando Librerias =============================
from pages.Modelo_ import bosque
from pages.Modelo_ import handle_outliers
from pages.Modelo_ import analizar_y_eliminar_ruido

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Título de la página
st.title(" :date: Predicciones")
## espacio en la parte superior de la pagina
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Componente para cargar un archivo Excel
uploaded_file = st.file_uploader(":file_folder: Cargar archivo Excel", type=["xlsx", "xls"])
# ================================================  Cargar Datos  ===========================================================
# Leer el conjunto de datos por defecto
df = pd.read_excel("./data/Diclofenaco-prediccion.xlsx", engine="openpyxl")
# Carga el archivo y crea un DataFrame si se ha cargado un archivo
if uploaded_file is not None:
    filename = uploaded_file.name
    st.write(filename)

    try:
        new_df = pd.read_excel(uploaded_file, engine="openpyxl")
        new_df.columns = df.columns
        # Verificar la consistencia de las columnas y tipos de datos
        #  verifica si la cantidad de columnas no es la misma.
        if len(df.columns) != len(new_df.columns):
            st.error("Error: El archivo cargado no tiene la misma cantidad de columnas que el archivo por defecto.")
        # verifica si los nombres de las columnas no son iguales.
        elif all(df.columns == new_df.columns):
            st.error("Error: El archivo cargado tiene nombres de columnas incorrectos. Corrigiendo...")
            new_df.columns = df.columns
            df = new_df
            st.success("El archivo se ha cargado con éxito.")
        #  verifica si los tipos de datos de las columnas no son iguales.
        elif not df.dtypes.equals(new_df.dtypes):
            st.error("Error: El archivo cargado no tiene el mismo formato de datos que el archivo por defecto.")
        else:
            # Guardar y actualizar el nuevo DataFrame
            new_df.to_excel("../data/Diclofenaco-prediccion.xlsx", index=False)
            df = new_df
            st.success("El archivo se ha cargado con éxito.")
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
else:
    st.warning("Ningún archivo se ha cargado, se utilizará un archivo por defecto.")

# Aplicar la función para manejar outliers
df['PRODUCTOS ALMACENADOS'] = handle_outliers(df['PRODUCTOS ALMACENADOS'])

# Hacer predicciones con el modelo entrenado
nuevos_datos = df[['MES', 'PRODUCTOS ALMACENADOS', 'GASTO DE ALMACENAMIENTO', 'DEMANDA DEL PRODUCTO', 'FESTIVIDAD']]  # Columnas relevantes para la predicción
predicciones = bosque.predict(nuevos_datos)

# Crear un nuevo DataFrame con las predicciones
df_predicciones = pd.DataFrame({"PRODUCTOS VENDIDOS": predicciones})

# Agregar la columna de predicciones al DataFrame original
df_con_predicciones = pd.concat([df, df_predicciones], axis=1)

# Crear una tabla con Plotly que incluya la columna de predicciones con formato de 2 decimales
fig_predicciones = go.Figure(data=[go.Table(
    header=dict(values=list(df_con_predicciones.columns), fill_color='darkslategray', line_color='white', align='center'),
    cells=dict(values=[df_con_predicciones[col] if col != 'PRODUCTOS VENDIDOS' else df_con_predicciones[col].apply(lambda x: f'{x:.2f}') for col in df_con_predicciones.columns], 
               fill_color=['dimgray' if col != 'PRODUCTOS VENDIDOS' else 'steelblue' for col in df_con_predicciones.columns],
               line_color='white', align='center',
               format=[None if col != 'PRODUCTOS VENDIDOS' else ',.2f' for col in df_con_predicciones.columns])
)])

# Estilo adicional para la tabla
fig_predicciones.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
)

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

with col1:
    # Mostrar la tabla con predicciones debajo de los gráficos
    st.subheader("Cantidad de Productos que se Venderán:")
    st.plotly_chart(fig_predicciones)
with col2:

    # Crear gráfico de dispersión entre Demanda del Producto y Productos Vendidos Predichos
    scatter_predicciones = px.scatter(df_con_predicciones, x='DEMANDA DEL PRODUCTO', y='PRODUCTOS VENDIDOS',
                                    title='Diagrama de Dispersión: Demanda del Producto vs Productos Vendidos',
                                    labels={'DEMANDA DEL PRODUCTO': 'DEMANDA DEL PRODUCTO', 'PRODUCTOS VENDIDOS': 'PRODUCTOS VENDIDOS'})
    # Mostrar el gráfico de dispersión
    st.plotly_chart(scatter_predicciones)

# ------------------------ Realizar Predicciones ---------------------------
# Dividir la página en dos columnas
left_column, right_column = st.columns(2)

# Columna izquierda: Gráfico de dispersión
with left_column:
    left_column.header("Gráfico de Dispersión")
    scatter_fig = px.scatter(df_con_predicciones, x="DEMANDA DEL PRODUCTO", y="PRODUCTOS VENDIDOS", title="Productos Almacenados vs Productos Vendidos")
    left_column.plotly_chart(scatter_fig)

# Columna derecha: Formulario para ingresar nuevos datos y botón de predicción
with right_column:
    st.header("Ingresar Datos para Predicción")
    # Filtrar las variables que no son consideradas ruido
    variables_no_ruido = nuevos_datos.columns
    new_data = {
        "MES": right_column.number_input("MES", min_value=1, max_value=100,key="mes_input"),
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