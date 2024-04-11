import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ==========================================  Configurar la página =========================================================
st.set_page_config(
    page_title="Análisis de Ventas",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)
#============================================================================================================================
# ==========================================  Baner =========================================================================
# Cargar el componente de BannerPersonalizado.html
with open("utils/Baner.html", "r", encoding="utf-8") as file:
    custom_banner_html = file.read()

# Agregar estilos CSS desde la carpeta utils
with open("utils/Baner_style.css", "r", encoding="utf-8") as file:
    custom_styles_css = file.read()
# Mostrar el componente de Banner en Streamlit con los estilos CSS
st.markdown("""
    <style>
        %s
    </style>
""" % custom_styles_css, unsafe_allow_html=True)

st.markdown(custom_banner_html, unsafe_allow_html=True)
#============================================================================================================================

# ==========================================  Titulo de la Pagina ===========================================================
st.markdown("## :date: Cargando el conjunto de datos de entrenamiento :shopping_trolley:")
## espacio en la parte superior de la pagin
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
#============================================================================================================================


# ================================================  Cargar Datos  ===========================================================

# Componente para cargar el conjunto de datos
uploaded_file = st.file_uploader(":file_folder: Cargar un archivo Excel", type=["xlsx", "xls"])

# Leer el conjunto de datos por defecto
df = pd.read_excel("data/Melsol-test.xlsx", engine="openpyxl")

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
        #  verifica si los tipos de datos de las columnas no son iguales.
        elif not df.dtypes.equals(new_df.dtypes):
            st.error("Error: El archivo cargado no tiene el mismo formato de datos que el archivo por defecto.")
        else:
            # Guardar y actualizar el nuevo DataFrame
            new_df.to_excel("./data/Melsol-test.xlsx", index=False)
            df = new_df
            st.success("El archivo se ha cargado con éxito.")
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
else:
    st.warning("Ningún archivo se ha cargado, se utilizará un archivo por defecto.")


# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Agregar opciones para modificar el conjunto de datos
operation = st.sidebar.radio("Seleccione una operación:", ["Actualizar Columna", "Crear Fila", "Actualizar Fila", "Eliminar Fila"])

# Modificación del conjunto de datos
if operation == "Actualizar Columna":
    # Crear un formulario para renombrar columnas
    with col1:
        form = st.form("rename_columns_form")
        col_rename_dict = {}
        for col in df.columns:
            new_col_name = form.text_input(f"Nuevo nombre para la columna '{col}':", col)
            col_rename_dict[col] = new_col_name

        # Agregar un botón al formulario
        submit_button = form.form_submit_button("Actualizar")

        # Si el botón se presiona, actualizar las columnas del conjunto de datos
        if submit_button:
            df.rename(columns=col_rename_dict, inplace=True)
            df.to_excel("data/Melsol-test.xlsx", index=False)
            st.success("Nombres de columnas actualizados.")
# Resto de las operaciones
else:
    if operation == "Crear Fila":
        # Obtener la última fila del DataFrame
        last_row = df.iloc[-1]

        # Crear una nueva fila con datos ingresados por el usuario
        with col1:
            new_row = {
                'MES': st.number_input("MES", value=last_row['MES']),
                'PRODUCTOS ALMACENADOS': st.number_input("PRODUCTOS ALMACENADOS", value=last_row['PRODUCTOS ALMACENADOS']),
                'GASTO DE MARKETING': st.number_input("GASTO DE MARKETING", value=0.4),
                'GASTO DE ALMACENAMIENTO': st.number_input("GASTO DE ALMACENAMIENTO", value=last_row['GASTO DE ALMACENAMIENTO']),
                'DEMANDA DEL PRODUCTO': st.number_input("DEMANDA DEL PRODUCTO", value=last_row['DEMANDA DEL PRODUCTO']),
                'FESTIVIDAD': st.number_input("FESTIVIDAD", value=last_row['FESTIVIDAD']),
                'PRECIO DE VENTA': st.number_input("PRECIO DE VENTA", value=last_row['PRECIO DE VENTA']),
                'PRODUCTOS VENDIDOS': st.number_input("PRODUCTOS VENDIDOS", value=last_row['PRODUCTOS VENDIDOS'])
            }

            # Botón de submit para crear la fila
            if st.button("Crear Fila"):
                # Agregar la nueva fila al conjunto de datos
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_excel("data/Melsol-test.xlsx", index=False)
                st.success("Nueva fila creada con éxito.")

    elif operation == "Actualizar Fila":
        with col1:
            # Agregar opciones para seleccionar una fila para actualizar
            row_to_update = st.selectbox("Selecciona la fila para actualizar:", df.index + 1 ) -1

            # Crear una nueva fila con datos ingresados por el usuario
            updated_row = {
                'MES': st.number_input("MES", value=df.loc[row_to_update, 'MES']),
                'PRODUCTOS ALMACENADOS': st.number_input("PRODUCTOS ALMACENADOS", value=df.loc[row_to_update, 'PRODUCTOS ALMACENADOS']),
                'GASTO DE MARKETING': st.number_input("GASTO DE MARKETING", value=df.loc[row_to_update, 'GASTO DE MARKETING']),
                'GASTO DE ALMACENAMIENTO': st.number_input("GASTO DE ALMACENAMIENTO", value=df.loc[row_to_update, 'GASTO DE ALMACENAMIENTO']),
                'DEMANDA DEL PRODUCTO': st.number_input("DEMANDA DEL PRODUCTO", value=df.loc[row_to_update, 'DEMANDA DEL PRODUCTO']),
                'FESTIVIDAD': st.number_input("FESTIVIDAD", value=df.loc[row_to_update, 'FESTIVIDAD']),
                'PRECIO DE VENTA': st.number_input("PRECIO DE VENTA", value=df.loc[row_to_update, 'PRECIO DE VENTA']),
                'PRODUCTOS VENDIDOS': st.number_input("PRODUCTOS VENDIDOS", value=df.loc[row_to_update, 'PRODUCTOS VENDIDOS'])
            }

            # Botón de submit para actualizar la fila
            if st.button("Actualizar Fila"):
                # Actualizar la fila seleccionada en el conjunto de datos
                df.loc[row_to_update] = updated_row
                df.to_excel("data/Melsol-test.xlsx", index=False)
                st.success("Fila actualizada con éxito.")

    # Operación Eliminar Fila
    elif operation == "Eliminar Fila":
        with col1:
            # Agregar opciones para seleccionar una fila para eliminar
            row_to_delete = st.selectbox("Selecciona la fila para eliminar:", df.index + 1) - 1

            # Crear un formulario para mostrar los valores de la fila antes de eliminar
            form_delete = st.form("delete_row_form")

            # Mostrar los valores de la fila en el formulario
            for col in df.columns:
                form_delete.write(f"{col}: {df.loc[row_to_delete, col]}")

            # Botón de submit para confirmar la eliminación
            submit_delete_button = form_delete.form_submit_button("Eliminar Fila")

            # Si se presiona el botón, eliminar la fila seleccionada del conjunto de datos
            if submit_delete_button:
                df = df.drop(row_to_delete).reset_index(drop=True)
                df.to_excel("data/Melsol-test.xlsx", index=False)
                st.success("Fila eliminada con éxito.")

# ================================= Componente para mostrar el conjunto de datos =============================

# Columna derecha: Visualizar el conjunto de datos 
with col2:
    st.write("### Conjunto de Datos:")
    
    # Crear una copia del DataFrame con índices desplazados en 1
    df_display = df.copy()
    df_display.index = df_display.index + 1
    
    # Crear una tabla con Plotly solo con las primeras 5 filas
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Index'] + list(df_display.columns), fill_color='darkslategray', line_color='white', align='center'),
        cells=dict(values=[df_display.index] + [df_display[col] for col in df_display.columns], fill_color='dimgray', line_color='white', align='center')
    )])

    # Estilo adicional para la tabla
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
    )

    st.plotly_chart(fig)




