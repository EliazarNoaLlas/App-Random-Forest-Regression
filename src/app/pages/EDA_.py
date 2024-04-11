import streamlit as st
import plotly.express as px
import pandas as pd

# ==========================================  Configurar la página
# =========================================================
st.set_page_config(
    page_title="Análisis Exploratorio de Datos (EDA)",
    page_icon=":bar_chart:",
    layout="wide"
)
#============================================================================================================================
# Cargar el componente de BannerPersonalizado.html
with open("./utils/Baner.html", "r", encoding="utf-8") as file:
    custom_banner_html = file.read()

# Agregar estilos CSS desde la carpeta utils
with open("./utils/Baner_style.css", "r", encoding="utf-8") as file:
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
st.markdown("## :bar_chart: Análisis Exploratorio de Datos (EDA) 👋")
## espacio en la parte superior de la pagin
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
#============================================================================================================================

# ================================================  Cargar Datos  ===========================================================

# Leer el conjunto de datos por defecto
df = pd.read_excel("data/Melsol-test.xlsx", engine="openpyxl")
# Mostrar la tabla de datos
st.write("## Conjunto de Datos Actual:")
st.write(df)


# Sidebar con componentes interactivos
st.sidebar.header("Configuración de Filtros:")

# Filtrar por rango de Productos Almacenados
almacenados_range = st.sidebar.slider("Filtrar por Rango de Productos Almacenados", min_value=0, max_value=30, value=(0, 30))
filtered_data = df[(df["PRODUCTOS ALMACENADOS"] >= almacenados_range[0]) & (df["PRODUCTOS ALMACENADOS"] <= almacenados_range[1])]

# Filtrar por Demanda del Producto
demanda_options = df["DEMANDA DEL PRODUCTO"].unique()
selected_demanda = st.sidebar.multiselect("Filtrar por Demanda del Producto", demanda_options, default=demanda_options)
filtered_data = filtered_data[filtered_data["DEMANDA DEL PRODUCTO"].isin(selected_demanda)]

# Mostrar estadísticas descriptivas
st.write("## Estadísticas Descriptivas:")
st.write(filtered_data.describe())

# Gráfico de dispersión
st.write("## Gráfico de Dispersión:")
scatter_fig = px.scatter(filtered_data, x="PRODUCTOS ALMACENADOS", y="PRODUCTOS VENDIDOS", color="DEMANDA DEL PRODUCTO")
st.plotly_chart(scatter_fig)

