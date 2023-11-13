import streamlit as st
from Inicio import df
import plotly.express as px

st.set_page_config(
    page_title="eda",
    page_icon="👋",
)

st.write("# Análisis Exploratorio de Datos (EDA) 👋")

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

