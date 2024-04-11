import numpy as np
import pandas as pd

# Configuración de la semilla para reproducibilidad
np.random.seed(42)

# Número de filas
num_filas = 100

# Generación de datos aleatorios
data = {
    'MES': np.random.randint(1, 13, size=num_filas),
    'PRODUCTOS ALMACENADOS': np.random.randint(1, 100, size=num_filas),
    'GASTO DE MARKETING': np.round(np.random.uniform(0.1, 0.5, size=num_filas), 2),
    'GASTO DE ALMACENAMIENTO': np.round(np.random.uniform(0.5, 20, size=num_filas), 2),
    'DEMANDA DEL PRODUCTO': np.random.randint(1, 7, size=num_filas),
    'FESTIVIDAD': np.random.choice([0, 1], size=num_filas),
    'PRECIO DE VENTA': np.round(np.random.uniform(10, 12, size=num_filas), 2)
}

# Cálculo de 'PRODUCTOS VENDIDOS' usando la fórmula del modelo (simulada)
data['PRODUCTOS VENDIDOS'] = np.maximum(0, data['DEMANDA DEL PRODUCTO'] - np.round(data['GASTO DE MARKETING'] * data['PRODUCTOS ALMACENADOS'] * 0.1, 2))

# Crear DataFrame
df_simulado = pd.DataFrame(data)

# Mostrar el DataFrame simulado
print(df_simulado)

# Especifica la ruta del directorio donde deseas guardar el archivo Excel
directorio_destino = "./Data"

# Especifica el nombre del archivo Excel
nombre_archivo_excel = "Datos_Entrenamiento.xlsx"

# Ruta completa del archivo Excel
ruta_archivo_excel = f"{directorio_destino}/{nombre_archivo_excel}"

# Guarda el DataFrame en un archivo Excel
df_simulado.to_excel(ruta_archivo_excel, index=False)

print(f"Los datos simulados se han guardado en: {ruta_archivo_excel}")
