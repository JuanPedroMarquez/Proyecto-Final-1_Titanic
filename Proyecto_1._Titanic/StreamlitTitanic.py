import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(page_title="Panel de Control de Salarios",layout="wide") #configuración de la página

#@st.cache_resource #decorador para cachear los datos
def cargar_datos(): #función para cargar los datos
    return pd.read_csv('titanic.csv') 

datos = cargar_datos() #cargar los datos en una variable, empezamos a arreglarla

#Limpieza de datos (Hasta línea 92)
##############################################################################################################
# Cargamos los datos
df_Age_knn = datos
# Creamos la columna lastname para guardar el apellido de los pasajeros
df_Age_knn['LastName'] = df_Age_knn['Name'].apply(lambda x: x.split(',')[0])
# Vamos a normalizar Fare para asegurarnos que no tenga un impacto desproporcionado en la distancia 
scaler = StandardScaler()
df_Age_knn['Fare'] = scaler.fit_transform(df_Age_knn[['Fare']])
# estas son las columnas que se van a usar como predictores de Age
columnas=['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked',"Age"]
# Aplicamos OneHotEncoder para codificar nuestras variables categóricas 
encoder = OneHotEncoder(drop='first', sparse_output=False)
df_encoded = pd.DataFrame(encoder.fit_transform(df_Age_knn[columnas]))
df_encoded.columns = encoder.get_feature_names_out(columnas)
# Creamos dos dataframes basados en la presencia o ausencia de Age
df_encoded['Age'] = df_Age_knn['Age']
df_with_age = df_encoded.dropna(subset=['Age'])
df_without_age = df_encoded[df_encoded['Age'].isna()].drop(columns='Age')
# Dividimos el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(df_with_age.drop(columns='Age'), df_with_age['Age'], test_size=0.2, random_state=357)
# Lista para almacenar los valores de MSE (Error Cuadrático Medio)
mse = []

# Rango de k para probar
k_range = range(1, 7)

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = -cross_val_score(knn, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse.append(scores.mean())
    print(f'scores = {scores}')
    

# Graficamos los valores 
plt.plot(k_range, mse)
plt.xlabel('k')
plt.ylabel('MSE')
plt.title('Regla del codo para determinar el valor óptimo de k')
plt.show()

# Determinar el mejor valor de k
best_k = k_range[mse.index(min(mse))]
print(f'Mejor número K: {best_k}')

# Creamos el modelo KNN con el mejor valor de k
knn = KNeighborsRegressor(n_neighbors=best_k)

# Ajustamos el modelo a los datos sin valores nulos
knn.fit(X_train, y_train)

# Imputamos los valores faltantes en la columna 'Age'
imputed_ages = knn.predict(df_without_age)
df_Age_knn.loc[df_Age_knn['Age'].isna(), 'Age'] = imputed_ages

df2 = df_Age_knn.copy()  # Renombramos el dataframe para resumir el nombre.
df2 = pd.DataFrame(df2)  # Convertimos el dataframe en un objeto de tipo DataFrame.

datos = cargar_datos() # Recuperamos los datos originales para no perder la información de la tarifa.
df2["Fare"] = datos["Fare"] # Deshacemos la normalización retornando los valores originales de la tarifa.

df2["Euro_Fare"] = df2["Fare"] * 95.42231463 * 1.1711   # Convertimos la tarifa de libras de 1912 a libras de abril del 2024,
                                                        # y de libras de abril 2024 a euros de abril 2024

df2["Embarked"].fillna(df2["Embarked"].mode()[0], inplace=True) # Sustituimos valores nulos de Embarked por la moda

df2["Binary_Cabin"] = [0 if pd.isnull(i) else 1 for i in df2["Cabin"]] # Sustituimos valores nulos de Cabin por 0 y los conocidos por 1

#Tras corregir todas las columnas, eliminamos las columnas que no vamos a utilizar
df3 = df2.copy()
df3.drop(columns=["PassengerId", "SibSp", "Parch", "Ticket", "Cabin", "Name", "LastName"], inplace=True)
#Añadimos algunas variables más de interés que se crearon en el análisis exploratorio
#Grupos de edad
def age_group(age):
    if age < 10:
        return 'Kid'
    elif 10 <= age < 18:
        return 'Teen'
    elif 18 <= age <= 45:
        return 'Adult'
    else:
        return 'Third age'

df3['AgeGroup'] = df3['Age'].apply(age_group)

#Nombre honorífico
df3["Honorific_Name"] = df2["Name"].apply(lambda x: x.split(". ")[0].split(" ")[-1])
##############################################################################################################

opcion = st.sidebar.radio( #crear un radio button en el sidebar
    "Acceder a información específica:",
    ["Inicio", "Distribución por Sexo", "Distribución por Grupo de Edad", "Distribución por Clase", "Distribución por Nombre Honorífico", "Comparativa por Sexo y Edad", "Comparativa por Clase y Sexo", "Comparativa por Clase y Edad", "Ver Dataset"]
)

st.sidebar.header("Filtros") #crear un header en el sidebar

def agregar_todos(lista): #función para agregar "Todos" a una lista
    return ["Todos"] + list(lista)

filtros = { #diccionario con los filtros
    'Supervivencia': st.sidebar.multiselect("Seleccione si el pasajero sobrevivió o no:", options=agregar_todos(df3['Survived'].unique()), default=["Todos"]),
    'Sexo': st.sidebar.multiselect("Seleccione el Sexo de los pasajeros:", options=agregar_todos(df3['Sex'].unique()), default=["Todos"]),
    'Grupo de Edad': st.sidebar.multiselect("Seleccione el Tramo de Edad deseado:", options=agregar_todos(df3['AgeGroup'].unique()), default=["Todos"]),
    'Puerto de Embarcación': st.sidebar.multiselect("Seleccione el Puerto de Embarcación de los pasajeros:", options=agregar_todos(df3['Embarked'].unique()), default=["Todos"]),
    'Clase': st.sidebar.multiselect("Seleccione la Clase del pasajero:", options=agregar_todos(df3['Pclass'].unique()), default=["Todos"]),
    'Nombre Honorífico': st.sidebar.multiselect("Selecciona el Nombre Honorífico del pasajero:", options=agregar_todos(df3['Honorific_Name'].unique()), default=["Todos"])
}

def aplicar_filtros(df, filtros): #función para aplicar los filtros
    for col, val in filtros.items():
        if "Todos" not in val:
            df = df[df[col].isin(val)]
    return df

datos_filtrados = aplicar_filtros(df3, filtros) #aplicar los filtros a los datos

st.title("Expediente Titanic: Análisis de Supervivencia") #título de la página
st.write("Por: Juan Pedro Márquez Gandía, Data Analyst Jr. del equipo de investigación UpgradeHub") #autor

def mostrar_inicio(): #función para mostrar la sección de inicio
    st.header("En este Panel de Control, encontrarás información acerca de las víctimas del Titanic.")

    st.write("En este menú de inicio, tienes una descripción de las variables del dataset, así como un par de tablas de correlaciones para empezar a echarle una ojeada a los datos.")
    st.write("En el menú lateral podrás navegar a través de las secciones de interés, pudiendo filtrar para obtener información específica.")

    st.write('Nuestro dataset contiene las siguientes variables:')
    st.image('Variables.PNG', use_column_width=False)

    st.write("Sin embargo, habiendo revisado el df, eliminamos algunas variables que más que agrupar los datos, los segregaban, dificultando la extracción de información útil.")
    st.write("Por lo tanto, la lista completa de variables con las que trabajaremos es la siguiente:")
    st.image("VariablesDF3.PNG", use_column_width=False)

    st.write("Las variables que hemos añadido son las 4 últimas, de la 7 a la 10, por los siguientes motivos:")
    st.write("'Binary_Cabin', que nos permite saber si se conoce o no la cabina del pasajero. Esta variable se creó porque el 77% de los datos de Cabin son nulos.")
    st.write("'Euro_Fare', que nos permite saber cuánto pagaron los pasajeros en euros de 2024. Esta variable se creó porque la tarifa original estaba en libras de 1912.")
    st.write("'AgeGroup', que nos ayuda a agrupar a los pasajeros por tramos de edad.")
    st.write("Y, finalmente, 'Honorific_Name', donde extrajimos el nombre honorífico o título del nombre de la persona. Esta variable se creó porque podría ser una variable útil, pues otorga cierta información de manera indirecta, como el sexo de la persona (si es hombre o mujer), su edad (si es más joven o más adulto), su oficio (Dr. para los educados, Rev. para estudios religiosos, Major., Col. y Capt. para rangos militares...), su estado civil (casado o no) y si proviene de una familia noble (condes, nobles...).")

    st.write('A continuación, vemos las siguientes correlaciones:')
    st.image('Correlaciones1.PNG', use_column_width=False)
    st.write("Lo interesante de esta tabla es que nos muestra que la supervivencia está correlacionada negativamente con la clase del pasajero (es decir, cuanto mayor sea el valor numérico de la clase, menor supervivencia asociada), con si el pasajero era hombre (menor ratio de supervivencia asociada) y, en menor medida, si el puerto del pasajero era Southampton. Por otro lado, está positivamente correlacionada con la tarifa que se pagó, con el hecho de que el pasajero fuese mujer y se conociese su cabina y, aunque en menor medida, si el pasajero era un niño o si su puerto de embarque era Cherburgo.")
    st.write("Esto nos arroja algunas sugerencias iniciales de que el poder adquisitivo (tarifa y clase), la edad (niños) y el sexo (mujeres, en concreto) fueron factores importantes a la hora de 'determinar' o influir en la supervivencia de dichos pasajeros.")
    
    st.write("En la siguiente tabla de correlaciones, nos centramos en el nombre honorífico del pasajero, la cual ha sido tratada de forma separada pues debido a la gran cantidad de nombres diferentes, las dimensiones de la tabla pueden dificultar su legibilidad.")
    st.image('Correlaciones2.PNG', use_column_width=False)
    st.write("En este caso, vemos que el nombre honorífico del pasajero está correlacionado positivamente con la supervivencia si está asociado con una mujer (Miss. y Mrs.), por lo que parece que el sexo femenino, y en cierta medida parece que los niños también, tuvieron algún tipo de prioridad en las evacuaciones del Titanic.")
    
    st.image("superv_sexo.png", use_column_width=False)
    st.write("En este gráfico puede compararse el número total de supervivientes por sexo, habiendo un total de 577 hombres y 314 mujeres a bordo. Se puede ver que las mujeres tuvieron un mayor ratio de supervivencia (Nº supervivientes > Nº fallecidos).")
    
    st.image("superv_edad.png", use_column_width=False)
    st.write("En este gráfico puede compararse el número total de supervivientes por grupo de edad, habiendo un total de 69 niños, 52 adolescentes, 662 adultos y 108 personas de 3ª edad. Se puede ver que los niños tuvieron una ratio de supervivencia positiva y que el mayor número de bajas fueron adultos.")
    
    st.image("superv_clase.png", use_column_width=False)
    st.write("En este gráfico puede compararse el número total de supervivientes por clase, habiendo un total de 216 pasajeros de 1ª clase, 184 de 2ª clase y 491 de 3ª clase. Se puede ver que la 3ª clase tuvo un mayor número de bajas y una baja ratio de supervivencia. La ratio de supervivencia de los pasajeros de 1ª clase fue positiva y tuvieron la menor cantidad de bajas. La ratio de la de 2ª clase estuvo cerca del 50%.")
    
    st.image("superv_nombre.png", use_column_width=True)
    st.write("En este gráfico puede compararse el número total de supervivientes por nombre honorífico. Se puede ver que los nombres honoríficos asociados a mujeres (Miss., Mrs., Mlle., Mme., Ms. y Lady.), niños (Master.) y la nobleza (Countess. y Lady.) tuvieron un mayor número de supervivientes y una ratio de supervivencia positiva, mientras que los asociados a hombres tuvieron mayor número de bajas humanas (4 de cada 5 fallecimientos) y una ratio de supervivencia negativa. Solamente los Mr. fallecidos suponen el 79,4% de los fallecidos totales, y si le añadimos las Miss. y Mrs. fallecidas ese porcentaje sube hasta el 94%.")

    df_grouped = df3.groupby(['AgeGroup', 'Sex'])['Survived'].mean().reset_index()
    age_mapping = {"Kid": 0, "Teen": 1, "Adult": 2, "Third_age": 3}
    df_grouped['AgeGroupOrder'] = df_grouped['AgeGroup'].map(age_mapping)
    df_grouped = df_grouped.sort_values('AgeGroupOrder')
    fig = px.line(df_grouped, x="AgeGroup", y="Survived", color="Sex", title="Relación entre la edad y sexo", template="plotly_dark", width=650, height=480)
    df_grouped = df_grouped.drop(columns='AgeGroupOrder')
    st.plotly_chart(fig, use_container_width=False)
    st.write("Las mujeres tuvieron una ratio de supervivencia superior con respecto a los hombres para todas las categorías de edad excepto en los niños menores de 10 años, en la que ambos tuvieron alrededor de un 60%.")
        
    df_grouped = df3.groupby(['Pclass', 'Sex'])['Survived'].mean().reset_index()
    fig2 = px.line(df_grouped, x="Pclass", y="Survived", color="Sex", title="Ratio de supervivientes por clase y sexo", template="plotly_dark", width=650, height=480)
    st.plotly_chart(fig2, use_container_width=False)
    st.write("A mayor clase del pasajero, mayor tasa de supervivencia asociada, en ambos sexos, y siendo las tasas femeninas superiores.")

    df_grouped = df3.groupby(['AgeGroup', 'Pclass'])['Survived'].mean().reset_index()
    age_mapping = {"Kid": 0, "Teen": 1, "Adult": 2, "Third_age": 3}
    df_grouped['AgeGroupOrder'] = df_grouped['AgeGroup'].map(age_mapping)
    df_grouped = df_grouped.sort_values('AgeGroupOrder')
    fig3 = px.line(df_grouped, x="AgeGroup", y="Survived", color="Pclass", title="Relación entre la edad y clase", template="plotly_dark", width=650, height=480)
    df_grouped = df_grouped.drop(columns='AgeGroupOrder')
    st.plotly_chart(fig3, use_container_width=False)
    st.write("Los pasajeros en 3ª clase tuvieron menor ratio de supervivencia. En los niños, los de segunda clase tuvieron mayor ratio que los de primera, pero menor para el resto de grupos de edad. A mayor edad, menores ratios de supervivencia. Peor grupo: pasajeros más mayores de 3ª clase.")
    
    st.header("Conclusiones")
    st.write("Es factible la hipótesis de que estas tasas de supervivencia tan elevadas en los grupos femeninos e infantiles, de clases con mayor poder adquisitivo e incluso miembros de la nobleza se debe a que gozaron de algún tipo de prioridad a la hora de ser evacuados del barco.")
    st.write("Le recordamos que puede usar el menú desplegable de la izquierda para acceder a la información que desee, incluyendo el dataset completo. Gracias por su atención.")

    st.header("¡Gracias por ver!")

def mostrar_distribucion_sexos(): #función para mostrar la distribución por sexo
    st.header("Distribución por Sexo del Pasajero")
    fig = px.histogram(datos_filtrados, x='Sex', title="Pasajeros según su sexo")
    st.plotly_chart(fig, use_container_width=True)

def mostrar_grupos_edad(): #función para mostrar la distribución por grupo de edad
    st.header("Distribución por Edad del Pasajero")
    fig = px.histogram(datos_filtrados, x='AgeGroup', title="Pasajeros según grupo de edad al que pertenecen")
    st.plotly_chart(fig, use_container_width=True)

def mostrar_clase(): #función para mostrar la distribución por clase de pasajero
    st.header("Distribución por Clase del Pasajero")
    fig = px.histogram(datos_filtrados, x='AgeGroup', title="Pasajeros según su Clase")
    st.plotly_chart(fig, use_container_width=True)

def mostrar_nombre_honor(): #función para mostrar la distribución por nombre honorífico
    st.header("Distribución por Nombre Honorífico del Pasajero")
    fig = px.histogram(datos_filtrados, x='Honorific_Name', title="Pasajeros según su Nombre de Honor")
    st.plotly_chart(fig, use_container_width=True)

def mostrar_comparativa_sexo_edad(): #función para mostrar la comparativa de supervivencia por sexo y edad
    st.header("Comparativa de Supervivencia por sexo y edad")
    df_grouped = datos_filtrados.groupby(['AgeGroup', 'Sex'])['Survived'].mean().reset_index()
    age_mapping = {"Kid": 0, "Teen": 1, "Adult": 2, "Third_age": 3}
    df_grouped['AgeGroupOrder'] = df_grouped['AgeGroup'].map(age_mapping)
    df_grouped = df_grouped.sort_values('AgeGroupOrder')
    fig = px.line(df_grouped, x="AgeGroup", y="Survived", color="Sex", title="Relación entre la edad y sexo", template="plotly_dark", width=800, height=400)
    df_grouped = df_grouped.drop(columns='AgeGroupOrder')
    st.plotly_chart(fig, use_container_width=True)

def mostrar_comparativa_clase_sexo(): #función para mostrar la comparativa de supervivencia por clase y sexo
    st.header("Comparativa de Supervivencia por clase y sexo")
    df_grouped = datos_filtrados.groupby(['Pclass', 'Sex'])['Survived'].mean().reset_index()
    fig = px.line(df_grouped, x="Pclass", y="Survived", color="Sex", title="Ratio de supervivientes por clase y sexo", template="plotly_dark", width=650, height=480)
    st.plotly_chart(fig, use_container_width=True)

def mostrar_comparativa_clase_edad(): #función para mostrar la comparativa de supervivencia por clase y sexo
    st.header("Comparativa de Supervivencia por clase y edad")
    df_grouped = datos_filtrados.groupby(['AgeGroup', 'Pclass'])['Survived'].mean().reset_index()
    age_mapping = {"Kid": 0, "Teen": 1, "Adult": 2, "Third_age": 3}
    df_grouped['AgeGroupOrder'] = df_grouped['AgeGroup'].map(age_mapping)
    df_grouped = df_grouped.sort_values('AgeGroupOrder')
    fig = px.line(df_grouped, x="AgeGroup", y="Survived", color="Pclass", title="Relación entre la edad y clase", template="plotly_dark", width=650, height=480)
    df_grouped = df_grouped.drop(columns='AgeGroupOrder')
    st.plotly_chart(fig, use_container_width=True)

def mostrar_dataset(): #función para mostrar el dataset
    st.header("Dataset Completo")
    st.write("Aquí puedes acceder al dataset con los datos filtrados, disponible para descargar en formato CSV.")
    st.dataframe(datos_filtrados)

opciones = { #diccionario con las opciones
    "Inicio": mostrar_inicio,
    "Distribución por Sexo": mostrar_distribucion_sexos,
    "Distribución por Grupo de Edad": mostrar_grupos_edad,
    "Distribución por Clase": mostrar_clase,
    "Distribución por Nombre Honorífico": mostrar_nombre_honor,
    "Comparativa por Sexo y Edad": mostrar_comparativa_sexo_edad,
    "Comparativa por Clase y Sexo": mostrar_comparativa_clase_sexo,
    "Comparativa por Clase y Edad": mostrar_comparativa_clase_edad,
    "Ver Dataset Completo": mostrar_dataset,
}

opciones[opcion]() #mostrar la opción seleccionada