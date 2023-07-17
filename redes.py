# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:49:52 2023

@author: marti
"""

# Librerias
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from scipy.stats import ks_2samp
from scipy.stats import gamma
from numpy.random import seed
from numpy.random import poisson
from scipy.stats import kstest
import sklearn.metrics as metrics
import seaborn as sns
from keras import regularizers

# Importamos datos
ruta_archivo_excel="/content/Base_Clientes Alemanes.xlsx"
dfRaw = pd.read_excel(ruta_archivo_excel)
print(dfRaw)
# Los analizamos
#Vemos su info
dfRaw.info()
dfRaw.describe()
#Contamos cuantos nulls hay
nulos_por_columna = dfRaw.isnull().sum()
print(nulos_por_columna)
duplicados_por_columna = dfRaw.duplicated()
print(duplicados_por_columna)

# Cambiamos los nombres de las columnas para entender mejor
nuevos_nombres = {
    1: 'Status of Existing Checking Account',
    2: 'Duration in Month',
    3: 'Credit History',
    4: 'Purpose',
    5: 'Credit Amount',
    6: 'Savings Account/Bonds',
    7: 'Present Employment Since',
    8: 'Installement Rate in Percentage of Disposable Income',
    9: 'Personal Status & Sex',
    10: 'Other Debtors/Guarantors',
    11: 'Present Residence Since',
    12: 'Property',
    13: 'Age in Years',
    14: 'Other Installment Plans',
    15: 'Housing',
    16: 'Number of Existing Credits at this Bank',
    17: 'Job',
    18: 'Number of People Being Liable to Provide Maintenance for',
    19: 'Telephone',
    20: 'Domestic Worker',
}
# Cambiar los nombres de las columnas
data_nombres = dfRaw.rename(columns=nuevos_nombres)
# guardamos la data y mostramos el conjunto de datos nuevo
print(data_nombres)

# Reemplazamos los valores de cada columna

data = data_nombres.replace({
      "A11" : "... < 0 DM",
      "A12":" 0 <= ... <  200 DM",
      "A13" : " ... >= 200 DM",
      "A14" : "no checking account (Dummy)",
      "A30":" no credits taken",
      "A31": "all credits at this bank paid back duly",
      "A32":" existing credits paid back duly till now",
      "A33" : "delay in paying off in the past",
      "A34" : "Other credits",
      "A40" : "car (new)",
      "A41" : "car (used)",
      "A42" : "furniture/equipment",
      "A43" : "radio/television",
      "A44" : "domestic appliances",
      "A45" : "repairs",
      "A46" : "education",
      "A47" : "(vacation - does not exist?)",
      "A48" : "retraining",
      "A49" : "business",
      "A50" : "others",
      "A61" : "... <  100 DM",
      "A62" :"100 <= ... <  500 DM",
      "A63":"500 <= ... < 1000 DM",
      "A64":" .. >= 1000 DM",
      "A65":" unknown/ no savings account",
      "A71":" unemployed",
      "A72":" ... < 1 year",
      "A73":"1 <= ... < 4 years",
      "A74":" 4 <= ... < 7 years",
      "A75":" .. >= 7 years",
      "A91": "male - divorced/separated",
      "A92": "female - divorced/separated/married",
      "A93": "male - single",
      "A94": "male - married/widowed",
      "A95": "female - single",
      "A101": "none",
      "A102": "co-applicant",
      "A103": "guarantor",
      "A121": "real estate",
      "A122":"if not A121: building society savings agreement/life insurance",
      "A123":"if not A121/A122 : car or other, not in attribute 6",
      "A124": "unknown / no property",
      "A141": "bank",
      "A142":"stores",
      "A143":"none",
      "A151": "rent",
      "A152": "own",
      "A153": "for free",
      "A171": "unemployed/unskilled - non-resident",
      "A172":"unskilled - resident",
      "A173": "skilled employee / official",
      "A174": "management/ self-employed/highly qualified employee/ officer",
      "A191": "no",
      "A192": "yes",
      "A201": "yes",
      "A202": "no",

      })
data['Purpose']

# Reemplazamos aceptación 1 good y 2 bad
data['Aceptación'] = data_nombres['Aceptación'].replace({
      1 : "Good",
      2: "Bad"
      })
data['Aceptación']

y=np.where(dfRaw['Aceptación']==2,1,0)
print(y)

# Anonizamos los datos
del data['Cliente']
data

#1) realiza análisis exploratorios en columnas  cuantiativas y cualitativas
#CUANTITATIVAS
data[
    [ 'Duration in Month',
      'Credit Amount',
      'Installement Rate in Percentage of Disposable Income',
      'Age in Years',
      'Number of Existing Credits at this Bank',
      'Number of People Being Liable to Provide Maintenance for',
      'Aceptación']
]#.dtypes
data
#Para las columnas cuantitativas, selecciona columnas específicas y realiza
#la categorización o transformación de datos, como la categorización de
#'Duración en mes' en estratos, la categorización de 'Cantidad de crédito',
#la segmentación de 'Edad en años' en grupos y el análisis de las columnas restantes.
#Luego, el código imprime los tipos de datos de estas columnas.
#CUALITATIVAS
data[
    ['Status of Existing Checking Account',
     'Credit History',
     'Purpose',
     'Savings Account/Bonds',
     'Personal Status & Sex',
     'Other Debtors/Guarantors',
     'Property',
     'Present Residence Since',
     'Other Installment Plans',
     'Housing',
     'Job',
     'Telephone',
     'Domestic Worker',
     'Aceptación']] # .dtypes
data
#Para columnas cualitativas, selecciona columnas específicas e imprime sus tipos de datos.
#%%2) graficos de exploratory data analysis
data_description = data.describe()
print(data_description)

#PASO 1: Variable aceptación recuento
sns.set_palette("pastel") #ajustes de color
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Aceptación')
plt.title("Recuento de aceptación", fontsize=20)
plt.show()
#un gráfico de recuento (countplot) de la variable "Aceptación" en un conjunto de datos llamado "data".
#analisis:
    #Hay 700 casos en los que un solicitante fue clasificado como 0 -> good
    #Hay 300 casos en los que un solicitante fue clasificado como -> bad

#PASO 2: GRAFICOS GENERALES: Histogramas
data.hist(figsize=(11, 11), color='pink', layout=(4,2))
plt.figure()
# utiliza para visualizar la frecuencia con la que ocurren diferentes valores dentro de un conjunto de datos
#y proporciona información sobre la forma y la dispersión de la distribución.

#PASO 3:GRAFICOS MÁS ESPECIFICOS: comprando variables
#Paso 3.a: Distribución de edad, distribución del monto de crédito y distribución de duración
plt.figure(figsize=(14, 12))
plt.subplot(221)
ax1 = sns.histplot(data=data, x='Age in Years', hue='Aceptación', multiple='stack', palette='pastel', kde=True)
ax1.set_title("Distribución de edad", fontsize=20)

plt.subplot(222)
ax2 = sns.histplot(data=data, x='Credit Amount', hue='Aceptación', multiple='stack', palette='pastel', kde=True)
ax2.set_title("Distribución del monto del crédito", fontsize=20)

plt.subplot(212)
ax3 = sns.histplot(data=data, x='Duration in Month', hue='Aceptación', multiple='stack', palette='pastel', kde=True, bins=10)
ax3.set_title("Distribución de duración", fontsize=20)

plt.show()
##analisis:
    #Los solicitantes de entre 20 y 30 años tienen más probabilidades de solicitar un préstamo
    #Es menos probable que los solicitantes soliciten un préstamo de alto crédito
    #Se han pagado más préstamos alrededor de 20 meses después de su emisión.
    #Es más probable que el banco reciba solicitantes entre 20 y 30 años y solicite préstamos entre 250 y 2500 DM

#Paso 3.b: Distribución Housing
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Housing', hue='Aceptación', palette=('pastel'))
plt.title("Distribución Housing", fontsize=20)
plt.show()
##analisis:
    #La mayoría de los solicitantes son dueños de una casa
    #Más de la mitad de los solicitantes propietarios de una vivienda clasificada como buena

#Paso 3.c: Dostribución por genero
countplotGenero = sns.countplot(data=data, x='Personal Status & Sex', hue='Aceptación', palette='pastel')
countplotGenero.set_title("Count de genero y status personal", fontsize=20)
countplotGenero.set_xlabel("")
boxplotGenero = sns.boxplot(data=data, x='Personal Status & Sex', y='Credit Amount', palette='pastel', hue='Aceptación')
boxplotGenero.set_title("Credit Amount por genero y status personal", fontsize=20)
boxplotGenero.set_xlabel("")
plt.show()

conteo_valores = data['Personal Status & Sex'].value_counts()
conteo_valores1 = data.groupby(['Personal Status & Sex', 'Aceptación']).size().unstack(fill_value=0)
print(conteo_valores)
print(conteo_valores1)
#Procentaje de rechazo hombre, casado o viudo, cateoria 4
porcentaje_de_rechazo_c4= 25/(67+25)
print(porcentaje_de_rechazo_c4) #27%
#Procentaje de rechazo homrbes solteros, categoria 3
porcentaje_de_rechazo_c3= 146/(402+146)
print(porcentaje_de_rechazo_c3) #26%
#Procentaje de rechazo mujeres divorsiadas/separadas/casadas, categoría 2
porcentaje_de_rechazo_c2= 109/(201+109)
print(porcentaje_de_rechazo_c2) #35%
#Procentaje de rexhazo hombres divrociados/separados, categoria 1
porcentaje_de_rechazo_c1= 20/(30+20)
print(porcentaje_de_rechazo_c1) #40%
#Count Hombres y mujeres
count_hombres= 548+92+50 #incluye categoria 1,3,4
count_mujeres = 310 #incluye categoria 2
print(count_hombres)
print(count_mujeres)
#Porcentaje de rechazo hombres y mujeres
procentaje_rechazo_hombres= (25+146+20)/count_hombres
procentaje_rechazo_mujeres= 109/count_mujeres
print(procentaje_rechazo_hombres)
print(procentaje_rechazo_mujeres)
##analisis
    #Vemos que hombres solteros son los que mas piden y los que menos indice de rechazo tienen
    #Hay 2 veces más solicitantes masculinos que femeninos en los datos
    #El procentaje de rechazo de un hombre es de 27% y de una mujer de 35%

#Paso 3.d: Distribución por job category
countplotJob = sns.countplot(data=data, x='Job', hue='Aceptación', palette='pastel')
countplotJob.set_title("Count Job", fontsize=20)
countplotJob.set_xlabel("")
plt.subplot()
##analisis:
    #vemos que los skilled son los que más piden

#Paso 3.e:Distribución por duración
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("Distribución por Duration", fontsize=20)
sns.countplot(data=data, x='Duration in Month', hue='Aceptación', palette='pastel', ax=ax1)
ax1.set_xlabel("")
sns.lineplot(data=data, x='Duration in Month', y='Credit Amount', hue='Aceptación', palette='pastel', ax=ax2)
ax2.set_xlabel("")
sns.pointplot(data=data, x='Duration in Month', y='Credit Amount', hue='Aceptación', palette='pastel', ax=ax3)
fig.tight_layout()
plt.show()
##analisis
    #La mayoría de los préstamos emitidos tenían una duración de 12 y 24 meses
    #La mayoría de los solicitantes que pagaron sus préstamos dentro de los 24 meses se clasifican como buenos
    #La mayoría de los solicitantes con una duración de préstamo superior a 24 meses se clasifican como malos

#Paso 3.f: Distribución por Proposito
plt.figure(figsize = (14,12))
plt.subplot(221)
ax1 = sns.countplot(data=data, x="Purpose", palette="pastel", hue = "Aceptación")
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45)
ax1.set_xlabel("")
ax1.set_title("Purposes Count", fontsize=20)
plt.subplot(222)
ax2 = sns.violinplot(data=data, x="Purpose", y="Age in Years", palette="pastel", hue = "Aceptación",split=True)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45)
ax2.set_xlabel("")
ax2.set_title("Purposes by Age", fontsize=20)
plt.subplot(212)
ax3 = sns.boxplot(data=data, x="Purpose", y="Credit Amount", palette="pastel", hue = "Aceptación")
ax3.set_xlabel("Purposes")
ax3.set_title("Credit Amount distribuition by Purposes", fontsize=20)
plt.subplots_adjust(hspace = 0.6, top = 0.8)
plt.show()
#analisis:
    #Una gran parte de los solicitantes solicitó préstamos para comprar automóviles, radios/tv.
    #Más de la mitad de los solicitantes solicitaron préstamos de menos de 5000 DM
    #Los solicitantes con préstamos de alto crédito tienen más probabilidades de clasificarse como malos

###CONLUSIONES GENERLES
    #Los préstamos con plazos de duración inferiores a 24 meses tienen más probabilidades de ser reembolsados
    #Es más seguro emitir préstamos con un monto de crédito inferior a 5,000 DM y un plazo de duración de menos de 24 meses
    #Los solicitantes que poseen una propiedad muestran que son financieramente independientes y son mejores candidatos para un préstamo
    #Los solicitantes con trabajos calificados y altamente calificados son candidatos más seguros para emitir préstamos#Los préstamos para automóviles son los préstamos más comunes con una alta relación de ganancia a pérdida emitidos por el banco (préstamo más rentable)
    #Es más rentable emitir préstamos de menos de 2500 marcos alemanes que préstamos de crédito más altos que tienen menos probabilidades de ser reembolsados.

#Calculamos IV (Information Value)
# Función cuantitativa
def get_woe_iv_bins(feature, target, input_df, bins_num=5):

    df = input_df[[feature,target]]
    bins = pd.qcut(df[feature], bins_num, duplicates='drop')
    df['bins'] = bins
    df = pd.pivot_table(df, columns=[target], values=[feature], index=['bins'], aggfunc='count')

    df.columns = df.columns.to_series().str.join('_')
    df = df.rename(columns={f"{feature}_Bad":"Bad",f"{feature}_Good":"Good"})

    df = df.reset_index()

    df['Total'] = df['Bad']+df['Good']
    df['Probabilidad Default'] = df['Bad'] /(df['Bad']+df['Good'])
    df['% / total_Bad'] = df['Bad']/df['Bad'].sum()
    df['% / total_Good'] = df['Good']/df['Good'].sum()
    df['WOE'] = np.log(df['% / total_Bad'] / df['% / total_Good'])
    df['IV_i'] = (df['WOE'] * (df['% / total_Bad'] - df['% / total_Good'] ))
    df['IV'] = np.sum(df['IV_i'])
    print(df)

    print(f"IV: {df['IV'][0]}")

get_woe_iv_bins(feature='Duration in Month', target='Aceptación', input_df=data, bins_num=5)
#IV (Infromation Value)= 0.21618295432812568
get_woe_iv_bins(feature='Credit Amount', target='Aceptación', input_df=data, bins_num=5)
#IV (Infromation Value): 0.093362 12556052204
get_woe_iv_bins(feature='Installement Rate in Percentage of Disposable Income', target='Aceptación', input_df=data, bins_num=5)
#IV (Infromation Value): 0.02556886483364345
get_woe_iv_bins(feature="Age in Years", target='Aceptación', input_df=data, bins_num=5)
#IV (Infromation Value): 0.06836957282974725
get_woe_iv_bins(feature="Number of Existing Credits at this Bank", target='Aceptación', input_df=data, bins_num=5)
#IV (Infromation Value): 0.0035847291092131702
get_woe_iv_bins(feature="Number of People Being Liable to Provide Maintenance for", target='Aceptación', input_df=data, bins_num=5)
#IV (Infromation Value): 0.0

#Funcion cualitativa
def get_woe_iv_categorical(feature, target, input_df):
    plt.hist(input_df[feature])
    df = input_df[[feature,target]]
    bins = input_df[feature]
    df['bins'] = bins
    df = pd.pivot_table(df, columns=[target], values=[feature], index=['bins'], aggfunc='count')
    df.columns = df.columns.to_series().str.join('_')
    df = df.rename(columns={f"{feature}_Bad":"Bad",f"{feature}_Good":"Good"})
    df = df.reset_index()
    df['Total'] = df['Bad']+df['Good']
    df['Probabilidad Default'] = df['Bad'] /(df['Bad']+df['Good'])
    df['% / total_Bad'] = df['Bad']/df['Bad'].sum()
    df['% / total_Good'] = df['Good']/df['Good'].sum()
    df['WOE'] = np.log(df['% / total_Bad'] / df['% / total_Good'])
    df['IV_i'] = (df['WOE'] * (df['% / total_Bad'] - df['% / total_Good'] ))
    df['IV'] = np.sum(df['IV_i'])
    print(f"IV: {df['IV'][0]}")

get_woe_iv_categorical('Status of Existing Checking Account','Aceptación',data)
#IV (Infromation Value): 0.6660115033513336
get_woe_iv_categorical('Credit History','Aceptación',data)
#IV (Infromation Value): 0.29323354739082624
get_woe_iv_categorical('Purpose','Aceptación',data)
#IV (Infromation Value): 0.16919506567307835
get_woe_iv_categorical('Savings Account/Bonds','Aceptación',data)
#IV (Infromation Value): 0.19600955690422667
get_woe_iv_categorical('Present Employment Since','Aceptación',data)
#IV (Infromation Value): 0.086433631026641
get_woe_iv_categorical('Personal Status & Sex','Aceptación',data)
#IV (Infromation Value): 0.04467067763379072
get_woe_iv_categorical('Other Debtors/Guarantors','Aceptación',data)
#IV (Infromation Value): 0.032019322019485055
get_woe_iv_categorical('Present Residence Since','Aceptación',data)
#IV (Infromation Value): 0.0035887731887050195
get_woe_iv_categorical('Property','Aceptación',data)
#IV (Infromation Value): 0.11263826240979674
get_woe_iv_categorical('Other Installment Plans','Aceptación',data)
#IV (Infromation Value): 0.05761454195564788
get_woe_iv_categorical('Housing','Aceptación',data)
#IV (Infromation Value): 0.08329343361549926
get_woe_iv_categorical('Job','Aceptación',data)
#IV (Infromation Value): 0.00876276570742829
get_woe_iv_categorical('Telephone','Aceptación',data)
#IV (Infromation Value): 0.0063776050286746735
get_woe_iv_categorical('Domestic Worker','Aceptación',data)
#IV (Infromation Value): 0.04387741201028899

#Dataset Significativo + Separación train y test
# Armamos el dataset signifcativo
# Nos quedamos con el top 5 de variables con mejor IV
df_significativo = data.copy()

#Variablles Cualitativas
del df_significativo['Present Employment Since']
del df_significativo['Personal Status & Sex']
del df_significativo['Other Debtors/Guarantors']
del df_significativo["Present Residence Since"]
del df_significativo["Property"]
del df_significativo["Age in Years"]
del df_significativo["Other Installment Plans"]
del df_significativo["Housing"]
del df_significativo['Number of Existing Credits at this Bank']
del df_significativo['Job']
del df_significativo['Telephone']
del df_significativo['Domestic Worker']
del df_significativo['Credit Amount']
del df_significativo['Installement Rate in Percentage of Disposable Income']
del df_significativo['Number of People Being Liable to Provide Maintenance for']


df_significativo

df_significativo['Duration in Month'] = pd.qcut(df_significativo['Duration in Month'],5, duplicates='drop')
df_significativo['Duration in Month']


df_significativo_sin_aceptacion = df_significativo.loc[:, df_significativo.columns != 'Aceptación'] #Eliminamos la columna "aceptacion"

df_significativo_sin_aceptacion

df_significativo_sin_aceptacion_oh_explicativo = pd.get_dummies(df_significativo_sin_aceptacion)
df_significativo_sin_aceptacion_oh_explicativo # Quedan 29 columnas

# Separacion train y test
Y = np.where(df_significativo['Aceptación']=='Good',0,1) #Metemos en la Y los valores de la columna aceptacion

# train (80%) y test (20%)

df_significativo_train, df_significativo_test = train_test_split(df_significativo,test_size= 0.2, random_state= 1, stratify=df_significativo['Aceptación']) #random state es la semilla, poniendole 1, la dejamos activada.


df_significativo_sin_aceptacion_train = df_significativo_train.loc[:, df_significativo_train.columns != 'Aceptación']
df_significativo_sin_aceptacion_test = df_significativo_test.loc[:, df_significativo_test.columns != 'Aceptación']

# Tranformamos en dummies
df_significativo_sin_aceptacion_oh_train = pd.get_dummies(df_significativo_sin_aceptacion_train)
df_significativo_sin_aceptacion_oh_test= pd.get_dummies(df_significativo_sin_aceptacion_test)

X_train_modelo = df_significativo_sin_aceptacion_oh_train
Y_train_modelo = np.where(df_significativo_train['Aceptación']=='Good',0,1)
X_test_modelo = df_significativo_sin_aceptacion_oh_test
Y_test_modelo = np.where(df_significativo_test['Aceptación']=='Good',0,1)

#Modelo
modelo = Sequential()

modelo.add(Dense(10, activation='softmax', input_shape=(29,)))

modelo.add(Dense(1, activation='sigmoid'))

modelo.compile(optimizer='sgd',
               loss='binary_crossentropy',
               metrics=['accuracy'])

history = modelo.fit(x=X_train_modelo, y=Y_train_modelo,
                     validation_data=(X_test_modelo, Y_test_modelo),
                     batch_size=100,
                     epochs=3000)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0,1)
plt.title("Pérdida del modelo")
plt.ylabel("Pérdida")
plt.xlabel("Épocas del modelo")
plt.legend(["train", "val"], loc="upper right")
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylim(0,1)
plt.title("Accuracy del modelo")
plt.ylabel("accuracy")
plt.xlabel("Épocas del modelo")
plt.legend(["train", "val"], loc="lower right")
plt.show()

Y_predicho = np.where(modelo.predict(X_test_modelo)>0.2,1,0).T[0]
datos_matriz = {'y': Y_test_modelo,
                'y_predicho': Y_predicho,
               }


df_matriz = pd.DataFrame(datos_matriz)
print(df_matriz)

matriz_de_confusion_modelo = pd.crosstab(df_matriz['y'], df_matriz['y_predicho'], rownames=['Actual'], colnames=['Predicho'])
matriz_de_confusion_modelo

matriz_de_confusion_modelo[0]
matriz_de_confusion_modelo[1]

# Matriz de confusion con test
Y_test_modelo_predicted = np.where(modelo.predict(X_test_modelo) > 0.2, 1, 0).T[0]
Y_test_modelo_predicted

data = {'y_actual': Y_test_modelo,
        'y_predicted': Y_test_modelo_predicted}

postprocess_df = pd.DataFrame(data)
confusion_matrix = pd.crosstab(postprocess_df['y_actual'], postprocess_df['y_predicted'], rownames=['Actual'], colnames=['Predicted'])

confusion_matrix

# tasa de error tipo 2
matriz_de_confusion_modelo[1][0]/(matriz_de_confusion_modelo[1][0]+matriz_de_confusion_modelo[0][0])
# tasa dse error tipo 1
matriz_de_confusion_modelo[0][1]/(matriz_de_confusion_modelo[0][1]+matriz_de_confusion_modelo[1][1])

# Indicador KS y ROC AUC

from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

def calculo_del_roc_y_ks (Y, Y_predicho):

  df=pd.DataFrame()
  df["real"]= Y
  df["Proba"]= Y_predicho

  class0= df[df["real"]==0]
  class1= df[df["real"]==1]

  KS= ks_2samp(class0["Proba"], class1["Proba"])
  ROC_AUC = roc_auc_score(df["real"], df["Proba"])

  print (f"KS: {KS.statistic:.4f}(p-valure:{KS.pvalue:3e})")
  print (f"ROC AUC:{ROC_AUC:.4f}")

  return KS.statistic, ROC_AUC


calculo_del_roc_y_ks (Y_test_modelo,modelo.predict(X_test_modelo).T[0])
