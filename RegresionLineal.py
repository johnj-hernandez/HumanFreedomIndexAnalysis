import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns'none)
stats = [
    "pf_ruleOfLaw",
    "pf_securityAndSafaty",
    "pf_movement",
    "pf_religion",
    "pf_association",
    "pf_expression",
    "pf_identity",
    "pf_score",
    "pf_rank",
    "ef_government",
    "ef_legal",
    "ef_money",
    "ef_trade",
    "ef_regulation",
    "ef_score",
    "ef_rank",
    "hf_score",
    "hf_rank",
    ]
data = pd.read_csv("hf.csv")
print(data)
data=data[data.countries=='Colombia']
print(data)
print(list(data))
df = data.drop(['countries','region'], axis=1)
datos=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
i=0
for stats_name in stats:
    X = df
    y = df[[stats_name]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    reg = LinearRegression()
    print(stats_name)
    reg.fit(X_train[['year']], y_train)
    y_predicted = reg.predict(X_test[['year']])
    MSE=mean_squared_error(y_test, y_predicted)
    print("Error cuadrado: ",MSE )
    rs=r2_score(y_test, y_predicted)
    print("R²: ",rs)
    print("Predicciones: ", y_predicted)
    datos[i].append(MSE)
    datos[i].append(rs)
    datos[i].append(y_predicted)
    ig, ax = plt.subplots()
    ax.scatter(y_test, y_predicted)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Datos')
    ax.set_ylabel('Prediccion')
    ax.set_title(stats_name)
    plt.show()
    i=i+1
tabla=pd.DataFrame(datos,index=stats,columns=["Error medio Cuadrado", "R^2", "Predicciones"])
print(tabla)
#export_csv = tabla.to_csv (r'C:\Users\ASUS\source\repos\ProyectoLenguaje\ProyectoLenguaje\export_dataframe.csv', header=True)
