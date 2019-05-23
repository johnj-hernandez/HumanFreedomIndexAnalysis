import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy 
##usamos la semilla 35 para hacer nuestro proyecto producible
numpy.random.seed(35)

#Grafico de codo
def elbowPlot(data,maxKClusters):
    inertias=list()
    for i in range(1,maxKClusters+1):
        myCluster=KMeans(n_clusters=i)
        myCluster.fit(data)   
        inertias.append(myCluster.inertia_)  
    plt.figure() 
    x=[i for i in range(1,maxKClusters+1)]
    y=[i for i in inertias]
    plt.plot(x,y, 'ro-', markersize=8, lw=2)
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()


######---------------------MAIN-------------------------------------
df = pd.read_csv("hf.csv")
df.head()
df.describe()

#agrupamos por año
print(df.groupby('year').size())

#filtramos por el año 2016 que es el mas reciente
datos2016=df[df['year']==2016]

#eliminamos los valores na 
datos2016=datos2016.dropna()

#eliminamos las 3 primeras columnas correspondientes a el año, pais y region 
datos2016_2=datos2016.drop(['year','countries','region'],axis=1)

#tomaremos 3 clusters
elbowPlot(datos2016_2,10)
cluster=KMeans(n_clusters=3)
cluster.fit(datos2016_2)
#analizamos los centros y les agregamos sus nombres de columna
centros=cluster.cluster_centers_
centros=pd.DataFrame(centros)
centros.columns=list(datos2016_2)

#Exportamos los datos
centros.to_csv("centros.csv")

datos2016['cluster']=cluster.labels_

datos2016.plot.scatter(x='pf_score',y='ef_score',c='cluster',colormap='viridis')


#ya tenemos los clusters en datos2016 y datos 2016_2 representa las columnas usadas para el clustering
datos2016[datos2016['countries']=='Colombia']

#filtamos paises de cada cluster
datos2016[datos2016['cluster']==2]

