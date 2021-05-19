import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans

df=pd.read_csv("Iris.csv")
print(df.describe())

sns.pairplot(df, hue='Species', markers='x')
plt.show()
df.drop("Id",inplace=True,axis=1)
df.drop("Species",inplace=True,axis=1)
sns.heatmap(df.corr(),linecolor='white',linewidths=1,annot=True)
within_cluster=[]
x=df.iloc[:,[0,1,2,3]].values

cluster_range=range(1,15)
for k in cluster_range:
    kn=KMeans(n_clusters=k)
    kn=kn.fit(df)
    within_cluster.append(kn.inertia_)
plt.plot(cluster_range,within_cluster,'go--',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of square')
plt.grid()
plt.show()
model=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
prediction=model.fit_predict(df)
plt.scatter(x[prediction==0,0],x[prediction==0,1],s=50,c='red',label ='Iris-setosa')
plt.scatter(x[prediction==1,0],x[prediction==1,1],s=50,c='blue',label ='Iris-versicolour')
plt.scatter(x[prediction==2,0],x[prediction==2,1],s=50,c='green',label ='Iris-virginica')

plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=100,c='Yellow',label='Centroid')
plt.legend()
plt.grid()
plt.show( )
