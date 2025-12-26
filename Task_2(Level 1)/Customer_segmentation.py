# CUSTOMER SEGMENTATION FOR SPOTIFY DATASET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#LOAD DATASET
df=pd.read_csv("spotify_data(2).csv",encoding="latin1")
print("\n       FIRST FIVE ROWS         ")
print(df.head())

#DATA CLEANING
print("\n       MISSING VALUES       ")
print(df.isnull().sum())
df=df.dropna()

#NUMERIC CONVERSION
df["streams"]=pd.to_numeric(df["streams"],errors="coerce")
df["in_spotify_playlists"]=pd.to_numeric(df["in_spotify_playlists"],errors="coerce")
df["in_spotify_charts"]=pd.to_numeric(df["in_spotify_charts"],errors="coerce")
df=df.dropna(subset=["streams"])

#DESCRIPTIVE STATISTICS
print("\n         DESCRIPTIVE STATISTICS        ")
print(df.describe())
print("\n AVERAGE STREAMS:",df["streams"].mean())
print("\n AVERAGE DANCEABILITY:",df["danceability_%"].mean())

#FEATURE SELECTION
features=["streams",
          "in_spotify_playlists",
          "in_spotify_charts",
          "danceability_%",
          "energy_%",
          "valence_%",
          "acousticness_%"
          ]
X=df[features]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#FIND OPTIMAL K
wcss=[]
for k in range(2,10):
    km=KMeans(n_clusters=k,random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(2,10),wcss,marker='o')
plt.title("ELBOW METHOD (CHOOSE OPTIMAL K)")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

k=4
model=KMeans(n_clusters=k,random_state=42)
df["Cluster"]=model.fit_predict(X_scaled)
print("\n     SAMPLE CLUSTER ASSIGNMENTS   ")
print(df[["track_name","artist(s)_name","Cluster"]].head())

#VISUALIZATIONS
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["streams"],y=df["danceability_%"],hue=df["Cluster"],palette="viridis",s=80)
plt.title("SPOTIFY TRACK SEGMENTATION")
plt.xlabel("Streams")
plt.ylabel("Danceability %")
plt.show()

#BAR CHART
cluster_avg=df.groupby("Cluster")["streams"].mean()
plt.figure(figsize=(8,6))
cluster_avg.plot(kind="bar")
plt.title("AVERAGE STREAMS PER CLUSTER(BAR CHART)")
plt.xlabel("Cluster")
plt.ylabel("Average Streams")
plt.show()

#CLUSTER INSIGHTS
cluster_summary=df.groupby("Cluster")[features].mean()
print("\n     CLUSTER SUMMARY      ")
print(cluster_summary)
print("\n   INSIGHTS   ")
for c,row in cluster_summary.iterrows():
    print(f"\nCluster {c}:")
    print(f"-Avg Streams:{row['streams']:.2f}")
    print(f"-Danceability:{row['danceability_%']:.2f}")
    print(f"-Energy:{row['energy_%']:.2f}")

    if row["streams"]>df["streams"].mean():
        print("  -> POPULAR / VIRAL SONGS")
    else:
        print("  -> LESS POPULAR TRACKS")

    if row["danceability_%"]>60:
        print(" -> HIGHLY DANCEABLE MUSIC")
    elif row["energy_%"]>70:
        print(" -> HIGH ENERGY TRACKS")

df.to_csv("cleaned_spotify_data.csv",index=False)
print("\n Cleaned dataset saved as: cleaned_spotify_data.csv")