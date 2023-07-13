# Projekt 2
## Analiza zbioru danych **"e-shop clothing 2008"**
## Kamień milowy 2: Model
#Autorzy: Laura Hoang, Piotr Bielecki

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import seaborn as sns

with open("e-shop data and description/e-shop clothing 2008 data description.txt", "r") as f:
    print(f.read())
    f.close()

#### Import danych

data = pd.read_csv('e-shop data and description/e-shop clothing 2008.csv', sep=';')
data

# =============================================================================
# Po poprzednim kamieniu milowym ustaliliśmy kilka kwestii:
# - zdecydowaliśmy się na klasteryzację po **kategoriach produktów**
# - utworzymy kolumnę z dniem tygodnia, gdyż taki format dat jaki mamy jest niewygodny do pracy i wnosi dla nas za mało informacji
# - usuniemy kolumny `price` i `price 2`, gdyż ten zbiór danych mógłby być przeznaczony do tego, aby w przyszłości przewidywać ceny produktów
# 
# #### Data preprocessing
# 
# Tak jak ustaliliśmy powyżej, dodamy kolumnę `weekday` z dniem tygodnia wygenerowanym z dat.
# =============================================================================

import datetime

weekdays = []
for (yr, mth, day) in zip(data.year, data.month, data.day):
    weekdays += [datetime.datetime(yr, mth, day).weekday()]

data['weekday'] = weekdays
cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols]

# 0- poniedzialek, 6- niedziela
data

# Kolumna `year` osiaga tylko jedna wartość (2008), przez co nie wnosi dla nas dużo informacji. Dlatego zostanie ona usunięta.

data = data.drop(columns=['year'])
data

# Tak jak ustaliliśmy, usuniemy również kolumny `price` i `price 2`.

data = data.drop(columns=['price', 'price 2'])
data

# zakodujemy dni tygodnia, miesiące i zmienną page 2 one hot encodingiem, usuniemy sessionID

encoded = pd.get_dummies(data['month'].astype(str))
data_preprocessed = data.drop('month', axis=1)
data_preprocessed = pd.concat([data_preprocessed, encoded], axis = 1)
data_preprocessed.rename(columns={'4':'april', '5':'may', '6':'june' , '7':'july', '8':'august'}, inplace=True)

encoded = pd.get_dummies(data['weekday'].astype(str))
data_preprocessed = data_preprocessed.drop('weekday', axis=1)
data_preprocessed = pd.concat([data_preprocessed, encoded], axis = 1)
data_preprocessed.rename(columns={'0':'Monday', '1':'Tuesday', '2':'Wednesday' , '3':'Thursday', '4':'Friday', '5':'Saturday', '6':'Sunday'}, inplace=True)
data_preprocessed
encoded = pd.get_dummies(data['page 2 (clothing model)'].astype(str))
data_preprocessed = data_preprocessed.drop('page 2 (clothing model)', axis=1)
data_preprocessed = pd.concat([data_preprocessed, encoded], axis = 1)
data_preprocessed = data_preprocessed.drop(columns = 'session ID')
data_preprocessed

# funkcja zwracająca metryki

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def plot_scores(model, X, max_clusters=20):
    
    cluster_num_seq = range(2, max_clusters+1)
    
    scores = [silhouette_score, calinski_harabasz_score, davies_bouldin_score]
    
    scores_values = count_clustering_scores(X, cluster_num_seq, model, scores)
    

    plt.plot(cluster_num_seq, scores_values[silhouette_score], 'bx-')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Miara metryki silhouette')
    plt.title('Wartość miary silhouette')
    plt.show()
        
    plt.plot(cluster_num_seq, scores_values[calinski_harabasz_score], 'bx-')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Miara metryki Calinski-Harabasz')
    plt.title('Wartość miary Calinski-Harabasz')
    plt.show()
    
    plt.plot(cluster_num_seq, scores_values[davies_bouldin_score], 'bx-')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Miara metryki Davies-Bouldin')
    plt.title('Wartość miary Davies-Bouldin')
    plt.show()

def count_clustering_scores(X, cluster_num, model, scores_fun_list):
    if isinstance(cluster_num, int):
        cluster_num_iter = [cluster_num]
    else:
        cluster_num_iter = cluster_num
        
    scores = {}    
    for x in scores_fun_list:
        scores[x] = []
        
    for k in cluster_num_iter:
        model.set_params(n_clusters = k)
        labels = model.fit_predict(X)
        for a in scores_fun_list:
            temp = scores[a]
            temp.append(a(X, labels))
            scores[a] = temp
    
    if isinstance(cluster_num, int):
        return scores[0]
    else:
        return scores

sample = data_preprocessed.sample(30000)

from sklearn.cluster import KMeans
plot_scores(KMeans(random_state=420, n_init = 20), sample)

from sklearn.cluster import MiniBatchKMeans

plot_scores(MiniBatchKMeans(random_state=420, n_init = 20), sample)

from sklearn.cluster import AgglomerativeClustering

plot_scores(AgglomerativeClustering(linkage = 'ward'), sample)

plot_scores(AgglomerativeClustering(linkage = 'average'), sample)

plot_scores(AgglomerativeClustering(linkage = 'complete'), sample)

plot_scores(AgglomerativeClustering(linkage = 'single'), sample)