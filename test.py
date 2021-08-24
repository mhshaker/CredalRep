from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering

X = np.array([9,9,9,9,9,9,9,9,9,9,9,9,9,3,3,3,3])
X = X.reshape(-1, 1)  
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [10, 2], [10, 4], [10, 0]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
res = kmeans.labels_
index = np.where(res == 1)

print(res)
print(index)



# kmeans.cluster_centers_

# pm = np.array([0,1,2,3,4,100,105])
# idx = [0,2,3,5,6]


# cluster_algorithm = AgglomerativeClustering(n_clusters=2)
# labels = cluster_algorithm.fit_predict(np.expand_dims(pm[idx], axis=-1))

# print(labels)

# idx_labels = [np.where(labels == e)[0] for e in set(labels)]
# idx_labels  # [array([3, 4], dtype=int64), array([0, 1, 2], dtype=int64)]
