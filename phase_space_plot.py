import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
# from scipy.spatial import distance
# from scipy.spatial.distance import cdist
from sklearn import manifold
# import collections
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import pdb

def biplot(X, labels, arrow_mul=1, text_mul=1.1):

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    x_data = X[:,0]
    y_data = X[:,1]

    pc0 = pca.components_[0]
    pc1 = pca.components_[1]

    plt.figure()
    plt.scatter(x_data, y_data)

    for i in range(pc0.shape[0]):
        plt.arrow(0, 0,
                  pc0[i]*arrow_mul, pc1[i]*arrow_mul,
                  color='r')
        plt.text(pc0[i]*arrow_mul*text_mul,
                 pc1[i]*arrow_mul*text_mul,
                 labels,
                 color='r')
    plt.show()


"""
df = pd.read_csv('chinen.csv')
specimen = set(df.iloc[:, 2])
amount = df.iloc[:, 4]
# specimen_count = collections.Counter(specimen)

spec_amount_dict = {}
for species in specimen:
    idx = np.where(df.iloc[:, 2] == species, True, False)
    spec_amount_dict[species] = np.sum(amount.iloc[idx])

sorted_list = sorted(spec_amount_dict.items(), key=lambda x:x[1])
sorted_amount_list = [item[1] for item in sorted_list][::-1]
sorted_specimen_list = [item[0] for item in sorted_list][::-1]
cum_ratio_list = np.cumsum(sorted_amount_list)/np.sum(sorted_amount_list)
focused_idx = np.where(cum_ratio_list < 0.9, True, False)

focused_specimen_list = sorted_specimen_list[:20]
focused_years_list = np.arange(1996, 2021)

result_df = ""
for year in focused_years_list:
    year_str = str(year)
    idx = [year_str in item for item in df.iloc[:, 0]]
    sub_df = df.iloc[idx, :]
    sub_df = sub_df.reset_index(drop=True)

    tmp_dict = {}
    for i in range(len(sub_df)):
        species = sub_df.iat[i, 2]
        if species in focused_specimen_list:
            if species in tmp_dict:
                tmp_dict[species] += sub_df.iat[i, 4]
            else:
                tmp_dict[species] = sub_df.iat[i, 4]
        else:
            if 10000 in tmp_dict:
                tmp_dict[10000] += sub_df.iat[i, 4]
            else:
                tmp_dict[10000] = sub_df.iat[i, 4]

    tmp_ser = pd.Series(tmp_dict)

    if isinstance(result_df, str):
        result_df = pd.DataFrame(tmp_ser)
        result_df = result_df.T
    else:
        result_df = result_df.append(tmp_ser, ignore_index=True)

nan_idx = np.isnan(result_df)
result_df[nan_idx] = 0
result_df.to_csv("year_feature_mat.csv")
"""

feature_df = pd.read_csv('year_feature_mat.csv')
feature_df = feature_df.iloc[:, 1:]
# feature_df = pd.read_csv('cross.csv', index_col=0)
labels = feature_df.columns
autoscaled_df = (feature_df - feature_df.mean()) / feature_df.std()

# biplot(autoscaled_df, labels, arrow_mul=6)

pca = PCA()
pca.fit(autoscaled_df)
score = pd.DataFrame(pca.transform(autoscaled_df), index=feature_df.index)

plt.scatter(score.iloc[:, 0], score.iloc[:, 1])
plt.show()

pdb.set_trace()

"""
dist_mat = distance_matrix(feature_df.values, feature_df.values)
# dist_mat = cdist(feature_df.values, feature_df.values, 'mahalanobis')
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
pos = mds.fit_transform(dist_mat)

pdb.set_trace()
cov_mat = np.matrix(np.cov(feature_df.values.T, dtype=np.float64))
inv_cov_mat = np.linalg.inv(cov_mat)
mah_dist_mat = np.zeros((len(feature_df), len(feature_df)))
for i in range(0, (len(feature_df)-1)):
    u = feature_df.values[i, ]
    for j in range(i, len(feature_df)):
        v = feature_df.values[j, ]
        dist = distance.mahalanobis(u, v, inv_cov_mat)
        if np.isnan(dist):
            pdb.set_trace()
        mah_dist_mat[i, j] = dist
"""

plt.scatter(pos[:, 0], pos[:, 1], marker = 'o')
labels = np.arange(1996, 2021)

for label, x, y in zip(labels, pos[:, 0], pos[:, 1]):
    plt.annotate(
        label,
        xy = (x, y),
    )
plt.show()

pos_df = pd.DataFrame(pos)
pos_df.columns = ['x', 'y']
pos_df.to_csv('pos.csv')

pdb.set_trace()
