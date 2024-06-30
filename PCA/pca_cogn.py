import pandas as pd
import os
import argparse
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns



# Parsing arguments from command line
parser = parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', help='Base directory', default = os.getcwd()) # working directory, data should be here. Default is the current directory
parser.add_argument('--data_path', help='Path to data file', default='data/CogProf.Final_FAB2xlsx')
parser.add_argument('--population', help="Set >ALL< or >AP_naive< for antipyschotic naive polution", default='ALL')

args = parser.parse_args()
base_dir = args.base_dir
data_path = args.data_path
ap_naive = args.population == 'AP_naive'

data = pd.read_excel(os.path.join(base_dir, args.data_path))
cogn_tests = data.columns[-7:]

# Transform features into the right datatype
data.Group = data.Group.astype("category")
data.Subgroup = data.Subgroup.astype("category")
data.AP_Naive = data.AP_Naive.astype("category")
data.ID = data.ID.astype(int)
data.Age = data.Age.apply(np.floor).astype(int)
data.Gender = data.Gender.astype("category")

# Antipsychotic naive
if ap_naive:
    data = data[data.AP_Naive==1]
    apdx = '_ap_naive'
else:
    apdx = '_'

# Separating out the target
y = data.Subgroup.values

# Separating out the features
x = data[cogn_tests].values

# PCA
pca = PCA(n_components=len(cogn_tests))
principalComponents = pca.fit_transform(x-x.mean(axis=0))
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC' + str(i+1) for i in range(len(cogn_tests))])
finalDf = pd.concat([principalDf.get(['PC1','PC2','PC3']), data['Subgroup'], data['Group']], axis = 1)

# Plot PCs
sns.set_theme(style="white", palette=["r","g","b"])
fig = plt.figure(figsize = (8,8))
ax = sns.scatterplot(data=finalDf, x='PC1', y='PC2', hue='Subgroup', alpha=0.7, style="Group")
ax.set_aspect('equal', adjustable='box')
plt.savefig('PCA/results/pc1pc2' + apdx + '.png')

fig = plt.figure(figsize = (8,8))
ax = sns.scatterplot(data=finalDf, x='PC1', y='PC3', hue='Subgroup', alpha=0.7, style="Group")
ax.set_aspect('equal', adjustable='box')
plt.savefig('PCA/results/pc1pc3' + apdx + '.png')

fig = plt.figure(figsize = (8,8))
ax = sns.scatterplot(data=finalDf, x='PC2', y='PC3', hue='Subgroup', alpha=0.7, style="Group")
ax.set_aspect('equal', adjustable='box')
plt.savefig('PCA/results/pc2pc3' + apdx + '.png')

# Plot variance explained
pca_expvar = pca.explained_variance_ratio_
pca_cums_ev = pca_expvar.cumsum()


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Components', fontsize = 15)
ax.set_ylabel('Cumulative Explained Variance', fontsize = 15)
ax.bar(['PC' + str(i+1) for i in range(len(cogn_tests))], pca_cums_ev)
plt.savefig('PCA/results/cumExpVar' + apdx + '.png')


# Loadings of all 6 PCs as heatmap
pcaComp = pca.components_
pcaCompDf = pd.DataFrame(pcaComp, index=['PC' + str(i+1) for i in range(len(cogn_tests))], columns=cogn_tests)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax = sns.heatmap( pcaCompDf , linewidth = 0.5 , cmap = 'coolwarm' )
plt.yticks(rotation=0)
plt.savefig('PCA/results/pcLoadings' + apdx + 'png',bbox_inches='tight')

# Loadings of first 3 PCs as barplot
pcaCompSelDf = pcaCompDf.transpose().get(['PC1','PC2','PC3'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax = pcaCompSelDf.plot.bar()
plt.ylim(-.45,.45)
ax.set_ylabel('Loadings', fontsize = 15)
plt.savefig('PCA/results/pc123overTests' + apdx + '.png',bbox_inches='tight')


