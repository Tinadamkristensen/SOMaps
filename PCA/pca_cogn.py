"""
Created on Wed May  4 15:37:27 2022

@author: FMAG0005
"""

#%%
import pandas as pd
import os
import argparse

#%%
# Parsing arguments from command line
parser = parser = argparse.ArgumentParser()
parser.add_argument('--basedir', help='Base directory', default = os.getcwd()) # working directory, data should be here. Default is the current directory
parser.add_argument('--cog_tests', help='Cognitive tests to include in the analysis') #cognitive tests to include into the analysis
args = parser.parse_args()
data = pd.read_excel('PCA_FAB_COG_W_DART.xlsx', error_bad_lines=False)

#%% Import the data
import missingno as msno
msno.matrix(data)

#%%
#data = data.drop(index = [37, 40, 81, 87, 113, 132, 158, 166, 172, 282, 283, 293, 302, 313])
#msno.matrix(data)
# eventually also 240, 245, 250

#%% Select choosen features
import numpy as np

# Transform features into the right datatype
data.Group = data.Group.astype("category")
data.Subgroup = data.Subgroup.astype("category")
data.ID_Num = data.ID_Num.astype(int)
data.Age = data.Age.apply(np.floor).astype(int)
data.Gender = data.Gender.astype("category")

# drop idx
drop_idx = [165]
data = data.drop(index = drop_idx)

# Select subset of data
data = data[["ID_Num","Group","Subgroup","Age"] + args.cog_tests]

# Separating out the target
y = data.Subgroup.values

# Separating out the features
x = data.drop(['Group','ID_Num','Subgroup','Age','Gender'],axis=1)

features = x.columns.values
x = x.loc[:,:].values

apdx = '_' + '_'.join(features[0:2])

data.to_csv('cogn_data' + apdx + '.csv')

msno.matrix(data)

#%% Prepare data
# Imputing missing values using a KNN Imputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
x = imputer.fit_transform(x)

# Standardizing the features, need for PCA!
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)

#%% Double check whether we have missing values
msno.matrix(pd.DataFrame(x))


#%% Perform PCA, choose the frist three PCs
from sklearn.decomposition import PCA

pca = PCA(n_components=6)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1','PC2','PC3','PC4','PC5','PC6'])

finalDf = pd.concat([principalDf.get(['PC1','PC2','PC3']), data['Subgroup'], data['Group']], axis = 1)

#%% Plot PCs
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize = (8,8))
ax = sns.scatterplot('PC1', 'PC2', data=finalDf, hue='Subgroup', palette=["r","g","b"], alpha=0.7, style="Group")
ax.set_aspect('equal', adjustable='box')
plt.savefig('pc1pc2' + apdx + '.png')
plt.show()
plt.clf()

fig = plt.figure(figsize = (8,8))
ax = sns.scatterplot('PC1', 'PC3', data=finalDf, hue='Subgroup', palette=["r","g","b"], alpha=0.7, style="Group")
ax.set_aspect('equal', adjustable='box')
plt.savefig('pc1pc3' + apdx + '.png')
plt.show()
plt.clf()

fig = plt.figure(figsize = (8,8))
ax = sns.scatterplot('PC2', 'PC3', data=finalDf, hue='Subgroup', palette=["r","g","b"], alpha=0.7, style="Group")
ax.set_aspect('equal', adjustable='box')
plt.savefig('pc2pc3' + apdx + '.png')
plt.show()
plt.clf()

#%% Plot variance explained
pca_expvar = pca.explained_variance_ratio_
pca_cums_ev = pca_expvar.cumsum()


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Components', fontsize = 15)
ax.set_ylabel('Cumulative Explained Variance', fontsize = 15)
ax.bar(['PC1','PC2','PC3','PC4','PC5','PC6'], pca_cums_ev)
plt.savefig('cumExpVar' + apdx + '.png')


#%% Loadings of all 6 PCs as heatmap
pcaComp = pca.components_
pcaCompDf = pd.DataFrame(pcaComp,index=['PC1','PC2','PC3','PC4','PC5','PC6'],columns=features)

import seaborn as sns

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax = sns.heatmap( pcaCompDf , linewidth = 0.5 , cmap = 'coolwarm' )
plt.yticks(rotation=0)
plt.savefig('pcLoadings' + apdx + 'png',bbox_inches='tight')

#%% Loadings of first 3 PCs as barplot
pcaCompSelDf = pcaCompDf.transpose().drop(['PC4','PC5','PC6'],1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax = pcaCompSelDf.plot.bar()
plt.ylim(-.45,.45)
ax.set_ylabel('Loadings', fontsize = 15)
plt.savefig('pc123overTests' + apdx + '.png',bbox_inches='tight')


