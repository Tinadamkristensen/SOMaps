import pandas as pd
import numpy as np
import os
import sys
from minisom import MiniSom
import matplotlib.pyplot as plt

# -- Load Data
apdx = '_WAIS_FIQ_CANTAB_RVPA'

data = pd.read_csv('cogn_data' + apdx + '.csv')
data = data.drop('Unnamed: 0',axis=1)

# -- Cognitive tests
# --
# SSPTotalerrors
# SWM Strategy
# SWM Totalerrors
# SWM between errors
# IEDTotalerrorsadj
# IEDEDSerrors
# IEDTotallatency
# RTIMeansimplereactiontime
# RVPMeanlatency
# --

if 'CANTAB_IEDTotalerrorsadj' in data:
    data = data.assign(CANTAB_IEDTotalerrorsadj_neg = -1*data.CANTAB_IEDTotalerrorsadj)
    data = data.drop('CANTAB_IEDTotalerrorsadj',1)

if 'CANTAB_SWM_Totalerrors' in data:
    data = data.assign(CANTAB_SWM_Totalerrors_neg =  -1*data.CANTAB_SWM_Totalerrors)
    data = data.drop('CANTAB_SWM_Totalerrors',1)
    
if 'CANTAB_SWM_Strategy' in data:
    data = data.assign(CANTAB_SWM_Strategy_neg =  -1*data.CANTAB_SWM_Strategy)
    data = data.drop('CANTAB_SWM_Strategy',1)

# -- Separating out the target
y = data.Subgroup.values
y = y.astype(int)
label_names = {0:'HC', 1:'UHR', 2:'FEP'}

# -- Separating out the features
x = data.drop(['Group','ID_Num','Subgroup','Age','Gender'],1)
features = x.columns.values

# -- Imputing missing values with a KNN imputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
x = imputer.fit_transform(x)

# -- Data normalization
x = (x - np.mean(x[y==0], axis=0)) / np.std(x[y==0], axis=0)

# -- SOM
n_neurons = 2 # number of neurons in x axis
m_neurons = 3 # number of neurons in y axis

apdx = apdx + '_' + str(n_neurons*m_neurons)

som = MiniSom(n_neurons, m_neurons, x.shape[1], sigma=1.5, learning_rate=.5, neighborhood_function='gaussian', random_seed=0) 

# -- Initialization
som.pca_weights_init(x) # initialize weights with PCA
som.train(x, 1000, verbose=True)  # random training, 1000 iterations

# -- Plot
plt.figure(figsize=(9, 9))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
plt.colorbar()

# Response patterns
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']

for cnt, xx in enumerate(x):
    w = som.winner(xx)  # getting the winner
    # place a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5,
             markers[y[cnt]-1],
             markerfacecolor='None',
             markeredgecolor=colors[y[cnt]-1],
             markersize=12, markeredgewidth=2)

plt.savefig('som_results/response_winner' + apdx + '.png')

# winner node coordinate (x and y) for each subject
w_x, w_y = zip(*[som.winner(d) for d in x])
w_x = np.array(w_x)
w_y = np.array(w_y)

winner_tab = pd.DataFrame()
winner_tab.insert(0, 'ID_Num',data.ID_Num)
winner_tab.insert(1,'winner_x',w_x)
winner_tab.insert(2,'winner_y',w_y)

winner_tab.to_csv ('som_results/winner_tab' + apdx + '.csv', index = False, header=True)

plt.figure(figsize=(9, 9))
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
plt.colorbar()

for c in np.unique(y):
    idx_target = y==c
    plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                s=50, c=colors[c-1], label=label_names[c])
plt.legend(loc='upper right')
plt.grid()
plt.savefig('som_results/response_winner_pattern' + apdx + '.png')

# -- Class distributions in each neuron
fig, axs  = plt.subplots(n_neurons,m_neurons,figsize=(8,8))             
labels_map = som.labels_map(x, [label_names[t] for t in y])

for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names.values()]
    axs[position[0], position[1]].pie(label_fracs, autopct='%.0f%%',labels = label_names.values(),startangle=90, colors = ["C2","C0","C1"])
    axs[position[0], position[1]].set_title(str(position))

plt.savefig('som_results/repronse_winner_distribution' + apdx + '.png')    
plt.show()

# Spyder plots of averages in cognitive profiles

labels_map = som.labels_map(x, [label_names[t] for t in y])
meanProfDf = pd.DataFrame(dict(tests = features))
cnt = 1

for position in labels_map.keys():
        meanProf = np.mean(x[(w_x==position[0]) & (w_y==position[1]),:],axis=0)
        meanProfDf[str(position)] = meanProf.tolist()
        
meanProfDf = meanProfDf.append(meanProfDf.iloc[0])

labels = meanProfDf.tests.values        
        
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(labels))

plt.figure(figsize=(8, 8))
plt.subplot(polar=True)

for position in labels_map.keys():
    plt.plot(label_loc, meanProfDf[str(position)], label=str(position))
    

lines, labels = plt.thetagrids(np.degrees(label_loc), labels=features)
plt.legend()
plt.savefig('som_results/repronse_mean_profiles' + apdx + '.png')

# Split cognitive profiles and plot spyder plots

fig, axs  = plt.subplots(n_neurons,m_neurons,figsize=(20,20), subplot_kw={'projection': 'polar'})

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(labels))

colors = plt.rcParams["axes.prop_cycle"]()

for position in labels_map.keys():
    c = next(colors)["color"]
    axs[position[0], position[1]].plot(label_loc, meanProfDf[str(position)], label=str(position), c=c, alpha=0.75)
    axs[position[0], position[1]].fill(label_loc, meanProfDf[str(position)], label=str(position), c=c, alpha=0.2)
    axs[position[0], position[1]].set_thetagrids(np.degrees(label_loc), labels=features)
    axs[position[0], position[1]].set_ylim(-4,2)
    
plt.savefig('som_results/repronse_mean_profiles_subfig' + apdx + '.png')
 
# Plot mean and standard deviation of cognitive profiles
plt.figure(figsize=(8, 8))

meanProfDf = pd.DataFrame(dict(tests = features))
stdProfDf = pd.DataFrame(dict(tests = features))

for position in labels_map.keys():
        meanProf = np.mean(x[(w_x==position[0]) & (w_y==position[1]),:],axis=0)
        meanProfDf[str(position)] = meanProf.tolist()
        
        stdProf = np.std(x[(w_x==position[0]) & (w_y==position[1]),:],axis=0)
        stdProfDf[str(position)] = stdProf.tolist()

for position in labels_map.keys():
    plt.errorbar(features, meanProfDf[str(position)],stdProfDf[str(position)], label=str(position))

plt.grid()
plt.xticks(rotation = 45,ha="right")
plt.legend()
plt.savefig('som_results/repronse_mean_profiles_lines' + apdx + '.png')
plt.show()
