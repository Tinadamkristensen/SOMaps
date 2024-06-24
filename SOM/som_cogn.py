import pandas as pd
import numpy as np
import os
import sys
import argparse
from minisom import MiniSom
import matplotlib.pyplot as plt

# Inverse tests
# =============================================================================
# SSPTotalerrors
# SWM Strategy
# SWM Totalerrors
# SWM between errors
# IEDTotalerrorsadj
# IEDEDSerrors
# IEDTotallatency
# RTIMeansimplereactiontime
# RVPMeanlatency
# =============================================================================

parser = parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', help='Base directory', default = os.getcwd()) # working directory, data should be here. Default is the current directory
parser.add_argument('--data_path', help='Path to data file', default='data/CogProf.Final.xlsx')
parser.add_argument('--population', help="Set >ALL< or >AP_naive< for antipyschotic naive polution", default='ALL')
parser.add_argument('--norm', help="Set to >ALL< or >HC< for normalization based on HC only", default='ALL')


args = parser.parse_args()
base_dir = args.base_dir
data_path = args.data_path
ap_naive = args.population == 'AP_naive'
hc_norm = args.norm == 'HC'

print(f'Running SOM analysis with:\n\t AP Naive Patients only: {ap_naive}, \n\t HC normalization: {hc_norm}')



data = pd.read_excel(os.path.join(base_dir, args.data_path))
data = data.rename(columns={'WAIS_FIQ':'FIQ',
            'CANTAB_RVPA':'RVPA',
            'CANTAB_SWM_Strategy':'SWM_Strat',
            'BACS_numbersequencetotalcorrect':'NS',
            'BACS_Fluency_Total':'Fluency',
            'BACS_Symbolnumbersnrcorrec':'Symb',
            'CANTAB_IEDTotalerrorsadj':'IED'})

cogn_tests = data.columns[-7:].values.tolist()

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
    apdx = '_all'


if 'IED' in cogn_tests:
    data = data.assign(IED_neg = -1*data.IED)
    cogn_tests[cogn_tests.index('IED')] = 'IED_neg' 
    data = data.drop(labels='IED', axis=1)

if 'SWM_Err_Err' in cogn_tests:
    data = data.assign(SWM_Err_neg =  -1*data.SWM_Err)
    cogn_tests[cogn_tests.index('SWM_Err')] = 'SWM_Err_neg' 
    data = data.drop(labels='SWM_Err', axis=1)
    
if 'SWM_Strat' in cogn_tests:
    data = data.assign(SWM_Strat_neg =  -1*data.SWM_Strat)
    cogn_tests[cogn_tests.index('SWM_Strat')] = 'SWM_Strat_neg' 
    data = data.drop(labels='SWM_Strat', axis=1)

# Separating out the target
y = data.Subgroup.values.astype(int)
label_names = {1:'HC', 2:'UHR', 3:'FEP'}

#  Separating out the features
x = data[cogn_tests].values

# data normalization
if hc_norm:
    x = (x - x[y==1].mean(axis=0)) / x[y==1].std(axis=0)
    apdx += '_hc_norm'
else:
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    apdx += '_pop_norm'


# This is the number of neurons in the SOM. This is very important to get right.
# If you increase the numbers, you will get a better fit, but also take the risk of overfitting.
# You get a "Net" with n x m neurons.
# Best to have some reference when choosing.

n_neurons = 2
m_neurons = 3

apdx += '_' + str(n_neurons*m_neurons)

# You might have to install MiniSom
# This is the SOM algorithm, with standard parameters
# Definition:  MiniSom(x, y, input_len, sigma=1.0, learning_rate=0.5, decay_function=asymptotic_decay, neighborhood_function='gaussian', topology='rectangular', activation_distance='euclidean', random_seed=None)
som = MiniSom(n_neurons, m_neurons, x.shape[1], sigma=1.5, learning_rate=.5, neighborhood_function='gaussian', random_seed=0) 

# We initiate it with PCA (faster) and train it for 1000 steps
som.pca_weights_init(x)
som.train(x, 1000, verbose=True)  # random training

#  Plot SOM

plt.figure(figsize=(9, 9))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
plt.colorbar()

# Plotting the response for each pattern
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
plt.savefig('SOM/results/response_winner' + apdx + '.png')

#  Scatter plot of points in certain node, colourcoded with classes. Each node (here as box) represents one cognitive profile.

# get the winner node coordinate (x and y) for each subject
w_x, w_y = zip(*[som.winner(d) for d in x])
w_x = np.array(w_x)
w_y = np.array(w_y)

winner_tab = pd.DataFrame()
winner_tab.insert(0, 'ID',data.ID)
winner_tab.insert(1,'winner_x',w_x)
winner_tab.insert(2,'winner_y',w_y)
winner_tab.to_csv ('SOM/results/winner_tab' + apdx + '.csv', index = False, header=True)

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
plt.savefig('SOM/results/response_winner_pattern' + apdx + '.png')

#  Plot the class distributions in each neuron
fig, axs  = plt.subplots(n_neurons,m_neurons,figsize=(8,8))
                         
labels_map = som.labels_map(x, [label_names[t] for t in y])

for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names.values()]
    axs[position[0], position[1]].pie(label_fracs, autopct='%.0f%%',labels = label_names.values(),startangle=90, colors = ["C2","C0","C1"])
    axs[position[0], position[1]].set_title(str(position))
plt.savefig('SOM/results/winner_distribution' + apdx + '.png')    

#  Make a spyder plot of averages in cognitive profiles
labels_map = som.labels_map(x, [label_names[t] for t in y])
meanProfDf = pd.DataFrame(dict(tests = cogn_tests))
cnt = 1

for position in labels_map.keys():
        meanProf = np.mean(x[(w_x==position[0]) & (w_y==position[1]),:],axis=0)
        meanProfDf[str(position)] = meanProf.tolist()
        
meanProfDf = pd.concat((meanProfDf, meanProfDf.loc[[0]]), axis=0)
labels = meanProfDf.tests.values        
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(labels))

plt.figure(figsize=(8, 8))
plt.subplot(polar=True)
for position in labels_map.keys():
    plt.plot(label_loc, meanProfDf[str(position)], label=str(position))

lines, labels = plt.thetagrids(np.degrees(label_loc), labels=cogn_tests + [''])
plt.legend()
plt.savefig('SOM/results/response_mean_profiles' + apdx + '.png')

#  Split cognitive profiles and plot spyder plots
fig, axs  = plt.subplots(n_neurons,m_neurons,figsize=(20,20), subplot_kw={'projection': 'polar'})
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(labels))
colors = plt.rcParams["axes.prop_cycle"]()

for position in labels_map.keys():
    c = next(colors)["color"]
    axs[position[0], position[1]].plot(label_loc, meanProfDf[str(position)], label=str(position), c=c, alpha=0.75)
    axs[position[0], position[1]].fill(label_loc, meanProfDf[str(position)], label=str(position), c=c, alpha=0.2)
    axs[position[0], position[1]].set_thetagrids(np.degrees(label_loc), labels=cogn_tests + [''])
    axs[position[0], position[1]].set_ylim(-4,2)
    
plt.savefig('SOM/results/se_mean_profiles_subfig' + apdx + '.png')
 
#  Plot mean and standard deviation of cognitive profiles
plt.figure(figsize=(8, 8))

meanProfDf = pd.DataFrame(dict(tests = cogn_tests))
stdProfDf = pd.DataFrame(dict(tests = cogn_tests))


for position in labels_map.keys():
        meanProf = np.mean(x[(w_x==position[0]) & (w_y==position[1]),:],axis=0)
        meanProfDf[str(position)] = meanProf.tolist()
        
        stdProf = np.std(x[(w_x==position[0]) & (w_y==position[1]),:],axis=0)
        stdProfDf[str(position)] = stdProf.tolist()

for position in labels_map.keys():
    plt.errorbar(cogn_tests, meanProfDf[str(position)],stdProfDf[str(position)], label=str(position))

plt.grid()
plt.xticks(rotation = 45,ha="right")
plt.legend()
plt.savefig('SOM/results/se_mean_profiles_lines' + apdx + '.png')