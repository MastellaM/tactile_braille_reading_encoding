from braille_reading_rsnn_Loihi_encoding import run_neuralnetwork_trainMN
from parameters.MN_params import MNparams_dict
import matplotlib.pyplot as plt
fig1,axis1 = plt.subplots(nrows = 1,ncols = 1)
import numpy as np
import pickle

collection = []

accuracy_list,spk_input,best_layers,best_MNparams,MNparams_coll,layers_update_coll = run_neuralnetwork_trainMN()
axis1.plot(np.array(accuracy_list).T[:,0],label = 'trainMN')
plt.figure()
time,idx = np.where(spk_input[0,:,:].cpu().detach().numpy())
plt.scatter(time,idx)
local_col = [accuracy_list,time,idx,best_layers,best_MNparams,MNparams_coll,layers_update_coll]
collection.append(local_col)
where_to_save = 'data/collection_trainMN.pkl'
with open(where_to_save, 'wb') as f:
    pickle.dump(local_col, f)

# with open('data/collection.pkl', 'wb') as f:
#     pickle.dump(collection, f)
plt.legend()
plt.show()

# print(accuracy_list)


