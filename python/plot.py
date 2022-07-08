#from braille_reading_rsnn_Loihi_encoding import run_neuralnetwork
MNparams_dict = [
    ['A2B',0,0,0],
    ['C2J',5,0,0],
    ['K',30,0,0],
    ['L',30,10,-0.6],
    ['M2O',5,10,-0.6],
    ['P2Q',5,5,-0.3],
    ['R',0,8,-0.1],
    ['S',5,-3,0.5],
    ['T',-80,0,0]
]
import matplotlib.pyplot as plt
fig1,axis1 = plt.subplots(nrows = 1,ncols = 1)
import numpy as np
import pickle
collection = []
#for i in range(len(MNparams_dict))[2:]:
   # accuracy_list,spk_input = run_neuralnetwork(a = MNparams_dict[i][1],A1=MNparams_dict[i][2],A2 = MNparams_dict[i][3])
   # axis1.plot(np.array(accuracy_list).T[:,0],label = MNparams_dict[i][0])
   # plt.figure()
   # time,idx = np.where(spk_input[0,:,:].cpu().detach().numpy())
   # plt.scatter(time,idx)
   # local_col = [accuracy_list,time,idx]
   # collection.append(local_col)
    #where_to_save = 'data/collection' + str(i) + '.pkl'
    #with open(where_to_save, 'wb') as f:
       # pickle.dump(local_col, f)
# with open('data/collection.pkl', 'wb') as f:
#     pickle.dump(collection, f)
plt.legend()
#plt.show()

#print(accuracy_list)
#plt.plot(accuracy_list)

with open('C:/Users/Startklar/OneDrive/Desktop/INI_burocracy/Viaggi/Telluride/Work/tactile_braille_reading_encoding/python/data/collection0.pkl', 'rb') as f:
    Collection0=pickle.load(f)
    f.close()

with open('C:/Users/Startklar/OneDrive/Desktop/INI_burocracy/Viaggi/Telluride/Work/tactile_braille_reading_encoding/python/data/collection1.pkl', 'rb') as f:
    Collection1=pickle.load(f)
    f.close()

plt.title('Train & Test accuracy')
plt.plot(np.array(Collection0[0][1]).T, color = 'r')
plt.plot(np.array(Collection1[0][1]).T, color = 'b')
plt.plot(np.array(Collection0[0][0]).T, color = 'g')
plt.plot(np.array(Collection1[0][0]).T, color = 'm')
plt.legend(['MN Tonic spiking - Test accuracy', 'MN Adaptation - Test accuracy', 'MN Tonic spiking - Train accuracy', 'MN Adaptation - Train accuracy'], loc ="lower right")
plt.show()