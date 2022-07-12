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
import torch
import io
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
# plt.legend()
#plt.show()

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def plot_trainedMN():
    with open('data/collection_trainMN.pkl','rb') as f:
        if torch.cuda.is_available():
            collection = pickle.load(f)
        else:
            collection = CPU_Unpickler(f).load()
        f.close()

    plt.plot(collection[0][0],label = 'training')
    plt.plot(collection[0][1],label = 'test')
    plt.figure()
    plt.imshow(collection[3][0].detach().cpu().numpy(),label = 'w1')
    plt.imshow(collection[3][1].detach().cpu().numpy(),label = 'w2')
    plt.imshow(collection[3][2].detach().cpu().numpy(),label = 'v1')
    plt.figure()
    import seaborn as nico_ocd
    plt.hist(collection[4][0].detach().cpu().numpy(),label = 'a')
    plt.hist(collection[4][1].detach().cpu().numpy(),label = 'A1')
    plt.hist(collection[4][2].detach().cpu().numpy(),label = 'A2')
    plt.legend()
    plt.xlabel('Neurons')
    plt.ylabel('Value')
    plt.figure()
    params_vector = collection[5]
    a_vector = np.zeros([len(params_vector),len(params_vector[0][0])])
    A1_vector = np.zeros([len(params_vector),len(params_vector[0][0])])
    A2_vector = np.zeros([len(params_vector),len(params_vector[0][0])])
    for epix,epoch in enumerate(params_vector):
        a_vector[epix,:] = epoch[0].detach().cpu().numpy()
        A1_vector[epix, :] = epoch[1].detach().cpu().numpy()
        A2_vector[epix, :] = epoch[2].detach().cpu().numpy()
    plt.plot(a_vector)
    plt.plot(A1_vector)
    plt.plot(A2_vector)
    plt.show()
#print(accuracy_list)
#plt.plot(accuracy_list)
def plot_fixedMN():
    with open('C:/Users/Startklar/OneDrive/Desktop/INI_burocracy/Viaggi/Telluride/Work/tactile_braille_reading_encoding/python/data/collection0.pkl', 'rb') as f:
        Collection0=pickle.load(f)
        f.close()

    with open('C:/Users/Startklar/OneDrive/Desktop/INI_burocracy/Viaggi/Telluride/Work/tactile_braille_reading_encoding/python/data/collection1.pkl', 'rb') as f:
        Collection1=pickle.load(f)
        f.close()

    with open('C:/Users/Startklar/OneDrive/Desktop/INI_burocracy/Viaggi/Telluride/Work/tactile_braille_reading_encoding/python/data/collection2.pkl', 'rb') as f:
        Collection2=pickle.load(f)
        f.close()

    with open('C:/Users/Startklar/OneDrive/Desktop/INI_burocracy/Viaggi/Telluride/Work/tactile_braille_reading_encoding/python/data/collection3.pkl', 'rb') as f:
        Collection3=pickle.load(f)
        f.close()

    with open('C:/Users/Startklar/OneDrive/Desktop/INI_burocracy/Viaggi/Telluride/Work/tactile_braille_reading_encoding/python/data/collection4.pkl', 'rb') as f:
        Collection4=pickle.load(f)
        f.close()

    with open('C:/Users/Startklar/OneDrive/Desktop/INI_burocracy/Viaggi/Telluride/Work/tactile_braille_reading_encoding/python/data/collection5.pkl', 'rb') as f:
        Collection5=pickle.load(f)
        f.close()

    with open('C:/Users/Startklar/OneDrive/Desktop/INI_burocracy/Viaggi/Telluride/Work/tactile_braille_reading_encoding/python/data/collection6.pkl', 'rb') as f:
        Collection6=pickle.load(f)
        f.close()

    with open('C:/Users/Startklar/OneDrive/Desktop/INI_burocracy/Viaggi/Telluride/Work/tactile_braille_reading_encoding/python/data/collection7.pkl', 'rb') as f:
        Collection7=pickle.load(f)
        f.close()

    with open('C:/Users/Startklar/OneDrive/Desktop/INI_burocracy/Viaggi/Telluride/Work/tactile_braille_reading_encoding/python/data/collection8.pkl', 'rb') as f:
        Collection8=pickle.load(f)
        f.close()

    plt.figure()
    plt.title('Test accuracy')
    plt.plot(np.array(Collection0[0][1]).T, color = 'r')
    plt.plot(np.array(Collection1[0][1]).T, color = 'b')
    plt.plot(np.array(Collection2[0][1]).T, color = 'k')
    plt.plot(np.array(Collection3[0][1]).T, color = 'g')
    plt.plot(np.array(Collection4[0][1]).T, color = 'm')
    plt.plot(np.array(Collection5[0][1]).T, color = 'violet')
    plt.plot(np.array(Collection6[0][1]).T, color = 'steelblue')
    plt.plot(np.array(Collection7[0][1]).T, color = 'yellow')
    plt.plot(np.array(Collection8[0][1]).T, color = 'sienna')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Tonic spiking', 'Spike frequency adaptation','Hyperpolarizing spiking','Hyperpolarizing bursting','Tonic bursting', 'Mixed mode', 'Basal bistability', 'Preferred frequency', 'Spike latency'], loc ="lower right")
    plt.figure()
    plt.title('Train accuracy')
    plt.plot(np.array(Collection0[0][0]).T, color = 'r')
    plt.plot(np.array(Collection1[0][0]).T, color = 'b')
    plt.plot(np.array(Collection2[0][0]).T, color = 'k')
    plt.plot(np.array(Collection3[0][0]).T, color = 'g')
    plt.plot(np.array(Collection4[0][0]).T, color = 'm')
    plt.plot(np.array(Collection5[0][0]).T, color = 'violet')
    plt.plot(np.array(Collection6[0][0]).T, color = 'steelblue')
    plt.plot(np.array(Collection7[0][0]).T, color = 'yellow')
    plt.plot(np.array(Collection8[0][0]).T, color = 'sienna')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Tonic spiking', 'Spike frequency adaptation','Hyperpolarizing spiking','Hyperpolarizing bursting','Tonic bursting', 'Mixed mode', 'Basal bistability', 'Preferred frequency', 'Spike latency'], loc ="lower right")
    plt.show()


plot_trainedMN()