from cProfile import label
from threading import Thread
#from MN_neuron import MN_neuron
# from torchvision.datasets import MNIST
# from torchvision.transforms import ToTensor
# from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.interpolate import interp1d
from utils import interpolate_data, normalize_data
from torchviz import make_dot
from torch import nn
from math import sqrt
from collections import namedtuple

'''
MN neuron as in : A Generalized Linear Integrate-and-Fire Neural Model Produces
Diverse Spiking Behaviors 2009 by Ştefan Mihalaş and Ernst Niebur
'''
from torch.utils.data import TensorDataset, DataLoader
from parameters.MN_params import MNparams_dict
from parameters.MN_params import INIT_MODE

if torch.cuda.is_available():
    print("Single GPU detected. Setting up the simulation there.")
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No GPU detected. Running on CPU.")
dtype = torch.float

def upsample(data, n=2):
    shp = data.shape
    tmp = data.reshape(shp + (1,))
    tmp = data.tile((1, 1, 1, n))
    return tmp.reshape((shp[0], n * shp[1], shp[2]))

class Network(object):
    def __init__(self):
        self.parameters = {'time_bin_size': 1,
                          'tau_mem': 0.02,
                          'tau_ratio': 2,
                          'fwd_weight_scale': 1,
                          'weight_scale_factor': 0.01,
                          'reg_spikes': 0.004,
                          'reg_neurons': 0.000001,
                          'nb_input_copies': 4,
                           'nb_encoding':12}

        self.train_params = {}

        self.loss_per_epoch = []
        self.accs_per_epoch = []

        self.generate_dataset()
        self.init_params()

    def plot_spikesplot_spikes(self,who = 'L0'):
        data = self.spikes_out[who].clone().detach().cpu().numpy()
        for sample_idx in range(data.shape[0]):
            t,idx = np.where(data[sample_idx])
            plt.scatter(t,idx+sample_idx*data.shape[2])
        plt.title(who)
    def load_analog_data(self):
        # data structure: [trial number] x ['key'] x [time] x [sensor_nr]
        import gzip
        file_name = 'data/tutorial5_braille_spiking_data.pkl.gz'
        with gzip.open(file_name, 'rb') as infile:
            data_dict = pickle.load(infile)

        max_time = int(54 * 25)  # ms
        time_bin_size = int(self.parameters['time_bin_size'])  # ms
        global time
        time = range(0, max_time, time_bin_size)

        letter_written = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                          'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        nb_channels = data_dict[0]['taxel_data'].shape[1]
        # nb_channels = 12  # We did it because Zenke takes 4 sensors
        # Extract data
        nb_repetitions = 50

        data = []
        labels = []

        data_dict = interpolate_data(data_dict)
        for i, letter in enumerate(letter_written):
            for repetition in np.arange(nb_repetitions):
                idx = i * nb_repetitions + repetition
                dat = 1.0 - data_dict[idx]['taxel_data'][:] / 255
                data.append(dat)
                labels.append(i)

        # Crop to same length
        data_steps = l = np.min([len(d) for d in data])
        data = torch.tensor(np.array([d[:l] for d in data]), dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        # Select nonzero inputs
        # nzid = [1, 2, 6, 10]
        # data = data[:, :, nzid]

        file_name = 'data/data_tactile_processes.pkl'
        with open(file_name, 'wb') as f:
            ...
            pickle.dump(data, f)

        file_name = 'data/labels_tactile_processes.pkl'
        with open(file_name, 'wb') as f:
            ...
            pickle.dump(data, f)

        # Standardize data
        rshp = data.reshape((-1, data.shape[2]))
        data = (data - rshp.mean(0)) / (rshp.std(0) + 1e-3)

        nb_upsample = 2
        data = upsample(data, n=nb_upsample)

        # Shuffle data
        idx = np.arange(len(data))
        np.random.seed(0)
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]

        # Peform train/test split
        a = int(0.8 * len(idx))
        x_train, x_test = data[:a], data[a:]
        y_train, y_test = labels[:a], labels[a:]

        ds_train = TensorDataset(x_train, y_train)
        ds_test = TensorDataset(x_test, y_test)

        return ds_train, ds_test, labels, nb_channels, data_steps

    def generate_dataset(self):
        ds_train, ds_test, labels, nb_channels, data_steps = self.load_analog_data()
        self.parameters['data_step'] = data_steps
        self.generator = DataLoader(ds_train, batch_size=128,
                                    shuffle=False, num_workers=2)
        self.parameters['labels'] = labels
        self.parameters['nb_channels'] = nb_channels
        self.parameters['nb_inputs'] = nb_channels*self.parameters['nb_input_copies']
        self.parameters['nb_outputs'] = self.parameters['nb_inputs']
    
    def init_params(self):
        self.tau_syn = self.parameters['tau_mem']/self.parameters['tau_ratio']
        self.parameters['alpha'] = float(np.exp(-self.parameters['time_bin_size'] * 0.001 / self.tau_syn))
        self.parameters['beta'] = float(np.exp(-self.parameters['time_bin_size'] * 0.001 / self.parameters['tau_mem']))

        fwd_weight_scale = self.parameters['fwd_weight_scale']
        rec_weight_scale = self.parameters['weight_scale_factor'] * fwd_weight_scale

        encoder_weight_scale = 1.0
        fwd_weight_scale = 3.0
        rec_weight_scale = 1e-2 * fwd_weight_scale

        # Encoder
        enc_gain = torch.empty((self.parameters['nb_inputs'],), device=device, dtype=dtype, requires_grad=False)
        enc_bias = torch.empty((self.parameters['nb_inputs'],), device=device, dtype=dtype, requires_grad=False)
        torch.nn.init.normal_(enc_gain, mean=0.0, std=encoder_weight_scale)  # TODO update this parameter
        torch.nn.init.normal_(enc_bias, mean=0.0, std=1.0)

        # MN Neurons
        a = torch.empty((self.parameters['nb_inputs'],), device=device, dtype=dtype).to(device)
        torch.nn.init.normal_(a, mean=MNparams_dict[INIT_MODE][0], std=fwd_weight_scale / np.sqrt(self.parameters['nb_inputs']))
        self.parameters['a'] = a

        A1 = torch.empty((self.parameters['nb_inputs'],), device=device, dtype=dtype).to(device)
        torch.nn.init.normal_(A1, mean=MNparams_dict[INIT_MODE][1], std=fwd_weight_scale / np.sqrt(self.parameters['nb_inputs']))
        self.parameters['A1'] = A1

        A2 = torch.empty((self.parameters['nb_inputs'],), device=device, dtype=dtype).to(device)
        torch.nn.init.normal_(A2, mean=MNparams_dict[INIT_MODE][2], std=fwd_weight_scale / np.sqrt(self.parameters['nb_inputs']))
        self.parameters['A2'] = A2

        w1 = torch.empty((self.parameters['nb_inputs'], self.parameters['nb_outputs']), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w1, mean=0.0, std=fwd_weight_scale / np.sqrt(self.parameters['nb_encoding']))
        self.train_params['w1'] = w1

        w2 = torch.empty((self.parameters['nb_inputs'], self.parameters['nb_outputs']), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w2, mean=0.0, std=fwd_weight_scale / np.sqrt(self.parameters['nb_encoding']))
        self.train_params['w2'] = w2
        # neurons.append({'mn' : MN_neuron(params['nb_channels']*params['nb_input_copies'], a, A1, A2).to(device)})

        # # Spiking network
        # layers = []
        # w1 = torch.empty((self.parameters['nb_inputs'], nb_hidden), device=device, dtype=dtype, requires_grad=True)
        # torch.nn.init.normal_(w1, mean=0.0, std=fwd_weight_scale / np.sqrt(self.parameters['nb_inputs']))
        # layers.append(w1)
        #
        # w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
        # torch.nn.init.normal_(w2, mean=0.0, std=fwd_weight_scale / np.sqrt(nb_hidden))
        # layers.append(w2)
        #
        # v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
        # torch.nn.init.normal_(v1, mean=0.0, std=rec_weight_scale / np.sqrt(nb_hidden))
        # layers.append(v1)

    def train(self, nb_epochs=300):
        self.neurons = {}
        self.neurons['MN'] = MN_neuron(self.parameters['nb_inputs'], self.parameters['nb_inputs'],
                                        a=self.parameters['a'], A1=self.parameters['A1'], A2=self.parameters['A2']).to(device)
        self.neurons['LIF'] = LIF_neuron(self.parameters['nb_inputs'], self.parameters['nb_inputs'],alpha = self.parameters['alpha'],beta = self.parameters['beta']).to(device)
        self.train_params['a'] = self.neurons['MN'].a
        self.train_params['A1'] = self.neurons['MN'].A1
        self.train_params['A2'] = self.neurons['MN'].A2


        torch.nn.init.eye_(self.neurons['MN'].linear.weight)

        optimizer = torch.optim.Adamax([self.train_params[param] for param in self.train_params.keys()], lr=1, betas=(0.9, 0.995))  # params['lr'] lr=0.0015

        log_softmax_fn = nn.LogSoftmax(dim=1)  # The log softmax function across output units
        loss_fn = nn.NLLLoss()  # The negative log likelihood loss function
        self.train_params_history = {'w1':[],'a':[],'A1':[],'A2':[],'w2':[]}
        for e in range(nb_epochs):
            self.accs_per_batch = []

            for x_local, y_local in self.generator:
                self.neurons['LIF'].initialize_state()
                self.neurons['MN'].initialize_state()

                x_local, y_local = x_local.to(device), y_local.to(device)
                self.run_snn(x_local)
                n_out_spikes = torch.sum(self.spikes_out['L2'], 1) # sum over time stamps
                log_p_y = log_softmax_fn(n_out_spikes)
                loss_val = loss_fn(log_p_y, y_local)

                optimizer.zero_grad()
                # print('a_grad',self.train_params['a'].grad)
                loss_val.backward()
                optimizer.step()
                self.loss_per_epoch.append(loss_val.item())

                # compare to labels
                _, am = torch.max(n_out_spikes, 1)  # argmax over output units
                # print('am',am)
                tmp = np.mean((y_local == am).detach().cpu().numpy())
                # print(tmp)

                for state in self.neurons['MN'].state:
                    state.detach_()
                self.accs_per_batch.append(tmp)
            # print('a_diff', (self.train_params['a'].clone().detach().cpu().numpy() - prev_a).max())
            # print('A1_diff', (self.train_params['A1'].clone().detach().cpu().numpy() - prev_A1).max())
            # print('A2_diff', (self.train_params['A2'].clone().detach().cpu().numpy() - prev_A2).max())
            # print('w1_diff', (self.train_params['w1'].clone().detach().cpu().numpy() - prev_w1).max())
            for train_key in self.train_params_history.keys():
                self.train_params_history[train_key].append(self.train_params[train_key].clone().detach().cpu().numpy())
            # params_learning_save.append(self.train_params)
            mean_accs = np.mean(self.accs_per_batch)
            print('epoch',e,'acc(%)', mean_accs * 100)
    def run_snn(self,x_local):
        TOTTIME = self.parameters['data_step']
        bs = x_local.shape[0]
        encoder_currents = x_local.tile((self.parameters['nb_input_copies'],))
        V = torch.zeros((bs,x_local.shape[1],encoder_currents.shape[2]), device=device)
        Th = torch.zeros((bs,x_local.shape[1],encoder_currents.shape[2]), device=device)
        i1 = torch.zeros((bs,x_local.shape[1],encoder_currents.shape[2]), device=device)
        i2 = torch.zeros((bs,x_local.shape[1],encoder_currents.shape[2]), device=device)
        spikes = torch.zeros((bs,x_local.shape[1],encoder_currents.shape[2]), device=device)
        self.spikes_out = {}
        for t in range(TOTTIME):

            spikes[:,t,:] = self.neurons['MN'](encoder_currents[:,t,:])
            V[:,t,:] = self.neurons['MN'].state.V
            Th[:,t,:] = self.neurons['MN'].state.Thr
            i1[:,t,:] = self.neurons['MN'].state.i1
            i2[:,t,:] = self.neurons['MN'].state.i2
        self.spikes_out['L0'] = spikes

        # self.plot_spikes()
        # plt.show()
        #
        h1 = torch.einsum("abc,cd->abd", (spikes, self.train_params['w1']))
        V = torch.zeros((bs,h1.shape[1],h1.shape[2]), device=device)
        I = torch.zeros((bs,h1.shape[1],h1.shape[2]), device=device)

        spikes = torch.zeros((bs,h1.shape[1],h1.shape[2]), device=device)
        for t in range(TOTTIME):
            spikes[:,t] = self.neurons['LIF'](h1[:,t])
            V[:,t] = self.neurons['LIF'].state.V
            I[:,t] = self.neurons['LIF'].state.I
        self.spikes_out['L1'] = spikes

        h2 = torch.einsum("abc,cd->abd", (spikes, self.train_params['w2']))
        V = torch.zeros((bs,h2.shape[1],h2.shape[2]), device=device)
        I = torch.zeros((bs,h2.shape[1],h2.shape[2]), device=device)

        spikes = torch.zeros((bs,h2.shape[1],h2.shape[2]), device=device)
        for t in range(TOTTIME):
            spikes[:,t] = self.neurons['LIF'](h2[:,t])
            V[:,t] = self.neurons['LIF'].state.V
            I[:,t] = self.neurons['LIF'].state.I
        self.spikes_out['L2'] = spikes


    def plot_train_params(self):
        for key in self.train_params_history.keys():
            for time in range(len(self.train_params_history[key])):
                plt.plot(self.train_params_history[key][time][:,0])
            plt.title(key)
            plt.figure()

    def save_results(self,name = ''):
        where_to_save = 'data/' + name + 'params.pkl'
        with open(where_to_save, 'wb') as f:
            pickle.dump(self.parameters, f)
        where_to_save = 'data/' + name + 'params_train.pkl'
        with open(where_to_save, 'wb') as f:
            pickle.dump(self.train_params, f)
        where_to_save = 'data/' + name + 'params_train_history.pkl'
        with open(where_to_save, 'wb') as f:
            pickle.dump(self.train_params_history, f)
class MN_neuron(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i1', 'i2', 'Thr'])

    def __init__(self, n_in, n_out, a, A1, A2):
        super(MN_neuron, self).__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.C = 1
        self.EL = -0.07
        self.Vr = -0.07
        self.R1 = 0
        self.R2 = 1
        self.Tr = -0.06
        self.Tinf = -0.05
        self.b = 10  # units of 1/s
        self.G = 50 * self.C  # units of 1/s
        self.k1 = 200  # units of 1/s
        self.k2 = 20  # units of 1/s
        self.dt = 1 / 1000
        self.a = nn.Parameter(torch.ones(n_out).to(device) * a, requires_grad=True).to(device)
        #self.A1 = A1 * self.C
        #self.A2 = A2 * self.C
        self.A1 = nn.Parameter(torch.ones(n_out).to(device) * A1, requires_grad=True).to(device)
        self.A2 = nn.Parameter(torch.ones(n_out).to(device) * A2, requires_grad=True).to(device)
        self.state = None
        self.n_out = n_out
    def initialize_state(self):
        self.state = None
    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones(x.shape[0], self.n_out, device=x.device) * self.EL,
                                          i1=torch.zeros(x.shape[0], self.n_out, device=x.device),
                                          i2=torch.zeros(x.shape[0], self.n_out, device=x.device),
                                          Thr=torch.ones(x.shape[0], self.n_out, device=x.device) * self.Tr, )

        # print('V_shape',self.state.V.shape)
        # print('x_shape',x.shape)
        V = self.state.V
        i1 = self.state.i1
        i2 = self.state.i2
        Thr = self.state.Thr

        i1 += -self.k1 * i1 * self.dt
        i2 += -self.k2 * i2 * self.dt
        V += self.dt * (self.linear(x) + i1 + i2 - self.G * (V - self.EL)) / self.C
        Thr += self.dt * (self.a * (V - self.EL) - self.b * (Thr - self.Tinf))

        spk = activation(V - Thr)

        i1 = (1 - spk) * i1 + (spk) * (self.R1 * i1 + self.A1)
        i2 = (1 - spk) * i2 + (spk) * (self.R2 * i2 + self.A2)
        Thr = (1 - spk) * Thr + (spk) * torch.max(Thr, torch.tensor(self.Tr))
        V = (1 - spk) * V + (spk) * self.Vr

        self.state = self.NeuronState(V=V, i1=i1, i2=i2, Thr=Thr)

        return spk

class LIF_neuron(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'I'])

    def __init__(self, n_in, n_out,alpha,beta):
        super(LIF_neuron, self).__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.C = 1
        self.thr = 1
        self.state = None
        self.n_out = n_out
        self.alpha = alpha
        self.beta = beta
        self.dt = 1 / 1000
        self.Vr = 0
    def initialize_state(self):
        self.state = None
    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones(x.shape[0], self.n_out, device=x.device),
                                          I=torch.zeros(x.shape[0], self.n_out, device=x.device),
                                          )

        V = self.state.V
        I = self.state.I
        I = self.alpha * I + self.linear(x)

        V += self.dt * (I - self.beta * (V)) / self.C

        spk = activation(V - self.thr)

        V = (1 - spk) * V + (spk) * self.Vr

        self.state = self.NeuronState(V=V, I = I)

        return spk



class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 20.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


activation = SurrGradSpike.apply
name = 'MN_LIF_LIF'
nico = Network()
nico.train()
nico.save_results(name = name)
nico.plot_train_params()