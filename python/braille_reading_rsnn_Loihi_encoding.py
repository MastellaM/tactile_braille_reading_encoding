#!/usr/bin/env python
# coding: utf-8

# ## Notebook intended to run the working networks for Braille reading

# In[181]:

from collections import namedtuple
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import random
import json

from sklearn.metrics import confusion_matrix

# ### Don't forget to select the threshold you want to work with and if you want to use pre-trained weights or train from scratch

# In[182]:

from matplotlib.gridspec import GridSpec
# set variables
# multiple_gpus = True # set to 'True' if more than 1 GPU available
use_nni_weights = False  # set to 'True' for use of weights from NNI optimization
use_seed = False  # set seed to achive reproducable results
threshold = "enc" # possible values are: 1, 2, 5, 10
run = "_3"  # run number for statistics
epochs = 300  # 300 # set the number of epochs you want to train the network here
torch.manual_seed(0)
# In[183]:


if torch.cuda.device_count() > 1:
    gpu_sel = 1
    gpu_av = [torch.cuda.is_available() for ii in range(torch.cuda.device_count())]
    print("Detected {} GPUs. The load will be shared.".format(torch.cuda.device_count()))
    if True in gpu_av:
        if gpu_av[gpu_sel]:
            device = torch.device("cuda:" + str(gpu_sel))
        else:
            device = torch.device("cuda:" + str(gpu_av.index(True)))
        # torch.cuda.set_per_process_memory_fraction(0.25, device=device)
    else:
        device = torch.device("cpu")
else:
    if torch.cuda.is_available():
        print("Single GPU detected. Setting up the simulation there.")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Running on CPU.")

# In[184]:


if use_seed:
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set to {}".format(seed))
else:
    print("Shuffle data randomly")

# In[185]:


dtype = torch.float

# In[186]:


letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# In[187]:


# replace torch.tile() by torch_tile()
def torch_tile(data, reps):
    np_tile = np.tile(data.cpu().detach().numpy(), reps)
    return torch.tensor(np_tile)


# In[188]:


def load_and_extract_augmented(params, file_name, taxels=None, letter_written=letters):
    """ From extract_data_icub_events by Alejandro"""

    max_time = int(54 * 25)  # ms
    time_bin_size = int(params['time_bin_size'])  # ms
    # global time
    # time = range(0, max_time, time_bin_size)
    ## Increase max_time to make sure no timestep is cut due to fractional amount of steps
    global time_step
    time_step = time_bin_size * 0.001
    data_steps = len(time)

    infile = open(file_name, 'rb')
    data_dict = pickle.load(infile)
    infile.close()
    # Extract data
    data = []
    labels = []
    bins = 1000  # [ms] 1000 ms in 1 second
    nchan = len(data_dict[1]['events'])  # number of channels/sensors
    for i, sample in enumerate(data_dict):
        dat = (sample['events'][:])
        events_array = np.zeros([nchan, round((max_time / time_bin_size) + 0.5), 2])
        for taxel in range(len(dat)):
            for event_type in range(len(dat[taxel])):
                if dat[taxel][event_type]:
                    indx = bins * (np.array(dat[taxel][event_type]))
                    indx = np.array((indx / time_bin_size).round(), dtype=int)
                    events_array[taxel, indx, event_type] = 1
        if taxels != None:
            events_array = np.reshape(np.transpose(events_array, (1, 0, 2))[:, taxels, :], (events_array.shape[1], -1))
            selected_chans = 2 * len(taxels)
        else:
            events_array = np.reshape(np.transpose(events_array, (1, 0, 2)), (events_array.shape[1], -1))
            selected_chans = 2 * nchan
        data.append(events_array)
        labels.append(letter_written.index(sample['letter']))

    # return data,labels
    data = np.array(data)
    labels = np.array(labels)

    data = torch.tensor(data, dtype=dtype)
    labels = torch.tensor(labels, dtype=torch.long)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, shuffle=True, stratify=labels,
                                                        random_state=42)  # if fix seed wanted add: random_state=42

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)

    return ds_train, ds_test, labels, selected_chans, data_steps


# In[189]:

# Insert MN_neuron class
class MN_neuron(nn.Module):
    # we save the state of the neuron in a namedtuple (Similar to a dictionary)
    NeuronState = namedtuple('NeuronState', ['V', 'i1', 'i2', 'Thr'])

    def __init__(self, n_in, n_out, a, A1, A2):
        super(MN_neuron, self).__init__()

        self.linear = nn.Linear(n_in, n_out, bias=False)
        # torch.nn.init.eye_(self.linear.weight)
        torch.nn.init.constant_(self.linear.weight, 2.0)
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

        # self.a = nn.Parameter(torch.tensor(a), requires_grad=True)
        self.a = nn.Parameter(torch.ones(1, n_out) * a, requires_grad=True)
        # torch.nn.init.constant_(self.a, a)
        self.A1 = A1 * self.C
        self.A2 = A2 * self.C

        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones(x.shape[0], n_out, device=x.device) * self.EL,
                                          i1=torch.zeros(x.shape[0], n_out, device=x.device),
                                          i2=torch.zeros(x.shape[0], n_out, device=x.device),
                                          Thr=torch.ones(x.shape[0], n_out, device=x.device) * self.Tr)

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


class SurrGradSpike(torch.autograd.Function):
    scale = 20.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


activation = SurrGradSpike.apply

# # Training
# We are gonna train the network to emit 4 and 8 spikes. For that, the loss function is defined as the MSE of the sum of spikes and the objective number of spikes (8 and 4)

# In[ ]:


from torch.nn import parameter

DEVICE = "cpu"
n_in = 1
n_out = 2
BATCH_SIZE = 1
TOTTIME = 1000
Ie = torch.tensor(1.5).view(1, 1)

Net = MN_neuron(n_in, n_out, 5., 0., 0.)

optimizer = torch.optim.SGD(params=Net.parameters(), lr=1e-2)
for epoch in range(200):
    spikes = 0
    voltages = []
    for t in range(100):
        spikes += Net(Ie)
        voltages.append(Net.state.V.clone().detach().cpu().numpy())

    voltages = np.stack(voltages)
    mse = torch.sum((spikes - torch.tensor([8, 4])) ** 2)

    optimizer.zero_grad()
    # Loss = torch.nn.MSELoss()
    # mse = Loss(spikes[t], torch.ones_like(spikes[t]))
    mse.backward()
    optimizer.step()

    for state in Net.state:
        state.detach_()

    # make_dot(mse, params=dict(Net.named_parameters()))

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

plt.plot(voltages[:, 0, :])

# In[ ]:


Net.a

#end of MN neuron
n_in = 32*12
n_out = 32*12
mn = MN_neuron(n_in, n_out, 5, 0, 0).to(device)

def run_snn(inputs, enc_params, layers):
    bs = inputs.shape[0]
    enc = torch.zeros((bs, nb_inputs), device=device, dtype=dtype)
    input_spk = torch.zeros((bs, nb_inputs), device=device, dtype=dtype)
    syn = torch.zeros((bs, nb_hidden), device=device, dtype=dtype)
    mem = -1e-3 * torch.ones((bs, nb_hidden), device=device, dtype=dtype)
    out = torch.zeros((bs, nb_hidden), device=device, dtype=dtype)

    enc_rec = []
    mem_rec = []
    spk_rec = []
    spk_input = []
    # enc_gain = enc_params[0]braille_reading_rsnn_Loihi
    # enc_bias = enc_params[1]
    # encoder_currents = torch.einsum("abc,c->ab", (inputs.tile((enc_fan_out,)), enc_gain))+enc_bias
    encoder_currents = enc_params[0] * (inputs.tile((nb_input_copies,)) + enc_params[1]) #TODO Understand why the enc_bias is inside the par.
    for t in range(nb_steps):
        # Compute encoder activity24
        # Input layer with CUBA neurons
        #new_enc = (beta * enc + (1.0 - beta) * encoder_currents[:, t]) * (1.0 - input_spk.detach())
        #new_enc = (beta * enc + encoder_currents[:, t]) * (1.0 - input_spk.detach())
        #input_spk = spike_fn(new_enc - 1.0) #TODO check if is new_enc or enc?

        #Input layer with MN neurons
        input_spk = mn(encoder_currents[:, t])
        print()
        tx, idx = np.where(input_spk[0].cpu().detach().numpy())
        plt.figure(figsize=(12,12))
        plt.scatter(tx, idx)
        plt.show()

        # Compute hidden layer activity
        #h1 = encoder_currents[:, t].mm(layers[0]) + torch.einsum("ab,bc->ac", (out, layers[2]))
        h1 = input_spk.mm(layers[0]) + torch.einsum("ab,bc->ac", (out, layers[2]))
        enc = new_enc
        # Up to here is what i pasted from zenke, dear lyes

        # LK: leak and integrate
        new_syn = alpha * syn + h1
        new_mem = beta * mem + new_syn
        # new_mem = beta*mem + new_syn*(1 - spk_rec[-1]) if t != 0 else alpha*syn + h1

        # LK: fire
        mthr = new_mem - 1.0
        out = spike_fn(mthr)
        rst = out.detach()  # We do not want to backprop through the reset

        mem = new_mem * (1.0 - rst)
        syn = new_syn

        enc_rec.append(new_enc)
        mem_rec.append(mem)
        spk_rec.append(out)
        spk_input.append(input_spk)

    # Now we merge the recorded membrane potentials into a single tensor
    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)
    enc_rec = torch.stack(enc_rec, dim=1)
    spk_input = torch.stack(spk_input, dim =1)
    #plt.plot(mem_rec[0,:,:].cpu().detach().numpy())
    #tx, idx = np.where(spk_rec[0].cpu().detach().numpy())
    #plt.figure(figsize=(12,12))
    #plt.scatter(tx, idx)
    #plt.show()
    time_range = np.array([i for i in range(494)])
    # plt.figure()
    counter = 0
    # colors = ['b','g','y','r']
    # for j in range(4):
    #     for i in range(24):
    #         spikes = spk_rec[j,:,i].cpu().detach().numpy()[spk_rec[j,:,i].cpu().detach().numpy() > 0.0]
    #         time_spikes = time_range[spk_rec[j,:,i].cpu().detach().numpy()>0.0]
    #         plt.plot(time_spikes,spikes+counter,'.',color = colors[j])
    #         counter += 1
    # plt.show()
    # print(layers[1].shape)
    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, layers[1]))
    flt = torch.zeros((bs, nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((bs, nb_outputs), device=device, dtype=dtype)
    s_out_rec = [out]  # out is initialized as zeros, so it is fine to start with this
    out_rec = [out]
    flt_rec = []
    for t in range(nb_steps):
        # LK: leak and integrate
        new_flt = alpha * flt + h2[:, t]
        new_out = beta * out + new_flt

        # LK: fire
        mthr_out = new_out - 1.0
        s_out = spike_fn(mthr_out)
        rst_out = s_out.detach()

        flt = new_flt
        out = new_out * (1.0 - rst_out)

        out_rec.append(out)
        s_out_rec.append(s_out)
        flt_rec.append(flt)
    out_rec = torch.stack(out_rec, dim=1)
    s_out_rec = torch.stack(s_out_rec, dim=1)
    flt_rec = torch.stack(flt_rec, dim = 1)
    other_recs = [mem_rec, spk_rec, s_out_rec]
    layers_update = layers
    enc_params_update = enc_params
    # print(enc_params_update[0].shape)
    # fig = plt.figure(dpi=150, figsize=(7, 3))
    # plot_voltage_traces(enc_rec[:, :, :24], spk_rec[:, :, :24], color="black", alpha=0.2)
    # plt.show()
    tx, idx = np.where(s_out_rec[0].cpu().detach().numpy())
    # plt.figure(figsize=(12,12))
    # print(idx.shape)
    # plt.scatter(tx, idx)
    # plt.show()
    # plt.imshow(layers[0].cpu().detach().numpy(), aspect='auto')
    # plt.colorbar()
    # plt.figure()
    # plt.plot(mem_rec[0,:,:].cpu().detach().numpy())
    # plt.figure()
    # plt.plot(spk_rec[0,:,:].cpu().detach().numpy())
    # plt.figure()
    # plt.plot(out_rec[0,:,:].cpu().detach().numpy())
    # plt.figure()
    # plt.plot(s_out_rec[0,:,:].cpu().detach().numpy())
    # plt.figure()
    # plt.imshow(layers[1].cpu().detach().numpy(),aspect = 'auto')
    # plt.colorbar()
    # plt.show()
    return out_rec, other_recs, layers_update, enc_params_update


# In[190]:


def load_layers(file, map_location, requires_grad=True, variable=False):
    if variable:

        lays = file

        for ii in lays:
            ii.requires_grad = requires_grad

    else:

        lays = torch.load(file, map_location=map_location)

        for ii in lays:
            ii.requires_grad = requires_grad

    return lays


# In[191]:

def plot_voltage_traces(mem, spk=None, dim=(3,5), spike_height=5, **kwargs):
    gs=GridSpec(*dim)
    if spk is not None:
        dat = 1.0*mem
        dat[spk>0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)
        ax.plot(dat[i], **kwargs)
        ax.axis("off")


### Here, this function is only used to define the global variables to be used in other functions
def build(params):
    global nb_input_copies
    nb_input_copies = params['nb_input_copies']  # Num of spiking neurons used to encode each channel 

    # Network parameters
    global nb_inputs
    nb_inputs = nb_channels * nb_input_copies
    global nb_hidden
    nb_hidden = 450
    global nb_outputs
    nb_outputs = len(np.unique(labels)) + 1
    global nb_steps
    nb_steps = data_steps

    tau_mem = params['tau_mem']  # ms
    tau_syn = tau_mem / params['tau_ratio']

    global alpha
    alpha = float(np.exp(-time_step / tau_syn))
    global beta
    beta = float(np.exp(-time_step / tau_mem))

    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = params['weight_scale_factor'] * fwd_weight_scale


# In[192]:


def build_and_predict(params, x, use_nni_weights):
    x = x.to(device)

    global nb_input_copies
    nb_input_copies = params['nb_input_copies']  # Num of spiking neurons used to encode each channel 

    # Network parameters
    global nb_inputs
    nb_inputs = nb_channels * nb_input_copies
    global nb_hidden
    nb_hidden = 450
    global nb_outputs
    nb_outputs = len(np.unique(labels)) + 1
    global nb_steps
    nb_steps = data_steps

    tau_mem = params['tau_mem']  # ms
    tau_syn = tau_mem / params['tau_ratio']

    global alpha
    alpha = float(np.exp(-time_step / tau_syn))
    global beta
    beta = float(np.exp(-time_step / tau_mem))

    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = params['weight_scale_factor'] * fwd_weight_scale

    # Spiking network
    if use_nni_weights:
        layers = load_layers('../NNI/SpyTorch_layers/best_test_' + file_type + '_thr_' + str(file_thr) + '_ref_' + str(
            file_ref) + '_' + optim_nni_experiment + '.pt', map_location=device)
    else:
        encoder_weight_scale = 1.0
        fwd_weight_scale = 3.0
        rec_weight_scale = 1e-2 * fwd_weight_scale

        # Parameters

        enc_params = []
        # Encoder
        enc_gain = torch.empty((nb_inputs,), device=device, dtype=dtype, requires_grad=False)
        enc_bias = torch.empty((nb_inputs,), device=device, dtype=dtype, requires_grad=False)
        torch.nn.init.normal_(enc_gain, mean=0.0, std=encoder_weight_scale)  # TODO update this parameter
        torch.nn.init.normal_(enc_bias, mean=0.0, std=1.0)

        enc_params.append(enc_gain)
        enc_params.append(enc_bias)

        layers = []

        w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w1, mean=0.0, std=fwd_weight_scale / np.sqrt(nb_inputs))
        layers.append(w1)

        w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w2, mean=0.0, std=fwd_weight_scale / np.sqrt(nb_hidden))
        layers.append(w2)

        v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(v1, mean=0.0, std=rec_weight_scale / np.sqrt(nb_hidden))
        layers.append(v1)

    # Make predictions
    output, _, _ = run_snn(x, enc_params, layers)
    m = torch.sum(others[-1], 1)  # sum over time
    _, am = torch.max(m, 1)  # argmax over output units

    return letters[am.detach().cpu().numpy()[0]]


# In[193]:


def train(params, dataset, lr=0.0015, nb_epochs=300, opt_parameters=None, layers=None, dataset_test=None, params_enc = None):
    ttc_hist = []

    if (opt_parameters != None) & (layers != None):
        parameters = opt_parameters  # The paramters we want to optimize
        layers = layers
    elif (opt_parameters != None) & (layers == None):
        parameters = opt_parameters
        layers = [w1, w2, v1]
    elif (opt_parameters == None) & (layers != None):
        parameters = [w1, w2, v1]
        layers = layers
    elif (opt_parameters == None) & (layers == None):  # default from tutorial 5
        parameters = [w1, w2, v1]
        layers = [w1, w2, v1]

    optimizer = torch.optim.Adamax(parameters, lr=0.0005, betas=(0.9, 0.995))  # params['lr'] lr=0.0015

    log_softmax_fn = nn.LogSoftmax(dim=1)  # The log softmax function across output units
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    generator = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    # The optimization loop
    loss_hist = []
    accs_hist = [[], []]
    layers_mean = [[],[],[]]
    enc_mean = [[],[]]
    for e in range(nb_epochs):
        local_loss = []
        # accs: mean training accuracies for each batch
        accs = []
        for x_local, y_local in generator:
            x_local, y_local = x_local.to(device), y_local.to(device)
            output, recs, layers_update,enc_params_update = run_snn(x_local, params_enc,layers)
            # print(len(layers_update))
            # print(layers_update[0].shape)
            _, spks, _ = recs
            # with output spikes
            m = torch.sum(recs[-1], 1)  # sum over time
            #m = torch.sum(output,1) #sum over time
            log_p_y = log_softmax_fn(m)

            # Here we can set up our regularizer loss
            reg_loss = params['reg_spikes'] * torch.mean(
                torch.sum(spks, 1))  # e.g., L1 loss on total number of spikes (original: 1e-3)
            reg_loss += params['reg_neurons'] * torch.mean(
                torch.sum(torch.sum(spks, dim=0), dim=0) ** 2)  # L2 loss on spikes per neuron (original: 2e-6)

            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) + reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())

            # compare to labels
            _, am = torch.max(m, 1)  # argmax over output units
            tmp = np.mean((y_local == am).detach().cpu().numpy())
            accs.append(tmp)



        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)

        # mean_accs: mean training accuracy of current epoch (average over all batches)
        mean_accs = np.mean(accs)
        accs_hist[0].append(mean_accs)

        # Calculate test accuracy in each epoch
        if dataset_test is not None:
            test_acc, test_ttc = compute_classification_accuracy(
                params,
                dataset_test,
                layers=layers_update,
                early=True,
                enc_params=params_enc
            )
            accs_hist[1].append(test_acc)  # only safe best test
            ttc_hist.append(test_ttc)

        if dataset_test is None:
            # save best training
            if mean_accs >= np.max(accs_hist[0]):
                best_acc_layers = []
                for ii in layers_update:
                    best_acc_layers.append(ii.detach().clone())
        else:
            # save best test
            if np.max(test_acc) >= np.max(accs_hist[1]):
                best_acc_layers = []
                for ii in layers_update:
                    best_acc_layers.append(ii.detach().clone())
        # print(len(layers_update))
        # print(layers_update[0].shape)

        for ii_idx,ii in enumerate(layers_update):
            layers_mean[ii_idx].append(np.mean(ii.cpu().detach().numpy(),axis=1))

        print(enc_params_update[0].shape)
        for jj_idx,jj in enumerate(enc_params_update):
            enc_mean[jj_idx].append(jj.cpu().detach().numpy())
            # print(enc_mean)

        print("Epoch {}/{} done. Train accuracy: {:.2f}%, Test accuracy: {:.2f}% , Loss: {:.2f}.".format(e + 1, nb_epochs,
                                                                                          accs_hist[0][-1] * 100,
                                                                                          accs_hist[1][-1] * 100,mean_loss))

    return loss_hist, accs_hist, best_acc_layers, ttc_hist,layers_mean,enc_mean


# In[194]:

def interpolate_data(data_dict,interpolate_size = 1000):
    from scipy.interpolate import interp1d
    for data in data_dict:
        data['taxel_data_interp'] = []

        for sensor_idx in range(data['taxel_data'].shape[1]):
            time_interp = np.arange(0, len(data_dict[0]['taxel_data'][:, sensor_idx]) - 1,
                                     len(data_dict[0]['taxel_data'][:, sensor_idx])/ interpolate_size)

            old_time = np.arange(0, len(data['taxel_data'][:, sensor_idx]))
            f = interp1d(old_time, data['taxel_data'][:, sensor_idx])
            data['taxel_data_interp'].append(f(time_interp))
        data['taxel_data_interp'] = np.array(data['taxel_data_interp']).T
    return data_dict


def build_and_train(params, ds_train, ds_test, epochs=epochs):
    global nb_input_copies
    nb_input_copies = params['nb_input_copies']  # Num of spiking neurons used to encode each channel 

    # Network parameters
    global nb_inputs
    nb_inputs = nb_channels * nb_input_copies
    global nb_hidden
    nb_hidden = 450
    global nb_outputs
    nb_outputs = len(np.unique(labels)) + 1
    global nb_steps
    nb_steps = data_steps

    tau_mem = params['tau_mem']  # ms
    tau_syn = tau_mem / params['tau_ratio']

    global alpha
    alpha = float(np.exp(-params['time_bin_size']*0.001 / tau_syn))
    global beta
    beta = float(np.exp(-params['time_bin_size']*0.001 / tau_mem))

    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = params['weight_scale_factor'] * fwd_weight_scale

    encoder_weight_scale = 1.0
    fwd_weight_scale = 3.0
    rec_weight_scale = 1e-2 * fwd_weight_scale

    enc_params = []
    # Encoder
    enc_gain = torch.empty((nb_inputs,), device=device, dtype=dtype, requires_grad=True)
    enc_bias = torch.empty((nb_inputs,), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(enc_gain, mean=0.0, std=encoder_weight_scale)  # TODO update this parameter
    torch.nn.init.normal_(enc_bias, mean=0.0, std=1.0)
    enc_params.append(enc_gain)
    enc_params.append(enc_bias)

    # Spiking network
    layers = []
    w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w1, mean=0.0, std=fwd_weight_scale / np.sqrt(nb_inputs))
    layers.append(w1)

    w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w2, mean=0.0, std=fwd_weight_scale / np.sqrt(nb_hidden))
    layers.append(w2)

    v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(v1, mean=0.0, std=rec_weight_scale / np.sqrt(nb_hidden))
    layers.append(v1)

    layers_init = []
    for ii in layers:
        layers_init.append(ii.detach().clone())

    opt_parameters = [w1, w2, v1,enc_gain,enc_bias]

    # a fixed learning rate is already defined within the train function, that's why here it is omitted
    loss_hist, accs_hist, best_acc_layers, ttc_hist,layers_mean,enc_mean = train(params, ds_train, nb_epochs=epochs, opt_parameters=opt_parameters,
                                                 layers=layers, dataset_test=ds_test,params_enc=enc_params)

    # best training and test at best training
    acc_best_train = np.max(accs_hist[0])  # returns max value
    acc_best_train = acc_best_train * 100
    idx_best_train = np.argmax(accs_hist[0])  # returns index of max value
    acc_test_at_best_train = accs_hist[1][idx_best_train] * 100

    # best test and training at best test
    acc_best_test = np.max(accs_hist[1])
    acc_best_test = acc_best_test * 100
    idx_best_test = np.argmax(accs_hist[1])
    acc_train_at_best_test = accs_hist[0][idx_best_test] * 100

    print("Final results: \n")
    print("Best training accuracy: {:.2f}% and according test accuracy: {:.2f}% at epoch: {}".format(acc_best_train,
                                                                                                     acc_test_at_best_train,
                                                                                                     idx_best_train + 1))  # only from training
    print("Best test accuracy: {:.2f}% and according train accuracy: {:.2f}% at epoch: {}".format(acc_best_test,
                                                                                                  acc_train_at_best_test,
                                                                                                  idx_best_test + 1))  # only from training

    return loss_hist, accs_hist, best_acc_layers,layers_mean,enc_mean


# In[195]:


def compute_classification_accuracy(params, dataset, layers=None, early=False,enc_params = None):
    """ Computes classification accuracy on supplied data in batches. """

    generator = DataLoader(dataset, batch_size=128,
                           shuffle=False, num_workers=2)
    accs = []
    multi_accs = []
    ttc = None

    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        if layers == None:
            layers = [w1, w2, v1]
            output, others, _,_ = run_snn(x_local, layers)
        else:
            output, others, _,_ = run_snn(x_local, enc_params, layers)
        # with output spikes
        m = torch.sum(others[-1], 1)  # sum over time
        _, am = torch.max(m, 1)  # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)

        if early:
            accs_early = []
            for t in range(output.shape[1] - 1):
                # with spiking output layer
                m_early = torch.sum(others[-1][:, :t + 1, :], 1)  # sum over time
                _, am_early = torch.max(m_early, 1)  # argmax over output units
                # compare to labels
                tmp_early = np.mean((y_local == am_early).detach().cpu().numpy())
                accs_early.append(tmp_early)
            multi_accs.append(accs_early)

    if early:
        mean_multi = np.mean(multi_accs, axis=0)
        if np.max(mean_multi) > mean_multi[-1]:
            if mean_multi[-2] == mean_multi[-1]:
                flattening = []
                for ii in range(len(mean_multi) - 2, 1, -1):
                    if mean_multi[ii] != mean_multi[ii - 1]:
                        flattening.append(ii)
                # time to classify
                ttc = time[flattening[0]]
            else:
                # time to classify
                ttc = time[-1]
        else:
            # time to classify
            ttc = time[np.argmax(mean_multi)]

    return np.mean(accs), ttc


# In[196]:


def ConfusionMatrix(params, dataset, save, layers=None, labels=letters):
    generator = DataLoader(dataset, batch_size=128,
                           shuffle=False, num_workers=2)
    accs = []
    multi_accs = []
    trues = []
    preds = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        if layers == None:
            layers = [w1, w2, v1]
            output, others, _ = run_snn(x_local, layers)
        else:
            output, others, _ = run_snn(x_local, layers)
        # with output spikes
        m = torch.sum(others[-1], 1)  # sum over time
        _, am = torch.max(m, 1)  # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
        trues.extend(y_local.detach().cpu().numpy())
        preds.extend(am.detach().cpu().numpy())

    # return trues, preds

    cm = confusion_matrix(trues, preds, normalize='true')
    cm_df = pd.DataFrame(cm, index=[ii for ii in labels], columns=[jj for jj in labels])
    plt.figure(figsize=(12, 9))
    sn.heatmap(cm_df,
               annot=True,
               fmt='.1g',
               # linewidths=0.005,
               # linecolor='black',
               cbar=False,
               square=False,
               cmap="YlGnBu")
    plt.xlabel('\nPredicted')
    plt.ylabel('True\n')
    plt.xticks(rotation=0)
    plt.show()


# Load augmented Braille data

# In[197]:

def load_spikes():
    file_dir_data = 'data/'
    file_type = 'data_braille_letters'
    file_thr = str(threshold)
    file_ref = 'Null'
    # file_name = file_dir_data + file_type + '_th' + file_thr + '_rp' + file_ref
    file_name = file_dir_data + file_type + '_th' + file_thr

    file_dir_params = 'parameters/'
    param_filename = 'parameters_th' + str(threshold)
    file_name_parameters = file_dir_params + param_filename + '.txt'
    params = {}
    with open(file_name_parameters) as file:
        for line in file:
            (key, value) = line.split()
            if key == 'time_bin_size' or key == 'nb_input_copies':
                params[key] = int(value)
            else:
                params[key] = np.double(value)


# Upsample
def upsample(data, n=2):
    shp = data.shape
    tmp = data.reshape(shp + (1,))
    tmp = data.tile((1, 1, 1, n))
    return tmp.reshape((shp[0], n * shp[1], shp[2]))


def load_analog_data():
    # data structure: [trial number] x ['key'] x [time] x [sensor_nr]
    import gzip
    file_name = 'data/tutorial5_braille_spiking_data.pkl.gz'
    with gzip.open(file_name, 'rb') as infile:
        data_dict = pickle.load(infile)

    max_time = int(54*25) #ms
    time_bin_size = int(params['time_bin_size']) # ms
    global time
    time = range(0,max_time,time_bin_size)


    letter_written = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                      'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # nb_channels = data_dict[0]['taxel_data'].shape[1]
    nb_channels = 12 #We did it because Zenke takes 4 sensors
    # Extract data
    nb_repetitions = 50

    data = []
    labels = []

    # data_dict = interpolate_data(data_dict,interpolate_size=500)
    for i, letter in enumerate(letter_written):
        for repetition in np.arange(nb_repetitions):
            idx = i * nb_repetitions + repetition
            dat = 1.0 - data_dict[idx]['taxel_data'][:] / 255
            data.append(dat)
            labels.append(i)




    # Crop to same length
    data_steps = l = np.min([len(d) for d in data])
    data = torch.tensor(np.array([d[:l] for d in data]), dtype=dtype)
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


# In[198]:


file_dir_params = 'parameters/'
param_filename = 'parameters_th' + str(threshold)
file_name_parameters = file_dir_params + param_filename + '.txt'
params = {}
with open(file_name_parameters) as file:
    for line in file:
        (key, value) = line.split()
        if key == 'time_bin_size' or key == 'nb_input_copies':
            params[key] = int(value)
        else:
            params[key] = np.double(value)


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """

    scale = params['scale']

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

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


spike_fn = SurrGradSpike.apply

# ### Train and test the network

# In[ ]:


if not use_nni_weights:
    ds_train, ds_test, labels, nb_channels, data_steps = load_analog_data()
    # ds_train, ds_test, labels, nb_channels, data_steps = load_and_extract_augmented(params, file_name, letter_written=letters)
    #
    loss_hist, acc_hist, best_layers,layers_mean,enc_mean = build_and_train(params, ds_train, ds_test, epochs=epochs)

# In[ ]:


if not use_nni_weights:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    viridis = cm.get_cmap('viridis', len(layers_mean[0]))
    colors = viridis(np.linspace(0,1,len(layers_mean[0])))
    titles = ['w1','w2','v1']
    for i in range(3):
        plt.figure()

        plt.plot(layers_mean[i])
        plt.title(titles[i])
        plt.xlabel('Epochs(#)')
    titles = ['enc_gain','enc_bias']
    for i in range(2):
        plt.figure()
        plt.plot(enc_mean[i])
        plt.title(titles[i])
        plt.xlabel('Epochs(#)')
    plt.show()
    plt.figure()
    plt.plot(range(1, len(acc_hist[0]) + 1),enc_mean)
    plt.figure()
    plt.plot(range(1, len(acc_hist[0]) + 1), 100 * np.array(acc_hist[0]), color='blue')
    plt.plot(range(1, len(acc_hist[1]) + 1), 100 * np.array(acc_hist[1]), color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(["Training", "Test"], loc='lower right')
    plt.show()

# ### Test the pre-trained network if you already have the pre-trained weights

# In[ ]:


if use_nni_weights:
    # saved from NNI:
    layers = load_layers("weights/SpyTorch_trained_weights_rec_th" + file_thr + run + ".pt", map_location=device)

    print("Input weights matrix: {}x{}".format(len(layers[0]), len(layers[0][0])))
    print("Hidden weights matrix: {}x{}".format(len(layers[2]), len(layers[2][0])))
    print("Output weights matrix: {}x{}".format(len(layers[1]), len(layers[1][0])))

# In[ ]:


if use_nni_weights:
    ds_train, ds_test, labels, nb_channels, data_steps = load_and_extract_augmented(params, file_name,
                                                                                    letter_written=letters)

    build(params)

    test_acc = compute_classification_accuracy(params, ds_test, layers=layers, early=True)

    print("Test accuracy: {}%".format(np.round(test_acc[0] * 100, 2)))
    print("Test accuracy as it comes, without rounding: {}".format(test_acc[0]))

# ### Confusion matrix

# In[ ]:


save = False

if use_nni_weights:
    # from SAVED layers (from NNI) corresponding to best test:
    ConfusionMatrix(params, ds_test, layers=load_layers("weights/SpyTorch_trained_weights_rec_th" + file_thr + ".pt",
                                                        map_location=device), save=save)
else:
    # from the JUST TRAINED layers corresponding to best test:
    ConfusionMatrix(params, ds_test, layers=best_layers, save=save)

# In[ ]:


torch.save(best_layers, "weights/SpyTorch_trained_weights_rec_th" + file_thr + run + ".pt")

# In[ ]:
