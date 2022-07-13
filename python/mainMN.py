from cProfile import label
from threading import Thread
#from MN_neuron import MN_neuron
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
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
        self.a = nn.Parameter(torch.ones(n_out) * a, requires_grad=True)
        #self.A1 = A1 * self.C
        #self.A2 = A2 * self.C
        self.A1 = nn.Parameter(torch.ones(n_out) * A1, requires_grad=True)
        self.A2 = nn.Parameter(torch.ones(n_out) * A2, requires_grad=True)
        self.state = None
        self.n_out = n_out

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones(x.shape[0], self.n_out, device=x.device) * self.EL,
                                          i1=torch.zeros(x.shape[0], self.n_out, device=x.device),
                                          i2=torch.zeros(x.shape[0], self.n_out, device=x.device),
                                          Thr=torch.ones(x.shape[0], self.n_out, device=x.device) * self.Tr, )

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

#--------------Load Data--------------------#

data = open("data/data_braille_letters_raw", "rb")
data_unpkl = pickle.load(data)
data.close()

#--------------Normalize & Interpolate Data--------------------#

data_norm= normalize_data(data_unpkl, 0.05)
braile_data = interpolate_data(data_norm)

#--------------Select a random data entry--------------------#

tx_data = braile_data[2000]
data = tx_data['taxel_data_interp']

#--------------Params --------------------#

DEVICE = "cpu"
n_in = np.shape(tx_data['taxel_data_interp'])[1]
BATCH_SIZE = 1
n_out =np.shape(tx_data['taxel_data_interp'])[1]
TOTTIME = np.shape(tx_data['taxel_data_interp'])[0]
a = 5.
A1=5.
A2=5.
epochs = 10

#--------------Instatiate Neurons --------------------#

Net = MN_neuron(n_in, n_out, a = a, A1=A1, A2=A2)
torch.nn.init.eye_(Net.linear.weight)

#--------------Training --------------------#

optimizer = torch.optim.RMSprop(
        params=[Net.a, Net.A1, Net.A2], lr=1, weight_decay=1e-6)
Loss = torch.nn.MSELoss()

list_A1 = []
list_A2 = []
list_a = []
for epoch in range (epochs):
    V = torch.zeros(TOTTIME,n_out, device=DEVICE)
    Th= torch.zeros(TOTTIME,n_out, device=DEVICE)
    i1= torch.zeros(TOTTIME, n_out,device=DEVICE)
    i2= torch.zeros(TOTTIME,n_out, device=DEVICE)
    spikes = torch.zeros(TOTTIME,n_out, device=DEVICE)

    for t in range(TOTTIME):
        input= torch.ones(1,n_in)*(data[t,:])
        spikes[t] = Net(input.float())
        V[t]= Net.state.V
        Th[t]= Net.state.Thr
        i1[t]= Net.state.i1
        i2[t]= Net.state.i2
    optimizer.zero_grad()
    #dot = make_dot(spikes[t], params = {'W': Net.linear.weight, 'a': a})
    #dot.format = 'png'
    #dot.render()
    mse = torch.sum((spikes.sum(dim=0)-torch.ones(n_out)*10)**2)
    mse.backward()
    optimizer.step()
    print("af: {}" .format(Net.a))
    print("A1f: {}".format(Net.A1))
    print("A2f: {}".format(Net.A2))
    list_a.append(Net.a.clone().detach().cpu().numpy())
    list_A1.append(Net.A1.clone().detach().cpu().numpy())
    list_A2.append(Net.A2.clone().detach().cpu().numpy())
   # print("grad: {}" .format(Net.a.grad))
    #print("we: {}" .format(Net.linear.weight))

    for state in Net.state:
        state.detach_()


#--------------Plotting --------------------#

V = V.detach().numpy()
Th = Th.detach().numpy()
i1 =i1.detach().numpy()
i2 = i2.detach().numpy()
spikes = spikes.detach().numpy()

plt.figure()
plt.title("MN response ")
plt.plot(V[:,6] ,  label = "V")
plt.plot(Th[:,6], label = "Th")
plt.legend()

plt.figure()
plt.title("analog inputs")
plt.plot(data)

spikes_arg=np.where(spikes)
plt.figure()
plt.scatter(spikes_arg[0], spikes_arg[1] ,label = "spikes")
plt.title("spiking inputs")



plt.show()