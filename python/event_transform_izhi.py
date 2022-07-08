'''
Copyright (C) 2021
Authors: Alejandro Pequeno-Zurro

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from scipy.interpolate import splrep, splev

def gen_spikes(A,scale,a,b,c,d,tau):
    ''' Convert data matrix A into spiking output using the Izhikevich neuron model.
        A: Data matrix where each row of A is a sensor and each column is a sample. The data is assumed to be sampled at 1 kHz.
        scale: gain on sensor output
        a,b,c,d,tau: Izhikevich neuron model parameters.
        Returns matrices of the analog output voltage (v), the recovery variable (u), and a binary spiking output (spikes)
    '''

    num_sens = len(A)
    num_samp = len(A[0])

    v0 = -65
    v = np.empty([num_sens, num_samp+1])
    u0 = v0*b
    u = np.empty([num_sens, num_samp+1])
    spikes = np.zeros([num_sens, num_samp+1],dtype=bool)

    for i in range(num_sens):
        for t in range(num_samp + 1):
            if t == 0:
                v[i][t] = v0
                u[i][t] = u0
            else:
                I = A[i][t-1] * scale
                if v[i][t-1] == 30:
                    v_prev = c
                else:
                    v_prev = v[i][t-1]
                u_prev = u[i][t-1]
                v[i][t] = v_prev + tau*(0.04*(v_prev**2)+(5*v_prev) + 140 - u_prev + I)
                u[i][t] = u_prev + tau*(a*((b*v[i][t])-u_prev))
                if v[i][t] >= 30:
                    v[i][t] = 30
                    u[i][t] = u[i][t] + d
                    spikes[i][t] = 1
    return v, u, spikes

def extract_data_icub_raw_integers(file_name):
    ''' Read the files and convert taxel data and labels
        file_name: filename of the dataset in format dict{'taxel_data':, 'letter':}
    '''
    data = []
    labels = []
    print("file name {}".format(file_name))
    with open(file_name, 'rb') as infile:
        data_dict = pickle.load(infile)
    for item in data_dict:
        dat = np.abs(255 - item['taxel_data'][:])
        data.append(dat)
        labels.append(item['letter'])
    return data, labels

def main():
    ''' Convert time-based data into event-based data '''

    tic = time.perf_counter()

    a_SA = 0.02
    b_SA = 0.2
    c_SA = -65
    d_SA = 8
    tau_SA = 1
    k_SA = 150
    SA_params = np.array([a_SA, b_SA, c_SA, d_SA, tau_SA, k_SA])

    a_RA = 0.02
    b_RA = 0.25
    c_RA = -65
    d_RA = 8
    tau_RA = 0.25
    k_RA = 7.5
    RA_params = np.array([a_RA, b_RA, c_RA, d_RA, tau_RA, k_RA])

    f_raw = 40  # Hz
    f_interp = 1000 # Hz

    full_data_save = True

    data_raw_temp, labels_raw = extract_data_icub_raw_integers('./data/data_braille_letters_raw')
    num_trials = len(data_raw_temp)
    num_samples = len(data_raw_temp[0])
    num_sensors = len(data_raw_temp[0][0])
    time_raw = np.arange(0,num_samples/f_raw,1/f_raw)
    time_interp = np.arange(0,num_samples/f_raw,1/f_interp)
    num_samples_interp = len(time_interp)
    data_raw = np.empty([num_trials, num_sensors,num_samples])
    data_raw_interp = np.empty([num_trials, num_sensors, num_samples_interp])
    for i in range(num_trials):
        num_samples_curr = min(len(data_raw_temp[i]),num_samples)
        for j in range(num_samples_curr):
            for k in range(num_sensors):
                data_raw[i][k][j] = data_raw_temp[i][j][k]
    for i in range(num_trials):
        for j in range(num_sensors):
            spl = splrep(time_raw,data_raw[i][j])
            data_raw_interp[i][j] = abs(splev(time_interp,spl))
    data_raw_max = np.amax(data_raw_interp)
    data_raw_min = np.amin(data_raw_interp)
    data_raw_scaled = (data_raw_interp - data_raw_min)/data_raw_max
    data_params = np.array([f_raw, f_interp, data_raw_max, data_raw_min])
    toc = time.perf_counter()
    print(f"Import, reformat, interpolate, and normalize the raw data: {toc - tic:0.4f} seconds")

    data_SA_v = np.empty([num_trials,num_sensors,num_samples_interp])
    data_SA_u = np.empty([num_trials, num_sensors, num_samples_interp])
    data_SA_spikes = np.empty([num_trials, num_sensors, num_samples_interp],dtype=bool)
    data_RA_v = np.empty([num_trials,num_sensors,num_samples_interp])
    data_RA_u = np.empty([num_trials, num_sensors, num_samples_interp])
    data_RA_spikes = np.empty([num_trials, num_sensors, num_samples_interp],dtype=bool)

    data = list()
    data_dict = {}
    data_dict['SA_params'] = SA_params
    data_dict['RA_params'] = RA_params
    data_dict['data_params'] = data_params
    data.append(data_dict)
    count = 0
    for i in range(num_trials):
        data_dict = {}
        v_SA, u_SA, spikes_SA = gen_spikes(data_raw_scaled[i],k_SA,a_SA,b_SA,c_SA,d_SA,tau_SA)
        data_SA_v[i] = v_SA[:,:-1]
        data_SA_u[i] = u_SA[:,:-1]
        data_SA_spikes[i] = spikes_SA[:,:-1]
        v_RA, u_RA, spikes_RA = gen_spikes(data_raw_scaled[i],k_RA,a_RA,b_RA,c_RA,d_RA,tau_RA)
        data_RA_v[i] = v_RA[:,:-1]
        data_RA_u[i] = u_RA[:,:-1]
        data_RA_spikes[i] = spikes_RA[:,:-1]
        count += 1

        data_dict['letter'] = labels_raw[i]
        data_dict['SA_spikes'] = data_SA_spikes[i]
        data_dict['RA_spikes'] = data_RA_spikes[i]

        if (full_data_save):
            data_dict['raw'] = data_raw[i]
            data_dict['interp'] = data_raw_interp[i]
            data_dict['scaled'] = data_raw_scaled[i]
            data_dict['SA_v'] = data_SA_v[i]
            data_dict['SA_u'] = data_SA_u[i]
            data_dict['RA_v'] = data_RA_v[i]
            data_dict['RA_u'] = data_RA_u[i]
        data.append(data_dict)

        if (count%500 == 0):
            print(count)
    toc2 = time.perf_counter()
    print(f"Convert the data to spiking activity using Izhikevich neuron model (SA and RA): {toc2 - toc:0.4f} seconds")

    spikes_per_trial_SA = np.sum(data_SA_spikes, axis=2)
    spikes_per_taxel_SA = sum(spikes_per_trial_SA)
    spikes_total_SA = sum(spikes_per_taxel_SA)
    spikes_per_trial_RA = np.sum(data_RA_spikes, axis=2)
    spikes_per_taxel_RA = sum(spikes_per_trial_RA)
    spikes_total_RA = sum(spikes_per_taxel_RA)

    events_filename_out = './data/data_braille_letters_izhi'
    full_data_filename_out = './data/data_braille_letters_izhi_full'

    if (full_data_save):
        with open(full_data_filename_out, 'wb') as outf:
            pickle.dump(data, outf)
    else:
        with open(events_filename_out, 'wb') as outf:
            pickle.dump(data, outf)

    toc3 = time.perf_counter()
    print(f"Save the converted data in a new file: {toc3 - toc2:0.4f} seconds")


if __name__ == "__main__":
    main()