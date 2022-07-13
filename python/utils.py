from scipy.interpolate import interp1d
import numpy as np

def interpolate_data(data_dict,interpolate_size = 1000):
    for data in data_dict:
        data['taxel_data_interp'] = []

        for sensor_idx in range(data['taxel_data'].shape[1]):
            time_interp = np.arange(0, len(data_dict[0]['taxel_data'][:, sensor_idx]) - 2,
                                     len(data_dict[0]['taxel_data'][:, sensor_idx])/ interpolate_size)
            old_time = np.arange(0, len(data['taxel_data'][:, sensor_idx]))
            f = interp1d(old_time, data['taxel_data'][:, sensor_idx])
            data['taxel_data_interp'].append(f(time_interp))
        data['taxel_data_interp'] = np.array(data['taxel_data_interp']).T
    return data_dict



def normalize_data (data_dict, scale):
    for i,data in enumerate( data_dict):
        data['taxel_data'] =  (255 - data['taxel_data']) * scale
    return data_dict