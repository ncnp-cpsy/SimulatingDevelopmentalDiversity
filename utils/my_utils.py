import time
import math
# import vis_funcs as vis_funcs
# import utils.vis_funcs as vis_funcs
import numpy as np
import pandas as pd

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def simple_optimizer(rnn, lr):
    for param in rnn.parameters():
        nparam.data -= lr * param.grad.data

def saving_tensor3d(tnsr, out_dir_name, prefix, suffix=None, delimiter=','):
    if tnsr.device.type != 'cpu': tnsr = tnsr.cpu()
    for i in range(tnsr.size(0)):
        if suffix == None: suffix = str(i)
        np.savetxt(out_dir_name + prefix + suffix,
                   tnsr[i,:,:], delimiter=delimiter, fmt='%.18f')
    return

def save_all_vars(all_vars, out_dir_name, prefix_dict=None, suffix='', delimiter=','):
    for key, value in all_vars.items():
        if type(prefix_dict) == dict:
            if key in prefix_dict.keys(): prefix=prefix_dict[key] 
            else: prefix=key
        else: prefix=key
        saving_tensor3d(value, out_dir_name=out_dir_name,
                        prefix=prefix, suffix=suffix, delimiter=delimiter)
        # print('tensor of ', key, ' is saved at ', out_dir_name + prefix + suffix)    

def save_all_vars_learn(all_vars, out_dir_name, prefix_dict=None, suffix='', delimiter=','):
    for key, value in all_vars.items():
        if type(prefix_dict) == dict:
            if key in prefix_dict.keys(): prefix=prefix_dict[key] 
            else: prefix=key
        else: prefix=key
        saving_tensor3d(value[int(suffix),:,:].view(1,value.size(1), value.size(2)), out_dir_name=out_dir_name,
                        prefix=prefix, suffix=suffix, delimiter=delimiter)
        # print('tensor of ', key, ' is saved at ', out_dir_name + prefix + suffix)    


def _softmax_transform_1dim(data, ref=10, sigma=0.05, lim=[-1, 1], mode='forward'):
    ''' pre-data is 1 dim, post data is ref dimendion'''
    
    rslt = []
    interval = (lim[1] - lim[0]) / (ref - 1)
    reference = [lim[0] + interval * i for i in range(ref)]

    if type(data) is list:
        data = np.array(data)

    if mode == 'forward':
        for row in data:
            temp = np.exp( (- (reference - row) ** 2) / sigma)
            rslt.append(temp / np.sum(temp))

    elif mode == 'inverse':
        for row in data:
            rslt.append(sum(reference * row))
    
    return np.array(rslt)

def softmax_transform(data, ref=10, sigma=0.05, lim=[-1, 1], mode='forward'):
    '''this function is run numpy array
    '''
    if mode == 'forward':
        for i in range(data.shape[1]):
            d = data[:,i]
            new_1dim = _softmax_transform_1dim(data=d, ref=ref, sigma=sigma, lim=lim, mode=mode)
            rslt = new_1dim if i ==0 else np.concatenate([rslt, new_1dim], axis=1)
    elif mode == 'inverse':
        for i in range(int(data.shape[1] / ref)):
            d = data[:, (i * ref):((i + 1) * ref)]
            new_1dim = _softmax_transform_1dim(data=d, ref=ref, sigma=sigma, lim=lim, mode=mode).reshape(data.shape[0],1)
            rslt = new_1dim if i == 0 else np.concatenate([rslt, new_1dim], axis=1)
    else:
        raise Exception('Error!')
    rslt = np.array(rslt)
    return rslt


def make_softmax_dir(data_dir='',
                     out_dir_name='',
                     size=0,
                     read_sep=',',
                     wright_sep=',',
                     fmt='%.18f',
                     trans=True):
    for i in range(size):
        data = np.loadtxt(
            data_dir + str(i),
            delimiter=read_sep,
            skiprows=0)
        data = softmax_transform(data) if trans else data
        np.savetxt(out_dir_name + str(i),
                   data,
                   fmt=fmt,
                   delimiter=wright_sep)


def reza_to_csv(filepath, dataname):
    df = vis_funcs.convertTxtFile(filepath + '/' + dataname)
    df.to_csv(filepath + '_csv/' + dataname, header=False, index=False)
    return

'''
# below is main code for converting of space of tab splited file to csv.
for i in range(11):
    filepath = 'data/hy_sim'
    dataname = 'test' + str(i)
    reza_to_csv(filepath, dataname)
'''

'''
# below is main code for test of softmax transformation.
filename = 'data/lr_by_yy_csv' + '/data' + str(0)
data_pre = np.loadtxt(filename,
                      delimiter=',',
                      # delimiter=' ', # if use space-splited, change this
                      skiprows=0)
data_for = softmax_transform(data=data_pre, ref=10, mode='forward')
data_inv = softmax_transform(data=data_for, ref=10, mode='inverse')

df_pre = pd.DataFrame(data_pre)
df_for = pd.DataFrame(data_for)
df_inv = pd.DataFrame(data_inv)

print(df_pre.describe())
print(df_for.describe())
print(df_inv.describe())
'''
