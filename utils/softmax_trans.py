import numpy as np
from my_utils import *

data_dir = 'data/wcst_200306/data'
# data_dir = 'data/wcst_200306/test'
out_dir_name = 'data/wcst_200306_20dim/data'
# out_dir_name = 'data/wcst_200306_20dim/test'
size = 20

for i in range(size):
    data = np.loadtxt(data_dir + str(i),
                      delimiter=' ', skiprows=0)
    data = softmax_transform(data)
    np.savetxt(out_dir_name + str(i), data, fmt='%.18f',
               delimiter=' ')

