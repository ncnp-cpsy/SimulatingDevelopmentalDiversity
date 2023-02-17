#!/usr/bin/env python
# coding: utf-8

import math
import re
import pandas as pd
pd.set_option('display.max_columns', 100)
import numpy as np
np.set_printoptions(np.inf)
from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
mpl.use('Agg')
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
# plt.style.use('seaborn-paper')
font = {'family' : 'meiryo'}

summary = {}

def setPlot(fig, title='', title_size=16, adjust=0.9, 
            fpass='../img/', fname='tmp.png', save=False, dpi=200):

    plt.tight_layout()
    fig.suptitle(t=title, fontsize=title_size)
    plt.subplots_adjust(top=adjust)
    
    if save==True:
        filename = fpass + fname
        plt.savefig(filename, dpi=dpi)
    # plt.show()
    plt.close()
    
    return

def get_summary():
    filename = '../mainrslt/py_summary.csv'
    df = pd.DataFrame(summary, index=[''])
    df.to_csv(filename, header=True, index=False)
    return(df)

    
def convertTxtFile(filepass):
    ''' converting txt file to dataframe
    '''
    
    fname = filepass  # fname = "../codeToShare/new0/data/data0"
    
    # read file
    with open(fname, mode="r") as f:
        contents = f.readlines()

    # converting
    for i in range(len(contents)):
        contents[i] = re.sub(r'\s+', ",", contents[i].lstrip().rstrip())
        if i == 0:
            csv = contents[i] + "\n"
        else:
            csv += contents[i] + "\n"

    # write file
    with open("tmp.csv", mode='w') as f:
        f.write(csv)
        
    # converting to data frame
    df = pd.read_csv('tmp.csv', header=None)
    return df


def conbineTxtFile(f1, f2, header_list=()):
    ''' conbining txt file to dataframe
    '''
    
    fname1 = f1
    fname2 = f2
    
    # read file
    with open(fname1, mode="r") as f:
        contents1 = f.readlines()
    with open(fname2, mode="r") as f:
        contents2 = f.readlines()

    # conbining
    
    l = min([len(contents1), len(contents2)])
    for i in range(l):
        contents1[i] = re.sub(r'\s+', ",", contents1[i].lstrip().rstrip())
        contents2[i] = re.sub(r'\s+', ",", contents2[i].lstrip().rstrip())
        if i == 0:
            csv = contents1[i] + "," + contents2[i] + "\n"
        else:
            csv += contents1[i] + "," + contents2[i] + "\n"

    # write file
    with open("tmp.csv", mode='w') as f:
        f.write(csv)
        
    # converting to data frame
    if len(header_list) != 0:
        df = pd.read_csv('tmp.csv', names=header_list)
    elif len(header_list) == 0:
        df = pd.read_csv('tmp.csv', header=None)
    return df


def make_new_row_dist(row):
    
    col_names = ['ctgry_border', 'ctgry_center',
                 'dist_home_euc', 'dist_home_x', 'dist_home_y', 
                 'dist_left_euc', 'dist_left_x', 'dist_left_y', 
                 'dist_right_euc', 'dist_right_x', 'dist_right_y', 
                 'rad_home', 'rad_right', 'rad_left']
    row_new = pd.Series(dict(zip(col_names, [None] * len(col_names))))
    
    # define center and border for H, L, R
    center = {
        'home':{'x':0, 'y':-0.4}, 
        'left':{'x':-0.5, 'y':0.5}, 
        'right':{'x':0.5,'y':0.5}
    }
    border = {'x':0, 'y':0}

    # catergorizing by border
    if(row['x'] >= border['x'] and row['y'] > border['y']):
        row_new['ctgry_border'] = 'R'
    elif(row['x'] < border['x'] and row['y'] > border['y']):
        row_new['ctgry_border'] = 'L'
    elif(row['y'] <= border['y']):
        row_new['ctgry_border'] = 'H'
    else:
        row_new['ctgry_border'] = None
    
    # cal distance between home and (x, y)
    row_new['dist_home_x'] = row['x'] - center['home']['x']
    row_new['dist_home_y'] = row['y'] - center['home']['y']
    row_new['dist_home_euc'] = math.sqrt(row_new['dist_home_x']**2 + row_new['dist_home_y']**2)
    
    # cal distance between left and (x, y)
    row_new['dist_left_x'] = row['x'] - center['left']['x']
    row_new['dist_left_y'] = row['y'] - center['left']['y']
    row_new['dist_left_euc'] = math.sqrt(row_new['dist_left_x']**2 + row_new['dist_left_y']**2)

    # cal distance between right and (x, y)
    row_new['dist_right_x'] = row['x'] - center['right']['x']
    row_new['dist_right_y'] = row['y'] - center['right']['y']
    row_new['dist_right_euc'] = math.sqrt(row_new['dist_right_x']**2 + row_new['dist_right_y']**2)
    
    # catergorizing by border
    dict_tmp = {
        'dist_home_euc': row_new['dist_home_euc'], 
        'dist_left_euc': row_new['dist_left_euc'], 
        'dist_right_euc': row_new['dist_right_euc'], 
               }
    max_key = min(dict_tmp, key=dict_tmp.get)
    if(max_key == 'dist_home_euc'):
        row_new['ctgry_center'] = 'H'
    elif(max_key == 'dist_left_euc'):
        row_new['ctgry_center'] = 'L'
    elif(max_key == 'dist_right_euc'):
        row_new['ctgry_center'] = 'R'
    else:
        row_new['ctgry_center'] = None
    
    # cal radian
    row_new['rad_home'] = math.atan2(row['y'] - center['home']['y'], row['x'] - center['home']['x'])
    row_new['rad_left'] = math.atan2(row['y'] - center['left']['y'], row['x'] - center['left']['x'])
    row_new['rad_right'] = math.atan2(row['y'] - center['right']['y'], row['x'] - center['right']['x'])
        
    return row_new

def make_new_row_move(df):
    
    col_names = ['move_euc', 'move_x', 'move_y', 'rad_move']
    df_new = pd.DataFrame(None, index=df.index, columns=col_names)
    
    for i in range(len(df_new)):
        if(i == 0):
            df_new['move_x'][i] = 0
            df_new['move_y'][i] = 0
            df_new['move_euc'][i] = 0
            df_new['rad_move'][i] = 0
        else:
            df_new['move_x'][i] = df['x'][i] - df['x'][i-1]
            df_new['move_y'][i] = df['y'][i] - df['y'][i-1]
            df_new['move_euc'][i] = math.sqrt(df_new['move_x'][i]**2 + df_new['move_y'][i]**2)
            df_new['rad_move'][i] = math.atan2(df['y'][i] - df['y'][i-1], df['x'][i] - df['x'][i-1])
    return(df_new[['move_euc', 'move_x', 'move_y', 'rad_move']].astype(np.float64))

def make_df_with_feutures(f):
    
    df = convertTxtFile(filepass=f)
    df.columns=['x', 'y']
    df_new = df.join(make_new_row_move(df))
    df_new = df_new.join(df.apply(make_new_row_dist, axis=1))
    
    return(df_new)

def make_df_tar_with_feutures(f):
    
    df = convertTxtFile(filepass=f)
    df.columns=['x', 'y']
    df_new = df.join(df.apply(make_new_row_dist, axis=1))
    
    return(df_new)

def count_error_move(df):

    df_tmp = df.iloc[:, :]

    error = pd.Series(dict(zip(['L->R', 'R->L', 'L->H->L', 'R->H->R'], [0] * 4)))
    position = 'ctgry_border'
    tmp_now = df_tmp[position][0]
    tmp_pre, tmp_prepre = '', ''

    for i in range(len(df_tmp)-1):

        i = i + 1
        if(df_tmp[position][i] != df_tmp[position][i-1]):

            tmp_prepre = tmp_pre
            tmp_pre = tmp_now
            tmp_now = df_tmp[position][i]

            # print(str(i) + ': ' + tmp_prepre + tmp_pre + tmp_now)

            if(tmp_pre == 'L' and tmp_now == 'R'):
                error['L->R'] = error['L->R'] + 1
            elif(tmp_pre == 'R' and tmp_now == 'L'):
                error['R->L'] = error['R->L'] + 1
            elif(tmp_prepre == 'L' and tmp_pre == 'H' and tmp_now == 'L'):
                error['L->H->L'] = error['L->H->L'] + 1
            elif(tmp_prepre == 'R' and tmp_pre == 'H' and tmp_now == 'R'):
                error['R->H->R'] = error['R->H->R'] + 1
                
    return(error)

def count_error_ctgry(df_out, df_tar):
    
    error = pd.Series(dict(zip(['H', 'L', 'R'], [0] * 3)))
    position = 'ctgry_border'
    
    for i in range(len(df_tar)):
        # print('out: ' + df_out[position][i] + '    tar: ' + df_tar[position][i])
        if df_out[position][i] != df_tar[position][i]:
            error[df_tar[position][i]] = error[df_tar[position][i]] + 1
            
    return(error)

def count_ctgry(df):
    ''' 
    this function returns the number of 'H', 'L', 'R' appeared in dataframe
    '''
    rslt = pd.Series(dict(zip(['H', 'L', 'R'], [0] * 3)))
    position = 'ctgry_border'
    
    for i in range(len(df)):
        rslt[df[position][i]] = rslt[df[position][i]] + 1
            
    return(rslt)

def cal_variance_of_data(f):

    df = make_df_tar_with_feutures(f=f)
    rslt = {}

    # cal variance of home, left, right
    df_tmp = df[df.ctgry_border=='H']
    rslt['home_x_mean'] = np.round(np.mean(df_tmp['dist_home_x']), 2)
    rslt['home_y_mean'] = np.round(np.mean(df_tmp['dist_home_y']), 2)
    rslt['home_x_var'] = np.round(np.std(df_tmp['dist_home_x']), 2)
    rslt['home_y_var'] = np.round(np.std(df_tmp['dist_home_y']), 2)

    df_tmp = df[df.ctgry_border=='L']
    rslt['left_x_mean'] = np.round(np.mean(df_tmp['dist_left_x']), 2)
    rslt['left_y_mean'] = np.round(np.mean(df_tmp['dist_left_y']), 2)
    rslt['left_x_var'] = np.round(np.std(df_tmp['dist_left_x']), 2)
    rslt['left_y_var'] = np.round(np.std(df_tmp['dist_left_y']), 2)

    df_tmp = df[df.ctgry_border=='R']
    rslt['right_x_mean'] = np.round(np.mean(df_tmp['dist_right_x']), 2)
    rslt['right_y_mean'] = np.round(np.mean(df_tmp['dist_right_y']), 2)
    rslt['right_x_var'] = np.round(np.std(df_tmp['dist_right_x']), 2)
    rslt['right_y_var'] = np.round(np.std(df_tmp['dist_right_y']), 2)

    rslt = pd.DataFrame(rslt, index=[None])

    # cal variance of steps
    cnt=0
    steps = None
    flag_LR, flag_H = False, False

    for i, ctg in enumerate(df['ctgry_border']):
        cnt = cnt + 1
        # check
        if (flag_H == False) and (flag_LR == False) and (ctg == 'L' or ctg == 'R'):
            flag_H = True
        if (flag_H == True) and (flag_LR == False) and (ctg == 'H'):
            flag_LR = True
        if (flag_H == True) and (flag_LR == True) and (ctg == 'H'):
            # add count
            if steps == None:
                steps = [cnt-1]
            else:
                steps.append(cnt-1)
            # modosu
            cnt = 0
            flag_LR, flag_H = False, False

    rslt['step_mean'] = np.round(np.mean(steps), 2)
    rslt['step_var'] = np.round(np.std(steps), 2)
    
    return(rslt)


def compare_tar_gen(filepass, tar_path, numSeq, summary_key=''):
    for i in range(numSeq):

        df_out = make_df_with_feutures(f=filepass + str(i))
        df_tar = make_df_tar_with_feutures(f=tar_path + str(i))

        if i==0:
            # cal denominator, which is deviated from df_tar
            rslt_tar = pd.DataFrame(count_ctgry(df=df_tar))
            rslt_tar.columns = ['data0']

            # cal numerator, which is deviated from output of RNN 
            rslt_out = pd.DataFrame(count_error_ctgry(df_out=df_out, df_tar=df_tar)).append(
                pd.DataFrame(count_error_move(df=df_out)))
            rslt_out.columns = ['data0']

        else:
            # cal denominator, which is deviated from df_tar
            df_add_tar = pd.DataFrame(count_ctgry(df=df_tar))
            df_add_tar.columns = ['data' + str(i)]
            rslt_tar = rslt_tar.join(df_add_tar)

            # cal numerator, which is deviated from output of RNN 
            df_add_out = pd.DataFrame(count_error_ctgry(df_out=df_out, df_tar=df_tar)).append(
                pd.DataFrame(count_error_move(df=df_out)))
            df_add_out.columns = ['data' + str(i)]
            rslt_out = rslt_out.join(df_add_out)

    rslt_out = rslt_out.T
    rslt_tar = rslt_tar.T
    rslt = pd.DataFrame(index=rslt_out.index, columns=rslt_out.columns)

    total = 0
    for i, x in enumerate(rslt_out.index):
        for j, y in enumerate(rslt_out.columns):
            if y == 'H' or y == 'R' or y == 'L':
                per = np.round(rslt_out.at[x, y]/rslt_tar.at[x, y]*100, 1)
                total = total + per
                rslt.at[x, y] = str(rslt_out.at[x, y]) + '/' + str(rslt_tar.at[x, y]) + '(' + str(per) + '%)'
            else:
                rslt.at[x, y] = rslt_out.at[x, y]

    if summary_key != '':
        summary[summary_key + '_L<->R_error_sum'] = np.sum(rslt[['L->R', 'R->L']].sum())
        summary[summary_key + '_L<->L_and_R<->R_error_sum'] = np.sum(rslt[['L->H->L', 'R->H->R']].sum())
        summary[summary_key + '_position_error_mean'] = total / (numSeq * 3)

    return(rslt)

def variance_of_data(f, numSeq):
    for i in range(numSeq):
        if i == 0:
            df = pd.DataFrame(cal_variance_of_data(f=f + str(i)))
        else:
            df_add = pd.DataFrame(cal_variance_of_data(f=f + str(i)))
            df = df.append(df_add)
    df.index = ['data' + str(i) for i in range(numSeq)]

    return(df)



##############################
### variance of peak times ###
##############################


def cal_var_peak(f, mode_step=False):
    df = make_df_with_feutures(f=f)
    peaks = []

    for idx, y in zip(df.index, df['y']):
        if (idx == 0):
            y_pre = y
        elif (idx == 1):
            delta = y - y_pre
            y_pre = y
        else:
            delta_pre = delta
            delta = y - y_pre
            y_pre = y
            if delta * delta_pre < 0:
                peaks.append(idx-1)
    not_peaks = list(set(df.index) ^ set(peaks))
    df_peaks_L = df.loc[peaks][df.loc[peaks]['ctgry_border'] == 'L']
    df_peaks_R = df.loc[peaks][df.loc[peaks]['ctgry_border'] == 'R']
    df_peaks_H = df.loc[peaks][df.loc[peaks]['ctgry_border'] == 'H']

    plt.plot(df['x'], df['y'])
    plt.scatter(df_peaks_L['x'], df_peaks_L['y'],
            marker='x', color='blue')
    plt.scatter(df_peaks_R['x'], df_peaks_R['y'],
            marker='o', color='red')
    plt.scatter(df_peaks_H['x'], df_peaks_H['y'],
            marker='^', color='green')
    plt.savefig('tmp.png')
    plt.close()

    steps = []
    for i in range(len(peaks) - 1):
        steps.append(peaks[i+1] - peaks[i])
    # print(steps)
    if mode_step==True:
        return(steps)

    # col_name_org = ['home_x', 'home_y', 'left_x', 'left_y', 'right_x', 'right_y', 'step']
    # col_name = [name + '_mean' for name in col_name_org]
    # col_name.append([name + '_var' for name in col_name_org])
    # rslt = pd.Series([None] * len(col_name), index=col_name)
    rslt = {}

    # mean
    rslt['home_x_mean'] = np.round(np.mean(df_peaks_H['x']), 2)
    rslt['home_y_mean'] = np.round(np.mean(df_peaks_H['y']), 2)
    rslt['left_x_mean'] = np.round(np.mean(df_peaks_L['x']), 2)
    rslt['left_y_mean'] = np.round(np.mean(df_peaks_L['y']), 2)
    rslt['right_x_mean'] = np.round(np.mean(df_peaks_R['x']), 2)
    rslt['right_y_mean'] = np.round(np.mean(df_peaks_R['y']), 2)
    rslt['step_mean'] = np.round(np.mean(steps), 2)

    # variance
    rslt['home_x_var'] = np.round(np.std(df_peaks_H['x']), 2)
    rslt['home_y_var'] = np.round(np.std(df_peaks_H['y']), 2)
    rslt['left_x_var'] = np.round(np.std(df_peaks_L['x']), 2)
    rslt['left_y_var'] = np.round(np.std(df_peaks_L['y']), 2)
    rslt['right_x_var'] = np.round(np.std(df_peaks_R['x']), 2)
    rslt['right_y_var'] = np.round(np.std(df_peaks_R['y']), 2)
    rslt['step_var'] = np.round(np.std(steps), 2)

    return(pd.DataFrame(rslt, index=[None]))

def make_var_df_peak(f, numSeq, summary_key=''):

    for i in range(numSeq):
        if i==0:
            rslt = cal_var_peak(f=f + str(i))
            rslt.index = ['data' + str(i)]
        else:
            tmp = cal_var_peak(f=f + str(i))
            tmp.index = ['data' + str(i)]
            rslt = rslt.append(tmp)

    if summary_key != '':
        col_name = ['home_x_var', 'home_y_var', 'left_x_var', 'left_y_var', 'right_x_var', 'right_y_var']
        # keys = [summary_key + '_peak_' + pos for pos in col_name]
        # values = rslt[col_name].mean()
        # summary.update(zip(keys, values))
        summary[summary_key + '_peak_var_mean'] = rslt[col_name].mean().mean()
        summary[summary_key + '_peak_var_step'] = rslt['step_var'].mean()
        # print(rslt.mean())

    return(rslt)






#####################
### KL divergence ###
#####################

def cal_discrete_freq_2D(f):
    
    lattice = {'col':10, 'row':10}
    lim = {'x':
           {'min': -1,
            'max': 1},
           'y':
           {'min': -1,
            'max': 1}
    }
    rslt = np.zeros((lattice['col'], lattice['row']), dtype = int)

    df = convertTxtFile(filepass=f)
    df.columns = ['x', 'y']

    for point in df.itertuples():
        # print(point)

        for i in range(0, lattice['col']):
            lim_x_min = lim['x']['min'] + ((lim['x']['max'] - lim['x']['min']) / lattice['col']) * i
            lim_x_max = lim['x']['min'] + ((lim['x']['max'] - lim['x']['min']) / lattice['col']) * (i + 1)
            # print('lim_x', lim_x_min, lim_x_max)
        
            for j in range(0, lattice['row']):
                lim_y_min = lim['y']['min'] + ((lim['y']['max'] - lim['y']['min']) / lattice['row']) * j
                lim_y_max = lim['y']['min'] + ((lim['y']['max'] - lim['y']['min']) / lattice['row']) * (j + 1)
                # print('lim_y', lim_y_min, lim_y_max)
            
                if (lim_x_min <= point.x) and (point.x < lim_x_max) and (lim_y_min <= point.y) and (point.y < lim_y_max):
                    rslt[i,j] = rslt[i,j] + 1
                    
    if rslt.sum() == len(df):
        print('not coverd point exist.')

    rslt = rslt / rslt.sum()
    return(rslt)

def cal_discrete_freq_time(f):

    lim = 30
    peaks = np.array(cal_var_peak(f=f, mode_step=True))
    rslt = np.array([0] * lim)

    for i in range(peaks.shape[0]):
        if peaks[i] < lim:
            rslt[peaks[i]] = rslt[peaks[i]] + 1
        else:
            print('not coverer step exist.')
            
    # print(pd.DataFrame(rslt))
    rslt = rslt / rslt.sum()
    
    return(rslt)

def cal_discrete_freq_trans(f):
    df = make_df_with_feutures(f=f)
    position = 'ctgry_border'
    
    col = ['HH', 'HL', 'HR', 'LH', 'LL', 'LR', 'RH', 'RL', 'RR']
    rslt = np.zeros((1,len(col)))
    rslt = pd.DataFrame(rslt, index=[''], columns=col)

    for i in range(len(df)-1):
        pre = df[position][i]
        post = df[position][i+1]

        if pre == 'H':
            if post == 'H':
                rslt['HH'] += 1
            elif post == 'L':
                rslt['HL'] += 1
            elif post == 'R':
                rslt['HR'] += 1
        elif pre == 'L':
            if post == 'H':
                rslt['LH'] += 1
            elif post == 'L':
                rslt['LL'] += 1
            elif post == 'R':
                rslt['LR'] += 1
        elif pre == 'R':
            if post == 'H':
                rslt['RH'] += 1
            elif post == 'L':
                rslt['RL'] += 1
            elif post == 'R':
                rslt['RR'] += 1

    rslt = rslt.to_numpy()
    rslt = rslt / rslt.sum()
    rslt = pd.DataFrame(rslt)
        
    return rslt
    

def plot_freq(pro, mode='2D', fname='tmp'):
    if mode == '2D':
        plt.plot(pro)
    elif mode == 'time':
        plt.plot(pro)
        # vis_funcs.setPlot(fig=fig, title_size=18, adjust=0.9,
        #                  fname='freq_' + fname + '.png', save=save)
    return()

def my_entropy(pro):
    eps = 0.000000001
    pro = np.array(pro) + eps
    tmp = - pro * np.log(pro)
    rslt = tmp.sum()
    return rslt
    
def my_kld(pk, qk):
    eps = 0.000000001
    pro_true = np.array(pk) + eps
    pro_tar = np.array(qk) + eps

    tmp = pro_true * (np.log(pro_true) - np.log(pro_tar))
    rslt = tmp.sum()
    
    return rslt
    
def cal_KLD(true, tar, mode='2D'):

    if (mode == '2D') or (mode == 'entropy'):
        pro_true = cal_discrete_freq_2D(f=true)
        pro_tar = cal_discrete_freq_2D(f=tar)
#        plot_freq(pro=pro_true, fname='true_2D')
#        plot_freq(pro=pro_tar, fname='tar_2D')

        true = []
        tar = []
        
        for i in range(pro_true.shape[0]):
            for j in range(pro_true.shape[1]):
                true.append(pro_true[i, j])
                tar.append(pro_tar[i, j])
        # print(pd.DataFrame(pro_true), pd.DataFrame(pro_tar))
    elif mode =='time':
        true = cal_discrete_freq_time(f=true)
        tar = cal_discrete_freq_time(f=tar)
#        plot_freq(pro=true, fname='true_time', mode='time')
#        plot_freq(pro=tar, fname='tar_time', mode='time')
    elif mode == 'trans':
        true = cal_discrete_freq_trans(f=true)
        tar = cal_discrete_freq_trans(f=tar)
        
    #tmp = stats.entropy(pk=true, qk=tar)
    if mode != 'entropy':
        tmp = my_kld(pk=true, qk=tar)
    elif mode == 'entropy':
        tmp = my_entropy(pro=tar)
    
    return(tmp)



def make_KLD_df(true, tar, numSeq, summary_key=''):
    # true dist is target sequence, tar dist is generated sequence by RNN
    col = ['entropy', '2D', 'time', 'trans']
    row = ['data' + str(i) for i in range(numSeq)]
    df = pd.DataFrame(index=row, columns=col)

    for i, c in enumerate(col):
        for j, r in enumerate(row):
            kld = cal_KLD(true=true + str(j), tar=tar + str(j), mode=c)
            df.loc[r, c] = kld

    if summary_key != '':
        summary[summary_key + '_entropy'] = df['entropy'].mean()
        summary[summary_key + '_kld_2D'] = df['2D'].mean()
        summary[summary_key + '_kld_time'] = df['time'].mean()
        summary[summary_key + '_kld_trans'] = df['trans'].mean()
    
    return(df)

########################
### dist of Z0 units ###
########################

def make_units_dist(filepass='/nas/taka/LR/190905_4/190916_161159/outputsClosed/', numSeq=4, unit='zValuePrior', step=0, summary_key=''):
    df = convertTxtFile(filepass = filepass + unit + str(0))
    row = ['data' + str(i) for i in range(numSeq)]
    col = [unit + str(step) + '_' + str(i) for i in range(len(df.columns))]
    
    rslt = pd.DataFrame(index=row, columns=col)

    for i in range(numSeq):
        df = convertTxtFile(filepass = filepass + unit + str(i))
        tmp = df.iloc[step, :].values
        rslt.loc[row[i], col] = tmp

    if summary_key != '':
        summary[summary_key + '_' + unit + str(step) + '_var'] = np.mean(rslt.std())
        # 列(つまり, 各ユニットごとでシーケンス間の不偏標準偏差をとり，その平均を返すようにしている
        
    return rslt

######################
### sigma activity ###
######################

def sigma_activity(filepass, numSeq, summary_key=''):
    for i in range(numSeq):
        if i == 0:
            df = convertTxtFile(filepass = filepass + str(i))
        else:
            df = df.append(convertTxtFile(filepass = filepass + str(i)))
    if summary_key != '':
        keys = [summary_key + '_sigma_mean_unit' + str(i) for i in range(len(df.columns))]
        values = df.mean().values.tolist()
        add_dic = zip(keys, values)
        summary.update(add_dic)
        summary[summary_key + '_sigma_mean_all_unit'] = np.mean(values)
    
    return None

###################
### convergence ###
###################

def plotConvergence(f, intl=None, intvl=None, suffix='', save=False):
    '''
    '''
    filelist = ["mainrslt/lowerBound", "mainrslt/lowerBoundRecError", 
                "mainrslt/lowerBoundRecTerm", "mainrslt/lowerBoundRegTerm"
               ]
    titlelist = ['lower bound', 'closed reconstruction error',
                 'reconstruction term', 'regularization term'
                ]
    
    fig = plt.figure(figsize=(14, (14/4)*len(filelist)))
    plt.style.use('seaborn-paper')
    palette = plt.get_cmap('Set1')
    
    for i in range(len(filelist)):
        
        df = convertTxtFile(filepass = f + filelist[i])
        if ((intl != None) and (intvl != None)):
            df = df.iloc[intl:intl+intvl,:].values
            
        ax = fig.add_subplot(len(filelist), 1, i+1)
        ax.set_title(titlelist[i], fontsize=16)
        ax.set_xlim(-20, len(df))
        ax.plot(df, marker='', color=palette(i+1), linewidth=1, alpha=0.5, label=titlelist[i])
        ax.legend(loc='lower right', fontsize=12)

    setPlot(fig=fig, title_size=18, adjust=0.9, 
            fname='convergence' + suffix + '.png', save=save)

    return



###################
#### sequences ####
###################

def plotComparingTimeSeries(output, target, numSeq, title='', header=(), save=False):
    
    fig = plt.figure(figsize=(14, 2 * math.ceil(numSeq/2)))
    
    for i in range(numSeq):
        f_out = output + str(i)
        f_target = target + str(i)
        df = conbineTxtFile(f1=f_out, f2 = f_target,  header_list=header)        
        
        ax = fig.add_subplot(math.ceil(numSeq/2), 2, i+1)
        ax.set_title("data " + str(i))
        ax.plot(df, marker='', linewidth=1, alpha=0.5)
        ax.legend(df.columns, loc='lower right')
        
        
    setPlot(fig=fig, 
            title=title, title_size=18, adjust=0.9, 
            fname=title + '_time.png', save=save)
    
    return


def plotComparing2D(output, target, numSeq, title="", lim=1.5, save=False):
    
    fig = plt.figure(figsize=(14, 4 * math.ceil(numSeq/4)))

    for i in range(numSeq):
        f_out = output + str(i)
        f_target = target + str(i)
        df_out = convertTxtFile(filepass=f_out)
        df_tar = convertTxtFile(filepass=f_target)
        
        if i == 0:
            ax1 = fig.add_subplot(math.ceil(numSeq/4), 4, i+1)
            ax1.plot(df_out[0], df_out[1], alpha=0.5, label='RNN')
            ax1.plot(df_tar[0], df_tar[1], alpha=0.5, label='target')
            ax1.set_xlim(-lim, lim)
            ax1.set_ylim(-lim, lim)
            ax1.set_aspect('equal')
            ax1.set_title('data ' + str(i), fontsize = 10)
            ax1.set_xlabel('x', fontsize = 8)
            ax1.set_ylabel('y', fontsize = 8)
            ax1.legend(loc = 'lower right')
        else:
            ax = fig.add_subplot(math.ceil(numSeq/4), 4, i+1, sharex=ax1, sharey=ax1)
            ax.plot(df_out[0], df_out[1], alpha=0.5, label='RNN')
            ax.plot(df_tar[0], df_tar[1], alpha=0.5, label='target')
            ax.set_title('data ' + str(i), fontsize = 10)
            ax.set_aspect('equal')
            ax.set_xlabel('x', fontsize = 8)
            ax.set_ylabel('y', fontsize = 8)
            ax.legend(loc = 'lower right')
        
    setPlot(fig=fig, 
        title=title, title_size=18, adjust=0.8, 
        fname=title + '_2D.png', save=save)
    
    return





##################
### Neural Activity ###
##################


def plotColorMapForNeuralActivity(f, seq, title='', suffix='', mode="outputsClosed/", header=(), intl=None, intvl=None, save=False):
    '''
    '''
    filepass = f
    seqNum = seq
    filelist = ["xValueClosed", "dValuePrior", "zValuePrior", "myuValuePrior", "sigmValuePrior"]
    
    j = 26
    fig = plt.figure(figsize=(14, 3 * len(filelist)))
    gs = GridSpec(len(filelist), j)
    
    # outputs of RNN and target
    f_out = filepass + mode + filelist[0] + str(seqNum)
    f_target = filepass + "data/data" + str(seqNum)
    df = conbineTxtFile(f1=f_out, f2 = f_target, header_list=header)
    if ((intl != None) and (intvl != None)):
        df = convertTxtFile(filepass = filepass + mode + filelist[0] + str(seqNum))
        df = df.iloc[intl:intl+intvl,:].values
    
    ax1 = plt.subplot(gs[0, :(j-2)])
    ax1.set_ylabel("outputs and data " + str(seqNum))
    ax1.set_xlim(-1, len(df)+1)
    ax1.plot(df, marker='', linewidth=1, alpha=0.5)
    ax1.legend(header, loc='lower right')
    
    # neural activity
    for i in range(len(filelist)-1):
        
        k = i + 1
        f = filepass + mode + filelist[k] + str(seqNum)
        df = convertTxtFile(filepass = f)
        if ((intl != None) and (intvl != None)):
            df = df.iloc[intl:intl+intvl,:].values
        
        ax_map = plt.subplot(gs[k, :(j-2)], sharex=ax1)
        ax_bar = plt.subplot(gs[k, (j-1)])
        sns.heatmap(df.T, cmap='OrRd', ax=ax_map, cbar_ax=ax_bar)# , vmin=0, vmax=1.5
        ax_map.set_ylabel(filelist[k])
        
    setPlot(fig=fig, 
            title=title, title_size=18, adjust=0.94,
            fname='neu_act_timeseries' + suffix + str(seqNum) + '.png', 
            save=save)
    return


def plotComparingNeuralActivity(f, numSeq, title="", save=False):

    fig, ax = plt.subplots(nrows=math.ceil(numSeq/2), ncols=2, figsize=(14, 2 * math.ceil(numSeq/2)))

    for i in range(numSeq):
        fname = f + str(i)
        df = convertTxtFile(filepass = fname)
        
        sns.heatmap(df.T, ax=ax[math.floor(i/2), i % 2], vmin=0, vmax=1.5, cmap='OrRd')
        ax[math.floor(i/2), i % 2].set_title("data" + str(i))
        ax[math.floor(i/2), i % 2].set_xlabel('time step')    

    setPlot(fig=fig, 
            title=title, title_size=18, adjust=0.9,
            fname='neu_act_for_many_seq.png', 
            save=save)
    
    return





###################
### free generation ###
###################


def plotFreeGeneration(f, num, initial=100000, interval=500, numVis=8, col=4, title="", lim=1, save=False):

    fig = plt.figure(figsize=(14, 4 * math.ceil(numVis/4)))
    df = convertTxtFile(filepass = f + "outputsClosed200KSteps/xValueClosed" + str(num))
    
    for i in range(numVis):
        
        l = initial + interval * i
        m = l + interval
        
        ax = fig.add_subplot(math.ceil(numVis/col), col, i+1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title('from ' + str(l) + ' to ' + str(m) + ' steps', fontsize = 12)
        ax.set_xlabel('x', fontsize = 10)
        ax.set_ylabel('y', fontsize = 10)
        ax.plot(df.iloc[l:m,0], df.iloc[l:m,1], alpha=0.5) # , size=(7,7)
        
    setPlot(fig=fig, 
            title=title, title_size=18, adjust=0.9,
            fname='free_gen_seq' + str(num) + '.png', 
            save=save)

    return





###########
### PCA ####
###########

def myPCA(f, intl=None, intvl=None):
    df = convertTxtFile(filepass = f)
    pca = PCA(n_components=2)
    if ((intl != None) and (intvl != None)):
        X = pca.fit_transform(df.iloc[intl:intl+intvl,:].values)
    else:
        X = pca.fit_transform(df.iloc[:,:].values)
    embed = pd.DataFrame(X)
    return(embed)

'''
def plotTimeSeriesOfPCA(f, title='PCA'):
'''
'''
    embed = myPCA(f= f)
    embed.plot(figsize=(14,4), title=title)
    return

def plot2DOfPCA(f, title='PCA'):
'''
'''
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(embed.iloc[:,0], embed.iloc[:,1], alpha=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
#    fig.show
    return(fig)
'''

def plot_PCA_2D_for_sequences(f, numSeq, title_ax='data', title_fig='', initial=None, interval=None, save=False):

    fig = plt.figure(figsize=(14, 2 * math.ceil(numSeq/2)))
        
    for i in range(numSeq):
        
        filepass=f + str(i)
        df=myPCA(f=filepass, intl=initial, intvl=interval)
        
        ax = fig.add_subplot(math.ceil(numSeq/4), 4, i+1)
        ax.scatter(df.iloc[:,0], df.iloc[:,1], alpha=0.5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title_ax +str(i))
        
    setPlot(fig=fig, 
            title=title_fig, title_size=18, adjust=0.86,
            fname='PCA_2D_for_mulit_seq_'+title_fig+'.png', 
            save=save)
    
    return

def plotPCAby2DwithDandZ(f, num=1, title='', initial=None, interval=None, save=False):
    '''
    '''
    filepass = f
    seqNum = num
    filelist = ["dValuePrior", "zValuePrior"]
    titlelist = ["d value", "z value"]
    
    fig = plt.figure(figsize=(8, 4))
    
    for i in range(len(filelist)):
        
        df = myPCA(f=filepass + filelist[i] + str(seqNum), intl=initial, intvl=interval)
        
        ax = fig.add_subplot(1, len(filelist), i+1)
        ax.scatter(df.iloc[:,0], df.iloc[:,1], alpha=0.5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(titlelist[i])

    setPlot(fig=fig, 
            title=title, title_size=18, adjust=0.86,
            fname='PCA_2D_for_one_seq_'+title+'.png', 
            save=save)
        
    return

def plotPCAbyTimeSeriesWithDandZ(f, num=1, title='', initial=None, interval=None, save=False):
    '''
    '''
    filepass = f
    seqNum = num
    filelist = ["xValueClosed", "dValuePrior", "zValuePrior"]
    titlelist = ['xValue (outputs of RNN)', 'initial 2 component of dValues', 'initial 2 component of zValues']
    intl = initial
    intvl = interval
    
    fig = plt.figure(figsize=(14, 9))
    
    df = convertTxtFile(filepass = filepass + filelist[0] + str(seqNum))
    if ((intl != None) and (intvl != None)):
        df = df.iloc[intl:intl+intvl,:].values
    ax1 = fig.add_subplot(len(filelist), 1, 1)
    ax1.plot(df)
    ax1.set_title(titlelist[0])
    ax1.set_xlabel('time step')
    plt.tight_layout()
    
    for i in range(len(filelist)-1):
        df = myPCA(f=filepass + filelist[i+1] + str(seqNum), intl=initial, intvl=interval)        
        ax = fig.add_subplot(len(filelist), 1, i+2, sharex=ax1)
        ax.plot(df)
        ax.set_title(titlelist[i+1])
        ax.set_xlabel('time step')


    setPlot(fig=fig, 
            title=title, title_size=18, adjust=0.93,
            fname='PCA_time_'+title+'.png', 
            save=save)
        
    return
