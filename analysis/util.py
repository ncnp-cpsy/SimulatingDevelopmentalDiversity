import re
import math

import pandas as pd
import numpy as np

# _/_/_/ Basic Functions _/_/_/


class CONFIG:
    img_out_dir = './'
    log_out_fname = img_out_dir + 'plotlog.txt'
    summary = {}
    category_name = 'ctgry_border'


def set_out_dir(filepath='./'):
    CONFIG.img_out_dir = filepath
    CONFIG.log_out_fname = CONFIG.img_out_dir + 'plotlog.txt'
    return


def get_out_dir():
    return CONFIG.img_out_dir


def get_log_dir():
    return CONFIG.log_out_fname


def clear_summary():
    CONFIG.summary  = {}
    return


def get_summary():
    return CONFIG.summary


def save_summary(filepath=None):
    if filepath is None:
        filepath = get_out_dir() + 'summary.csv'
    df = pd.DataFrame(CONFIG.summary, index=[''])
    df.to_csv(filepath, header=True, index=False)
    with open(get_log_dir(), mode='a') as f:
        print(df, file=f)
    return df


def get_category_name():
    return CONFIG.category_name


# _/_/_/ Preprocessin _/_/_/


def get_align_length(mode='short', df_out=None, df_tar=None):
    if type(mode) is int:
        length = mode
    elif type(mode) is str:
        if mode == 'out':
            if type(df_out) is pd.DataFrame:
                length = len(df_out)
            else:
                length = None
        elif mode == 'tar':
            if type(df_tar) is pd.DataFrame:
                length = len(df_tar)
            else:
                length = None
        elif mode == 'short':
            if (type(df_out) is pd.DataFrame) and (type(df_tar) is pd.DataFrame):
                if len(df_out) <= len(df_tar):
                    length = len(df_out)
                else:
                    length = len(df_tar)
            elif (not type(df_out) is pd.DataFrame) \
                 and (type(df_tar) is pd.DataFrame):
                length = len(df_tar)
            elif (type(df_out) is pd.DataFrame) \
                 and (not type(df_tar) is pd.DataFrame):
                length = len(df_out)
            else:
                length = None
        else: length = None
    else: length = None

    if type(df_out) is pd.DataFrame: df_out = df_out[:length]
    if type(df_tar) is pd.DataFrame: df_tar = df_tar[:length]

    return df_out, df_tar, length


# _/_/_/ File Loader _/_/_/


def read_sequential_data(filepath, skiprows=0, max_rows=None, sep='reza'):
    if sep == ' ': df = read_ssv(filepath=filepath, skiprows=skiprows, max_rows=max_rows)
    elif sep == ',': df = read_csv(filepath=filepath, skiprows=skiprows, max_rows=max_rows)
    elif sep == 'reza': df = read_reza_format(filepath=filepath, skiprows=skiprows, max_rows=max_rows)
    else: df = None

    return df


def read_ssv(filepath, skiprows=0, max_rows=None):
    data = np.loadtxt(filepath, delimiter=' ',
                      skiprows=skiprows, max_rows=max_rows)
    df = pd.DataFrame(data)
    return df


def read_csv(filepath, skiprows=0, max_rows=None):
    data = np.loadtxt(filepath, delimiter=',',
                      skiprows=skiprows, max_rows=max_rows)
    df = pd.DataFrame(data)
    return df


def read_reza_format(filepath, skiprows=0, max_rows=None):
    ''' converting txt file (reza-san's format) to dataframe
    '''

    fname = filepath
    # print(fname, skiprows, max_rows)

    # read file
    contents = []
    with open(fname) as f:
        for i, line in enumerate(f):
            if i >= skiprows:
                content = re.sub(r'\s+', ",", line.lstrip().rstrip())
                contents.append(content)

                if i == skiprows:
                    csv = contents[i - skiprows] + "\n"
                else:
                    csv += contents[i - skiprows] + "\n"
            if max_rows != None:
                if i == max_rows - 1:
                    break

    # write file
    # tmp_fname = "tmp_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".csv"
    tmp_fname = get_out_dir() + "tmp.csv"
    with open(tmp_fname, mode='w') as f:
        f.write(csv)

    # converting to data frame
    df = pd.read_csv(tmp_fname, header=None)
    # print(df.shape)

    return df


def convert_reza_to_csv(filepath, dataname):
    df = read_sequential_data(filepath + '/' + dataname)
    df.to_csv(filepath + '_csv/' + dataname, header=False, index=False)
    return


def conbine_reza_files(f1, f2, header_list=()):
    ''' conbining txt file (reza-san's format) to dataframe
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


def make_df_tar_with_features(f, skiprows=0, max_rows=None):
    df = read_sequential_data(
        filepath=f,
        skiprows=skiprows,
        max_rows=max_rows)
    df.columns=['x', 'y']
    df_new = df.join(df.apply(make_new_row_dist, axis=1))
    return(df_new)


def make_new_row_dist(row):
    col_names = [
        'ctgry_border', 'ctgry_center', 'dist_home_euc', 'dist_home_x',
        'dist_home_y', 'dist_left_euc', 'dist_left_x', 'dist_left_y',
        'dist_right_euc', 'dist_right_x', 'dist_right_y', 'rad_home',
        'rad_right', 'rad_left'
    ]
    row_new = pd.Series(dict(zip(col_names, [None] * len(col_names))))

    # define center and border for H, L, R
    center = {
        'home': {
            'x': 0.0,
            'y': -0.75,
        },
        'left': {
            'x': -0.4,
            'y': 0.35,
        },
        'right': {
            'x': 0.4,
            'y': 0.35
        }
    }
    border = {'x': 0, 'y': 0}

    # catergorizing by border line
    if (row['x'] >= border['x'] and row['y'] > border['y']):
        row_new['ctgry_border'] = 'R'
    elif (row['x'] < border['x'] and row['y'] > border['y']):
        row_new['ctgry_border'] = 'L'
    elif (row['y'] <= border['y']):
        row_new['ctgry_border'] = 'H'
    else:
        row_new['ctgry_border'] = None

    # cal distance between home and (x, y)
    row_new['dist_home_x'] = row['x'] - center['home']['x']
    row_new['dist_home_y'] = row['y'] - center['home']['y']
    row_new['dist_home_euc'] = math.sqrt(row_new['dist_home_x']**2 +
                                         row_new['dist_home_y']**2)

    # cal distance between left and (x, y)
    row_new['dist_left_x'] = row['x'] - center['left']['x']
    row_new['dist_left_y'] = row['y'] - center['left']['y']
    row_new['dist_left_euc'] = math.sqrt(row_new['dist_left_x']**2 +
                                         row_new['dist_left_y']**2)

    # cal distance between right and (x, y)
    row_new['dist_right_x'] = row['x'] - center['right']['x']
    row_new['dist_right_y'] = row['y'] - center['right']['y']
    row_new['dist_right_euc'] = math.sqrt(row_new['dist_right_x']**2 +
                                          row_new['dist_right_y']**2)

    # catergorizing by border
    dict_tmp = {
        'dist_home_euc': row_new['dist_home_euc'],
        'dist_left_euc': row_new['dist_left_euc'],
        'dist_right_euc': row_new['dist_right_euc'],
    }
    max_key = min(dict_tmp, key=dict_tmp.get)
    if (max_key == 'dist_home_euc'):
        row_new['ctgry_center'] = 'H'
    elif (max_key == 'dist_left_euc'):
        row_new['ctgry_center'] = 'L'
    elif (max_key == 'dist_right_euc'):
        row_new['ctgry_center'] = 'R'
    else:
        row_new['ctgry_center'] = None

    # cal radian
    row_new['rad_home'] = math.atan2(row['y'] - center['home']['y'],
                                     row['x'] - center['home']['x'])
    row_new['rad_left'] = math.atan2(row['y'] - center['left']['y'],
                                     row['x'] - center['left']['x'])
    row_new['rad_right'] = math.atan2(row['y'] - center['right']['y'],
                                      row['x'] - center['right']['x'])

    return row_new
