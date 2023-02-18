import os
import math

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import seaborn as sns

import analysis.util as util

mpl.use('Agg')
# plt.style.use('ggplot')  # 'seaborn-paper'
plt.style.use('seaborn-whitegrid')
font = {'family': 'meiryo'}


def set_plot(fig,
             title='',
             title_size=16,
             adjust=0.9,
             fname='tmp.png',
             format='png',
             save=False,
             dpi=200):

    # plt.tight_layout()
    if not title == "":
        fig.suptitle(t=title, fontsize=title_size)
    # plt.subplots_adjust(top=adjust)

    if save:
        filepath = util.get_out_dir() + fname
        with open(util.get_log_dir(), mode='a') as f:
            print('file is saved as ' + filepath, file=f)
        plt.savefig(filepath, format=format, dpi=dpi)
    # plt.show()
    plt.close('all')

    return


#_/_/_/ convergence _/_/_/


def plot_convergence(f,
                     skiprows=0,
                     max_rows=None,
                     suffix='',
                     save=False,
                     summary_key='',
                     point=None,
                     lim_list=None,
                     rolling_steps=100,
                     rolling_epoch=0):
    '''
    '''
    filelist_tmp = [
        'mainrslt/loss', "mainrslt/elbo_loss", "mainrslt/weighted_elbo_loss",
        "mainrslt/nll_loss", "mainrslt/kld_loss",
        "mainrslt/nonweighted_elbo_loss", "mainrslt/log_prior",
        "mainrslt/log_posterior", "mainrslt/log_generation",
        "mainrslt/normal_elbo_loss", "mainrslt/kld_loss_non_weighted",
        "mainrslt/kld_loss_non_weighted_l0",
        "mainrslt/kld_loss_non_weighted_l1",
        "mainrslt/kld_loss_non_weighted_l2", "mainrslt/kld_loss_init",
        "mainrslt/kld_loss_other", "mainrslt/kld_loss_learn",
        "mainrslt/kld_loss_regl", "mainrslt/valid_loss"
    ]
    titlelist_tmp = [
        'loss', 'lower_bound', "weighted_elbo_loss", 'reconstruction term',
        'regularization term', "nonweighted_elbo_loss", "log_prior",
        "log_posterior", "log_generation", "normal_elbo_loss",
        'KLD Loss (non-weighted-by-MP)', 'KLD Loss L0 (non-weighted-by-MP)',
        'KLD Loss L1 (non-weighted-by-MP)', 'KLD Loss L2 (non-weighted-by-MP)',
        'KLD Loss (only initial step, weighted)', 'KLD Loss (other, weighted)',
        'KLD Loss (Learn of double prior model)',
        'KLD Loss (Regl of double prior model)',
        'validation loss (closed reconstruction error)'
    ]

    filelist, titlelist = [], []
    for i, fname in enumerate(filelist_tmp):
        if os.path.exists(f + fname):
            filelist.append(filelist_tmp[i])
            titlelist.append(titlelist_tmp[i])

    if len(filelist) != 0:
        fig = plt.figure(figsize=(14, 3 * len(filelist)))

        for i in range(len(filelist)):
            df = util.read_sequential_data(filepath=f + filelist[i],
                                           skiprows=skiprows,
                                           max_rows=max_rows)

            # PLOT
            ax = fig.add_subplot(len(filelist), 1, i + 1)
            ax.set_title(titlelist[i], fontsize=16)
            ax.plot(df, marker='', linewidth=1, alpha=0.5, label=titlelist[i])
            if lim_list is not None:
                ax.set_ylim(-0.01, lim_list[i])
            ax.set_xlim(-20, len(df))
            if point is not None:
                for p in point:
                    ax.plot([p, p],
                            ax.get_ylim(),
                            linestyle='dashed',
                            color='red')
            ax.legend(fontsize=12)

            # METRIX
            summ = {}
            summary_str = summary_key + '_' + titlelist[i] + '_' + str(
                rolling_steps) + '-steps_'
            summ[summary_str + 'value'] = df.iloc[rolling_epoch - 1, 0]
            summ[summary_str +
                 'mean'] = df[rolling_epoch -
                              rolling_steps:rolling_epoch].mean()[0]
            summ[summary_str +
                 'std'] = df[rolling_epoch -
                             rolling_steps:rolling_epoch].std()[0]
            util.get_summary().update(summ)
            # print('Degub...', '\n', summary_str,
            # '\ndf.iloc[rolling_epoch-1,0]: ', df.iloc[rolling_epoch-1, 0], 'this is correct!',
            # '\ndf[rolling_epoch-rolling_steps:rolling_epoch]: ',
            # df[rolling_epoch-rolling_steps:rolling_epoch])

        set_plot(fig=fig,
                 title_size=18,
                 adjust=0.9,
                 fname='convergence' + suffix + '.png',
                 save=save)

    else:
        with open(util.get_log_dir(), mode='a') as f:
            print('loss file do not exists', file=f)

    return


def plot_comparing_sequences(output,
                             target,
                             num_seq,
                             title='',
                             header=(),
                             save=False):

    fig = plt.figure(figsize=(14, 2 * math.ceil(num_seq / 2)))

    for i in range(num_seq):
        f_out = output + str(i)
        f_target = target + str(i) if target is not None else None
        df = util.conbine_reza_files(f1=f_out,
                                     f2=f_target,
                                     header_list=header)

        ax = fig.add_subplot(math.ceil(num_seq / 2), 2, i + 1)
        ax.set_title("data " + str(i))
        ax.plot(df, marker='', linewidth=1, alpha=0.5)
        ax.legend(df.columns, loc='lower right')

    set_plot(fig=fig,
             title=title,
             title_size=18,
             adjust=0.9,
             fname=title + '_time.png',
             save=save)

    return


def plot_plane_comparing(output,
                         num_seq,
                         target=None,
                         align_length='short',
                         title="",
                         lim=1.5,
                         save=False):

    fig = plt.figure(figsize=(14, 4 * math.ceil(num_seq / 4)))
    for i in range(num_seq):
        df_out = util.read_sequential_data(filepath=output + str(i))
        df_tar = util.read_sequential_data(
            filepath=target + str(i)) if not target is None else None

        df_out, df_tar, _ = util.get_align_length(mode=align_length,
                                                  df_out=df_out,
                                                  df_tar=df_tar)

        if i == 0:
            ax1 = fig.add_subplot(math.ceil(num_seq / 4), 4, i + 1)
            ax1.plot(df_out[0], df_out[1], alpha=0.5, label='RNN')
            if not df_tar is None:
                ax1.plot(df_tar[0], df_tar[1], alpha=0.5, label='target')
            ax1.set_xlim(-lim, lim)
            ax1.set_ylim(-lim, lim)
            ax1.set_aspect('equal')
            ax1.set_title('data ' + str(i), fontsize=10)
            ax1.set_xlabel('x', fontsize=8)
            ax1.set_ylabel('y', fontsize=8)
            ax1.legend(loc='lower right')
        else:
            ax = fig.add_subplot(math.ceil(num_seq / 4),
                                 4,
                                 i + 1,
                                 sharex=ax1,
                                 sharey=ax1)
            ax.plot(df_out[0], df_out[1], alpha=0.5, label='RNN')
            if not df_tar is None:
                ax.plot(df_tar[0], df_tar[1], alpha=0.5, label='target')
            ax.set_title('data ' + str(i), fontsize=10)
            ax.set_aspect('equal')
            ax.set_xlabel('x', fontsize=8)
            ax.set_ylabel('y', fontsize=8)
            ax.legend(loc='lower right')

    set_plot(fig=fig,
             title=title,
             title_size=18,
             adjust=0.8,
             fname=title + '_2D.png',
             save=save)

    return


def plot_neural_activity_and_true(output,
                                  seq,
                                  target=None,
                                  true=None,
                                  read_suffix='',
                                  title='',
                                  suffix='',
                                  header=(),
                                  vmin_list=None,
                                  vmax_list=None,
                                  align_length='short',
                                  skiprows=0,
                                  max_rows=None,
                                  save=False):

    filepath = output
    filepath_target = target
    filepath_true = true
    num_seq = seq

    filelist_tmp = [
        "xValueInversed", "dValuePrior", "zValuePrior", "myuValuePrior",
        "sigmaValuePrior"
    ]
    if read_suffix != '':
        filelist_tmp = [fname + read_suffix for fname in filelist_tmp]

    filelist = []
    for i, fname in enumerate(filelist_tmp):
        # print(filepath + fname + str(num_seq))
        if os.path.exists(filepath + fname + str(num_seq)):
            filelist.append(filelist_tmp[i])

    j = 26
    fig = plt.figure(figsize=(14, 3 * len(filelist)))
    gs = GridSpec(len(filelist), j)

    # outputs of RNN and target
    df_out = util.read_sequential_data(filepath=filepath + filelist[0] +
                                       str(num_seq),
                                       skiprows=skiprows,
                                       max_rows=max_rows)
    if target is not None:
        df_tar = util.read_sequential_data(
            filepath=filepath_target + str(num_seq),
            skiprows=0,
            max_rows=None
            #skiprows=skiprows, max_rows=max_rows
        )
    else:
        df_tar = None

    df_out, df_tar, max_rows_tmp = util.get_align_length(mode=align_length,
                                                         df_out=df_out,
                                                         df_tar=df_tar)
    if max_rows is None:
        max_rows = max_rows_tmp

    # plot RNN output
    ax1 = plt.subplot(gs[0, :(j - 2)])
    ax1.set_ylabel("outputs and data " + str(num_seq))
    if vmin_list is None and vmax_list is None:
        ax1.set_xlim(-1, max_rows + 1)
        ax1.set_ylim(-1.2, 1.2)
    else:
        ax1.set_xlim(-1, max_rows + 1)
        ax1.set_ylim(vmin_list[0], vmax_list[0])

    # ax1.plot(df_out, marker='', linewidth=1, alpha=0.5)
    df_out.plot(marker='', linewidth=1, alpha=0.5, ax=ax1)
    if not df_tar is None:
        # ax1.plot(df_tar, marker='', linewidth=1, alpha=0.5)
        df_tar.plot(marker='', linewidth=1, alpha=0.5, ax=ax1)

    # converting rnn output to state
    marker = ['|', '1', '2']
    color = ['w', 'r', 'b']

    df_state = util.make_df_tar_with_features(
        f=filepath + filelist[0] + str(num_seq),
        skiprows=skiprows,
        max_rows=max_rows).loc[:, util.get_category_name()]

    df_state = pd.concat([
        pd.DataFrame(df_state),
        pd.DataFrame([-1.0 for _ in df_state], columns=['y_rnn_state']),
        pd.DataFrame([i for i in range(len(df_state))], columns=['step'])
    ],
                         axis=1)
    for i, c in enumerate(set(df_state[util.get_category_name()])):
        m = '|'
        if c == 'H': mc = color[0]
        elif c == 'L': mc = color[1]
        elif c == 'R': mc = color[2]
        df_selected = df_state[df_state[util.get_category_name()] == c]
        ax1.scatter(data=df_selected,
                    x='step',
                    y='y_rnn_state',
                    marker=m,
                    label=c,
                    c=mc,
                    s=30)

    # plot stat of target data
    if target != None:
        df_tar_state = util.make_df_tar_with_features(
            f=filepath_target + str(num_seq),
            skiprows=0,
            max_rows=None
            # skiprows=skiprows, max_rows=max_rows
        ).loc[:, util.get_category_name()]
        df_tar_state = pd.concat([
            pd.DataFrame(df_tar_state),
            pd.DataFrame([-0.9
                          for _ in df_tar_state], columns=['y_tar_state']),
            pd.DataFrame([i for i in range(len(df_tar_state))],
                         columns=['step'])
        ],
                                 axis=1)
        for i, c in enumerate(set(df_tar_state[util.get_category_name()])):
            m = '|'
            if c == 'H': mc = color[0]
            elif c == 'L': mc = color[1]
            elif c == 'R': mc = color[2]
            df_selected = df_tar_state[df_tar_state[util.get_category_name()]
                                       == c]
            ax1.scatter(data=df_selected,
                        x='step',
                        y='y_tar_state',
                        marker=m,
                        label=c,
                        c=mc,
                        s=30)

    # ture variables
    if true != None:
        df_true = pd.read_csv(filepath_true + str(num_seq),
                              skiprows=list(range(1, 1 + skiprows)),
                              nrows=max_rows,
                              header=0,
                              sep=',')
        # df_true = df_true.join(df_true.apply(make_new_row_dist, axis=1))

        if 'omega' not in df_true.columns:
            df_true = pd.read_csv(filepath_true + str(num_seq),
                                  skiprows=list(range(1, 1 + skiprows)),
                                  nrows=max_rows,
                                  header=0,
                                  sep=' ')

        df_true = pd.concat([
            df_true,
            pd.DataFrame([i for i in range(len(df_true))], columns=['step']),
            pd.DataFrame([1.1 for _ in range(len(df_true))],
                         columns=['y_omega']),
            pd.DataFrame([1.0 for _ in range(len(df_true))],
                         columns=['y_theta']),
            pd.DataFrame([0.9 for _ in range(len(df_true))],
                         columns=['y_sigma']),
            pd.DataFrame([-0.9 for _ in range(len(df_true))],
                         columns=['y_true_state'])
        ],
                            axis=1)

        # omega
        ax1.scatter(data=df_true,
                    x='step',
                    y='y_omega',
                    marker='|',
                    label='omega',
                    c=df_true.omega,
                    cmap='OrRd',
                    s=30,
                    norm=Normalize(vmin=0, vmax=5.0))
        # theta
        ax1.scatter(
            data=df_true,
            x='step',
            y='y_theta',
            marker='|',
            label='theta',
            c=df_true.theta,
            cmap='cool',
            s=30,
            #cool_r, cool, OrRd, Greens
            norm=Normalize(vmin=0, vmax=1.0))
        '''
        # sigma
        ax1.scatter(data=df_true, x='step', y='y_sigma',
                    marker='|', label='sigma',
                    c=df_true.target_sigma, cmap='OrRd', s=30,
                    norm=Normalize(vmin=0, vmax=0.06))
        '''
        '''
        # target state using true data
        for i, c in enumerate(set(df_true.state)):
            m = '|' # marker[i]
            # df_selected = df_true[df_true['state'] == c]
            df_selected = df_true[df_true[util.get_category_name()] == c]
            if c == 'Home': mc = color[0]
            elif c == 'Left': mc = color[1]
            elif c == 'Right': mc = color[2]
            ax1.scatter(data=df_selected, x='step', y='y_true_state',
                        marker=m, label=c, c=mc, s=30)
        '''

    ax1.legend(header, loc='lower right')

    # neural activity
    for i in range(len(filelist) - 1):
        k = i + 1
        df = util.read_sequential_data(filepath=filepath + filelist[k] +
                                       str(num_seq),
                                       skiprows=skiprows,
                                       max_rows=max_rows)

        ax_map = plt.subplot(gs[k, :(j - 2)], sharex=ax1)
        ax_bar = plt.subplot(gs[k, (j - 1)])
        if vmin_list is None and vmax_list is None:
            sns.heatmap(df.T, cmap='OrRd', ax=ax_map, cbar_ax=ax_bar)
        else:
            sns.heatmap(df.T,
                        cmap='OrRd',
                        ax=ax_map,
                        vmin=vmin_list[k],
                        vmax=vmax_list[k],
                        cbar_ax=ax_bar)
        ax_map.set_ylabel(filelist[k])

    if type(skiprows) == int and type(max_rows) == int:
        title = title + ': ' + str(skiprows) + '-' + str(max_rows) + 'steps'
    set_plot(fig=fig,
             title=title,
             title_size=18,
             adjust=0.94,
             fname='neural_activity' + suffix + str(num_seq) + '.png',
             save=save)

    return


def plot_free_generation(f,
                         intl=100000,
                         intvl=500,
                         numVis=8,
                         col=4,
                         title="",
                         lim=1,
                         save=False):

    fig = plt.figure(figsize=(14, 4 * math.ceil(numVis / 4)))
    df = util.read_sequential_data(filepath=f,
                                   skiprows=0,
                                   max_rows=intl + intvl * numVis)

    fig = plt.figure(figsize=(14, 4 * math.ceil(numVis / 4)))
    df = util.read_sequential_data(filepath=f,
                                   skiprows=0,
                                   max_rows=intl + intvl * numVis)

    for i in range(numVis):

        l = intl + intvl * i
        m = l + intvl

        ax = fig.add_subplot(math.ceil(numVis / col), col, i + 1)
        ax.set_aspect('equal')
        ax.set_title('from ' + str(l) + ' to ' + str(m) + ' steps',
                     fontsize=12)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.plot(df.iloc[l:m, 0], df.iloc[l:m, 1], alpha=0.5)  # , size=(7,7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    title = title + '_from_' + str(intl) + '_to_' + str(intl + intvl * numVis)
    set_plot(fig=fig,
             title=title,
             title_size=18,
             adjust=0.9,
             fname=title + '.png',
             save=save)

    return


def plot_out_mean_std(dir_output,
                      target=None,
                      num_seq=0,
                      read_suffix='',
                      title='',
                      save=True):
    fname_vars = [
        'xValueInversed', 'meanValue', 'varianceValue', 'zValuePrior',
        'myuValuePrior', 'sigmaValuePrior'
    ]
    fname_vars = [tmp + read_suffix for tmp in fname_vars]

    exist_x_mean = os.path.exists(dir_output + fname_vars[1] + str(num_seq))

    df_x = util.read_sequential_data(filepath=dir_output + fname_vars[0] +
                                     str(num_seq))
    if exist_x_mean:
        df_x_mean = util.read_sequential_data(filepath=dir_output +
                                              fname_vars[1] + str(num_seq))
        df_x_std = util.read_sequential_data(filepath=dir_output +
                                             fname_vars[2] + str(num_seq))
        df_x_std = np.sqrt(df_x_std)

    df_z = util.read_sequential_data(filepath=dir_output + fname_vars[3] +
                                     str(num_seq))
    df_z_mean = util.read_sequential_data(filepath=dir_output + fname_vars[4] +
                                          str(num_seq))
    df_z_std = util.read_sequential_data(filepath=dir_output + fname_vars[5] +
                                         str(num_seq))

    df_tar = util.read_sequential_data(
        filepath=target + str(num_seq)) \
        if not target=='' is None else None

    row_size = len(df_x.columns) + len(df_z.columns)

    fig = plt.figure(figsize=(16, 2.3 * row_size))

    # generated variables
    for i in range(len(df_x.columns)):
        ax = fig.add_subplot(row_size, 1, i + 1)

        arg_mean = df_x_mean.loc[:, i].values if exist_x_mean else None
        arg_std = df_x_std.loc[:, i].values if exist_x_mean else None
        arg_true = df_tar.loc[:, i].values if not df_tar is None else None

        plot_out_mean_std_one_seq(pred=df_x.loc[:, i].values,
                                  ax=ax,
                                  pred_mean=arg_mean,
                                  pred_std=arg_std,
                                  true=arg_true)

    # hidden variables
    for i in range(len(df_z.columns)):
        ax = fig.add_subplot(row_size, 1, len(df_x.columns) + i + 1)
        plot_out_mean_std_one_seq(pred=df_z.loc[:, i].values,
                                  ax=ax,
                                  pred_mean=df_z_mean.loc[:, i].values,
                                  pred_std=df_z_std.loc[:, i].values,
                                  true=None)

    title = title + str(num_seq)
    set_plot(fig=fig,
             title=title,
             title_size=18,
             adjust=0.9,
             fname='seq_std_' + title + '.png',
             save=save)

    return


def plot_out_mean_std_one_seq(pred,
                              ax,
                              pred_mean=None,
                              pred_std=None,
                              true=None):
    # fig, ax = plt.subplots(figsize=(16, 4))

    sns.lineplot(np.arange(pred.shape[0]),
                 pred.flatten(),
                 color="blue",
                 ax=ax,
                 label="prediction")

    if type(pred_mean) == np.ndarray:
        sns.lineplot(np.arange(pred_mean.shape[0]),
                     pred_mean.flatten(),
                     color="black",
                     ax=ax,
                     label="pred_mean")

    if type(pred_std) == np.ndarray:
        ax.plot(np.arange(pred_std.shape[0]),
                (pred - 1.96 * pred_std**(1 / 2)).flatten(),
                alpha=0.3,
                color='gray',
                label=".95 interval")
        ax.plot(np.arange(pred_std.shape[0]),
                (pred + 1.96 * pred_std**(1 / 2)).flatten(),
                alpha=0.3,
                color='gray')

    if type(true) == np.ndarray:
        sns.lineplot(np.arange(true.shape[0]),
                     true.flatten(),
                     color="red",
                     ax=ax,
                     label="true state")

    # ax.set_title("simulation data")
    ax.legend()

    return
