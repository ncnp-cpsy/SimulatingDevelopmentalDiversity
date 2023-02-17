import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_2D_sequence(df, max_time_step=None, out_dir='./tmp.png'):
    fig = plt.figure(figsize=(16, 4))
    gs = GridSpec(1, 3)

    ax1 = plt.subplot(gs[0, 0])
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    if max_time_step is None:
        ax1.plot(df.iloc[:, 0], df.iloc[:, 1], alpha=0.5)
    if max_time_step is not None:
        ax1.plot(df.iloc[:max_time_step, 0],
                 df.iloc[:max_time_step, 1],
                 alpha=0.5)

    ax2 = plt.subplot(gs[0, 1:3], sharey=ax1)
    ax2.set_xlabel('time step')
    if max_time_step is not None:
        ax2.set_xlim(0, max_time_step)
    ax2.plot(df)

    # plt.show()
    plt.savefig(out_dir)
    plt.close()


def plot_sequence(df, out_dir='/tmp.png'):
    plt.figure(figsize=(16, 4))
    df.plot()
    plt.savefig(out_dir)
    plt.close('all')


def plot_matrix(filepass, out_dir, col=9, row=8):
    fig = plt.figure(figsize=(2 * col, 2 * row))

    for i in range(row * col):
        df = pd.read_csv(filepass + str(i), header=None)
        print('now: ', i)

        ax1 = fig.add_subplot(row, col, i + 1)
        ax1.plot(df.iloc[:, 0], df.iloc[:, 1], alpha=0.5)
        ax1.set_xlim(-1.0, 1.0)
        ax1.set_ylim(-1.0, 1.0)
        ax1.set_aspect('equal')
        ax1.set_title('data ' + str(i), fontsize=10)
        ax1.set_xlabel('x', fontsize=8)
        ax1.set_ylabel('y', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir)
    plt.close()


def plot_losses(loss, out_dir):
    plt.figure()
    plt.plot(loss)
    plt.savefig(out_dir)
    plt.close('all')
