import sys

import analysis.util as util
import analysis.visualizer as vis


def analyze_tutorial(
        filepath,
        out_dir,
        epoch,
        train_tar_path=None,
        test_tar_path=None,
        train_true_path=None,
        test_true_path=None,
        num_seq_tar=4,
        num_seq_free=None,
        num_seq_ereg=4,
        skiprows=1500,
        max_rows=2000):

    print('target file: ' + filepath,
          '\nout_dir: ' + out_dir,
          '\nepoch: ' + str(epoch),
          '\ntrain data: ', train_tar_path,
          '\ntest data: ', test_tar_path,
          '\ntrain true data: ', train_true_path,
          '\ntest true data: ', test_true_path)

    # setting
    util.set_out_dir(filepath=out_dir)
    util.clear_summary()
    if train_tar_path is None:
        train_tar_path = filepath + "data/data"
    if test_tar_path is None:
        test_tar_path = filepath + "data/test"
    if num_seq_free is None:
        num_seq_free = num_seq_tar

    # convergence
    vis.plot_convergence(
        f=filepath, save=True,
        summary_key='conv',
        rolling_steps=100,
        rolling_epoch=epoch
    )

    ## Neural Activity For Posterior Generation
    for i in range(num_seq_tar):
        vis.plot_neural_activity_and_true(
            output=filepath + 'posterior_generation/',
            target=train_tar_path,
            true=train_true_path,
            title='learning for data' + str(i),
            suffix='_post_data',
            header=('RNN x', 'RNN y', 'target x', 'target y'),
            seq=i,
            save=True)

    # target reconstruction
    out = filepath + "target_generation/xValueInversed"
    vis.plot_comparing_sequences(
        output=out,
        target=train_tar_path,
        header=('RNN x', 'RNN y', 'target x', 'target y'),
        num_seq=num_seq_tar,
        title='target_generation',
        save=True)
    vis.plot_plane_comparing(
        output=out,
        target=train_tar_path,
        num_seq=num_seq_tar,
        lim=1,
        title='target_generation',
        save=True)

    for i in range(num_seq_tar):
        vis.plot_neural_activity_and_true(
            output=filepath + 'target_generation/',
            target=train_tar_path,
            true=train_true_path,
            title='Target Generation for data' + str(i),
            suffix='_tar_gen_data',
            header=('RNN x', 'RNN y', 'target x', 'target y'),
            seq=i, save=True)
        vis.plot_out_mean_std(
            dir_output=filepath + 'target_generation/',
            num_seq=i,
            target=train_tar_path,
            title='tar_')

    # free generation
    for i in range(num_seq_free):
        vis.plot_free_generation(
            f=filepath + 'free_generation/xValueInversed' + str(i),
            title='free_generation_seq' + str(i),
            intl=0, intvl=200,
            lim=1, save=True)
        vis.plot_neural_activity_and_true(
            output=filepath + 'free_generation/',
            title='free gen from 100k: seq' + str(i),
            suffix='_free_gen_data',
            header=('RNN x', 'RNN y', 'target x', 'target y'),
            skiprows=skiprows,
            max_rows=max_rows,
            seq=i, save=True)

    # error regression for train data
    out = filepath + "error_regression/xValueInversed_1step_pred_train"
    tar = train_tar_path
    vis.plot_comparing_sequences(
        output=out,
        target=tar,
        header=('RNN x', 'RNN y', 'target x', 'target y'),
        num_seq=num_seq_tar,
        title='ereg_train',
        save=True)
    vis.plot_plane_comparing(
        output=out,
        target=tar,
        num_seq=num_seq_tar,
        lim=1,
        title='ereg_train',
        save=True)

    for i in range(num_seq_tar):
        vis.plot_neural_activity_and_true(
            output=filepath + 'error_regression/',
            target=train_tar_path,
            true=train_true_path,
            title='Error Regression for data' + str(i),
            suffix='_ereg_test_',
            header=('RNN x', 'RNN y', 'target x', 'target y'),
            seq=i,
            read_suffix='_1step_pred_train',
            save=True
        )

    # error regression for test data
    out = filepath + "error_regression/xValueInversed_1step_pred_test"
    tar = test_tar_path
    vis.plot_comparing_sequences(
        output=out,
        target=tar,
        header=('RNN x', 'RNN y', 'target x', 'target y'),
        num_seq=num_seq_ereg,
        title='ereg_test',
        save=True)
    vis.plot_plane_comparing(
        output=out,
        target=tar,
        num_seq=num_seq_ereg,
        lim=1,
        title='ereg_test',
        save=True)

    for i in range(num_seq_ereg):
        vis.plot_neural_activity_and_true(
            output=filepath + 'error_regression/',
            target=test_tar_path,
            true=test_true_path,
            title='Error Regression for test' + str(i),
            suffix='_ereg_test_',
            header=('RNN x', 'RNN y', 'target x', 'target y'),
            seq=i, read_suffix='_1step_pred_test', save=True)

    return


def test():
    args = sys.argv
    filepath = '~/200630-181818_500/'

    train_tar_path = 'data/wcst/200914/stoch_sm-const_bias-const_small/data/data'
    test_tar_path = 'data/wcst/200914/stoch_sm-const_bias-const_small/data/test'
    train_true_path = 'data/wcst/200914/stoch_sm-const_bias-const_small/data_true/data_true'
    test_true_path = 'data/wcst/200914/stoch_sm-switch_bias-switch_small/data_true/test_true'

    out_dir='../img/'
    # out_dir = args[2]

    # epoch=20000
    epoch = int(args[3])
    num_seq_tar = 4
    num_seq_free = num_seq_tar
    num_seq_ereg = 4
    skiprows = 1500
    max_rows = 2000

    analyze_tutorial(
        filepath=filepath,
        out_dir=out_dir,
        epoch=epoch,
        train_tar_path=train_tar_path,
        test_tar_path=test_tar_path,
        train_test_path=train_true_path,
        test_true_path=test_true_path,
        num_seq_tar=num_seq_tar,
        num_seq_free=num_seq_free,
        num_seq_ereg=num_seq_ereg,
        skiprows=skiprows,
        max_rows=max_rows)
    return


if __name__ == '__main__':
    test()
