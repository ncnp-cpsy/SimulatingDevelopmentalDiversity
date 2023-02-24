"""
Change CONFIG class which reflecting learning hyper parameter (etc., meta-prior, the number of neural units). The detail of CONFIG class was written in `config.py`.
"""

import time
import os
from datetime import datetime

from config import ConfigSoftmax as CONFIG
from analysis.analysis_tutorial import analyze_tutorial
from utils.my_utils import timeSince


def main():
    run_one_condition(
        model_class=CONFIG.model_class,
        params=CONFIG.params,
        test_only=True  # change it if do test only.
    )
    print('all_done')
    return


def run_one_condition(model_class,
                      params,
                      test_only=False,
                      epoch_list_test=None,
                      do_plot=True):
    start = time.time()
    print('#####################',
          '\n### start time: ',
          datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
          '\n### model class: ',
          model_class.__name__,
          '\n### out dir: ',
          params['out_dir_name'])
    model = model_class(params=params)

    if not test_only:
        # train and test
        model.train()
        print('### finish time (train and test): ',
              datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
              '\n### running time: (train and test)', timeSince(start))
    else:
        # test only
        epoch_list = epoch_list_test if epoch_list_test is not None \
                     else [int(tmp * params['test_every']) for tmp \
                           in range(1, int(params['epoch_size'] / \
                                           params['test_every'] + 1))]

        for epoch in epoch_list:
            model_path = params['model_path'] if 'model_path' in params.keys() \
                         else params['out_dir_name'] + 'saves/model_state_dict_' + str(epoch) + '.pth'
            print('### model path: ', model_path)
            model.load_model(saved_epoch=epoch,
                             model_path=model_path,
                             load_params=False)
            model.test(epoch=epoch)
            model.sample_all_z(epoch=epoch)

        print('### finish time (test): ',
              datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
              '\n### running time: (test)', timeSince(start))

    if do_plot:
        plot_rslt(model=model, params=params)

    print('### finish time (one condition): ',
          datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
          '\n### running time: (one condition)',
          timeSince(start))

    return


def plot_rslt(model, params):
    # plot results # use model.plot_all_dir() method as alternative
    # model.plot_all_dir()

    for epoch in [int(tmp * params['test_every']) for tmp \
                  in range(1, int(params['epoch_size'] / params['test_every'] + 1))]:
        os.system(
            'mkdir ' + params['out_dir_name'] + str(epoch) + '/img')
        os.system(
            'cp -r ' + \
            params['out_dir_name'] + 'mainrslt ' + \
            params['out_dir_name'] + str(epoch) + '/')

        analyze_tutorial(
            filepath=params['out_dir_name'] + str(epoch) + '/',
            out_dir=params['out_dir_name'] + str(epoch) + '/img/',
            epoch=epoch,
            train_tar_path=params['train_data_path'],
            test_tar_path=params['test_data_path'],
            train_true_path=params['train_true_path'],
            test_true_path=params['test_true_path'],
            num_seq_tar=params['test_data_size_tar'],
            num_seq_ereg=params['test_data_size_ereg_test']
        )

    return


if __name__ == '__main__':
    main()
