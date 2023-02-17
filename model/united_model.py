import os
import copy
import pickle
import time
from datetime import datetime
from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np
import torch

from sklearn import preprocessing

from utils.my_data_loader import *
from utils.my_utils import timeSince, save_all_vars, make_softmax_dir
from utils.plot import *

from model.pvrnn import *
from model.loss import *
from model.regression import *

torch.set_printoptions(threshold=10000)


class Const(object):
    PADDING_TOKEN = -100.0
    FILE_FORMAT = '%.24f'


class Model(metaclass=ABCMeta):
    """
    Classes that provide learning and prediction methods regardless of the
    model type or implementation method, such as RNN or PyTorch.
    """
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass


class ModelRNNTG(Model):
    """This class is training and test for rnn by tani group
    """

    def __init__(self, params):
        """
        constructer
        :param params: hyper parameters for model and training
        """
        super().__init__(params)
        if self.params['max_time_step'] == None:
            self.params['max_time_step'] = self.get_max_length(
                data_dir=self.params['train_data_path'],
                data_size=self.params['data_size'],
                sep=self.params['sep_load'])

        self.out_dir_name = params['out_dir_name']
        self.init_seed(seed=self.params['seed'])
        self.rnn = self.build_model()

        self.save_prefix_dict = None

        return

    def init_seed(self, seed, fixed_gpu_seed=True):
        torch.manual_seed(seed)
        if fixed_gpu_seed:
            if self.params['device'] != 'cpu':
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(seed)
        return

    def make_directries(self):
        epoch_size = self.params['epoch_size']
        test_every = self.params['test_every']

        # directries for saving
        os.system('mkdir -pv ' + self.out_dir_name)
        os.system('mkdir ' + self.out_dir_name + 'saves/')
        os.system('mkdir ' + self.out_dir_name + 'mainrslt/')
        for epoch in [int(tmp * test_every) for tmp in range(1, int(epoch_size / test_every + 1))]:
            self.make_directries_epoch(epoch=epoch)

    def make_directries_epoch(self, epoch):
        print('mkdir ' + self.out_dir_name + str(epoch) + '/learning/')
        os.makedirs(self.out_dir_name + str(epoch), exist_ok=True)
        os.system('mkdir -pv' + self.out_dir_name + str(epoch))
        os.system('mkdir ' + self.out_dir_name + str(epoch) + '/learning/')
        os.system('mkdir ' + self.out_dir_name + str(epoch) + '/target_generation/')
        os.system('mkdir ' + self.out_dir_name + str(epoch) + '/posterior_generation/')
        os.system('mkdir ' + self.out_dir_name + str(epoch) + '/free_generation/')
        os.system('mkdir ' + self.out_dir_name + str(epoch) + '/error_regression/')

    def train(self):
        # setting for training
        epoch_size = self.params['epoch_size']
        test_every = self.params['test_every']
        device = self.params['device']

        # log and output directory
        self.make_directries()
        with open(self.out_dir_name + 'log.txt', mode='a') as f:
            for key, values in self.params.items():
                print(key, ':\t', values, file=f)

        # training setting
        optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()
        dataloader = self.get_dataloader(
            data_dir=self.params['train_data_path'],
            data_size=self.params['data_size'],
            mini_batch_size=self.params['mini_batch_size'],
            max_time_step=self.params['max_time_step'],
            shuffle=False,
            sep=self.params['sep_load'],
            is_plotting=True, save_name='data_train')
        self.clear_loss_dict_now()
        self.save_model(epoch=0)
        start = time.time()

        # main
        for epoch in range(1, epoch_size + 1):
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.to(device)
                # print('data size: ', data.size(), '\nlabel size: ', labels.size())

                optimizer.zero_grad()
                output = self.rnn(sequence_number=labels, target=data)
                loss = self.criterion(output, data)

                loss.backward()
                optimizer.step()
                self.save_loss(now_loss=loss.item(), is_mini_batch=True)

                # saving outputs during learning
                if epoch % test_every == 0:
                    if device != 'cpu': output = output.cpu()
                    for idx in range(labels.size(0)):
                        now_sequence_number = int(labels[idx])
                        now_output = output[idx,:,:].detach().numpy().reshape(
                            output.size(1), output.size(2))
                        save_all_vars_learn(
                            self.rnn.all_vars_detached,
                            out_dir_name=self.out_dir_name + str(epoch) + '/learning/',
                            prefix_dict=self.save_prefix_dict,
                            delimiter=self.params['sep_load'],
                            suffix=str(now_sequence_number))
                        now_output = self.inverse_preprocess(now_output)
                        np.savetxt(
                            self.out_dir_name + str(epoch) + '/learning/xValueInversed' + str(now_sequence_number),
                            now_output, delimiter=self.params['sep_load'], fmt=Const.FILE_FORMAT)

            self.save_loss(now_loss=loss.item(), is_mini_batch=False)

            if epoch % self.params['print_every'] == 0: self.print_log(epoch, loss, start)
            if epoch % self.params['save_every'] == 0: self.save_model(epoch)
            if epoch % test_every == 0:
                self.save_model(epoch)
                self.test(epoch)

        with open(self.out_dir_name + 'log.txt', mode='a') as f:
            print('all done. %s' % (timeSince(start)), file=f)

        return

    def test(self, epoch):
        self.make_directries_epoch(epoch=epoch)

        start = time.time()
        with open(self.out_dir_name + 'log.txt', mode='a') as f:
            print('\n\n\ntest start time: ', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=f)
            for key, values in self.params.items():
                print(key, ': ', values, file=f)

        max_time_step = self.params['max_time_step']

        test_data_size_tar = self.params['test_data_size_tar'] \
                             if 'test_data_size_tar' in self.params.keys() \
                                else self.params['test_data_size']
        test_data_size_free = self.params['test_data_size_free'] \
                             if 'test_data_size_free' in self.params.keys() \
                                else self.params['test_data_size']
        test_data_size_ereg_train = self.params['test_data_size_ereg_train'] \
                                    if 'test_data_size_ereg_train' in self.params.keys() \
                                       else self.params['test_data_size']
        test_data_size_ereg_test = self.params['test_data_size_ereg_test'] \
                                    if 'test_data_size_ereg_test' in self.params.keys() \
                                       else self.params['test_data_size']
        traversal_values = self.params['traversal_values'] if 'traversal_values' in self.params.keys() else 0

        # prediction of learning data
        dataloader = self.get_dataloader(
            data_dir=self.params['train_data_path'],
            data_size=self.params['data_size'],
            mini_batch_size=self.params['data_size'],
            max_time_step=self.params['max_time_step'],
            shuffle=False,
            sep=self.params['sep_load'],
            is_preprocessing=False,
            is_plotting=False,
            save_name='data_train')

        if test_data_size_tar != 0:
            for batch_idx, (data, labels) in enumerate(dataloader):
                for i in range(labels.size(0)):
                    sequence_number = labels[i].view(1)
                    target = data[i,:,:].view(1, data.size(1), data.size(2))
                    target = target[target != Const.PADDING_TOKEN].view(1, -1, target.size(2))

                    # target generation
                    output = self.target_generation(
                        sequence_number=sequence_number,
                        max_time_step=max_time_step,
                        write_file=True,
                        out_filepath=self.out_dir_name + str(epoch) + '/target_generation/',
                        suffix=str(int(sequence_number.item()))
                    )

                    # posterior generation (only PVRNN)
                    if 'PVRNN' in self.rnn.__class__.__name__:
                        output = self.posterior_generation(
                            sequence_number=sequence_number,
                            max_time_step=max_time_step,
                            write_file=True,
                            out_filepath=self.out_dir_name + str(epoch) + '/posterior_generation/',
                            suffix=str(int(sequence_number.item()))
                        )

                    with open(self.out_dir_name + 'log.txt', mode='a') as f:
                        print('Target Generations for data' + str(i) + ' done. %s' % (timeSince(start)), file=f)
                    if (batch_idx * labels.size(0)) + i + 1 == test_data_size_tar: break
                else: continue
                break

        # free generation
        if test_data_size_free != 0:
            for sequence_number in range(test_data_size_free):
                output = self.free_generation(
                    max_time_step=self.params['free_gen_step'],
                    write_file=True,
                    out_filepath=self.out_dir_name + str(epoch) + '/free_generation/',
                    suffix=str(int(sequence_number)))
                with open(self.out_dir_name + 'log.txt', mode='a') as f:
                    print('Free Generations for data' + str(i) + ' done. %s' % (timeSince(start)), file=f)

        # latent state traversal
        if traversal_values > 0:
            self.sample_all_z(epoch=epoch)

        # error regression to learning data
        if test_data_size_ereg_train != 0:
            for batch_idx, (data, labels) in enumerate(dataloader):
                for i in range(labels.size(0)):
                    sequence_number = labels[i].view(1)
                    target = data[i,:,:].view(1, data.size(1), data.size(2))
                    target = target[target != Const.PADDING_TOKEN].view(1, -1, target.size(2))
                    output = self.error_regression(
                        target,
                        write_file=True,
                        out_filepath=self.out_dir_name + str(epoch) + '/error_regression/',
                        suffix='_train' + str(int(sequence_number.item())))
                    with open(self.out_dir_name + 'log.txt', mode='a') as f:
                        print('Error regression for data' + str(i) + ' done. %s' % (timeSince(start)), file=f)
                    if (batch_idx * labels.size(0)) + i + 1 == test_data_size_ereg_train: break
                else: continue
                break

        # error regression to unknown data
        if test_data_size_ereg_test != 0:
            dataloader = self.get_dataloader(
                data_dir=self.params['test_data_path'],
                data_size=test_data_size_ereg_test,
                mini_batch_size=test_data_size_ereg_test,
                max_time_step=self.params['max_time_step'],
                shuffle=False, sep=self.params['sep_load'],
                is_plotting=True, save_name='data_test')

            for batch_idx, (data, labels) in enumerate(dataloader):
                for i in range(labels.size(0)):
                    sequence_number = labels[i].view(1)
                    target = data[i,:,:].view(1, data.size(1), data.size(2))
                    # print('target size: ', target.size(), '\nsequence_number size: ', sequence_number.size())

                    # error regression
                    output = self.error_regression(
                        target, write_file=True,
                        out_filepath=self.out_dir_name + str(epoch) + '/error_regression/',
                        suffix='_test' + str(int(sequence_number.item())))

                    with open(self.out_dir_name + 'log.txt', mode='a') as f:
                        print('test' + str(i) + ' done. %s' % (timeSince(start)), file=f)

            with open(self.out_dir_name + 'log.txt', mode='a') as f:
                print('all test done. %s' % (timeSince(start)), file=f)

        return

    def clear_loss_dict_now(self):
        self.loss_dict_now = {
            'loss': 0,
            'loss_grad': 0}
        return

    def save_loss(self, now_loss, is_mini_batch=False):
        if is_mini_batch == True:
            self.loss_dict_now['loss'] += self.criterion.loss
            self.loss_dict_now['loss_grad'] += now_loss
        else:
            for name, l in self.loss_dict_now.items():
                fname = self.out_dir_name + 'mainrslt/' + name
                with open(fname, mode='a') as f:
                    f.write('{:.24f}'.format(l) + '\n')
            self.clear_loss_dict_now()
        return

    def predict(self, target):
        prediction = self.error_regression(target)
        return prediction

    def error_regression(self, target,
                         write_file=False, out_filepath='', suffix=''):
        # Evacuating orignal rnn model
        param_dict = copy.deepcopy(self.rnn.state_dict())
        rnn_tmp = self.build_model()
        rnn_tmp.load_state_dict(param_dict)
        if self.params['device'] != 'cpu': rnn_tmp.cuda()

        # Regression
        regressor = self.get_regressor(rnn=rnn_tmp)
        output = regressor.regress(target=target)
        if self.params['device'] != 'cpu': output = output.cpu()

        # File saving
        if write_file == True:
            # Editting save_prefix_dict for saving postdiciton and prediction
            save_prefix_dict = {}
            keys_list = list(self.save_prefix_dict.keys())
            suffix_list = ['_1step_pred', '_1step_post']
            for key_old in keys_list:
                value_old = self.save_prefix_dict[key_old]
                for _suffix in suffix_list:
                    save_prefix_dict[key_old + _suffix] = value_old + _suffix
            print('prefix_dict\n', save_prefix_dict)

            # Save all variables
            save_all_vars(
                regressor.all_vars_detached,
                out_dir_name=out_filepath,
                prefix_dict=save_prefix_dict,
                suffix=suffix,
                delimiter=self.params['sep_load'])

            # Trans and save normal output
            now_output = self.inverse_preprocess(
                output.detach().numpy().reshape(output.size(1), output.size(2)))
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)

            # Trans and save prediction output
            now_output = self.inverse_preprocess(
                regressor.all_vars_detached['all_x' + suffix_list[0]].numpy().reshape(
                    output.size(1), output.size(2)))
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix_list[0] + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)

            # Trans and save postdiction output
            now_output = self.inverse_preprocess(
                regressor.all_vars_detached['all_x' + suffix_list[1]].numpy().reshape(
                    output.size(1), output.size(2)))
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix_list[1] + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)

        return output

    def free_generation(self, max_time_step,
                        write_file=False, out_filepath='', suffix=''):

        output = self.rnn(
            sequence_number=torch.tensor(-1).view(1),
            max_time_step=max_time_step,
            closed_threshold=-1)
        if self.params['device'] != 'cpu': output = output.cpu()

        if write_file==True:
            save_all_vars(
                self.rnn.all_vars_detached,
                out_dir_name=out_filepath,
                prefix_dict=self.save_prefix_dict,
                delimiter=self.params['sep_load'],
                suffix=suffix)
            now_output = self.inverse_preprocess(
                output.detach().numpy().reshape(output.size(1), output.size(2)))
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)

        return output

    def target_generation(self, sequence_number, max_time_step,
                          closed_threshold=0,
                          write_file=False, out_filepath='', suffix=''):
        '''
        MTRNN -> closed threshold = 0 (all input is not used)
        PV-RNN -> closed threhold = 1 (only time step 1, using posterior)
        '''
        output = self.rnn(
            sequence_number=sequence_number,
            max_time_step=max_time_step,
            target=None,
            closed_threshold=closed_threshold)
        if self.params['device'] != 'cpu': output = output.cpu()

        if write_file==True:
            save_all_vars(
                self.rnn.all_vars_detached,
                out_dir_name=out_filepath,
                prefix_dict=self.save_prefix_dict,
                suffix=suffix,
                delimiter=self.params['sep_load'])
            now_output = self.inverse_preprocess(
                output.detach().numpy().reshape(output.size(1), output.size(2)))
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)

        return output

    def get_optimizer(self):
        return torch.optim.Adam(self.rnn.parameters(), lr=self.params['lr'])

    def get_criterion(self):
        criterion = MTRNNLoss(padding_token=Const.PADDING_TOKEN)
        return criterion

    def get_regressor(self, rnn):
        regressor = Regressor(
            model=rnn,
            criterion=self.get_criterion(),
            lr=self.params['ereg_lr'],
            ws=self.params['ereg_window_size'],
            itrtn=self.params['ereg_iteration'],
            pred=self.params['ereg_pred_step']
        )
        return regressor

    def preprocess_data(
            self, data_dir_pre, data_dir_post, data_size,
            max_time_step=None, sep=',', is_plotting=False, save_name='data'):
        '''
        # normalizing data between -1 and 1
        self.normalize_data(
            data_dir_pre=data_dir_pre,
            data_dir_post=data_dir_post,
            data_size=data_size,
            max_time_step=max_time_step,
            sep=sep)
        '''
        # padding
        self.padding_sequence(
            data_dir_pre=data_dir_pre,
            data_dir_post=data_dir_post,
            data_size=data_size,
            max_time_step=max_time_step, sep=sep)

        return

    def inverse_preprocess(self, output):
        # output = self.inverse_normalize_data(output)
        return output

    def get_max_length(self, data_dir, data_size, sep=','):
        for idx in range(data_size):
            now_data = np.loadtxt(
                data_dir + str(idx),
                delimiter=sep, skiprows=0)
            if idx == 0:
                max_length = now_data.shape[0]
            elif (idx > 0 and now_data.shape[0] > max_length):
                max_length = now_data.shape[0]
            # print('idx: ', idx, 'now_length', now_data.shape[0], 'max: ', max_length)
        return max_length

    def padding_sequence(
            self, data_dir_pre, data_dir_post, data_size,
            sep=',', max_time_step=None):
        padding_token=Const.PADDING_TOKEN
        max_length = max_time_step

        # padding
        for idx in range(data_size):
            now_data = np.loadtxt(
                data_dir_pre + str(idx),
                delimiter=sep, skiprows=0)
            now_length = now_data.shape[0]
            if now_length < max_length:
                add_data = np.full((max_length - now_length, now_data.shape[1]), padding_token)
                # print('now_data', now_data.shape, 'add_data', add_data.shape)
                now_data = np.concatenate(
                    [now_data, add_data], axis=0)
            np.savetxt(
                data_dir_post + str(idx),
                now_data, fmt=Const.FILE_FORMAT, delimiter=sep)
        return

    def normalize_data(self, data_dir_pre, data_dir_post, data_size, max_time_step, sep):
        self.scaler_list = []
        x_dim = np.loadtxt(data_dir_pre + str(0)).shape[1]

        # combining all sequences
        data_all = np.full((data_size, max_time_step, x_dim), np.nan)
        for idx in range(data_size):
            data_add = np.loadtxt(
                data_dir_pre + str(idx), delimiter=sep, skiprows=0, max_rows=max_time_step)
            data_all[idx, :, :] = data_add
        data_all = np.where(data_all == Const.PADDING_TOKEN, np.nan, data_all)

        # trans
        for idx in range(x_dim):
            scaler = preprocessing.MinMaxScaler(
                    feature_range=(-1, 1), copy=True)

            data_tmp = data_all[:, :, idx].reshape(-1, 1)
            data_tmp = scaler.fit_transform(data_tmp)
            data_all[:, :, idx] = data_tmp.reshape(data_size, max_time_step)
            print('dim of ', str(idx), scaler.data_max_, scaler.data_min_)

            self.scaler_list.append(scaler)

        # save
        data_all = np.where(np.isnan(data_all), Const.PADDING_TOKEN, data_all)
        for idx in range(data_size):
            np.savetxt(
                data_dir_post + str(idx),
                data_all[idx, :, :], fmt=Const.FILE_FORMAT, delimiter=sep)

        return

    def inverse_normalize_data(self, output):
        is_torch = type(output) is torch.Tensor
        if is_torch:
            device = output.device
            output = output.to('cpu').numpy()

        for idx in range(output.shape[1]):
            scaler = self.scaler_list[idx]
            output_tmp = output[:, idx].reshape(-1, 1)
            output_tmp = scaler.inverse_transform(output_tmp)
            output[:, idx] = output_tmp.reshape(-1)

        output = torch.from_numpy(output).to(device) if is_torch else output
        return output

    def get_dataloader(
            self, data_dir, data_size, mini_batch_size,
            max_time_step=None, shuffle=True, sep=',',
            data_dir_post=None,
            is_preprocessing=True,
            is_plotting=False, save_name='data'):

        if data_dir_post is None:
            data_dir_post=self.out_dir_name + 'data_preprocessed/' + save_name
        if not os.path.exists(data_dir_post + str(0)):
            os.system('mkdir ' + self.out_dir_name + 'data_preprocessed/')
            is_preprocessing=True

        # preprocessing
        if is_preprocessing == True:
            self.preprocess_data(
                data_dir_pre=data_dir,
                data_dir_post=data_dir_post,
                data_size=data_size,
                max_time_step=max_time_step,
                sep=sep,
                is_plotting=is_plotting,
                save_name=save_name)

        # dataloader
        data_set = MyDataset(
            data_num=data_size,
            transform=None,
            filename=data_dir_post,
            sep=sep,
            max_time_step=max_time_step)
        dataloader = torch.utils.data.DataLoader(
            data_set, batch_size=mini_batch_size, shuffle=shuffle)

        return dataloader

    def print_log(self, epoch, loss, start):
        with open(self.out_dir_name + 'log.txt', mode='a') as f:
            print(
                '%s (%d %d%%) %.4f' % (timeSince(start),
                                       epoch,
                                       epoch / self.params['epoch_size'] * 100,
                                       loss), file=f)
            # print('kld', kld_loss, 'nll', nll_loss, file=f)
            # print(rnn.all_vars_detached)

        with open(self.out_dir_name + 'param.txt', mode='a') as f:
            print('### epoch: ', str(epoch), file=f)
            for name, param in self.rnn.named_parameters():
                print(name, param.data, file=f)
        return

    def build_model(self):
        model = RNNTG()
        if self.params['device'] != 'cpu': rnn.cuda()
        return

    def save_model(self, epoch):
        fn = self.out_dir_name + 'saves/params.pickle'
        with open(fn, 'wb') as f:
            pickle.dump(self.params, f)

        fn = self.out_dir_name + 'saves/model_state_dict_' + str(epoch) + '.pth'
        torch.save(self.rnn.state_dict(), fn)
        with open(self.out_dir_name + 'log.txt', mode='a') as f:
            print('Saved model to '+ fn, file=f)

        return

    def load_model(self, saved_epoch, model_path='', load_params=True):
        fn = self.out_dir_name + 'saves/params.pickle'
        if os.path.exists(fn) and load_params==True:
            with open(fn, 'rb') as f:
                self.params = pickle.load(f)
            with open(self.out_dir_name + 'log.txt', mode='a') as f:
                print('params is loaded.', file=f)
                for key, values in self.params.items():
                    print(key, '\t:\t', values, file=f)

        if model_path == '':
            model_path = self.out_dir_name + \
                         'saves/model_state_dict_' + \
                         str(saved_epoch) + '.pth'
        rnn = self.build_model()
        rnn.load_state_dict(torch.load(model_path))
        self.rnn = rnn
        return

    def plot_all_dir(self):
        # training data
        self.plot_one_dir(
            data_dir_tar=self.params['train_data_path'],
            data_dir_out=self.out_dir_name + 'data_train',
            data_size=self.params['data_size'],
            sep=self.params['sep_load'])

        # test data
        self.plot_one_dir(
            data_dir_tar=self.params['test_data_path'],
            data_dir_out=self.out_dir_name + 'data_test',
            data_size=self.params['test_data_size'],
            sep=self.params['sep_load'])

        # predictions
        for epoch in [int(tmp * self.params['test_every']) for tmp in range(1, int(self.params['epoch_size'] / self.params['test_every'] + 1))]:
            os.system('mkdir -pv ' + self.out_dir_name + str(epoch) + '/img/')

            # during learning
            self.plot_one_dir(
                data_dir_tar=self.out_dir_name + str(epoch) + '/learning/xValueInversed',
                data_dir_out=self.out_dir_name + str(epoch) + '/img/learning_' + str(epoch),
                data_size=self.params['data_size'],
                sep=self.params['sep_load'])

            # posterior generation
            self.plot_one_dir(
                data_dir_tar=self.out_dir_name + str(epoch) + '/posterior_generation/xValueInversed',
                data_dir_out=self.out_dir_name + str(epoch) + '/img/posterior_' + str(epoch),
                data_size=self.params['data_size'],
                sep=self.params['sep_load'])

            # target generation
            self.plot_one_dir(
                data_dir_tar=self.out_dir_name + str(epoch) + '/target_generation/xValueInversed',
                data_dir_out=self.out_dir_name + str(epoch) + '/img/target_' + str(epoch),
                data_size=self.params['data_size'],
                sep=self.params['sep_load'])

            # free generation
            self.plot_one_dir(
                data_dir_tar=self.out_dir_name + str(epoch) + '/free_generation/xValueInversed',
                data_dir_out=self.out_dir_name + str(epoch) + '/img/free_' + str(epoch),
                data_size=self.params['data_size'],
                sep=self.params['sep_load'])

            # error regression for training data
            self.plot_one_dir(
                data_dir_tar=self.out_dir_name + str(epoch) + '/error_regression/xValueInversed_1step_post_train',
                data_dir_out=self.out_dir_name + str(epoch) + '/img/ereg_train_' + str(epoch),
                data_size=self.params['data_size'],
                sep=self.params['sep_load'])

            # error regression for test data
            self.plot_one_dir(
                data_dir_tar=self.out_dir_name + str(epoch) + '/error_regression/xValueInversed_1step_post_test',
                data_dir_out=self.out_dir_name + str(epoch) + '/img/ereg_test_' + str(epoch),
                data_size=self.params['test_data_size'],
                sep=self.params['sep_load'])


        return

    def plot_one_dir(self, data_dir_tar, data_dir_out, data_size, sep=' '):
        for idx in range(data_size):
            filepath = data_dir_tar + str(idx)
            if os.path.exists(filepath):
                now_data = np.loadtxt(
                    filepath, delimiter=sep,
                    skiprows=0, max_rows=self.params['max_time_step'])
                plot_2D_sequence(
                    df=pd.DataFrame(now_data),
                    max_time_step=self.params['max_time_step'],
                    out_dir=data_dir_out + '_' + str(idx) + '.png')
            # else: print('plot wasn\'t done because', filepath, 'is not found.')
        return


class ModelPVRNNTanh(ModelRNNTG):
    def __init__(self, params):
        if 'num_iw' not in params: params['num_iw'] = 1
        if 'num_mc' not in params: params['num_mc'] = 1

        super().__init__(params)

        '''
        self.save_prefix_dict = {
            'all_x': 'xValue',
            'all_d': 'dValue',
            'all_z': 'zValue',
            'u_hid': 'hValue',
            'all_mu': 'myuValue',
            'all_sigma': 'sigmaValue',
            'all_mu_inference': 'myuValuePosterior',
            'all_sigma_inference': 'sigmaValuePosterior',
            'all_mu_prior': 'myuValuePrior',
            'all_sigma_prior': 'sigmaValuePrior',
            'kld_element': 'kld_element'
        }
        '''

        # in Reza-san's code,
        self.save_prefix_dict = {
            'all_x': 'xValue',
            'all_d': 'dValuePrior',
            'all_z': 'zValuePrior',
            'u_hid': 'hValuePrior',
            'all_mu': 'myuValuePrior',
            'all_sigma': 'sigmaValuePrior',
            'all_mu_inference': 'myuValue',
            'all_sigma_inference': 'sigmaValue',
            'all_mu_prior': 'myuValueTruePrior',
            'all_sigma_prior': 'sigmaValueTruePrior',
            'kld_element': 'kld_element'
            # 'dValuePrior', hValue, hValuePrior, noise, rhoValue, sigmMultNoise, zValue, zValuePrior
        }

        return

    def build_model(self):
        rnn = PVRNNTanh(
            x_dim=self.params['x_dim'],
            h_dim=self.params['h_dim'],
            z_dim=self.params['z_dim'],
            time_scale=self.params['time_scale'],
            data_size=self.params['data_size'],
            max_time_step=self.params['max_time_step'],
            device=self.params['device'],
            num_iw=self.params['num_iw'],
            num_mc=self.params['num_mc'],
            initial_gaussian_regularizer=self.params['initial_gaussian_regularizer'],
            use_hidden_for_posterior=self.params['use_hidden_for_posterior'],
            use_bottom_up_signal=self.params['use_bottom_up_signal']
        )
        if self.params['device'] != 'cpu': rnn.cuda()
        return rnn

    def get_criterion(self, rnn=None, meta_prior=None):
        if rnn is None: rnn = self.rnn
        if meta_prior is None: meta_prior = self.params['meta_prior']
        init_meta_prior = self.params['init_meta_prior'] \
                          if self.params['initial_gaussian_regularizer'] else None

        criterion = PVRNNTanhLoss(
            model=rnn, meta_prior=meta_prior,
            init_meta_prior=init_meta_prior,
            padding_token=Const.PADDING_TOKEN)

        return criterion

    def get_regressor(self, rnn):
        betas = self.params['ereg_betas'] if 'ereg_betas' in self.params.keys() \
                else (0.5, 0.999)
        regressor = RegressorPVRNN(
            model=rnn,
            criterion=self.get_criterion(
                rnn=rnn, meta_prior=self.params['ereg_meta_prior']),
            lr=self.params['ereg_lr'],
            betas=betas,
            ws=self.params['ereg_window_size'],
            itrtn=self.params['ereg_iteration'],
            pred=self.params['ereg_pred_step']
        )
        return regressor

    def target_generation(self, sequence_number, max_time_step,
                          closed_threshold=0,
                          write_file=False,
                          out_filepath='', suffix=''):
        output = super().target_generation(
            sequence_number=sequence_number,
            max_time_step=max_time_step,
            closed_threshold=1, # using posterior during 1 step
            # closed_threshold=max_time_step/4,
            write_file=write_file,
            out_filepath=out_filepath,
            suffix=suffix)

        return output

    def posterior_generation(self, sequence_number, max_time_step,
                          write_file=False,
                          out_filepath='', suffix=''):
        output = super().target_generation(
            sequence_number=sequence_number,
            max_time_step=max_time_step,
            # closed_threshold=max_time_step, # using posterior during full step
            closed_threshold=max_time_step/4,
            write_file=write_file,
            out_filepath=out_filepath,
            suffix=suffix)
        return output

    def sample_one_z(self,
                     mode,
                     target_idx,
                     generator_other_z,
                     fixed_value=None,
                     write_file=False, out_filepath='', suffix=''):
        output = self.rnn.latent_space_traversal(
            mode=mode,
            target_idx=target_idx,
            generator_other_z=generator_other_z,
            fixed_value=fixed_value,
            batch_size=1,
            max_time_step=None
        )

        if self.params['device'] != 'cpu': output = output.cpu()

        if write_file==True:
            save_all_vars(
                self.rnn.all_vars_detached,
                out_dir_name=out_filepath,
                prefix_dict=self.save_prefix_dict,
                suffix=suffix,
                delimiter=self.params['sep_load'])
            now_output = self.inverse_preprocess(
                output.detach().numpy().reshape(output.size(1), output.size(2)))
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)
        return output

    def sample_all_z(self, epoch):
        generator_other_z_list = ['prior'] #'zeros', 'randn'
        mode_list = [
            'constant_z', 'constant_sigma', 'constant_z_only_initial', 'traversal_z'
            #, 'constant_noise', 'traversal_z', 'traversal_noise'
        ]
        start_value = -1
        stop_value = 1
        for generator_other_z in generator_other_z_list:
            for mode in mode_list:
                # fixed_value_list = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                fixed_value_list = np.linspace(
                    start_value, stop_value, self.params['traversal_values']).tolist() \
                    if 'constant' in mode else [0.0]

                for idx_fix, fix in enumerate(fixed_value_list):
                    fname = self.out_dir_name + str(epoch) + '/latent_space_traversal/' + mode + '_' + \
                            str(generator_other_z) + '_fix' + str(idx_fix) + '/'
                    os.system('mkdir -pv ' + fname)

                    for idx_z in range(sum(self.params['z_dim'])):
                        output = self.sample_one_z(
                            mode=mode,
                            generator_other_z=generator_other_z,
                            target_idx=idx_z,
                            fixed_value=fix,
                            write_file=True,
                            out_filepath=fname,
                            suffix=str(int(idx_z))
                        )
        return

    def clear_loss_dict_now(self):
        self.loss_dict_now ={
            'elbo_loss': 0,
            'elbo_loss_grad': 0,
            'kld_loss': 0,
            'kld_loss_init': 0,
            'kld_loss_other': 0,
            'kld_loss_non_weighted': 0,
            'nll_loss': 0
        }
        for i, _ in enumerate(self.params['z_dim']):
            self.loss_dict_now['kld_loss_non_weighted_l' + str(i)] = 0

        return

    def save_loss(self, now_loss, is_mini_batch=False):
        if is_mini_batch == True:
            self.loss_dict_now['elbo_loss'] += self.criterion.elbo_loss.item()
            self.loss_dict_now['elbo_loss_grad'] += now_loss
            self.loss_dict_now['kld_loss'] += self.criterion.kld_loss.item()
            self.loss_dict_now['kld_loss_non_weighted'] \
                += sum(self.criterion.kld_loss_non_weighted).item()
            for i, _ in enumerate(self.params['z_dim']):
                self.loss_dict_now['kld_loss_non_weighted_l' + str(i)] = self.criterion.kld_loss_non_weighted[i].item()
            self.loss_dict_now['kld_loss_init'] += self.criterion.kld_loss_init.item()
            self.loss_dict_now['kld_loss_other'] += self.criterion.kld_loss_other.item()
            self.loss_dict_now['nll_loss'] += self.criterion.nll_loss.item()

        else:
            for name, loss in self.loss_dict_now.items():
                fname = self.out_dir_name + 'mainrslt/' + name
                with open(fname, mode='a') as f:
                    f.write('{:.18f}'.format(loss) + '\n')
            self.clear_loss_dict_now()
        return


class ModelPVRNNSoftmax(ModelPVRNNTanh):
    def __init__(self, params):
        super().__init__(params)
        self.use_softmax_preprocess \
            = True if 'use_softmax_preprocess' not in params \
            else params['use_softmax_preprocess']

        return

    def build_model(self):
        rnn = PVRNNSoftmax(
            x_dim=self.params['x_dim_reza_model'],
            real_x_dim=self.params['x_dim'],
            h_dim=self.params['h_dim'],
            z_dim=self.params['z_dim'],
            time_scale=self.params['time_scale'],
            data_size=self.params['data_size'],
            max_time_step=self.params['max_time_step'],
            device=self.params['device'],
            num_iw=self.params['num_iw'],
            num_mc=self.params['num_mc'],
            initial_gaussian_regularizer=self.params['initial_gaussian_regularizer'],
            use_hidden_for_posterior=self.params['use_hidden_for_posterior'],
            use_bottom_up_signal=self.params['use_bottom_up_signal']
        )
        if self.params['device'] != 'cpu': rnn.cuda()
        return rnn

    def get_criterion(self, rnn=None, meta_prior=None):
        if rnn is None: rnn = self.rnn
        if meta_prior is None: meta_prior = self.params['meta_prior']
        init_meta_prior = self.params['init_meta_prior'] \
                          if self.params['initial_gaussian_regularizer'] else None

        criterion = PVRNNSoftmaxLoss(
            model=rnn, meta_prior=meta_prior,
            init_meta_prior=init_meta_prior,
            padding_token=Const.PADDING_TOKEN)

        return criterion

    def error_regression(self, target,
                         write_file=False, out_filepath='', suffix=''):
        # Evacuating orignal rnn model
        param_dict = copy.deepcopy(self.rnn.state_dict())
        rnn_tmp = self.build_model()
        rnn_tmp.load_state_dict(param_dict)
        if self.params['device'] != 'cpu': rnn_tmp.cuda()

        # Regression
        regressor = self.get_regressor(rnn=rnn_tmp)
        output = regressor.regress(target=target, use_best_loss=False)
        if self.params['device'] != 'cpu': output = output.cpu()

        # File saving
        if write_file==True:
            # Editting save_prefix_dict for saving postdiciton and prediction
            save_prefix_dict = {}
            keys_list = list(self.save_prefix_dict.keys())
            suffix_list = ['_1step_pred', '_1step_post']
            for key_old in keys_list:
                value_old = self.save_prefix_dict[key_old]
                for _suffix in suffix_list:
                    save_prefix_dict[key_old + _suffix] = value_old + _suffix
            # print('prefix_dict\n', save_prefix_dict)

            # Save all variables
            save_all_vars(
                regressor.all_vars_detached,
                out_dir_name=out_filepath,
                prefix_dict=save_prefix_dict,
                suffix=suffix,
                delimiter=self.params['sep_load'])

            # Trans and save normal output
            now_output = self.inverse_preprocess(output)
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)

            # Trans and save prediction output
            now_output = self.inverse_preprocess(
                regressor.all_vars_detached['all_x' + suffix_list[0]])
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix_list[0] + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)

            # Trans and save postdiction output
            now_output = self.inverse_preprocess(
                regressor.all_vars_detached['all_x' + suffix_list[1]])
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix_list[1] + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)

        return output

    def free_generation(self, max_time_step,
                        write_file=False, out_filepath='', suffix=''):
        output = super().free_generation(
            max_time_step=max_time_step,
            write_file=False,
            out_filepath=out_filepath,
            suffix=suffix)

        if write_file==True:
            save_all_vars(
                self.rnn.all_vars_detached,
                out_dir_name=out_filepath,
                prefix_dict=self.save_prefix_dict,
                suffix=suffix,
                delimiter=self.params['sep_load'])
            now_output = self.inverse_preprocess(output)
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)

        return output

    def target_generation(self, sequence_number, max_time_step,
                          closed_threshold=0,
                          write_file=False, out_filepath='', suffix=''):
        output = super().target_generation(
            sequence_number=sequence_number,
            max_time_step=max_time_step,
            write_file=False,
            out_filepath=out_filepath,
            suffix=suffix)

        if write_file==True:
            save_all_vars(
                self.rnn.all_vars_detached,
                out_dir_name=out_filepath,
                prefix_dict=self.save_prefix_dict,
                suffix=suffix,
                delimiter=self.params['sep_load'])
            now_output = self.inverse_preprocess(output)
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)

        return output

    def posterior_generation(self, sequence_number, max_time_step,
                          write_file=False, out_filepath='', suffix=''):
        output = super().posterior_generation(
            sequence_number=sequence_number,
            max_time_step=max_time_step,
            write_file=False,
            out_filepath=out_filepath,
            suffix=suffix)

        if write_file==True:
            save_all_vars(
                self.rnn.all_vars_detached,
                out_dir_name=out_filepath,
                prefix_dict=self.save_prefix_dict,
                suffix=suffix,
                delimiter=self.params['sep_load'])
            now_output = self.inverse_preprocess(output)
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)
        return output

    def sample_one_z(self,
                     mode,
                     generator_other_z,
                     target_idx,
                     fixed_value=None,
                     write_file=False, out_filepath='', suffix=''):
        output = self.rnn.latent_space_traversal(
            mode=mode,
            target_idx=target_idx,
            generator_other_z=generator_other_z,
            fixed_value=fixed_value,
            batch_size=1,
            max_time_step=None
        )

        if self.params['device'] != 'cpu': output = output.cpu()

        if write_file==True:
            save_all_vars(
                self.rnn.all_vars_detached,
                out_dir_name=out_filepath,
                prefix_dict=self.save_prefix_dict,
                suffix=suffix,
                delimiter=self.params['sep_load'])
            now_output = self.inverse_preprocess(output)
            np.savetxt(
                out_filepath + 'xValueInversed' + suffix,
                now_output,
                delimiter=self.params['sep_load'],
                fmt=Const.FILE_FORMAT)
        return output

    def preprocess_data(
            self, data_dir_pre, data_dir_post, data_size,
            max_time_step=None, sep=',', is_plotting=False, save_name='data'):

        '''
        # normalizing data between -1 and 1
        self.normalize_data(
            data_dir_pre=data_dir_pre,
            data_dir_post=data_dir_post + '_minmax',
            data_size=data_size,
            max_time_step=max_time_step,
            sep=sep)
        '''

        # softmax transformation
        make_softmax_dir(
            data_dir=data_dir_pre,
            out_dir_name=data_dir_post,
            size=data_size,
            read_sep=sep,
            wright_sep=sep,
            trans=self.use_softmax_preprocess,
            fmt=Const.FILE_FORMAT)

        # padding
        self.padding_sequence(
            data_dir_pre=data_dir_post,
            data_dir_post=data_dir_post,
            data_size=data_size,
            max_time_step=max_time_step, sep=sep)

        return

    def inverse_softmax_trans(self, output):
        if type(output) == torch.Tensor:
            output = output.detach().numpy().reshape(-1, output.size(2))
        output = softmax_transform(output, mode='inverse')
        output = torch.tensor(output)
        return output

    def inverse_preprocess(self, output):
        output = self.inverse_softmax_trans(output)
        # output = self.inverse_normalize_data(output)
        return output
