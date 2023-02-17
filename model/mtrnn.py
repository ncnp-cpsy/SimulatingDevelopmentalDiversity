import torch

from model.cell import CellMTRNN, MaskMaker
from model.rnntg import RNNTG


class MTRNN(RNNTG):
    def __init__(self, h_dim, time_scale, data_size, device):
        """
        This class is designed in that the lowest hidden layer is equal to
        input and output layer. So, if you use 1 layer mtrnn (this is similar
        to CTRNN), you must set h_dim of the lowest layer is equal to
        input_size and output_size. Implementaion of MTRNN using MaskLinear,
        due to fast caliculation.
        """

        super().__init__()

        # hyper-parameters
        self.h_dim = h_dim
        self.num_time_layer = len(self.h_dim)
        self.time_scale = time_scale
        self.data_size = data_size
        self.device = device

        # recurrent model
        self.recurrence = CellMTRNN(h_dim=h_dim,
                                    time_scale=time_scale,
                                    mask=self.get_mask_matrix(),
                                    device=device)

        # for saving
        self.all_vars_detached_keys = ['all_x', 'hidden', 'u_hid']
        self.all_vars_non_detached_keys = ['all_x']

        return

    def get_mask_matrix(self):
        mask = MaskMaker.make_mask_matrix(h_dim=self.h_dim,
                                          mode='mtrnn_normal')
        return mask

    def _initialize(self, sequence_number, use_saved_u=False):
        batch_size = sequence_number.size(0)

        u_hid = self.get_init_u_hid(batch_size, use_saved_u=use_saved_u)
        hidden = self.recurrence.activation(u_hid)

        return hidden, u_hid

    def _one_step_forward(self, hidden, u_hid, step, pb=None):
        combat = hidden if pb is None else torch.cat([hidden, pb], dim=1)
        hidden, u_hid = self.recurrence(combat, u_hid)
        self.save_vars(hidden=hidden, u_hid=u_hid, step=step)
        return hidden, u_hid

    def forward(self, sequence_number, target=None,
                max_time_step=None, closed_threshold=0,
                pb=None, use_saved_u=False):

        # setting
        self.clear_vars()
        sequence_number, max_time_step, closed_threshold, target \
            = self.get_forward_setting(
                sequence_number=sequence_number,
                target=target,
                max_time_step=max_time_step,
                closed_threshold=closed_threshold)

        # initialize (this hidden variables are not used for outputs.)
        hidden, u_hid = self._initialize(sequence_number, use_saved_u=use_saved_u)

        '''
        print('target: ', target)
        print('initialized based on sequence_number : ', sequence_number,
              'closed threshold', closed_threshold,
              'initialized u_hid: ', u_hid,
              'initialized hidden', hidden)
        '''

        # forward pass from time step 1 to max_time_step
        for step in range(1, max_time_step + 1):
            # input
            if (step <= closed_threshold):
                hidden[:, :self.h_dim[0]] = target[:, step-1, :] \
                                            if step == 1 else target[:, step-2, :]
                # print('time ', str(step), ', now open loop')

            # recurrence
            hidden, u_hid = self._one_step_forward(hidden=hidden, u_hid=u_hid, step=step, pb=pb)

            '''# for debug
            print('time step: ', step,
                  '\noutput at this time: ', hidden[:, :self.h_dim[0]],
                  '\nu_hid: ', u_hid,
                  '\nhidden', hidden)
            '''

        output = self.get_return_output()
        return output

    def save_vars(self, hidden, u_hid, step):
        # saving
        self.save_vars_detached(
            [hidden[:, :self.h_dim[0]], hidden, u_hid], step=step)
        self.save_vars_non_detached(
            [hidden[:, :self.h_dim[0]]], step=step)
        return

    def get_return_output(self):
        return self.all_vars_non_detached['all_x']

    def get_x_dim(self):
        return self.h_dim[0]
