"""Cells for MTRNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(threshold=10000)


class MaskMaker(object):
    @classmethod
    def make_mask_matrix(cls,
                         h_dim,
                         mode='',
                         z_dim=0,
                         x_dim=0,
                         pb_size=0,
                         connect_layer=[]):
        if mode == 'mtrnn_normal':
            mask = cls.make_mtrnn_normal_mask(h_dim=h_dim)
        elif mode == 'mtrnn_pb':
            mask = cls.make_mtrnn_pb_mask(h_dim=h_dim, pb_size=pb_size)
        elif mode == 'sctrnn_init':
            mask = cls.make_sctrnn_init_mask(h_dim=h_dim, x_dim=x_dim)
        elif mode == 'sctrnn_pb':
            mask = cls.make_sctrnn_pb_mask(h_dim=h_dim,
                                           x_dim=x_dim,
                                           pb_size=pb_size)
        elif mode == 'pvrnn_recurrence':
            mask = cls.make_pvrnn_recurrence_mask(h_dim=h_dim, z_dim=z_dim)
        elif mode == 'pvrnn_recurrence_off-bottom-up':
            mask = cls.make_pvrnn_recurrence_offsignal_mask(h_dim=h_dim,
                                                            z_dim=z_dim)
        elif mode == 'pvrnn_inference':
            mask = cls.make_pvrnn_inference_mask(h_dim=h_dim, z_dim=z_dim)
        elif mode == 'pvrnn_progressive':
            mask = cls.make_pvrnn_progressive_mask(h_dim=h_dim,
                                                   z_dim=z_dim,
                                                   connect_layer=connect_layer)
        else:
            print('mask make error!!!!!!!!!')

        return mask

    @classmethod
    def make_mtrnn_normal_mask(cls, h_dim):
        """ Mask of normal MTRNN
        """
        num_time_layer = len(h_dim)

        if num_time_layer == 1:
            mask = torch.ones((h_dim[0], h_dim[0]))
        else:
            mask = torch.zeros((sum(h_dim), sum(h_dim)))

            for i in range(num_time_layer):
                if i == 0:
                    from_size = h_dim[i] + h_dim[i + 1]
                    from_start_idx = 0
                    to_start_idx = 0
                elif i > 0 and i < num_time_layer - 1:
                    from_size = h_dim[i - 1] + h_dim[i] + h_dim[i + 1]
                    to_start_idx += h_dim[i - 1]
                    if i != 1: from_start_idx += h_dim[i - 2]

                elif i == num_time_layer - 1:
                    from_size = h_dim[i - 1] + h_dim[i]
                    to_start_idx += h_dim[i - 1]
                    if i != 1: from_start_idx += h_dim[i - 2]

                mask[to_start_idx:to_start_idx+h_dim[i], \
                     from_start_idx:from_start_idx+from_size] \
                     = torch.ones((h_dim[i], from_size))
        return mask

    @classmethod
    def make_mtrnn_pb_mask(cls, h_dim, pb_size):
        num_time_layer = len(h_dim)

        # mask from hidden to hidden
        mask_from_h = cls.make_mtrnn_normal_mask(h_dim)

        # mask from pb units to hidden units
        mask_from_pb = torch.zeros((sum(h_dim), pb_size))
        mask_from_pb[-1 * h_dim[num_time_layer-1]:,:] \
            = torch.ones((h_dim[num_time_layer-1],pb_size))

        mask = torch.cat([mask_from_h, mask_from_pb], dim=1)

        return mask

    @classmethod
    def make_sctrnn_init_mask(cls, h_dim, x_dim):
        num_time_layer = len(h_dim)

        # mask from hidden to hidden
        mask_from_h = cls.make_mtrnn_normal_mask(h_dim)

        # mask from h to x
        mask_from_x = torch.zeros((sum(h_dim), x_dim))
        mask_from_x[:h_dim[0], :x_dim] = torch.ones((h_dim[0], x_dim))

        mask = torch.cat([mask_from_x, mask_from_h], dim=1)

        return mask

    @classmethod
    def make_sctrnn_pb_mask(cls, h_dim, x_dim, pb_size):
        num_time_layer = len(h_dim)

        # mask from hidden and input to hidden layer
        mask_from_xh = cls.make_sctrnn_init_mask(h_dim=h_dim, x_dim=x_dim)

        # mask from pb units to hidden units
        mask_from_pb = torch.zeros((sum(h_dim), pb_size))
        mask_from_pb[-1 * h_dim[num_time_layer-1]:,:] \
            = torch.ones((h_dim[num_time_layer-1], pb_size))

        mask = torch.cat([mask_from_xh, mask_from_pb], dim=1)

        return mask

    @classmethod
    def make_pvrnn_inference_mask(cls, h_dim, z_dim):
        """
        This is mask for posterior (inference model) and prior model of PVRNN.
        From d(hidden) to z, and only same layer is passed and other layer is
        disconnected.
        """
        num_time_layer = len(h_dim)

        mask = torch.zeros((sum(z_dim), sum(h_dim)))
        from_start_idx, to_start_idx = 0, 0
        for i in range(num_time_layer):
            mask[to_start_idx:to_start_idx+z_dim[i], \
                 from_start_idx:from_start_idx+h_dim[i]] \
                 = torch.ones((z_dim[i], h_dim[i]))
            to_start_idx += z_dim[i]
            from_start_idx += h_dim[i]

        return mask

    @classmethod
    def make_pvrnn_recurrence_from_z_to_h(cls, h_dim, z_dim):
        num_time_layer = len(h_dim)

        mask = torch.zeros((sum(h_dim), sum(z_dim)))
        from_start_idx, to_start_idx = 0, 0
        for i in range(num_time_layer):
            mask[to_start_idx:to_start_idx+h_dim[i], \
                        from_start_idx:from_start_idx+z_dim[i]] \
                        = torch.ones((h_dim[i], z_dim[i]))
            to_start_idx += h_dim[i]
            from_start_idx += z_dim[i]
        return mask

    @classmethod
    def make_pvrnn_recurrence_mask(cls, h_dim, z_dim):
        num_time_layer = len(h_dim)

        # mask from hidden to hidden
        mask_from_h = cls.make_mtrnn_normal_mask(h_dim)

        # mask from z to h
        mask_from_z = cls.make_pvrnn_recurrence_from_z_to_h(h_dim=h_dim,
                                                            z_dim=z_dim)
        mask = torch.cat([mask_from_h, mask_from_z], dim=1)

        return mask

    @classmethod
    def make_pvrnn_recurrence_offsignal_mask(cls, h_dim, z_dim):
        num_time_layer = len(h_dim)

        # mask from hidden to hidden
        if num_time_layer == 1:
            mask_from_h = torch.ones((h_dim[0], h_dim[0]))
        else:
            mask_from_h = torch.zeros((sum(h_dim), sum(h_dim)))
            start_idx = 0
            for i in range(num_time_layer):
                if i < num_time_layer - 1: from_size = h_dim[i] + h_dim[i + 1]
                elif i == num_time_layer - 1: from_size = h_dim[i]
                mask_from_h[start_idx:start_idx+h_dim[i], \
                            start_idx:start_idx+from_size] \
                            = torch.ones((h_dim[i], from_size))
                start_idx += h_dim[i]

        # mask from z to h
        mask_from_z = cls.make_pvrnn_recurrence_from_z_to_h(h_dim=h_dim,
                                                            z_dim=z_dim)

        mask = torch.cat([mask_from_h, mask_from_z], dim=1)

        return mask

    @classmethod
    def make_pvrnn_progressive_mask(cls, h_dim, z_dim, connect_layer):
        num_time_layer = len(h_dim)

        # mask from hidden to hidden
        mask_from_h = cls.make_mtrnn_normal_mask(h_dim=h_dim)

        # mask from z to h
        mask_from_z = cls.make_pvrnn_recurrence_from_z_to_h(h_dim=h_dim,
                                                            z_dim=z_dim)

        from_start_idx, to_start_idx = 0, 0
        for i in range(num_time_layer):
            if connect_layer[i]:
                # connected
                mask_from_z[to_start_idx:to_start_idx+h_dim[i], \
                     from_start_idx:from_start_idx+z_dim[i]] \
                     = torch.ones((h_dim[i], z_dim[i]))
            else:
                # disconnected
                mask_from_z[to_start_idx:to_start_idx+h_dim[i], \
                     from_start_idx:from_start_idx+z_dim[i]] \
                     = torch.zeros((h_dim[i], z_dim[i]))
            to_start_idx += h_dim[i]
            from_start_idx += z_dim[i]

        mask = torch.cat([mask_from_h, mask_from_z], dim=1)
        return mask


class MaskedLinear(nn.Linear):
    """
    same as Linear except has a configurable mask on the weights
    This class is written by https://github.com/karpathy/pytorch-made.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        # self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8)))
        self.mask.data.copy_(mask)
        # self.weight.data[self.mask==0] = self.mask[self.mask==0]

        # print('mask is set.', '\nmask', self.mask)
        # print('\nweight', self.weight, '\nbias', self.bias)
        return

    def forward(self, input):
        output = F.linear(input, self.mask * self.weight, self.bias)
        '''
        print('forward calc mask linear.',
              '\nmask: ', self.mask,
              '\nmask size: ', self.mask.size(),
              '\nweight: ', self.weight,
              '\nbias: ', self.bias,
              '\ninput: ', input,
              '\ninput size: ', input.size(),
              '\noutput:', output)
        '''
        return output


class CellMTRNN(nn.Module):
    ''' one step recurrence using multi time-scale model.
    '''
    def __init__(self, h_dim, time_scale, mask, device):
        super().__init__()

        self.h_dim = h_dim
        self.time_scale = time_scale
        self.num_time_layer = len(time_scale)
        self.mask = mask
        self.device = device

        # time scale
        time_scale = torch.tensor([
            time_scale[i] for i, size in enumerate(self.h_dim)
            for j in range(size)
        ]).to(self.device)
        time_scale_inv = torch.ones(sum(self.h_dim)).to(
            self.device) / time_scale
        self.time_scale = time_scale
        self.time_scale_inv = time_scale_inv

        # layer
        masked_layer = MaskedLinear(mask.shape[1], mask.shape[0])
        masked_layer.set_mask(mask)
        self.layer = masked_layer

        # activation function
        self.activation = nn.Tanh()

        '''
        print(
        'inv of time scale: ', self.time_scale_inv,
        '\nmask: ', self.layer.mask)
        '''

    def set_mask(self, mask):
        self.layer.set_mask(mask)
        return

    def forward(self, hidden, u_hid):
        # time continuous calculation
        u_hid = (1 - self.time_scale_inv) * u_hid + \
                (self.time_scale_inv) * self.layer(hidden)

        # activation
        hidden = self.activation(u_hid)

        return hidden, u_hid
