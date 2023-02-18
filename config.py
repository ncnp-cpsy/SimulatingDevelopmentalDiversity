from model.united_model import ModelPVRNNSoftmax


class ConfigSoftmax():
    model_class = ModelPVRNNSoftmax
    data_dir_common = 'data/wcst/stoch_limited-sm_bias-switch_always_reversal/'
    data_size = 4
    epoch_size = 20000
    run_name = 'sample/002'

    params = {
        # Model name.
        'model_class_name': model_class.__name__,

        # Write directry
        'out_dir_name': './rslt/' + run_name + '/',

        # Data directory
        'train_data_path': data_dir_common + 'data/data',
        'test_data_path': data_dir_common + 'data/test',
        'train_true_path': data_dir_common + 'data_true/data_true',
        'test_true_path': data_dir_common + 'data_true/test_true',
        'sep_load': ' ',

        # Setting for training
        'lr': 0.001,
        'seed': 425,
        'data_size': data_size,
        'mini_batch_size': data_size,  # int(data_size / 2),
        'epoch_size': epoch_size,
        'print_every': epoch_size / 100,
        'save_every': epoch_size / 10,
        'test_every': epoch_size / 2,

        # Hyper parameters for common model
        'max_time_step': 16,  # set None if align data length
        'h_dim': [20, 10, 10],  # contexts (hidden) units of RNN
        'time_scale': [2, 8, 32],
        'device': 'cpu',  # 'cpu' or 'cuda'

        # Hyper parameters for PVRNN
        'x_dim': 2,  # dimension of sequential data
        'x_dim_reza_model': 20,
        'z_dim': [2, 2, 2],
        'init_sigma': 1,
        'meta_prior': [1.0, 1.0, 1.0],
        'init_meta_prior': [0.01, 0.01, 0.01],
        'initial_gaussian_regularizer': True,
        'use_hidden_for_posterior': True,
        'use_bottom_up_signal': True,

        # Error regression etc.
        'test_gen_size': 4,
        'free_gen_step': 2000,
        'test_data_size': 4,
        'ereg_lr': 0.1,
        'ereg_window_size': 15,
        'ereg_iteration': 5,
        'ereg_pred_step': 1,
        'ereg_meta_prior': [0.000001, 0.000001, 0.000001],
        'traversal_values': 21,
    }
