from model.united_model import ModelPVRNNSoftmax


class ConfigSoftmax():
    model_class = ModelPVRNNSoftmax
    data_dir_common = 'data/wcst/limited_sm/'
    data_size = 18
    epoch_size = 10000  # 200000
    run_name = 'sample/short-epoch'

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
        'print_every': epoch_size / 1000,
        'save_every': epoch_size / 400,
        'test_every': epoch_size / 2,

        # Hyper parameters for common model
        'max_time_step': 512,  # set None if align data length
        'h_dim': [20, 10, 10],  # contexts (hidden) units of RNN
        'time_scale': [2, 8, 32],
        'device': 'cpu',  # 'cpu' or 'cuda'

        # Hyper parameters for PVRNN
        'x_dim': 2,  # dimension of sequential data
        'x_dim_reza_model': 20,
        'z_dim': [2, 2, 2],
        'meta_prior': [1.0, 1.0, 1.0],
        'init_meta_prior': [0.01, 0.01, 0.01],
        'initial_gaussian_regularizer': True,
        'use_hidden_for_posterior': True,
        'use_bottom_up_signal': True,

        # Error regression etc.
        'test_data_size': 18,
        'free_gen_step': 10000,
        'test_data_size_tar': 18,
        'test_data_size_free': 18,
        'test_data_size_ereg_train': 18,
        'test_data_size_ereg_test': 18,
        'ereg_lr': 0.09,
        'ereg_window_size': 15,
        'ereg_iteration': 20,
        'ereg_pred_step': 1,
        'ereg_meta_prior': [0.001, 0.01, 0.1],
        'traversal_values': 21,
    }
