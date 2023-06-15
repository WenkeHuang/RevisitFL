best_args = {
    'fl_cifar10': {
        'sgd': {
            0.5: {
            },
        },
        'fpl': {
            0.5: {
                'infoNCET': 0.2
            },
        },
        'fedlc': {
            0.5: {
                'tau': 1.0
            },
        },
        'fedavg': {
            0.5: {
            },
        },
        'fedcos': {
            0.5: {
            },
        },
        'fedprox': {
            0.5: {
                'mu': 0.01,
            },
        },
        'fednova': {
            0.5: {
                'rho': 0.9,
            },
        },
        'moon': {
            0.5: {
                'temperature': 0.5,
                'mu': 5
            },
        },
        'fedopt': {
            0.5: {
                'global_lr': 0.5
            },
        },
        'fedproto': {
            0.5: {
                'mu': 0.5
            },
        },
        'fedproc': {
            0.5: {

            },
        },
        'fedrs': {
            0.5: {
                # 'alpha': 0.3
                'alpha': 0.5
                # 'alpha': 0.7
                # 'alpha': 0.6
                # 'alpha': 0.4
            },
        },
        'scaffold': {
            0.5: {
                'global_lr': 0.05
            },
        },
        'feddyn': {
            0.5: {
                # 'reg_lamb': 1e-3
                # 'reg_lamb': 1e-4
                'reg_lamb': 1e-2
                # 'reg_lamb': 1e-1
                # 'reg_lamb': 1
            },
        },
        'fedavgnorm': {
            0.5: {
                't': 0.35
            },
        },
        'fedlogexp': {
            0.5: {
                'w': 1.0
            },
        },
        'fedournormlogexp': {
            0.5: {
                # 't': 0.35,
                'w': 1.0
            },
        },
    },
    'fl_cifar100': {
        'sgd': {
            0.5: {
            },
        },
        'fpl': {
            0.5: {
                'infoNCET': 0.02
            },
        },
        'fedproto': {
            0.5: {
                'mu': 0.5
            },
        },
        'fedlc': {
            0.5: {
                'tau': 1.0
            },
        },
        'fedavg': {
            0.5: {
            },
        },
        'fedcos': {
            0.5: {
            },
        },
        'fedavgnorm': {
            0.5: {
                't': 0.1
            },
        },
        'fedlogexp': {
            0.5: {
                'w': 1.0
            },
        },
        'fedournormlogexp': {
            0.5: {
                # 't': 0.1,
                'w': 1.0
            },
        },
        'fedprox': {
            0.5: {
                'mu': 0.001,
            },
        },
        'fednova': {
            0.5: {
                'rho': 0.9,
            },
        },
        'scaffold': {
            0.5: {
                'global_lr': 0.05
            },
        },
        'feddyn': {
            0.5: {
                # 'reg_lamb': 1e-3
                # 'reg_lamb': 1e-4
                'reg_lamb': 1e-2
                # 'reg_lamb': 1e-1
                # 'reg_lamb': 1
            },
        },
        'moon': {
            0.5: {
                'temperature': 0.5,
                'mu': 1
            },
        },
        'fedopt': {
            0.5: {
                'global_lr': 0.5
            },
        },
        'fedproc': {
            0.5: {

            },
        },
        'fedrs': {
            0.5: {
                'alpha': 0.5
            },
        },
    },
    'fl_tiny_imagenet': {
        'sgd': {
            0.5: {
            },
        },
        'fpl': {
            0.5: {
                'infoNCET': 0.02
            },
        },
        'fedproto': {
            0.5: {
                'mu': 0.5
            },
        },
        'fedlc': {
            0.5: {
                'tau': 1.0
            },
        },
        'fedavg': {
            0.5: {
            },
        },
        'fedcos': {
            0.5: {
            },
        },
        'fedavgnorm': {
            0.5: {
                't': 0.05
            },
        },
        'fedlogexp': {
            0.5: {
                'w': 1.0
            },
        },
        'fedournormlogexp': {
            0.5: {
                # 't': 0.1,
                'w': 1.0
            },
        },
        'fedprox': {
            0.5: {
                'mu': 0.001,
            },
        },
        'fednova': {
            0.5: {
                'rho': 0.9,
            },
        },
        'scaffold': {
            0.5: {
                'global_lr': 0.05
            },
        },
        'feddyn': {
            0.5: {
                'reg_lamb': 1e-2
            },
        },
        'moon': {
            0.5: {
                'temperature': 0.5,
                'mu': 1
            },
        },
        'fedopt': {
            0.5: {
                'global_lr': 0.5
            },
        },
        'fedproc': {
            0.5: {
            },
        },
        'fedalign': {
            0.5: {
                'mu': 0.45
            },
        },
        'fedrs': {
            0.5: {
                'alpha': 0.5
            },
        },
        'feddc': {
            0.5: {
                'alpha_coef': 1e-2
            },
        },
    },
    'fl_mnist': {
        'sgd': {
            0.5: {
            },
        },
        'fpl': {
            0.5: {
                'infoNCET': 0.2
            },
        },
        'fedlc': {
            0.5: {
                'tau': 1.0
            },
        },
        'fedavg': {
            0.5: {
            },
        },
        'fedcos': {
            0.5: {
            },
        },
        'fedprox': {
            0.5: {
                'mu': 0.01,
            },
        },
        'fednova': {
            0.5: {
                'rho': 0.9,
            },
        },
        'moon': {
            0.5: {
                'temperature': 0.5,
                'mu': 5
            },
        },
        'fedopt': {
            0.5: {
                'global_lr': 0.5
            },
        },
        'fedproto': {
            0.5: {
                'mu': 0.5
            },
        },
        'fedproc': {
            0.5: {

            },
        },
        'fedrs': {
            0.5: {
                # 'alpha': 0.3
                'alpha': 0.5
                # 'alpha': 0.7
                # 'alpha': 0.6
                # 'alpha': 0.4
            },
        },
        'scaffold': {
            0.5: {
                'global_lr': 0.05
            },
        },
        'feddyn': {
            0.5: {
                # 'reg_lamb': 1e-3
                # 'reg_lamb': 1e-4
                'reg_lamb': 1e-2
                # 'reg_lamb': 1e-1
                # 'reg_lamb': 1
            },
        },
        'fedavgnorm': {
            0.5: {
                't': 0.35
            },
        },
        'fedlogexp': {
            0.5: {
                'w': 1.0
            },
        },
        'fedournormlogexp': {
            0.5: {
                # 't': 0.35,
                'w': 1.0
            },
        },
    },
}
