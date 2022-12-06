best_args = {
    'fl_cifar10': {
        'sgd': {
            0.5: {
            },
        },
        'fedavg': {
            0.5: {
            },
        },
        'fedprox': {
            0.5: {
                'mu': 0.01,
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
                'global_lr': 0.5 # 0.6 0.25 0.1 1.0 2.0  0.6
            },
        },
        'fedproto': {
            0.5: {
                'mu': 1
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
        'feddc': {
            0.5: {
                'alpha_coef': 1e-2
            },
        },
        'fedavgnorm': {
            0.5: {
                't': 0.4
            },
        },
        # 'fedours': {
        #     0.5: {
        #         't': 0.35,
        #         'w': 0.05
        #     },
        # },
        # 'fedoursnoexp': {
        #     0.5: {
        #         't': 0.35,
        #         'w': 0.4
        #     },
        # },
        # 'fedournormexp': {
        #     0.5: {
        #         't': 0.35,
        #         'w': 1.0
        #     },
        # },
        'fedournormlogexp': {
            0.5: {
                't': 0.35,
                'w': 1.0
            },
        },
        # 'fedournorm': {
        #     0.5: {
        #         't': 0.35,
        #         'w': 0.4
        #     },
        # },
        # 'fedinfonce': {
        #     0.5: {
        #         'T': 1
        #     },
        # },
        # 'barlow': {
        #     0.5: {
        #         'lambd': 0.0051
        #     },
        # },
        # 'ours': {
        #     0.5: {
        #     },
        # },
        # 'ours_reg': {
        #     0.5: {
        #         'reserve_p': 0.8,
        #     },
        # },
        # 'agem': {
        #     0.5: {
        #
        #     },
        # },
        # 'cos': {
        #     0.5: {
        #
        #         # 'grad_scale':1.0,
        #     },
        # },
        # 'fedlocal': {
        #     0.5: {
        #
        #         'm': 0.999,
        #         'log_weight': 0.5,
        #     },
        # },
        # 'fedreg': {
        #     0.5: {
        #
        #         # 'reserve_p':0.3
        #     },
        # },
    },
    'fl_cifar100': {
        'sgd': {
            0.5: {
            },
        },
        'fedavg': {
            0.5: {
            },
        },
        'fedavgnorm': {
            0.5: {
                't': 0.04
            },
        },
        # 'fedours': {
        #     0.5: {
        #         't': 0.03,
        #         'w': 0.05
        #     },
        # },
        # 'fedoursnoexp': {
        #     0.5: {
        #         't': 0.03,
        #         'w': 0.05
        #     },
        # },
        'fedprox': {
            0.5: {
                'mu': 0.001,
            },
        },
        'moon': {
            0.5: {
                'temperature': 0.5,
                'mu': 1
            },
        },
        'fedproc': {
            0.5: {

            },
        },
        # },
        # 'fedinfonce': {
        #     0.5: {
        #
        #         'T': 1
        #     },
        # },
        # 'barlow': {
        #     0.5: {
        #
        #         'lambd': 0.0051
        #     },
        # },
        # 'ours': {
        #     0.5: {
        #
        #     },
        # },
        # 'ours_reg': {
        #     0.5: {
        #
        #         'reserve_p': 0.8,
        #     },
        # },
        # 'agem': {
        #     0.5: {
        #
        #     },
        # },
        # 'cos': {
        #     0.5: {
        #         # 'grad_scale':1.0,
        #     },
        # },
        # 'fedlocal': {
        #     0.5: {
        #         'm': 0.999,
        #         'log_weight': 0.5,
        #     },
        # },
        # 'fedreg': {
        #     0.5: {
        #         # 'reserve_p':0.3
        #     },
        # },
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
}
