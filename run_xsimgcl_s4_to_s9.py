#!/usr/bin/env python3
"""Run XSimGCL S4-S9 experiments sequentially in one Python entrypoint."""

from SELFRec import SELFRec
from util.conf import ModelConf


def run_models(models):
    for model in models:
        print('=' * 80)
        print(f'Running {model}')
        conf = ModelConf(f'./conf/{model}.yaml')
        SELFRec(conf).execute()


if __name__ == '__main__':
    default_models = [
        'LightGCN_neg', 'LightGCN_dyn',
        'SGL_neg', 'SGL_dyn',
        'SimGCL_neg', 'SimGCL_dyn',
        'XSimGCL_neg', 'XSimGCL_dyn',
    ]
    run_models(default_models)
