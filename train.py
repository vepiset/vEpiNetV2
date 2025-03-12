import numpy as np
import pandas as pd

from base_trainer.net_work import Train
from train_config import config as cfg

import setproctitle
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

setproctitle.setproctitle("train_spike_model")


def main():
    n_fold = 5

    data = pd.read_csv(cfg.DATA.data_file)
   
    for fold in range(n_fold):
        ###build dataset
        train_ind = data[data['train_val'] == 0].index.values
        train_data = data.iloc[train_ind].copy()
        val_ind = data[data['train_val'] == 1].index.values
        val_data = data.iloc[val_ind].copy()

    
        trainer = Train(train_df=train_data,
                        val_df=val_data,
                        fold=fold)

        ### train
        trainer.custom_loop()



if __name__ == '__main__':
    main()
