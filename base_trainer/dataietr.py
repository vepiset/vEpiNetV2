import random
import copy
import numpy as np
import sys
import os

sys.path.append('.')
from utils.logger import logger


class AlaskaDataIter():
    def __init__(self, df,
                 training_flag=True, shuffle=True):

        self.training_flag = training_flag
        self.shuffle = shuffle
        self.raw_data_set_size = None

        self.df = df
        logger.info(' contains%d samples  %d pos' % (len(self.df), np.sum(self.df['target'] == 1)))
        logger.info(' contains%d samples' % len(self.df))

        if training_flag:
            # sample there
            self.df = self.filter(self.df)

        logger.info(' After filter contains%d samples  %d pos' % (len(self.df), np.sum(self.df['target'] == 1)))
        logger.info(' After filter contains%d samples' % len(self.df))

        self.leads_nm = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5',
                         'T6',
                         'Fz', 'Cz', 'Pz',
                         'PG1', 'PG2', 'A1', 'A2',
                         'EKG1', 'EKG2', 'EMG1',
                         'EMG2', 'EMG3', 'EMG4']

        self.leads_dict = {value: index for index, value in enumerate(self.leads_nm)}


        self.left_brain = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'T5', 'T3', 'F7']
        self.right_brain = ['Fp2', 'F4', 'C4', 'P4', 'O2', 'T6', 'T4', 'F8']

    def filter(self, df):

        df = copy.deepcopy(df)
        pos_indx = df['target'] == 1
        pos_df = df[pos_indx]

        neg_indx = df['target'] == 0
        neg_df = df[neg_indx]

        neg_df = neg_df.sample(frac=1)

        dst_df = neg_df
        for i in range(1):
            dst_df = dst_df._append(pos_df)
        dst_df.reset_index()

        return dst_df

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def norm(self, wave):

        wave[:23, ...] = wave[:23, ...] / 1e-3
        wave[23:, ...] = wave[23:, ...] * 1e-2

        # 心电和肌电
        heart_wave = wave[23, :] - wave[24, :]

        muscle_wave1 = wave[25, :] - wave[26, :]

        muscle_wave2 = wave[27, :] - wave[28, :]

        heart_muscle = np.stack([heart_wave, muscle_wave1, muscle_wave2], axis=0)

        wave_26 = np.concatenate([wave[:23, ...], heart_muscle], axis=0)


        return wave_26

    def roll(self, waves, strength=1600 // 2):

        start = random.randint(-strength, strength)
        waves = np.roll(waves, start, axis=1)

        return waves

    def xshuffle(self, wave):

        # 获取数组的形状
        n_channels, n_samples = wave.shape

        # 创建一个包含0到n_channels-1的数组，表示通道的索引
        channel_indices = np.arange(n_channels)

        # 将通道索引数组进行shuffle
        np.random.shuffle(channel_indices)

        # 对原始数组进行0通道shuffle
        shuffled_wave = wave[channel_indices]

        return shuffled_wave

    def avg_lead(self, waves):

        # copy一份，防止原地修改
        waves = copy.deepcopy(waves)

        meadn = np.mean(waves[:19, :], axis=0)
        data = waves[:19, :] - meadn

        return data

    def union_polor_lead(self, waves):
        waves = copy.deepcopy(waves)

        left_leads = self.left_brain
        right_leads = self.right_brain
        a1_lead = ['A1']
        a2_lead = ['A2']

        left_lead_indx = [self.leads_dict[x] for x in left_leads]
        right_leads_indx = [self.leads_dict[x] for x in right_leads]
        a1_lead_indx = [self.leads_dict[x] for x in a1_lead]
        a2_lead_indx = [self.leads_dict[x] for x in a2_lead]
        left_lead = waves[left_lead_indx] - waves[a1_lead_indx]
        right_lead = waves[right_leads_indx] - waves[a2_lead_indx]

        data = np.concatenate([left_lead, right_lead], axis=0)
        return data

    def bipolar_lead(self, waves):
        waves = copy.deepcopy(waves)

        left_leads = self.left_brain
        right_leads = self.right_brain

        leads = []
        for i in range(len(left_leads)):
            if i < len(left_leads) - 1:
                tmp_lead = waves[self.leads_dict[left_leads[i]]] - waves[self.leads_dict[left_leads[i + 1]]]
            else:
                tmp_lead = waves[self.leads_dict[left_leads[i]]] - waves[self.leads_dict[left_leads[0]]]
            leads.append(tmp_lead)

        for i in range(len(right_leads)):
            if i < len(right_leads) - 1:
                tmp_lead = waves[self.leads_dict[right_leads[i]]] - waves[self.leads_dict[right_leads[i + 1]]]
            else:
                tmp_lead = waves[self.leads_dict[right_leads[i]]] - waves[self.leads_dict[right_leads[0]]]
            leads.append(tmp_lead)

        data = np.concatenate([leads], axis=0)

        return data

    def lead(self, waves):

        avg_lead = self.avg_lead(waves)
        union_polor_lead = self.union_polor_lead(waves)
        bipolor_lead = self.bipolar_lead(waves)

        return avg_lead, union_polor_lead, bipolor_lead

        #return avg_lead

    def single_map_func(self, dp, is_training):
        """Data augmentation function."""

        fname = dp['file_path']
        label = dp['target']
        video_feature = np.array(eval(dp['video_feature']), dtype=float)
        try:
            waves = np.load(fname)
        except:
            print("=====fname:", fname)
            waves = np.zeros(shape=[29, 2000])
            label = 0
            video_feature = np.array([0.0] * 6, dtype=float)

        waves = self.norm(waves)

        avg_lead, union_polor_lead, bipolor_lead = self.lead(waves)

        if is_training and random.uniform(0, 1) < 1:
            waves[:19, :] = self.xshuffle(waves[:19, :])
            avg_lead = self.xshuffle(avg_lead)
            union_polor_lead = self.xshuffle(union_polor_lead)
            bipolor_lead = self.xshuffle(bipolor_lead)

        waves = np.concatenate([waves, avg_lead, union_polor_lead, bipolor_lead], axis=0)

        label = np.expand_dims(label, -1)

        C, L = waves.shape

        if L < 2000:
            waves = np.pad(waves, ((0, 0), (0, 2000 - L)), 'constant', constant_values=0)
        elif L > 2000:
            waves = waves[:, 2000]
        waves = np.ascontiguousarray(waves)

        return waves, label,video_feature
