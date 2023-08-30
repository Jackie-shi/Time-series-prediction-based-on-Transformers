import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
matplotlib.use('Agg')
DATASET_DIR = '/home/shijiayi/AFT/dataset'
ETFS_DIR = '/home/shijiayi/AFT/dataset/etfs'
STOCKS_DIR = '/home/shijiayi/AFT/dataset/stocks'

def dataset_report():
    rows = []
    dirs = [ETFS_DIR, STOCKS_DIR]
    for idx, dir in enumerate(dirs):
        files_list = os.listdir(dir)
        for file_name in files_list:
            file_path = os.path.join(dir, file_name)
            df = pd.read_csv(file_path)
            # get number of lines & is null
            data_num = len(df)
            is_null = df.isnull().any().tolist()
            flag = 1 if True in is_null else 0
            rows.append([idx, file_name, data_num, flag])
    
    cols = ['Dir', 'Stock Code', 'Lines', 'IS_null']
    df_report = pd.DataFrame(rows, columns=cols)
    df_report.to_csv(os.path.join(DATASET_DIR, 'report.csv'), index=False)

class RawDataProcess(object):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read_data(self):
        df = pd.read_csv(self.file_path)
        return df

    def process(self):
        df = self.read_data()
        # fill the 0.0 Open price
        now_close = 0.0
        for idx, row in df.iterrows():
            prev_close = now_close
            if row['Open'] == 0.:
                df.loc[idx, 'Open'] = prev_close
            now_close = row['Close']
        df = df.drop([0])

        df.set_index('Date', inplace=True)
        processed_df = df.copy()
        # get Moving Average
        column_names = list(df.columns)
        for name in column_names:
            processed_df[name] = df[name].rolling(10).mean()

        # import pdb
        # pdb.set_trace()
        # processed_df = processed_df.dropna() 
        # target_df = pd.DataFrame(processed_df['Open'], index=processed_df.index)
        plt.figure(figsize=(30, 15))
        # plt.plot(range(20), prediction[:20], color='red', label='prediction')
        # plt.plot(range(20), real[:20], color='blue', label='real')
        # plt.legend()
        # plt.xlabel("time")
        # plt.ylabel("value")
        # plt.savefig(os.path.join('imgs/', saved_name+'_predict2.jpg'))
        


        # import sys
        # sys.exit()
        # get return
        processed_df = processed_df.pct_change()
        processed_df = processed_df.dropna()
        
        return processed_df

    def get_trainable_data(self, data, window_size, predict_size, stride):
        encoder_input = []
        decoder_input = []
        decoder_output = []

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        # save the scaler
        joblib.dump(scaler, os.path.join(DATASET_DIR, 'scaler.pkl'))

        target_df = pd.DataFrame(data['Open'], index=data.index)
        _ = scaler.fit_transform(target_df)
        joblib.dump(scaler, os.path.join(DATASET_DIR, 'target_scaler.pkl'))

        df_scaled = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        open = df_scaled['Open'].tolist()
        features = df_scaled.values.tolist()
        for i in range(0, len(features) - window_size - predict_size + 1, stride):
            en_input_data = features[i:i+window_size]
            de_input_start = i + window_size - 1
            de_output_start = de_input_start + 1
            de_input_data = open[de_input_start:de_input_start+predict_size]
            de_output_data = open[de_output_start:de_output_start+predict_size]
            
            encoder_input.append(en_input_data)
            decoder_input.append(de_input_data)
            decoder_output.append(de_output_data)
        
        return np.array(encoder_input), np.array(decoder_input)[:, :, np.newaxis], np.array(decoder_output)[:, :, np.newaxis]

class MyDataset(Dataset):
    def __init__(self, encoder_input, decoder_input, decoder_output):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.decoder_output = decoder_output
    
    def __len__(self):
        return self.encoder_input.shape[0]
    
    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_input[idx], self.decoder_output[idx]

if __name__ == '__main__':
    rdp = RawDataProcess(os.path.join(ETFS_DIR, 'SMH.csv'))
    data = rdp.process()
    x, y, z = rdp.get_trainable_data(data, 10, 4)
    print(x.shape)
    print(y.shape)
    print(z)
