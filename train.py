import os
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import numpy as np
import argparse
import model
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
matplotlib.use('Agg')

import joblib
from data_utils import RawDataProcess, MyDataset
import globals
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET_DIR = '/home/shijiayi/AFT/dataset'
ETFS_DIR = '/home/shijiayi/AFT/dataset/etfs'
STOCKS_DIR = '/home/shijiayi/AFT/dataset/stocks'
MODEL_DIR = '/home/shijiayi/AFT/model_save'

def train_func(train_dataloader, model):
    total_loss = 0
    for enc_inputs, dec_inputs, dec_outputs in train_dataloader:
        '''
        enc_inputs: [batch_size, src_len, feature_num]
        dec_inputs: [batch_size, tgt_len, 1]
        dec_outputs: [batch_size, tgt_len, 1]
        '''
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        # outputs: [batch_size * tgt_len, 1]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        dec_outputs = dec_outputs.float()
        loss = globals.criterion(outputs, dec_outputs.view(-1, dec_outputs.shape[-1]))
        total_loss += loss.item()
        globals.optimizer.zero_grad()
        loss.backward()
        globals.optimizer.step()
    return total_loss / len(train_dataloader)

def validate_func(val_dataloader, model):
    total_loss = 0
    for enc_inputs, dec_inputs, dec_outputs in val_dataloader:
        '''
        enc_inputs: [batch_size, src_len, feature_num]
        dec_inputs: [batch_size, tgt_len, 1]
        dec_outputs: [batch_size, tgt_len, 1]
        '''
        model.eval()
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        # outputs: [batch_size * tgt_len, 1]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        dec_outputs = dec_outputs.float()
        loss = globals.criterion(outputs, dec_outputs.view(-1, dec_outputs.shape[-1]))
        total_loss += loss.item()
    return total_loss / len(val_dataloader)

def train(mymodel, train_dataloader, val_dataloader, saved_name):
    globals.criterion = nn.L1Loss()
    globals.optimizer = optim.Adam(mymodel.parameters(), lr=globals.lr, weight_decay=globals.weight_decay)

    # print model
    print(mymodel)

    writer = SummaryWriter(log_dir="./runs/loss3")
    best_score = 100.
    for epoch in range(globals.epochs):
        train_loss = train_func(train_dataloader, mymodel)
        val_loss = validate_func(val_dataloader, mymodel)
        writer.add_scalars(
            main_tag='loss',
            tag_scalar_dict={
                'train_loss': train_loss,
                'val_loss': val_loss
            },
            global_step=epoch
        )
        logging.info('Epoch: {}\t train loss: {:.6f}\t val loss: {:.6f}'.format(epoch+1, train_loss, val_loss))
        if val_loss < best_score:
            logging.info("Save model...")
            torch.save({'model': mymodel.state_dict()}, os.path.join(MODEL_DIR, saved_name+'_best_model.pth'))
            best_score = val_loss
    writer.close()

def greedy_decoder(model, enc_input, start_data):
    '''
    Get prediction data according to the enc_input
    
    enc_input: [1, seq_len, feature_num]
    start_data: [1, 1, 1]
    '''
    enc_outputs, _ = model.encoder(enc_input.to(device))
    next_data = start_data
    i = 0
    dec_input = next_data.to(device)
    result = []
    while i < globals.predict_size:         
        dec_outputs, _, _ = model.decoder(dec_input, enc_outputs)
        projected = model.projection(dec_outputs)
        next_data = projected.data[0,-1,0]
        dec_input = torch.cat([dec_input, next_data.view([1, 1, -1]).to(device)], dim=1)
        result.append(next_data.cpu().detach().numpy().tolist())
        i += 1     
    return result

def inference(model, infer_dataloader, saved_name):
    prediction = []
    real = []
    loss = 0.
    for enc_inputs, dec_inputs, dec_outputs in infer_dataloader: 
        model.eval()
        start_data = enc_inputs[0, -1, 0].reshape(1, 1, -1)
        pred_data = greedy_decoder(model, enc_inputs, start_data)
        prediction.extend(pred_data)
        real.extend(dec_outputs.view(-1, dec_outputs.shape[-1]).squeeze(-1).cpu().detach().numpy().tolist())    
    for i in range(len(prediction)):
        loss += abs(prediction[i] - real[i])
    logging.info("Test loss: {:.6f}".format(loss / len(prediction)))
    loaded_scaler = joblib.load(os.path.join(DATASET_DIR, 'target_scaler.pkl'))
    prediction = loaded_scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
    real = loaded_scaler.inverse_transform(np.array(real).reshape(-1, 1))

    prediction = prediction.squeeze(-1)
    real = real.squeeze(-1)
    # print(real[:20])
    plt.figure(figsize=(30, 15))
    plt.plot(range(20), prediction[210:230], color='red', label='prediction')
    plt.plot(range(20), real[210:230], color='blue', label='real')
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("value")
    plt.savefig(os.path.join('imgs/', saved_name+'_predict2.jpg'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--is_train', action='store_true')
    parser.add_argument('-i', '--infer', action='store_true')
    args = parser.parse_args()

    mymodel = model.Transformer(
        d_time=2,
        enc_layers=2,
        dec_layers=2,
        d_k=4,
        d_v=4,
        d_ff=32,
        n_heads=2
    ).to(device)

    files_dict = {
        # "etfs": ['EWY' , 'IVV', 'EWT', 'DGT', 'EWD'],
        "stocks": ['NAV'] #, 'AA', 'DTE', 'MRO', 'CPT']
    }
    file_path_list = []
    for key in list(files_dict.keys()):
        if key == 'etfs':
            for name in files_dict[key]:
                file_path_list.append(os.path.join(ETFS_DIR, name+'.csv'))
        else:
            for name in files_dict[key]:
                file_path_list.append(os.path.join(STOCKS_DIR, name+'.csv'))
    for path in file_path_list:
        file_name = os.path.basename(path).split('.')[0]
        logging.info("Train on {}".format(file_name))
        rdp = RawDataProcess(path)
        data = rdp.process()
        enc_inputs, dec_inputs, dec_outputs = rdp.get_trainable_data(
            data, 
            window_size=globals.window_size, 
            predict_size=globals.predict_size,
            stride=globals.stride,
        )

        split_ratio = 0.8
        train_num = int(enc_inputs.shape[0] * split_ratio)
        val_test_num = enc_inputs.shape[0] - train_num
        val_num = val_test_num - val_test_num // 2
        train_dataset = MyDataset(enc_inputs[:train_num], dec_inputs[:train_num], dec_outputs[:train_num])
        val_dataset = MyDataset(enc_inputs[train_num:train_num+val_num], dec_inputs[train_num:train_num+val_num], dec_outputs[train_num:train_num+val_num])
        test_dataset = MyDataset(enc_inputs[train_num+val_num:], dec_inputs[train_num+val_num:], dec_outputs[train_num+val_num:])

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        logging.info("Train number: {} \t Validation number: {} \t Test number: {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))

        if args.is_train:
            train(mymodel, train_dataloader, val_dataloader, file_name)
        
        if args.infer:
            model_path = os.path.join(MODEL_DIR, file_name+'_best_model.pth')
            mymodel.load_state_dict(torch.load(model_path)['model'])
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            inference(mymodel, test_dataloader, file_name)


if __name__ == '__main__':
    main()

    
    