import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Modelss import Transformer
from tqdm import tqdm
torch.manual_seed(123)



def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    train_pred, _ = load_data(opt.data + 'train_pred.pkl', 'train')


    #print('[Info] Loading dev data...')
    #dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')
    test_pred, _ = load_data(opt.data + 'test_pred.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=False)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    trainpred = get_dataloader(train_pred, opt.batch_size, shuffle=False)
    testpred = get_dataloader(test_pred, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types, trainpred, testpred
"""
def load_data2(name):
    data = torch.load(name,map_location=torch.device('cuda'))
    return data
"""
def load_data1(opt):
    def load_data2(name):
        with open(name, 'rb') as f:    
            data = pickle.load(f, encoding='latin-1')
            return data
    
    attn1=load_data2(opt.data + 'attn/' + 'attn1.pkl').unsqueeze_(0)
    attn2=load_data2(opt.data +'attn/' + 'attn2.pkl').unsqueeze(0)
    attn3=load_data2(opt.data + 'attn/' +'attn3.pkl').unsqueeze(0)
    attn4=load_data2(opt.data + 'attn/' +'attn4.pkl').unsqueeze(0)
    attn5=load_data2(opt.data + 'attn/' +'attn5.pkl').unsqueeze(0)
    attn6=load_data2(opt.data + 'attn/' +'attn6.pkl').unsqueeze(0)
    attn7=load_data2(opt.data +'attn/' + 'attn7.pkl').unsqueeze(0)
    attn8=load_data2(opt.data +'attn/' + 'attn8.pkl').unsqueeze(0)
    enc1=load_data2(opt.data + 'attn/' +'enc1.pkl').unsqueeze(0)
    enc2=load_data2(opt.data +'attn/' + 'enc2.pkl').unsqueeze(0)
    enc3=load_data2(opt.data +'attn/' + 'enc3.pkl').unsqueeze(0)
    enc4=load_data2(opt.data +'attn/' + 'enc4.pkl').unsqueeze(0)
    enc5=load_data2(opt.data +'attn/' + 'enc5.pkl').unsqueeze(0)
    enc6=load_data2(opt.data +'attn/' + 'enc6.pkl').unsqueeze(0)
    enc7=load_data2(opt.data +'attn/' + 'enc7.pkl').unsqueeze(0)
    enc8=load_data2(opt.data + 'attn/' +'enc8.pkl').unsqueeze(0)
    attn=torch.cat((attn1.,attn2.unsqueeze(0),attn3.unsqueeze(0),attn4.unsqueeze(0),attn5.unsqueeze(0),attn6.unsqueeze(0),attn7.unsqueeze(0),attn8.unsqueeze(0)),dim=0)
    enc=torch.cat((enc1.unsqueeze(0),enc2.unsqueeze(0),enc3.unsqueeze(0),enc4.unsqueeze(0),enc5.unsqueeze(0),enc6.unsqueeze(0),enc7.unsqueeze(0),enc8.unsqueeze(0)),dim=0)
    return attn, enc

 
def train_epoch(model, training_data,trainpred, attn, enc, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """
    model.train()
    enc_out_ll=[]
    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    """ prepare data """
    for i in training_data:
         event_time, time_gap, event_type , xy= map(lambda x: x.to(opt.device), i)
    for i in trainpred:
        event_time_pred, time_gap_pred, event_type_pred, xy_pred= map(lambda x: x.to(opt.device), i)

    """ forward """
    optimizer.zero_grad()

    dec_out, prediction = model(event_type, event_time, xy,event_time_pred, event_type_pred, xy_pred,attn, enc)

    """ update parameters """
    optimizer.step()

    """ backward """
    # negative log-likelihood
    event_ll, non_event_ll = Utils.log_likelihood(model, dec_out,event_time_pred, event_type_pred)
    event_loss = -torch.sum(event_ll - non_event_ll)

    # type prediction
    pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type_pred, pred_loss_func)

    # time prediction
    se = Utils.time_loss(prediction[1], event_time_pred)

    # SE is usually large, scale it to stabilize training
    scale_time_loss = 100
    loss = event_loss + pred_loss + se / scale_time_loss
    loss.backward()

    """ update parameters """
    optimizer.step()

    """ note keeping """
    total_event_ll += -event_loss.item()
    total_time_se += se.item()
    total_event_rate += pred_num_event.item()
    total_num_event += event_type.ne(Constants.PAD).sum().item()
    # we do not predict the first event
    total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
    
    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def eval_epoch(model, validation_data,testpred,attn, enc, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """
    

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():

        """ prepare data """
        for i in validation_data:
            event_time, time_gap, event_type , xy= map(lambda x: x.to(opt.device), i)
        for i in testpred:
            event_time_pred, time_gap_pred, event_type_pred, xy_pred= map(lambda x: x.to(opt.device), i)


        """ forward """
        dec_out, prediction = model(event_type, event_time, xy,event_time_pred, event_type_pred, xy_pred,attn, enc)

        """ compute loss """
        event_ll, non_event_ll = Utils.log_likelihood(model, dec_out, event_time_pred, event_type_pred)
        event_loss = -torch.sum(event_ll - non_event_ll)
        _, pred_num = Utils.type_loss(prediction[0], event_type_pred, pred_loss_func)
        se = Utils.time_loss(prediction[1], event_time_pred)

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
 
        rmse = np.sqrt(total_time_se / total_num_pred)
        return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def train(model, training_data, validation_data,trainpred,testpred, attn, enc, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data,trainpred, attn, enc, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time = eval_epoch(model, validation_data,testpred,attn, enc, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))


        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, acc=valid_type, rmse=valid_time))

        scheduler.step()


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda')
    torch.autograd.set_detect_anomaly(True)
    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types, trainpred,testpred = prepare_dataloader(opt)
    attn, enc = load_data1(opt)

    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader,trainpred, testpred,attn, enc, optimizer, scheduler, pred_loss_func, opt)
    
if __name__ == '__main__':
    main()
