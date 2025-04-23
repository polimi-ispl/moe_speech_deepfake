import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from src.audio_utils import read_audio


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class LoadTrainData_LCNN(Dataset):
    def __init__(self, list_IDs, labels, win_len, fs=16000):
        '''self.list_IDs    : list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = win_len
        self.fs = fs
        self.win_len_samples = int(self.win_len * self.fs)

        df = pd.DataFrame(labels.items(), columns=['path', 'label'])
        self.real_list = list(df[df['label'] == 0]['path'])
        self.fake_list = list(df[df['label'] == 1]['path'])

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Balance real and fake samples in a batch
        if index % 2 == 0:
            track_name = random.choice(self.real_list)
        else:
            track_name = random.choice(self.fake_list)

        x, fs = read_audio(track_name, trim=False)
        y = self.labels[track_name]
        audio_len = len(x)

        if audio_len < self.win_len_samples:
            x = pad(x, self.win_len_samples)
            audio_len = len(x)

        last_valid_start_sample = audio_len - self.win_len_samples
        if not last_valid_start_sample == 0:
            start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        else:
            start_sample = 0
        x_win = x[start_sample : start_sample + self.win_len_samples]
        x_win = Tensor(x_win)

        return x_win, y


class LoadEvalData_LCNN(Dataset):
    def __init__(self, list_IDs, labels, win_len, fs=16000):
        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = win_len
        self.fs = fs
        self.win_len_samples = int(self.win_len * self.fs)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        track = self.list_IDs[index]
        x, fs = read_audio(track, trim=False)
        y = self.labels[track]
        audio_len = len(x)

        if audio_len < self.win_len_samples:
            x = pad(x, self.win_len_samples)

        # TEST ON THE MIDDLE WINDOW
        start_sample = int(0.5*(len(x) - self.win_len_samples))
        x_win = x[start_sample: start_sample + self.win_len_samples]
        x_inp = Tensor(x_win)

        return x_inp, y, track


def train_epoch(train_loader, model, optim, criterion, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    model.train()

    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader)):

        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy


def valid_epoch(data_loader, model, criterion, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0

    model.eval()

    for batch_x, batch_y in tqdm(data_loader, total=len(data_loader)):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)

    valid_accuracy = (num_correct / num_total) * 100
    running_loss /= num_total
    return running_loss, valid_accuracy


def eval_model(model, data_loader, save_path, device):

    model.eval()
    total_gating_weights = None
    batch_count = 0

    for batch_x, batch_y, utt_id in tqdm(data_loader, total=len(data_loader)):

        fname_list = []
        pred_list = []
        label_list = []

        batch_x = batch_x.to(device)

        batch_out, gating_weights = model(batch_x)
        batch_out = nn.Softmax(dim=1)(batch_out)
        batch_score = (batch_out[:, 0]).data.cpu().numpy().ravel()

        if total_gating_weights is None:
            total_gating_weights = np.sum(gating_weights.detach().cpu().numpy(), axis=0)
        else:
            total_gating_weights += np.sum(gating_weights.detach().cpu().numpy(), axis=0)
            batch_count += gating_weights.shape[0]

        averaged_gating_weights = total_gating_weights / batch_count  # shape: (num_experts)
        np.save(save_path.replace('.txt', '.npy'), averaged_gating_weights)

        fname_list.extend(utt_id)
        pred_list.extend(batch_score.tolist())
        label_list.extend(batch_y.tolist())

        with open(save_path, 'a+') as fh:
            for f, pred, lab in zip(fname_list, pred_list, label_list):
                fh.write('{} {} {}\n'.format(f, pred, lab))
        fh.close()
    print('Scores saved to {}'.format(save_path))