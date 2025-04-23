import os
from glob import glob
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.lcnn_model_2 import LCNN
from src.moe_model import Classic_MOE, Enhanced_MOE
from torch.utils.data import DataLoader

from src.training_utils import train_epoch, valid_epoch, LoadTrainData_LCNN
from src.utils import read_yaml, seed_everything


def load_asvspoof_data(file_name, base_path, label_dir):
    df = pd.read_csv(os.path.join(label_dir, file_name), sep=' ', header=None)
    df['label'] = df[4].map({'bonafide': 0, 'spoof': 1})
    df['path'] = df[1].apply(lambda x: os.path.join(base_path, x + '.flac'))
    return df[['path', 'label']]


def load_for_data(base_dir, subset):
    real_files = glob(os.path.join(base_dir, subset, 'real', '*.wav'))
    fake_files = glob(os.path.join(base_dir, subset, 'fake', '*.wav'))

    df_real = pd.DataFrame(real_files, columns=['path'])
    df_real['label'] = 0

    df_fake = pd.DataFrame(fake_files, columns=['path'])
    df_fake['label'] = 1

    return pd.concat([df_real, df_fake])


def main(config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    expert_1 = LCNN()
    expert_2 = LCNN()
    expert_3 = LCNN()
    expert_4 = LCNN()

    expert_1 = (expert_1).to(device)
    expert_2 = (expert_2).to(device)
    expert_3 = (expert_3).to(device)
    expert_4 = (expert_4).to(device)

    expert_1.load_state_dict(torch.load(f'{config["save_model_folder"]}/LCNN_ASVSPOOF.pth', map_location=device))
    expert_2.load_state_dict(torch.load(f'{config["save_model_folder"]}/LCNN_FOR.pth', map_location=device))
    expert_3.load_state_dict(torch.load(f'{config["save_model_folder"]}/LCNN_ADD.pth', map_location=device))
    expert_4.load_state_dict(torch.load(f'{config["save_model_folder"]}/LCNN_INTHEWILD.pth', map_location=device))

    moe_model = Classic_MOE(experts=[expert_1, expert_2, expert_3, expert_4])
    # moe_model = Enhanced_MOE(experts=[expert_1, expert_2, expert_3, expert_4])
    moe_model = (moe_model).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, moe_model.parameters()),
                                  lr=0.0001,
                                  weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=config['T_max'],
                                                           eta_min=config['eta_min'])

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # LOAD DATASETS
    df_train_list = []
    df_dev_list = []

    # ASVSPOOF 2019
    label_dir = f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_cm_protocols'
    base_paths = {
        'train': f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_train/flac/',
        'dev': f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_dev/flac/',
        'eval': f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_eval/flac/'
    }

    df_train_asv = load_asvspoof_data('ASVspoof2019.LA.cm.train.trn.txt', base_paths['train'], label_dir)
    df_dev_asv = load_asvspoof_data('ASVspoof2019.LA.cm.dev.trl.txt', base_paths['dev'], label_dir)

    df_train_list.append(df_train_asv)
    df_dev_list.append(df_dev_asv)

    # FAKE OR REAL
    base_dir = f'{config["for_path"]}/for-original-wav'

    df_train_for = load_for_data(base_dir, 'training')
    df_dev_for = load_for_data(base_dir, 'validation')

    df_train_list.append(df_train_for)
    df_dev_list.append(df_dev_for)

    # ADD 2022
    df_train_ADD = pd.read_csv(f'{config["add2022_path"]}/ADD_train_dev/label/train_label.txt', sep=' ', header=None)
    df_train_ADD['path'] = f'{config["add2022_path"]}/ADD_train_dev/train/' + df_train_ADD[0].astype(str)
    df_train_ADD['label'] = df_train_ADD[1].map({'genuine': 0, 'fake': 1})
    df_train_ADD = df_train_ADD[['path', 'label']]

    df_dev_ADD = pd.read_csv(f'{config["add2022_path"]}/ADD_train_dev/label/dev_label.txt', sep=' ', header=None)
    df_dev_ADD['path'] = f'{config["add2022_path"]}/ADD_train_dev/dev/' + df_dev_ADD[0].astype(str)
    df_dev_ADD['label'] = df_dev_ADD[1].map({'genuine': 0, 'fake': 1})
    df_dev_ADD = df_dev_ADD[['path', 'label']]

    df_train_list.append(df_train_ADD)
    df_dev_list.append(df_dev_ADD)

    # IN THE WILD
    df_meta = pd.read_csv(f'{config["InTheWild_path"]}/meta.csv')
    df_meta['path'] = df_meta['file'].apply(lambda x: f'{config["InTheWild_path"]}/wavs/' + x)
    df_meta['label'] = df_meta['label'].map({'spoof': 1, 'bona-fide': 0})
    df_meta = df_meta[['path', 'label']]

    df_train_itw, df_dev_itw = train_test_split(df_meta, test_size=0.4, stratify=df_meta['label'],
                                                random_state=config['seed'])
    df_dev_itw, df_eval_itw = train_test_split(df_dev_itw, test_size=0.5, stratify=df_dev_itw['label'],
                                               random_state=config['seed'])

    df_train_list.append(df_train_itw)
    df_dev_list.append(df_dev_itw)

    df_train = pd.concat(df_train_list, ignore_index=True)
    df_dev = pd.concat(df_dev_list, ignore_index=True)

    # Define training dataloader
    d_label_trn = dict(zip(df_train['path'], df_train['label']))
    file_train = list(df_train['path'])

    train_set = LoadTrainData_LCNN(list_IDs=file_train, labels=d_label_trn, win_len=config['win_len'])
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True,
                              num_workers=config['num_thread'], prefetch_factor=config['prefetch_factor'])
    del train_set, d_label_trn

    # Define validation dataloader
    d_label_dev = dict(zip(df_dev['path'], df_dev['label']))
    file_dev = list(df_dev['path'])

    dev_set = LoadTrainData_LCNN(list_IDs=file_dev, labels=d_label_dev, win_len=config['win_len'])
    dev_loader = DataLoader(dev_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_thread'],
                            prefetch_factor=config['prefetch_factor'])
    del dev_set, d_label_dev

    # Start training and validation
    best_acc = 0
    best_loss = 100
    early_stopping = 0
    for epoch in range(config['num_epochs']):
        if early_stopping < config['early_stopping']:

            for param_group in optimizer.param_groups:
                print(f"Learning Rate: {param_group['lr']}")

            running_loss, train_accuracy = train_epoch(train_loader, moe_model, optimizer, criterion, device)
            with torch.no_grad():
                valid_loss, valid_accuracy = valid_epoch(dev_loader, moe_model, criterion, device)

            scheduler.step(valid_loss)
            print(f'Epoch: {epoch} - Train Loss: {running_loss:.5f} - Val Loss: '
                  f'{valid_loss:.5f} - Train Acc: {train_accuracy:.2f} - Val Acc: {valid_accuracy:.2f}')

            if valid_loss < best_loss:
                print(f'Best model found at epoch {epoch}')
                torch.save(moe_model.state_dict(), os.path.join(config['save_model_folder'], config['moe_model_name']))
                early_stopping = 0

                best_loss = min(valid_loss, best_loss)
                best_acc = valid_accuracy
            else:
                early_stopping += 1
        else:
            print(f'Training stopped after {epoch} epochs - Best Val Acc {best_acc:.2f}')
            break


if __name__ == '__main__':

    this_folder = Path(__file__).parent

    config_path = this_folder / 'config' / 'lcnn_config.yaml'
    config = read_yaml(config_path)

    seed_everything(config['seed'])

    config['moe_model_name'] = f'MOE_CLASSIC.pth'
    print(f"Model name: {config['moe_model_name']}")

    main(config)
