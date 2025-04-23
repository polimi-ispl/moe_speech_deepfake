import os
from glob import glob
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.lcnn_model import LCNN
from src.training_utils import train_epoch, valid_epoch, LoadTrainData_LCNN
from src.utils import read_yaml, init_weights, seed_everything


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

    model = LCNN()

    model = (model).to(device)
    model.apply(init_weights)

    if os.path.exists(os.path.join(config['save_model_folder'], config['model_pretrained'])):
        model.load_state_dict(torch.load(os.path.join(config['save_model_folder'], config['model_pretrained']), map_location=device))
        print('Model loaded : {}'.format(config['model_pretrained']))
    else:
        print('No pretrained model loaded')

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['eta_min'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    criterion = criterion.to(device)

    # LOAD DATASETS
    df_train_list = []
    df_dev_list = []
    df_eval_list = []

    if config['training_dataset'] == 'ASVSPOOF19':

        label_dir = f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_cm_protocols'
        base_paths = {
            'train': f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_train/flac/',
            'dev': f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_dev/flac/',
            'eval': f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_eval/flac/'
        }

        df_train_asv = load_asvspoof_data('ASVspoof2019.LA.cm.train.trn.txt', base_paths['train'], label_dir)
        df_dev_asv = load_asvspoof_data('ASVspoof2019.LA.cm.dev.trl.txt', base_paths['dev'], label_dir)
        df_eval_asv = load_asvspoof_data('ASVspoof2019.LA.cm.eval.trl.txt', base_paths['eval'], label_dir)

        df_train_list.append(df_train_asv)
        df_dev_list.append(df_dev_asv)
        df_eval_list.append(df_eval_asv)


    elif config['training_dataset'] == 'FakeOrReal':

        base_dir = f'{config["for_path"]}/for-original-wav'

        df_train_for = load_for_data(base_dir, 'training')
        df_dev_for = load_for_data(base_dir, 'validation')
        df_eval_for = load_for_data(base_dir, 'testing')

        df_train_list.append(df_train_for)
        df_dev_list.append(df_dev_for)
        df_eval_list.append(df_eval_for)


    elif config['training_dataset'] == 'ADD2022':

        df_train_ADD = pd.read_csv(f'{config["add2022_path"]}/ADD_train_dev/label/train_label.txt', sep=' ', header=None)
        df_train_ADD['path'] = f'{config["add2022_path"]}/ADD_train_dev/train/' + df_train_ADD[0].astype(str)
        df_train_ADD['label'] = df_train_ADD[1].map({'genuine': 0, 'fake': 1})
        df_train_ADD = df_train_ADD[['path', 'label']]

        df_dev_ADD = pd.read_csv(f'{config["add2022_path"]}/ADD_train_dev/label/dev_label.txt', sep=' ', header=None)
        df_dev_ADD['path'] = f'{config["add2022_path"]}/ADD_train_dev/dev/' + df_dev_ADD[0].astype(str)
        df_dev_ADD['label'] = df_dev_ADD[1].map({'genuine': 0, 'fake': 1})
        df_dev_ADD = df_dev_ADD[['path', 'label']]

        df_eval_ADD = pd.read_csv(f'{config["add2022_path"]}/ADD2022_test/track1_label.txt', sep=' ', header=None)
        df_eval_ADD['path'] = f'{config["add2022_path"]}/ADD2022_test/track1test/' + df_eval_ADD[0].astype(str)
        df_eval_ADD['label'] = df_eval_ADD[1].map({'genuine': 0, 'fake': 1})
        df_eval_ADD = df_eval_ADD[['path', 'label']]

        df_train_list.append(df_train_ADD)
        df_dev_list.append(df_dev_ADD)
        df_eval_list.append(df_eval_ADD)


    elif config['training_dataset'] == 'InTheWild':

        df_meta = pd.read_csv(f'{config["InTheWild_path"]}/meta.csv')
        df_meta['path'] = df_meta['file'].apply(lambda x: f'{config["InTheWild_path"]}/wavs/' + x)
        df_meta['label'] = df_meta['label'].map({'spoof': 1, 'bona-fide': 0})
        df_meta = df_meta[['path', 'label']]

        df_train_itw, df_dev_itw = train_test_split(df_meta, test_size=0.4, stratify=df_meta['label'], random_state=config['seed'])
        df_dev_itw, df_eval_itw = train_test_split(df_dev_itw, test_size=0.5, stratify=df_dev_itw['label'], random_state=config['seed'])

        df_train_list.append(df_train_itw)
        df_dev_list.append(df_dev_itw)
        df_eval_list.append(df_eval_itw)


    else:
        raise ValueError(f'Training dataset {config["training_dataset"]} not recognized')


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

            running_loss, train_accuracy = train_epoch(train_loader, model, optimizer, criterion, device)
            with torch.no_grad():
                valid_loss, valid_accuracy = valid_epoch(dev_loader, model, criterion, device)

            scheduler.step(valid_loss)
            print(f'Epoch: {epoch} - Train Loss: {running_loss:.5f} - Val Loss: '
                        f'{valid_loss:.5f} - Train Acc: {train_accuracy:.2f} - Val Acc: {valid_accuracy:.2f}')

            if valid_loss < best_loss:
                print(f'Best model found at epoch {epoch}')
                torch.save(model.state_dict(), os.path.join(config['save_model_folder'], config['model_name']))
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

    # Decide which dataset to train on ['ASVSPOOF19', 'FakeOrReal', 'ADD2022', 'InTheWild']
    config['training_dataset'] = 'ASVSPOOF19'

    config['model_name'] = f'LCNN_{config["training_dataset"]}.pth'
    print(f"Model name: {config['model_name']}")

    main(config)

