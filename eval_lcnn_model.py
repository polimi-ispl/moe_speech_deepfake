import os
import random
from glob import glob
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.lcnn_model import LCNN
from src.training_utils import eval_model, LoadEvalData_LCNN
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


def main(model_path, save_folder, config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LCNN()

    model = (model).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print('EVALUATION - Model loaded : {}'.format(os.path.basename(model_path)))

    # ASVSPOOF 2019 DATASET
    label_dir = f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_cm_protocols'
    base_paths = {
        'train': f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_train/flac/',
        'dev': f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_dev/flac/',
        'eval': f'{config["asvspoof_path"]}/LA/ASVspoof2019_LA_eval/flac/'
    }

    df_eval_asv = load_asvspoof_data('ASVspoof2019.LA.cm.eval.trl.txt', base_paths['eval'], label_dir)

    # FAKE OR REAL DATASET
    base_dir = f'{config["for_path"]}/for-original-wav'
    df_eval_for = load_for_data(base_dir, 'testing')

    # ADD 2022 DATASET
    df_eval_ADD = pd.read_csv(f'{config["add2022_path"]}/ADD2022_test/track1_label.txt', sep=' ', header=None)
    df_eval_ADD['path'] = f'{config["add2022_path"]}/ADD2022_test/track1test/' + df_eval_ADD[0].astype(str)
    df_eval_ADD['label'] = df_eval_ADD[1].map({'genuine': 0, 'fake': 1})
    df_eval_ADD = df_eval_ADD[['path', 'label']]

    # IN THE WILD DATASET
    df_meta = pd.read_csv(f'{config["InTheWild_path"]}/meta.csv')
    df_meta['path'] = df_meta['file'].apply(lambda x: f'{config["InTheWild_path"]}/wavs/' + x)
    df_meta['label'] = df_meta['label'].map({'spoof': 1, 'bona-fide': 0})
    df_meta = df_meta[['path', 'label']]

    _, df_dev_itw = train_test_split(df_meta, test_size=0.4, stratify=df_meta['label'], random_state=config['seed'])
    _, df_eval_itw = train_test_split(df_dev_itw, test_size=0.5, stratify=df_dev_itw['label'], random_state=config['seed'])

    # PURDUE DATASET
    df_fake1 = glob(f'{config["purdue_path"]}/*/*.wav')
    df_fake2 = glob(f'{config["purdue_path"]}/*/*/*.wav')

    df_lj = glob(f'{config["ljspeech_path"]}/*.wav')
    df_ls1 = glob(f'{config["librispeech_path"]}/dev-clean/*/*/*.flac')
    df_ls2 = glob(f'{config["librispeech_path"]}/train-clean-100/*/*/*.flac')
    df_ls2 = random.sample(df_ls2, 11126 - len(df_ls1)) # take the correct number of files

    df_fake = pd.DataFrame(df_fake1 + df_fake2, columns=['path'])
    df_fake['label'] = 1

    df_real = pd.DataFrame(df_lj + df_ls1 + df_ls2, columns=['path'])
    df_real['label'] = 0

    df_purdue = pd.concat([df_fake, df_real], ignore_index=True)

    # TIMIT-TTS DATASET
    df_timit1 = glob(f'{config["timit_tts_path"]}/CLEAN/single_speaker/*/*.wav')
    df_timit2 = glob(f'{config["timit_tts_path"]}/CLEAN/multi_speaker/*/*/*.wav')
    df_timit_real = glob(f'{config["vidtimit_path"]}/*/audio/*.wav')

    df_fake = pd.DataFrame(df_timit1 + df_timit2, columns=['path'])
    df_fake['label'] = 1
    df_real = pd.DataFrame(df_timit_real, columns=['path'])
    df_real['label'] = 0

    df_timit = pd.concat([df_fake, df_real], ignore_index=True)

    # EVALUATE MODEL

    for dataset, df_eval in zip(['asvspoof', 'fakeorreal', 'inthewild', 'purdue', 'timit', 'add2022'], [df_eval_asv, df_eval_for, df_eval_itw, df_purdue, df_timit, df_eval_ADD]):

        result_path = os.path.join(save_folder, dataset, os.path.basename(model_path).replace('.pth', '.txt'))

        if not os.path.exists(os.path.join(save_folder, dataset)):
            os.makedirs(os.path.join(save_folder, dataset))

        if os.path.exists(result_path):
            print("Save path exists - Deleting file")
            os.remove(result_path)

        d_label_eval = dict(zip(df_eval['path'], df_eval['label']))
        file_eval = list(df_eval['path'])

        eval_set = LoadEvalData_LCNN(list_IDs=file_eval, labels=d_label_eval, win_len=config['win_len'])
        eval_loader = DataLoader(eval_set, batch_size=config['batch_size'], shuffle=False, drop_last=False,
                                  num_workers=config['num_thread'], prefetch_factor=config['prefetch_factor'])
        del eval_set, d_label_eval

        eval_model(model, eval_loader, result_path, device)


if __name__ == '__main__':

    this_folder = Path(__file__).parent

    config_path = this_folder / 'config' / 'lcnn_config.yaml'
    config = read_yaml(config_path)

    seed_everything(config['seed'])

    model_list = glob(os.path.join(config['save_model_folder'], '*.pth'))
    model_list = [os.path.basename(x) for x in model_list]

    for model_name in model_list:
        model_path = os.path.join(config['save_model_folder'], model_name)

        print(f"Evaluating model {model_name}")
        main(model_path, config['save_results_folder'], config)



