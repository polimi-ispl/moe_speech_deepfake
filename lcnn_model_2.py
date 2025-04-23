import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class LCNN(nn.Module):
    def __init__(self, num_class=2):
        super(LCNN, self).__init__()

        # Feature Extraction Part (First Part)
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5),
                               padding=(2, 2), stride=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1))
        self.batchnorm6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(3, 3),
                               padding=(1, 1), stride=(1, 1))
        self.maxpool9 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batchnorm10 = nn.BatchNorm2d(48)
        self.conv11 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(1, 1),
                                padding=(0, 0), stride=(1, 1))
        self.batchnorm13 = nn.BatchNorm2d(48)
        self.conv14 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.maxpool16 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv17 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1),
                                padding=(0, 0), stride=(1, 1))
        self.batchnorm19 = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.batchnorm22 = nn.BatchNorm2d(32)
        self.conv23 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                                padding=(0, 0), stride=(1, 1))
        self.batchnorm25 = nn.BatchNorm2d(32)
        self.conv26 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.maxpool28 = nn.AdaptiveMaxPool2d((16, 8))

        # Classification Part (Second Part)
        self.fc29 = nn.Linear(32 * 16 * 8, 128)
        self.batchnorm31 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.7)
        self.fc32 = nn.Linear(64, num_class)

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )
        self.softmax = nn.Softmax(dim=1)

    def mfm2(self, x):
        out1, out2 = torch.chunk(x, 2, 1)
        return torch.max(out1, out2)

    def mfm3(self, x):
        n, c, y, z = x.shape
        out1, out2, out3 = torch.chunk(x, 3, 1)
        res1 = torch.max(torch.max(out1, out2), out3)
        tmp1 = out1.flatten()
        tmp1 = tmp1.reshape(len(tmp1), -1)
        tmp2 = out2.flatten()
        tmp2 = tmp2.reshape(len(tmp2), -1)
        tmp3 = out3.flatten()
        tmp3 = tmp3.reshape(len(tmp3), -1)
        res2 = torch.cat((tmp1, tmp2, tmp3), 1)
        res2 = torch.median(res2, 1)[0]
        res2 = res2.reshape(n, -1, y, z)
        return torch.cat((res1, res2), 1)


    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x)+1e-6
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)

        # Forward pass through the Feature Extraction Part
        features = self.feature_extraction(x)
        # Forward pass through the Classification Part
        logits, embeddings = self.classification(features)
        output = self.softmax(logits)
        return output, embeddings

    def get_embeddings(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x)+1e-6
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)

        x = self.feature_extraction(x)

        x = self.conv17(x)
        x = self.mfm2(x)
        x = self.batchnorm19(x)
        x = self.conv20(x)
        x = self.mfm2(x)
        x = self.batchnorm22(x)
        x = self.conv23(x)
        x = self.mfm2(x)
        x = self.batchnorm25(x)
        x = self.conv26(x)
        x = self.mfm2(x)
        x = self.maxpool28(x)

        x = x.view(-1, 32 * 16 * 8)
        emb = self.mfm2((self.fc29(x)))
        # x = self.batchnorm31(x)
        # logits = self.fc32(x)
        return emb

    def feature_extraction(self, x):
        # Part 1: Feature Extraction
        x = self.conv1(x.unsqueeze(1))
        x = self.mfm2(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.mfm2(x)
        x = self.batchnorm6(x)
        x = self.conv7(x)
        x = self.mfm2(x)
        x = self.maxpool9(x)
        x = self.batchnorm10(x)
        x = self.conv11(x)
        x = self.mfm2(x)
        x = self.batchnorm13(x)
        x = self.conv14(x)
        x = self.mfm2(x)
        x = self.maxpool16(x)

        return x

    def classification(self, x):

        x = self.conv17(x)
        x = self.mfm2(x)
        x = self.batchnorm19(x)
        x = self.conv20(x)
        x = self.mfm2(x)
        x = self.batchnorm22(x)
        x = self.conv23(x)
        x = self.mfm2(x)
        x = self.batchnorm25(x)
        x = self.conv26(x)
        x = self.mfm2(x)
        x = self.maxpool28(x)

        # Part 2: Classification
        x = x.view(-1, 32 * 16 * 8)
        emb_128 = self.fc29(x)
        x = self.mfm2(emb_128)
        emb_64 = self.batchnorm31(x)
        logits = self.fc32(emb_64)

        return logits, emb_64

    def freeze_residual_part(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.maxpool3.parameters():
            param.requires_grad = False
        for param in self.conv4.parameters():
            param.requires_grad = False
        for param in self.batchnorm6.parameters():
            param.requires_grad = False
        for param in self.conv7.parameters():
            param.requires_grad = False
        for param in self.maxpool9.parameters():
            param.requires_grad = False
        for param in self.batchnorm10.parameters():
            param.requires_grad = False
        for param in self.conv11.parameters():
            param.requires_grad = False
        for param in self.batchnorm13.parameters():
            param.requires_grad = False
        for param in self.conv14.parameters():
            param.requires_grad = False
        for param in self.maxpool16.parameters():
            param.requires_grad = False


    def unfreeze_residual_part(self):
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.maxpool3.parameters():
            param.requires_grad = True
        for param in self.conv4.parameters():
            param.requires_grad = True
        for param in self.batchnorm6.parameters():
            param.requires_grad = True
        for param in self.conv7.parameters():
            param.requires_grad = True
        for param in self.maxpool9.parameters():
            param.requires_grad = True
        for param in self.batchnorm10.parameters():
            param.requires_grad = True
        for param in self.conv11.parameters():
            param.requires_grad = True
        for param in self.batchnorm13.parameters():
            param.requires_grad = True
        for param in self.conv14.parameters():
            param.requires_grad = True
        for param in self.maxpool16.parameters():
            param.requires_grad = True


    def freeze_processing_part(self):
        for param in self.conv17.parameters():
            param.requires_grad = False
        for param in self.batchnorm19.parameters():
            param.requires_grad = False
        for param in self.conv20.parameters():
            param.requires_grad = False
        for param in self.batchnorm22.parameters():
            param.requires_grad = False
        for param in self.conv23.parameters():
            param.requires_grad = False
        for param in self.batchnorm25.parameters():
            param.requires_grad = False
        for param in self.conv26.parameters():
            param.requires_grad = False
        for param in self.maxpool28.parameters():
            param.requires_grad = False

        # for param in self.mfm2.parameters():
        #     param.requires_grad = False
        for param in self.fc29.parameters():
            param.requires_grad = False
        for param in self.batchnorm31.parameters():
            param.requires_grad = False
        for param in self.fc32.parameters():
            param.requires_grad = False

    def unfreeze_processing_part(self):
        for param in self.conv17.parameters():
            param.requires_grad = True
        for param in self.batchnorm19.parameters():
            param.requires_grad = True
        for param in self.conv20.parameters():
            param.requires_grad = True
        for param in self.batchnorm22.parameters():
            param.requires_grad = True
        for param in self.conv23.parameters():
            param.requires_grad = True
        for param in self.batchnorm25.parameters():
            param.requires_grad = True
        for param in self.conv26.parameters():
            param.requires_grad = True
        for param in self.maxpool28.parameters():
            param.requires_grad = True

        # for param in self.mfm2.parameters():
        #     param.requires_grad = True
        for param in self.fc29.parameters():
            param.requires_grad = True
        for param in self.batchnorm31.parameters():
            param.requires_grad = True
        for param in self.fc32.parameters():
            param.requires_grad = True