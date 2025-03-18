import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import torchaudio

import math

import torch
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
import pdb
import numpy as np

from torch.autograd import Variable


class ResNet50(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        h = self.features(x)

        regmap8 = self.avgpool8(h)

        logits = self.fc(regmap8.squeeze(-1).squeeze(-1))

        # return logits, regmap8
        return logits


class ResNet18_audio(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_audio, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        h = self.features(x)

        regmap8 = self.avgpool8(h)

        logits = self.fc(regmap8.squeeze(-1).squeeze(-1))

        # return logits, regmap8
        return logits, regmap8


class ResNet18(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        [B, C, T, H, W] = x.shape

        for ii in range(T):
            if ii == 0:
                h = self.features(x[:, :, ii, :, :])
                regmap8 = self.avgpool8(h)
                logits = self.fc(regmap8.squeeze(-1).squeeze(-1))
            else:
                h = self.features(x[:, :, ii, :, :])
                regmap8 = self.avgpool8(h)
                logits = logits + self.fc(regmap8.squeeze(-1).squeeze(-1))

        return logits / T


class ResNet18_ScoreFuse_FaceAudio(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_ScoreFuse_FaceAudio, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

        # Audio 
        resnet_audio = models.resnet18(pretrained=pretrained)
        num_ftrs = resnet_audio.fc.in_features
        self.features_audio = nn.Sequential(*list(resnet_audio.children())[:-2])
        self.fc_audio = nn.Linear(num_ftrs, 2)

    def forward(self, x, x_audio):

        [B, C, T, H, W] = x.shape

        for ii in range(T):
            if ii == 0:
                h = self.features(x[:, :, ii, :, :])
                regmap8 = self.avgpool8(h)
                logits = self.fc(regmap8.squeeze(-1).squeeze(-1))
            else:
                h = self.features(x[:, :, ii, :, :])
                regmap8 = self.avgpool8(h)
                logits = logits + self.fc(regmap8.squeeze(-1).squeeze(-1))

        # audio
        h = self.features_audio(x_audio)
        regmap8 = self.avgpool8(h)
        logits_audio = self.fc_audio(regmap8.squeeze(-1).squeeze(-1))

        return logits / T, logits_audio, (logits / T + logits_audio) / 2


class OpenFace_1DCNN_2GRU(nn.Module):

    def __init__(self, ):
        super(OpenFace_1DCNN_2GRU, self).__init__()

        self.CNN1D = nn.Sequential(
            nn.Conv1d(43, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(p=0.2),
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.rnn = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bias=False,
                          bidirectional=False)

        #  binary CE
        self.fc = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        [B, C, T] = x.shape  # [B, 43, T=64]

        h = self.CNN1D(x)  # [B, 128, T=8]

        # pdb.set_trace()

        h = h.permute(0, 2, 1)

        rnn_out, _ = self.rnn(h)

        # pdb.set_trace()

        logits = self.fc(rnn_out[:, -1, :])  # rnn_out [B, 7, 128]

        return logits


class OpenFace_MLP_RNN(nn.Module):

    def __init__(self, ):
        super(OpenFace_MLP_RNN, self).__init__()

        #  binary CE
        self.MLP = nn.Sequential(
            nn.Linear(43, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bias=False,
                          bidirectional=False)

        #  binary CE
        self.fc = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        [B, C, T] = x.shape  # [B, 43, T=64]

        x_temp = self.MLP(x.permute(0, 2, 1).reshape(B * T, C))

        h = x_temp.reshape(B, T, 128)

        rnn_out, _ = self.rnn(h)

        # pdb.set_trace()

        logits = self.fc(rnn_out[:, -1, :])  # rnn_out [B, 7, 128]

        return logits


class OpenFace_MLP_MLP(nn.Module):

    def __init__(self, ):
        super(OpenFace_MLP_MLP, self).__init__()

        #  binary CE
        self.MLP = nn.Sequential(
            nn.Linear(43, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        #  binary CE
        self.MLP2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16, eps=1e-6),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        [B, C, T] = x.shape  # [B, 43, T=64]

        x_temp = self.MLP(x.permute(0, 2, 1).reshape(B * T, C))

        h = x_temp.reshape(B, T, 1)

        logits = self.MLP2(h.squeeze(-1))  # rnn_out [B, 7, 128]

        return logits, h


class Affect2_MLP_MLP(nn.Module):

    def __init__(self, ):
        super(Affect2_MLP_MLP, self).__init__()

        #  binary CE
        self.MLP = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        #  binary CE
        self.MLP2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16, eps=1e-6),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        [B, C, T] = x.shape  # [B, 43, T=64]

        x_temp = self.MLP(x.permute(0, 2, 1).reshape(B * T, C))

        h = x_temp.reshape(B, T, 1)

        logits = self.MLP2(h.squeeze(-1))  # rnn_out [B, 7, 128]

        return logits


class Affect7_MLP_MLP(nn.Module):

    def __init__(self, ):
        super(Affect7_MLP_MLP, self).__init__()

        #  binary CE
        self.MLP = nn.Sequential(
            nn.Linear(7, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        #  binary CE
        self.MLP2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16, eps=1e-6),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        [B, C, T] = x.shape  # [B, 43, T=64]

        x_temp = self.MLP(x.permute(0, 2, 1).reshape(B * T, C))

        h = x_temp.reshape(B, T, 1)

        logits = self.MLP2(h.squeeze(-1))  # rnn_out [B, 7, 128]

        return logits


class OpenFace_Affect7_MLP_MLP(nn.Module):

    def __init__(self, ):
        super(OpenFace_Affect7_MLP_MLP, self).__init__()

        #  binary CE
        self.MLP = nn.Sequential(
            nn.Linear(50, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        #  binary CE
        self.MLP2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16, eps=1e-6),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        [B, C, T] = x.shape  # [B, 43, T=64]

        x_temp = self.MLP(x.permute(0, 2, 1).reshape(B * T, C))

        h = x_temp.reshape(B, T, 1)

        logits = self.MLP2(h.squeeze(-1))  # rnn_out [B, 7, 128]

        return logits, h


class OpenFace_MLP_MLP_96frame(nn.Module):

    def __init__(self, ):
        super(OpenFace_MLP_MLP_96frame, self).__init__()

        #  binary CE
        self.MLP = nn.Sequential(
            nn.Linear(43, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        #  binary CE
        self.MLP2 = nn.Sequential(
            nn.Linear(96, 48),
            nn.LayerNorm(48, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(48, 24),
            nn.LayerNorm(24, eps=1e-6),
            nn.ReLU(),
            nn.Linear(24, 2),
        )

    def forward(self, x):
        [B, C, T] = x.shape  # [B, 43, T=64]

        x_temp = self.MLP(x.permute(0, 2, 1).reshape(B * T, C))

        h = x_temp.reshape(B, T, 1)

        logits = self.MLP2(h.squeeze(-1))  # rnn_out [B, 7, 128]

        return logits


class OpenFaceGaze_MLP_MLP(nn.Module):

    def __init__(self, ):
        super(OpenFaceGaze_MLP_MLP, self).__init__()

        #  binary CE
        self.MLP = nn.Sequential(
            nn.Linear(8, 16),
            nn.LayerNorm(16, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 32),
            nn.LayerNorm(32, eps=1e-6),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        #  binary CE
        self.MLP2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16, eps=1e-6),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        [B, C, T] = x.shape  # [B, 43, T=64]

        x_temp = self.MLP(x.permute(0, 2, 1).reshape(B * T, C))

        h = x_temp.reshape(B, T, 1)

        logits = self.MLP2(h.squeeze(-1))  # rnn_out [B, 7, 128]

        return logits


class OpenFaceGaze_AllMLP(nn.Module):

    def __init__(self, ):
        super(OpenFaceGaze_AllMLP, self).__init__()

        #  binary CE
        self.MLP = nn.Sequential(
            nn.Linear(8 * 64, 256),
            nn.LayerNorm(256, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        [B, C, T] = x.shape  # [B, 43, T=64]

        logits = self.MLP(x.view(B, -1))  # rnn_out [B, 7, 128]

        return logits


class OpenFaceAU_MLP_MLP(nn.Module):

    def __init__(self, ):
        super(OpenFaceAU_MLP_MLP, self).__init__()

        #  binary CE
        self.MLP = nn.Sequential(
            nn.Linear(35, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        #  binary CE
        self.MLP2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16, eps=1e-6),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        [B, C, T] = x.shape  # [B, 43, T=64]

        x_temp = self.MLP(x.permute(0, 2, 1).reshape(B * T, C))

        h = x_temp.reshape(B, T, 1)

        logits = self.MLP2(h.squeeze(-1))  # rnn_out [B, 7, 128]

        return logits


class ResNet18_GRU(nn.Module):

    def __init__(self, pretrained=True, GRU_layers=1):
        super(ResNet18_GRU, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        self.num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.rnn = nn.GRU(input_size=self.num_ftrs, hidden_size=self.num_ftrs // 2, num_layers=GRU_layers,
                          batch_first=True,
                          bias=False, bidirectional=False)

        #  binary CE
        self.fc = nn.Linear(self.num_ftrs // 2, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        [B, C, T, H, W] = x.shape

        fs = Variable(torch.zeros(B, T, self.num_ftrs)).cuda()

        for ii in range(T):
            h = self.features(x[:, :, ii, :, :])
            embedding = self.avgpool8(h).squeeze(-1).squeeze(-1)
            fs[:, ii, :] = embedding

        # pdb.set_trace()

        out, hidden = self.rnn(fs)

        logits = self.fc(out[:, -1, :])

        return logits, out[:, -1, :]


class ResNet18_BiGRU(nn.Module):

    def __init__(self, pretrained=True, GRU_layers=1):
        super(ResNet18_BiGRU, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        self.num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.rnn = nn.GRU(input_size=self.num_ftrs, hidden_size=self.num_ftrs // 2, num_layers=GRU_layers,
                          batch_first=True,
                          bias=False, bidirectional=True)

        #  binary CE
        self.fc = nn.Linear(self.num_ftrs, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        [B, C, T, H, W] = x.shape

        fs = Variable(torch.zeros(B, T, self.num_ftrs)).cuda()

        for ii in range(T):
            h = self.features(x[:, :, ii, :, :])
            embedding = self.avgpool8(h).squeeze(-1).squeeze(-1)
            fs[:, ii, :] = embedding

        # pdb.set_trace()

        out, hidden = self.rnn(fs)

        logits = self.fc(out[:, -1, :])

        return logits


# Audio Wave2Vec model, only finetune the last transformer block and fc layer
class w2v2_model(nn.Module):
    def __init__(self, ):
        super(w2v2_model, self).__init__()

        base_model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        list_of_freezable_layers = [
            base_model.feature_extractor,
            base_model.encoder.feature_projection,
            base_model.encoder.transformer.pos_conv_embed,
            base_model.encoder.transformer.layer_norm,
            base_model.encoder.transformer.layers[0:10]
        ]
        for x in list_of_freezable_layers:
            for p in x.parameters(): p.requires_grad = False

        self.wav2vec2 = base_model
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 2, bias=True),
        )

    def forward(self, x):
        features, _ = self.wav2vec2(
            x)  # IMPORTANT: model out is a tuple. Always use features,_ to receive  the model output.
        logits = self.classifier(torch.mean(features, 1))

        return logits
