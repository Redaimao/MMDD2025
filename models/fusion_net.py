from copy import deepcopy

import argparse

import torch

from models.ResNet_Xception import *
from models import ResNet_Xception
import sys

sys.path.append("..")
from modules.transformer import TransformerEncoder


class SimpleConcat(nn.Module):
    def __init__(self, dim=1):
        super(SimpleConcat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class SELayer(nn.Module):
    """
    SE-concatenation: first concatenate all the embeddings from different modality then perform SE attention.
    reference: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """

    def __init__(self, args):
        super(SELayer, self).__init__()
        # = args.reduction   #=16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(args.channel, args.channel // args.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(args.channel // args.reduction, args.channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FusionModule(nn.Module):
    def __init__(self, args):
        super(FusionModule, self).__init__()

        self.fusion_type = args.fusion_type  # 'concat' 'transformer' 'senet'
        self.modalities = args.modalities
        self.is_fusion = args.fusion
        combined_dim = len(self.modalities) * 64

        if 'v' in self.modalities:  # openface modality
            # self.v = True
            self.vision_model = getattr(ResNet_Xception, args.v_model)()
            self.v_projector = nn.Conv1d(args.v_dim, 64, kernel_size=1, padding=0, bias=False)

        if 'a' in self.modalities:  # audio modality
            # self.a = True
            self.audio_model = getattr(ResNet_Xception, args.a_model)()
            self.a_projector = nn.Conv1d(args.a_dim, 64, kernel_size=1, padding=0, bias=False)

        if 'f' in self.modalities:  # face modality
            # self.f = True
            self.face_model = getattr(ResNet_Xception, args.f_model)()
            self.f_projector = nn.Conv1d(args.f_dim, 64, kernel_size=1, padding=0, bias=False)

        if self.fusion_type == 'concat':
            self.fusion = SimpleConcat()
        elif self.fusion_type == 'transformer':
            self.fusion = self.transformer_attention(args)
        elif self.fusion_type == 'senet':
            self.fusion = self.se_fusion(args)
        else:
            raise Exception("Undefined fusion type!")

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(combined_dim // 2, 2),
        )

    def se_fusion(self, params):
        fusion_nets = []
        for i in range(len(self.modalities)):
            fusion_nets.append(SELayer(params).to(params.device))
        return fusion_nets

    def transformer_attention(self, params):
        fusion_nets = []
        for i in range(len(self.modalities)):
            fusion_nets.append(TransformerEncoder(embed_dim=params.embed_dim,
                                                  num_heads=params.num_heads,
                                                  layers=params.layers,
                                                  attn_dropout=params.attn_dropout,
                                                  relu_dropout=params.relu_dropout,
                                                  res_dropout=params.res_dropout,
                                                  embed_dropout=params.embed_dropout,
                                                  attn_mask=params.attn_mask).to(params.device))
        return fusion_nets

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self, vision, audio, face):
        feature_list = []
        if 'v' in self.modalities:
            v_logit, v_hidden = self.vision_model(vision)
           # print('v_hidden shape', v_hidden.shape)
            v_hidden = self.v_projector(v_hidden)  # #[b, 64, 1]
           # print('v_hidden shape', v_hidden.shape)
            feature_list.append(v_hidden)
        else:
            v_logit = None

        if 'a' in self.modalities:
            a_logit, a_hidden = self.audio_model(audio)
            #print('a_hidden shape', a_hidden.shape)

            a_hidden = self.a_projector(a_hidden.squeeze(-1))  # #[b, 64, 1]
           # print('a_hidden shape', a_hidden.shape)
            feature_list.append(a_hidden)
        else:
            a_logit = None

        if 'f' in self.modalities:
            f_logit, f_hidden = self.face_model(face)
          #  print('f_hidden shape', f_hidden.shape)

            f_hidden = self.f_projector(f_hidden.unsqueeze(-1))  # [b, 64, 1]
         #   print('f_hidden shape', f_hidden.shape)
            feature_list.append(f_hidden)
        else:
            f_logit = None

        if self.is_fusion:
            if self.fusion_type == 'concat':
                # oncat all the features and input to a classifier
                fused_logit = self.classifier(self.fusion(feature_list).squeeze(-1))
            elif self.fusion_type == 'senet':
                # each modality is input into its senet module, the the outputs are concatenated for classification
                res = [m(x.unsqueeze(-1)) for x, m in zip(feature_list, self.fusion)]
                res_tensor = torch.cat(res, dim=1).squeeze(-1).squeeze(-1)  # torch.Size([8, 192])
                fused_logit = self.classifier(res_tensor)
            elif self.fusion_type == 'transformer':
                # transformer attention between x1-x2 or x1-x2-x3
                if len(feature_list) == 2:
                    x1, x2 = feature_list
                    x1, x2 = x1.permute(2, 0, 1), x2.permute(2, 0, 1)
                    x1_to_x2 = self.fusion[0](x1, x2, x2).squeeze(0)
                    x2_to_x1 = self.fusion[1](x2, x1, x1).squeeze(0)
                    res_tensor = torch.cat([x1_to_x2, x2_to_x1], dim=-1)
                else:
                    x1, x2, x3 = feature_list
                    x1, x2, x3 = x1.permute(2, 0, 1), x2.permute(2, 0, 1), x3.permute(2, 0, 1)
                    x1_to_x2 = self.fusion[0](x1, x2, x2).squeeze(0)
                    x2_to_x3 = self.fusion[1](x2, x3, x3).squeeze(0)
                    x3_to_x1 = self.fusion[2](x3, x1, x1).squeeze(0)
                    res_tensor = torch.cat([x1_to_x2, x2_to_x3, x3_to_x1], dim=-1)
                fused_logit = self.classifier(res_tensor)
            else:
                fused_logit = None
        else:
            fused_logit = None
        #print("logit shape:", v_logit.shape, a_logit.shape, f_logit.shape, fused_logit.shape)
        return v_logit, a_logit, f_logit, fused_logit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--device', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  # default=0.001
    parser.add_argument('--batchsize', type=int, default=16, help='initial batchsize')  # 32
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='gamma of optim.lr_scheduler.StepLR, decay of lr')  # 0.1
    parser.add_argument('--echo_batches', type=int, default=1, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=30, help='total training epochs')
    parser.add_argument('--log', type=str, default="CNN_AllMLP_RLtrail_OpenFaceGaze", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    # config for individual modal
    parser.add_argument('--fusion', action='store_true', default='false', help='true when fusion module is used')
    parser.add_argument('--fusion_modal', type=str, default='FusionModule')

    parser.add_argument('--modalities', type=str, default='vaf', help='modalities in v-affect+openface, a-audio, '
                                                                      'f-face frames')

    parser.add_argument('--v_model', type=str, default='OpenFace_MLP_MLP')
    parser.add_argument('--a_model', type=str, default='ResNet18_audio')
    parser.add_argument('--f_model', type=str, default='ResNet18_GRU')

    # dimensions for each modality (the embedding size)
    parser.add_argument('--v_dim', type=int, default=64)
    parser.add_argument('--a_dim', type=int, default=512)
    parser.add_argument('--f_dim', type=int, default=256)

    # train with fusion parameters
    parser.add_argument('--fusion_type', type=str, default='mlpmix', help='modality fusion type in '
                                                                               'concat/transformer/senet/mlpmix')
    parser.add_argument('--concat_dim', type=int, default=-1, help='concatenation dim for concat fusion method')
    # config for transformer
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='attention dropout (for audio)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--layers', type=int, default=4,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')
    # config for senet
    parser.add_argument('--channel', type=int, default=64, help='channel dimension for linear layer')
    parser.add_argument('--reduction', type=int, default=16, help='linear dimension reduction')
 
    args = parser.parse_args()

    v_input = torch.rand(8, 43, 64).cuda()
    a_input = torch.rand(8, 3, 224, 224).cuda()
    f_input = torch.rand(8, 3, 32, 224, 224).cuda()

    fusion_model = FusionModule(args).to('cuda')
    v_logit, a_logit, f_logit, fused_logit = fusion_model.forward(v_input, a_input, f_input)
    print(v_logit.shape, a_logit.shape, f_logit.shape, fused_logit.shape)
