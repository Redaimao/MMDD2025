from __future__ import print_function, division
import argparse

from data_loader.Load_RLT_face_audio_OpenFace_Affect import *

from data_loader import *
from models.fusion_net import FusionModule
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import cv2
from utils import AvgrageMeter, performances
import torch.utils.data


# feature  -->   [ batch, channel, height, width ]
def get_train_dataset_loader(args):
    if args.train_dataset == "RLT":
        train_data = getattr(Load_RLT_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    elif args.train_dataset == "BagOfLies":
        train_data = getattr(Load_BagOfLies_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    elif args.train_dataset == "BoxOfLies":
        train_data = getattr(Load_BoxOfLies_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    elif args.train_dataset == "MU3D":
        train_data = getattr(Load_MU3D_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    else:
        raise Exception("Train dataset name not exists!")
        # train_data = None

    return train_data


def get_test_dataset_loader(args):
    if args.test_dataset == "RLT":
        test_data = getattr(Load_RLT_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([Normaliztion(), ToTensor_test()]))
    elif args.test_dataset == "BagOfLies":
        test_data = getattr(Load_BagOfLies_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([Normaliztion(), ToTensor_test()]))
    elif args.test_dataset == "BoxOfLies":
        test_data = getattr(Load_BoxOfLies_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([Normaliztion(), ToTensor_test()]))
    elif args.test_dataset == "MU3D":
        test_data = getattr(Load_MU3D_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([Normaliztion(), ToTensor_test()]))
    else:
        raise Exception("Test dataset name not exists!")
        # test_data = None

    return test_data


def FeatureMap2Heatmap(x, x2):
    ## initial images
    ## initial images
    org_img = x[0, :, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log + '/' + args.log + '_visual.jpg', org_img)

    org_img = x2[0, :, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log + '/' + args.log + '_audio.jpg', org_img)


# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    # os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(
        args.log + '/' + args.fusion_type + "_" + args.modalities + "_" + args.train_dataset + "_to_" +
        args.test_dataset + str(args.test_list.split('/')[-1].split('.')[0]) + '_fusion_test.txt',
        'w')

    echo_batches = args.echo_batches

    print("Deception Detection!!!:\n ")

    log_file.write('Deception Detection!!!:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?

    print('train from scratch!\n')
    log_file.write('train from scratch!\n')
    log_file.flush()

    # model = ResNet18_GRU(pretrained=True, GRU_layers=1)
    # model = ResNet18_BiGRU(pretrained=True, GRU_layers=2)
    # model = ResNet18(pretrained=True)
    model = FusionModule(args)

    # # model = OpenFaceAU_MLP_MLP()
    # # model = OpenFaceGaze_MLP_MLP()
    # # model = OpenFaceGaze_AllMLP()

    model = model.cuda()
    # model = model.to(device[0])
    # model = nn.DataParallel(model, device_ids=device, output_device=device[0])
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00005)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    print(model)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()  # possible warining that scheduler.step is before optimizer.step()---> should be after it!!!
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_absolute = AvgrageMeter()
        loss_contra = AvgrageMeter()
        loss_absolute_RGB = AvgrageMeter()
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        train_data = get_train_dataset_loader(args)
     
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs = sample_batched['video_x'].cuda()
            inputs_audio = sample_batched['audio_x'].cuda()
            inputs_OpenFace = sample_batched['OpenFace_x'].cuda()
            inputs_affect = sample_batched['x_affect'].cuda()

            DD_label = sample_batched['DD_label'].cuda()
            inputs_OpenFace = torch.cat((inputs_affect, inputs_OpenFace), dim=1)
            # join affect feature to openface  feature

            optimizer.zero_grad()

            log_file.write('\n')
            log_file.flush()

            # logits =  model(inputs)
            # inputs_OpenFace combined with affect features are the behavioral
            v_logit, a_logit, f_logit, fused_logit = model(inputs_OpenFace, inputs_audio, inputs)
            # logits =  model(inputs_OpenFace[:,:8,:])
            # logits =  model(inputs_OpenFace[:,8:,:])

            # pdb.set_trace()

            # pdb.set_trace()
            loss_global = criterion(fused_logit, DD_label.squeeze(-1)) * 0.5
            if 'v' in args.modalities and v_logit is not None:
                v_loss = criterion(v_logit, DD_label.squeeze(-1))
                loss_global += v_loss
            if 'a' in args.modalities and a_logit is not None:
                a_loss = criterion(a_logit, DD_label.squeeze(-1))
                loss_global += a_loss
            if 'f' in args.modalities and f_logit is not None:
                f_loss = criterion(f_logit, DD_label.squeeze(-1))
                loss_global += f_loss

            loss = loss_global

            loss.backward()

            optimizer.step()

            n = inputs.size(0)
            loss_absolute.update(loss_global.data, n)
            loss_contra.update(loss_global.data, n)
            loss_absolute_RGB.update(loss_global.data, n)

            if i % echo_batches == echo_batches - 1:  # print every 50 mini-batches

                # visualization
                FeatureMap2Heatmap(inputs[:, :, 1, :, :], inputs_audio)

                # log written
                print('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f , CE1= %.4f , CE2= %.4f \n' % (
                    epoch + 1, i + 1, lr, loss_absolute.avg, loss_contra.avg, loss_absolute_RGB.avg))

        # whole epoch average
        log_file.write('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f , CE1= %.4f , CE2= %.4f \n' % (
            epoch + 1, i + 1, lr, loss_absolute.avg, loss_contra.avg, loss_absolute_RGB.avg))
        log_file.flush()

        epoch_test = 1
        if epoch % epoch_test == epoch_test - 1:  # test every 5 epochs
            model.eval()

            with torch.no_grad():

                ###########################################
                '''                test             '''
                ##########################################
                # # differenet clip num for each video, it cannot stack into large batachsize, set = 1
                test_data = get_test_dataset_loader(args)
                # test_data = getattr(data_loader, args.test_dataset + '_test')(args.test_list, args.test_root,
                #                                                               transform=transforms.Compose(
                #                                                                   [Normaliztion(),
                #                                                                    ToTensor_test()]))
                # test_data = RLtraial_test(args.test_list, args.test_root,
                #                           transform=transforms.Compose([Normaliztion(), ToTensor_test()]))
                dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

                map_score_list = []

                for i, sample_batched in enumerate(dataloader_test):

                    inputs, DD_label = sample_batched['video_x'].cuda(), sample_batched['DD_label'].cuda()
                    inputs_OpenFace = sample_batched['OpenFace_x'].cuda()
                    inputs_audio = sample_batched['audio_x'].cuda()
                    inputs_affect = sample_batched['x_affect'].cuda()

                    optimizer.zero_grad()

                    # pdb.set_trace()

                    for clip_t in range(inputs_OpenFace.shape[1]):
                        _, _, _, fused_logit = model(
                            torch.cat((inputs_affect[:, clip_t, :, :], inputs_OpenFace[:, clip_t, :, :]), dim=1),
                            inputs_audio[:, clip_t, :, :, :],
                            inputs[:, clip_t, :, :, :, :])

                        # _, _, _, fused_logit = model(inputs_OpenFace[:, clip_t, :, :],
                        #                              inputs_audio[:, clip_t, :, :, :],
                        #                              inputs[:, clip_t, :, :, :, :])

                        # logits = model(inputs_OpenFace[:, clip_t, :, :])
                        # logits  =  model(inputs_OpenFace[:,clip_t,:8,:])
                        # logits  =  model(inputs_OpenFace[:,clip_t,8:,:])
                        if args.fusion:
                            if clip_t == 0:
                                logits_accumulate = F.softmax(fused_logit, -1)
                            else:
                                logits_accumulate += F.softmax(fused_logit, -1)
                        else:
                            raise Exception("testing for fusion only!")
                        # todo: test for other single modality models with/without fusion
                    logits_accumulate = logits_accumulate / inputs_OpenFace.shape[1]
                    for test_batch in range(inputs_audio.shape[0]):
                        map_score_list.append(
                            '{} {}\n'.format(logits_accumulate[test_batch][1], DD_label[test_batch][0]))

                test_filename = args.log + '/' + args.fusion_type + "_" + args.modalities + "_" + \
                                args.train_dataset + "_to_" + args.test_dataset + "_" + \
                                str(args.test_list.split('/')[-1].split('.')[0]) + "_scores.txt"
                with open(test_filename, 'w') as file:
                    file.writelines(map_score_list)

                    ##########################################################################
                #       performance evaluation
                ##########################################################################
                ACC_RLtrial, AUC_RLtrial, EER_RLtrial = performances(test_filename)
                # if ACC_RLtrial > best_val_acc:
                #     best_val_acc = ACC_RLtrial
                #     # save best model for DG initial
                #     torch.save({
                #         'epoch': epoch,
                #         'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'best_acc': best_val_acc,
                #     }, "/pth_to_checkpoint/checkpoint/DG_Tr_{}_Te_{}.pt".format(args.train_dataset,
                #                                                                                   args.test_dataset))

                print('epoch:%d, ACC= %.4f, AUC= %.4f, EER= %.4f\n' % (
                    epoch + 1, ACC_RLtrial, AUC_RLtrial, EER_RLtrial))
                log_file.write('\n epoch:%d, ACC= %.4f, AUC= %.4f, EER= %.4f\n\n' % (
                    epoch + 1, ACC_RLtrial, AUC_RLtrial, EER_RLtrial))

                log_file.flush()

    print('Finished Training')
    log_file.close()


if __name__ == "__main__":
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
    parser.add_argument('--log', type=str, default="logsintra", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    # dataset dirs
    parser.add_argument('--train_dataset', type=str, default='RLT')
    parser.add_argument('--train_root', type=str, default='',
                        help='train dataset root dir')
    parser.add_argument('--train_list', type=str, default='',
                        help='train feature list')

    parser.add_argument('--test_dataset', type=str, default='RLT')
    parser.add_argument('--test_root', type=str, default='',
                        help='test data root')
    parser.add_argument('--test_list', type=str, default='',
                        help='test feature list')

    # config for individual modal
    parser.add_argument('--fusion', action='store_true', default='false', help='true when fusion module is used')
    parser.add_argument('--fusion_modal', type=str, default='FusionModule')

    parser.add_argument('--modalities', type=str, default='vaf', help='modalities in v-affect+openface, a-audio, '
                                                                      'f-face frames')

    parser.add_argument('--v_model', type=str, default='OpenFace_Affect7_MLP_MLP')  # OpenFace_MLP_MLP
    parser.add_argument('--a_model', type=str, default='ResNet18_audio')
    parser.add_argument('--f_model', type=str, default='ResNet18_GRU')

    # dimensions for each modality (the embedding size)
    parser.add_argument('--v_dim', type=int, default=64)
    parser.add_argument('--a_dim', type=int, default=512)
    parser.add_argument('--f_dim', type=int, default=256)

    # train with fusion parameters
    parser.add_argument('--fusion_type', type=str, default='concat', help='modality fusion type in '
                                                                          'concat/transformer/senet/mlpmix')
    parser.add_argument('--concat_dim', type=int, default=-1, help='concatenation dim for concat fusion method')
    # config for transformer
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='attention dropout (for audio)')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--layers', type=int, default=2,
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
    train_test()
