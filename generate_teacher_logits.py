# -*- coding: utf-8 -*-
# Developed by Jiapeng Xie
import os

from pointcept.datasets import collate_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim
from pointcept.models.default import DefaultSegmentorV2
from tqdm import tqdm
from functools import partial
from icecream import ic
import random
from network.CA_BEV_Unet import CA_Unet
from network.A_BEV_Unet import BEV_Unet
from network.ptBEVnet import ptBEVnet
from dataloader.dataset import collate_fn_BEV, SemKITTI, get_SemKITTI_label_name, spherical_dataset, voxel_dataset
from utils.log_util import get_logger, make_log_dir
from config.config import load_config_data
from utils.warmupLR import warmupLR
# ignore weird np warning
import warnings

warnings.filterwarnings("ignore")


def main(arch_config, data_config):
    print("arch_config: ", arch_config)
    print("data_config: ", data_config)
    configs = load_config_data(arch_config)
    ic(configs)
    # parameters
    data_cfg = configs['data_loader']
    model_cfg = configs['model_params']
    train_cfg = configs['train_params']
    fea_compre = model_cfg['grid_size'][2]

    # torch.cuda.set_device(1)s
    pytorch_device = "cuda"
    print("Training in device: ", pytorch_device)

    ignore_label = data_cfg['ignore_label']

    # save
    model_save_path = make_log_dir(arch_config, data_config, train_cfg['name'])

    # log
    logger = get_logger(model_save_path + '/train.log')

    if data_cfg['dataset_type'] == 'polar':
        fea_dim = 9
        circular_padding = True
    elif data_cfg['dataset_type'] == 'traditional':
        fea_dim = 7
        circular_padding = False
    else:
        raise NotImplementedError

    # prepare miou fun
    unique_label, unique_label_str, _ = get_SemKITTI_label_name(data_config)

    # prepare bev model
    if model_cfg['use_co_attention']:
        my_BEV_model = CA_Unet(n_class=len(unique_label),
                               n_height=fea_compre,
                               residual=data_cfg['residual'],
                               input_batch_norm=model_cfg['use_norm'],
                               dropout=model_cfg['dropout'],
                               circular_padding=circular_padding)
    else:
        my_BEV_model = BEV_Unet(n_class=len(unique_label),
                                n_height=fea_compre,
                                residual=data_cfg['residual'],
                                input_batch_norm=model_cfg['use_norm'],
                                dropout=model_cfg['dropout'],
                                circular_padding=circular_padding)
    my_model = ptBEVnet(my_BEV_model,
                        grid_size=model_cfg['grid_size'],
                        fea_dim=fea_dim,
                        ppmodel_init_dim=model_cfg['ppmodel_init_dim'],
                        kernal_size=1,
                        fea_compre=fea_compre)

    model_load_path = train_cfg['model_load_path']
    if os.path.exists(model_load_path):
        logger.info("Load model from: " + model_load_path)
        my_model.load_state_dict(torch.load(model_load_path))
    else:
        logger.info("No pretrained model found, train from scratch.")

    my_model.to(pytorch_device)



    # prepare mamba model
    mamba_model =  DefaultSegmentorV2(num_classes=4,backbone_out_channels=64,backbone=dict(
        type='MambaMOS',
        in_channels=5,
        gather_num=8,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        mlp_ratio=4,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('ScanNet', 'S3DIS', 'Structured3D')),criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=0),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=0)
    ])
    checkpoint = torch.load('/data3/ccy/offline_weight_kd_bev/pretrain/model_best.pth')
    mamba_model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()})
    for _, value in mamba_model.named_parameters():
        value.requires_grad = False
    mamba_model.to(pytorch_device)
    mamba_model.eval()

    # prepare dataset
    train_pt_dataset = SemKITTI(data_config_path=data_config,
                                data_path=data_cfg['data_path'] + '/sequences/',
                                imageset='train',
                                return_ref=data_cfg['return_ref'],
                                residual=data_cfg['residual'],
                                residual_path=data_cfg['residual_path'],
                                drop_few_static_frames=data_cfg['drop_few_static_frames'])
    if data_cfg['dataset_type'] == 'polar':
        train_dataset = spherical_dataset(train_pt_dataset,
                                          grid_size=model_cfg['grid_size'],
                                          rotate_aug=data_cfg['rotate_aug'],
                                          flip_aug=data_cfg['flip_aug'],
                                          transform_aug=data_cfg['transform_aug'],
                                          fixed_volume_space=data_cfg['fixed_volume_space'])
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=data_cfg['batch_size'],
                                                       collate_fn=collate_fn_BEV,
                                                       shuffle=data_cfg['shuffle'],
                                                       num_workers=data_cfg['num_workers'],
                                                       pin_memory=True)
                                                    

    epoch = 0
    my_model.train()
    global_iter = 0
    while epoch < 1:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        for i_iter, (train_vox_label, train_grid, train_pt_labs, train_pt_fea, data_dict,indexlist) in enumerate(train_dataset_loader):
            ###### mamba teacher forward
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            pred = torch.zeros((segment.size, 4)).cuda()
            fea = torch.zeros((segment.size, 64)).cuda()
            tn = data_dict.pop("tn")
            final_scan_mask = tn.squeeze(1) == 0

            for i in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                with torch.no_grad():
                    time1 = time.time()
                    output_dict = mamba_model(input_dict)
                    time2 = time.time()
                    print(time2-time1)
                    pred_part = output_dict["seg_logits"]  # (n, k)
                    fea_part = output_dict["fea"]
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        fea[idx_part[bs:be], :] += fea_part[bs:be]
                        bs = be

            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                fea = fea[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
                tn = data_dict["origin_tn"]

                final_scan_mask = tn.squeeze(1) == 0
            teacher_fea = fea[final_scan_mask]
            teacher_pred = pred[final_scan_mask]

            teacher_fea = teacher_fea.cpu().detach().numpy()
            teacher_preda = teacher_pred.cpu()
            teacher_preda = teacher_preda.detach()
            teacher_preda = teacher_preda.numpy()
            seq = indexlist[0][0]
            seqindex = indexlist[0][1]
            filename = f'{seq}_{seqindex}.npy'
            file_path = os.path.join('yourpath', filename)
            np.save(file_path, teacher_preda)
            pbar.update(1)
            global_iter += 1
        pbar.close()
        epoch += 1


if __name__ == '__main__':
    arch_config_path = "config/KDMOS-semantickitti.yaml"
    data_config_path = "config/semantic-kitti-MOS.yaml"
    main(arch_config_path, data_config_path)