# -*- coding: utf-8 -*-

import os
import time
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim
import torch_scatter
from tqdm import tqdm
from functools import partial
from icecream import ic
import random
from network.CA_BEV_Unet import CA_Unet
from network.A_BEV_Unet import BEV_Unet
from network.ptBEVnet import ptBEVnet
from dataloader.dataset import collate_fn_BEV, SemKITTI, get_SemKITTI_label_name, spherical_dataset, voxel_dataset
from utils import lovasz_losses
from utils.log_util import get_logger, make_log_dir
from config.config import load_config_data
from utils.warmupLR import warmupLR
import kd_Utils as kd
# ignore weird np warning
import warnings

warnings.filterwarnings("ignore")

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('size:{:.3f} MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)
    
def set_seed(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label,ignore_label = None):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 1)
    if ignore_label is not None:
        if ignore_label in unique_label:
            unique_label = unique_label[unique_label != ignore_label]
            hist = hist[unique_label, :]
            hist = hist[:, unique_label]
    return hist


def main(arch_config, data_config):
    rank = 0
    seed = 114514
    set_seed(seed)
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
    pytorch_device = torch.device("cuda:3" )
    print("Training in device: ", pytorch_device)

    ignore_label = data_cfg['ignore_label']

    # torch.backends.cudnn.benchmark = True  

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
    unique_label_str = ['static', 'movable', 'moving']
    # prepare model
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
    #  my_model.cuda()

    # prepare dataset
    train_pt_dataset = SemKITTI(data_config_path=data_config,
                                data_path=data_cfg['data_path'] + '/sequences/',
                                imageset='train',
                                return_ref=data_cfg['return_ref'],
                                residual=data_cfg['residual'],
                                residual_path=data_cfg['residual_path'],
                                drop_few_static_frames=data_cfg['drop_few_static_frames'])
    val_pt_dataset = SemKITTI(data_config_path=data_config,
                              data_path=data_cfg['data_path'] + '/sequences/',
                              imageset='val',
                              return_ref=data_cfg['return_ref'],
                              residual=data_cfg['residual'],
                              residual_path=data_cfg['residual_path'],
                              drop_few_static_frames=False)
    if data_cfg['dataset_type'] == 'polar':
        train_dataset = spherical_dataset(train_pt_dataset,
                                          grid_size=model_cfg['grid_size'],
                                          rotate_aug=data_cfg['rotate_aug'],
                                          flip_aug=data_cfg['flip_aug'],
                                          transform_aug=data_cfg['transform_aug'],
                                          fixed_volume_space=data_cfg['fixed_volume_space'])
        val_dataset = spherical_dataset(val_pt_dataset,
                                        grid_size=model_cfg['grid_size'],
                                        fixed_volume_space=data_cfg['fixed_volume_space'])
    elif data_cfg['dataset_type'] == 'traditional':
        train_dataset = voxel_dataset(train_pt_dataset,
                                      grid_size=model_cfg['grid_size'],
                                      rotate_aug=data_cfg['rotate_aug'],
                                      flip_aug=data_cfg['flip_aug'],
                                      fixed_volume_space=data_cfg['fixed_volume_space'])
        val_dataset = voxel_dataset(val_pt_dataset,
                                    grid_size=model_cfg['grid_size'],
                                    fixed_volume_space=data_cfg['fixed_volume_space'])
    else:
        raise NotImplementedError
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=data_cfg['batch_size'],
                                                       collate_fn=collate_fn_BEV,
                                                       shuffle=data_cfg['shuffle'],
                                                       num_workers=data_cfg['num_workers'],
                                                       pin_memory=True,
                                                       worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
                                                       )
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=data_cfg['batch_size'],
                                                     collate_fn=collate_fn_BEV,
                                                     shuffle=False,
                                                     num_workers=data_cfg['num_workers'],
                                                     pin_memory=True,
                                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
                                                     )

    # optimizer
    # optimizer = optim.Adam(my_model.parameters(), lr=train_cfg["learning_rate"])
    if train_cfg["optimizer"] == 'Adam':
        optimizer = optim.Adam(my_model.parameters(),
                               lr=train_cfg["learning_rate"],
                               weight_decay=train_cfg["weight_decay"])
    elif train_cfg["optimizer"] == 'SGD':
        optimizer = optim.SGD(my_model.parameters(),
                              lr=train_cfg["learning_rate"],
                              momentum=train_cfg["momentum"],
                              weight_decay=train_cfg["weight_decay"])
    elif train_cfg["optimizer"] == 'AdamW':
        optimizer = optim.AdamW(my_model.parameters(),
                                lr=train_cfg["learning_rate"],
                                weight_decay=train_cfg["weight_decay"])
    else:
        raise NotImplementedError
    # Use warmup learning rate
    # post decay and step sizes come in epochs, and we want it in steps
    steps_per_epoch = len(train_dataset_loader)
    up_steps = int(train_cfg["wup_epochs"] * steps_per_epoch)
    final_decay = train_cfg["lr_decay"] ** (1 / steps_per_epoch)
    scheduler = warmupLR(optimizer=optimizer,
                         lr=train_cfg["learning_rate"],
                         warmup_steps=up_steps,
                         momentum=train_cfg["momentum"],
                         decay=final_decay)

    #loss_weight = torch.tensor([1.0014, 296.4371]).to(pytorch_device)
    #ls_CrossEntropy = torch.nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255)
    ls_CrossEntropy = torch.nn.CrossEntropyLoss(ignore_index=0)
    ls_lovasz = lovasz_losses.Lovasz_softmax_PointCloud()
    softloss = torch.nn.KLDivLoss(reduction="none")
    # training
    epoch = 0
    best_voxel_val_miou = 0
    best_val_loss = 9999999999
    my_model.train()
    global_iter = 0
    exce_counter = 0

    check_iter = train_cfg['checkpoint_every_n_steps']

    while epoch < train_cfg['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        for i_iter, (train_vox_label, train_grid, train_pt_labs, train_pt_fea, indexlist) in enumerate(train_dataset_loader):
            # validation
            if global_iter % train_cfg['eval_every_n_steps'] == 0 and global_iter != 0 and epoch>train_cfg['start_valid_epoch']:
                my_model.eval()
                voxel_hist_list = []
                val_loss_list = []
                with torch.no_grad():
                    for i_iter_val, (val_vox_label, val_grid, val_pt_labs, val_pt_fea,_) in enumerate(
                            val_dataset_loader):
                        val_pt_fea_ten = [i.to(pytorch_device) for i in val_pt_fea]
                        val_grid_ten = [i.to(pytorch_device) for i in val_grid]
                        val_vox_label_ten = val_vox_label.to(pytorch_device)

                        voxel_out, pt_out = my_model(val_pt_fea_ten, val_grid_ten, pytorch_device)

                        loss = lovasz_losses.lovasz_softmax(torch.nn.functional.softmax(voxel_out).detach(), val_vox_label_ten,
                                              ignore=0) + \
                               ls_CrossEntropy(voxel_out.detach(), val_vox_label_ten)
                        val_loss_list.append(loss.detach().cpu().numpy())

                        voxel_predict_labels = torch.argmax(voxel_out, dim=1)
                        voxel_predict_labels = voxel_predict_labels.cpu().detach().numpy()
                        for count, i_val_grid in enumerate(val_grid):
                            voxel_hist_list.append(fast_hist_crop(
                                voxel_predict_labels[
                                    count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][
                                                                                         :, 2]],
                                val_pt_labs[count], unique_label,ignore_label=0))

                my_model.train()
                voxel_iou = per_class_iu(sum(voxel_hist_list))
                logger.info('Validation per class iou (voxel): ')
                for class_name, class_iou in zip(unique_label_str, voxel_iou):
                    logger.info('%s : %.2f%%' % (class_name, class_iou * 100))
                voxel_val_miou = voxel_iou[2]

                logger.info(
                    'Current voxel val miou is %.3f while the best voxel val miou is %.3f' % (
                        voxel_val_miou, best_voxel_val_miou))
                logger.info('Current val loss is %.3f while the best val loss is %.3f' % (
                    np.mean(val_loss_list), best_val_loss))
                # save model if performance is improved
                if best_voxel_val_miou < voxel_val_miou:
                    best_voxel_val_miou = voxel_val_miou
                    logger.info("best voxel val miou model saved.")
                    torch.save(my_model.state_dict(), model_save_path + '/' + train_cfg['name'] + '_best_voxel_miou.pt')
                if np.mean(val_loss_list) < best_val_loss:
                    best_val_loss = np.mean(val_loss_list)
                    logger.info("best val loss model saved.")
                    torch.save(my_model.state_dict(), model_save_path + '/' + train_cfg['name'] + '_bestloss.pt')

                logger.info('%d exceptions encountered during last training\n' % exce_counter)
                exce_counter = 0
                loss_list = []
            # training
            try:
                batchsize = len(indexlist)
                teacher_predlist= []
                for i in range(0,batchsize):
                    teacher_pred = np.load(data_cfg['teacher_logits_path']+indexlist[i][0]+'_'+str(indexlist[i][1])+'.npy').reshape((-1, 4))
                    teacher_predlist.append(teacher_pred)

                teacher_pred = np.concatenate(teacher_predlist, axis=0)
                teacher_pred = torch.from_numpy(teacher_pred).to(pytorch_device)
                train_pt_fea_ten = [i.to(pytorch_device) for i in train_pt_fea]
                train_grid_ten = [i.to(pytorch_device) for i in train_grid]
                train_vox_label_ten = train_vox_label.to(pytorch_device)
                # forward + backward + optimize
                optimizer.zero_grad()  # zero the parameter gradients
                t0 = time.time()
                voxel_out, pt_out = my_model(train_pt_fea_ten, train_grid_ten, pytorch_device)
                t1 = time.time()
                # pointlabel
                tensor_list = [torch.from_numpy(arr) for arr in train_pt_labs]
                train_pt_labs = torch.cat(tensor_list, dim=0).to(pytorch_device)
                train_pt_labs = train_pt_labs.squeeze(1)



                loss = lovasz_losses.lovasz_softmax(torch.nn.functional.softmax(voxel_out), train_vox_label_ten,
                                                    ignore=0) + \
                       ls_CrossEntropy(voxel_out, train_vox_label_ten)


                # kl_loss
                loss_weight = torch.tensor([0, 1.0716, 22.0882, 421.3364], dtype=torch.float).to(pytorch_device)
                train_pt_labs = train_pt_labs.to(torch.long)
                assert torch.all(train_pt_labs >= 0) and torch.all(train_pt_labs < len(loss_weight)), "train_pt_labs values out of range"
                weight_label = loss_weight[train_pt_labs]

                loss_weight = weight_label.view(-1, 1)
                mask = (loss_weight == 421.3364).float()
                num_ones = mask.sum().item()
                # pt_out = normali  ze(pt_out)
                # teacher_pred = normalize(teacher_pred)
                # nckdloss
                # pt_out = normalize(pt_out)
                # teacher_pred = normalize(teacher_pred)
                gt_mask = kd._get_gt_mask(pt_out, train_pt_labs)
                other_mask = kd._get_other_mask(pt_out, train_pt_labs)
                pred_teacher_part2 = F.softmax(teacher_pred / 4 - 1000.0 * gt_mask, dim=1)
                log_pred_student_part2 = F.log_softmax(pt_out / 4 - 1000.0 * gt_mask, dim=1)+1e-8
                nckd_loss = (
                        F.kl_div(log_pred_student_part2, pred_teacher_part2+1e-8, reduction="none")
                )
                nckd_loss = nckd_loss * loss_weight
                #nckd_loss = nckd_loss.sum(dim=-1)  

                #nckd_loss = nckd_loss.mean()  

                nckd_loss = nckd_loss.sum() / loss_weight.sum()

                pred_student = F.softmax(pt_out / 4, dim=1) + 1e-8
                pred_teacher = F.softmax(teacher_pred / 4, dim=1) + 1e-8

                pred_student = kd.cat_mask(pred_student, gt_mask, other_mask)
                pred_teacher = kd.cat_mask(pred_teacher, gt_mask, other_mask)
                log_pred_student = torch.log(pred_student)
                tckd_loss = (
                    F.kl_div(log_pred_student, pred_teacher, reduction="none")
                )

                # tckd_loss = tckd_loss.sum() /train_pt_labs.shape[0]
                tckd_loss = tckd_loss * mask
                tckd_loss = tckd_loss.sum() / num_ones
            

                kd_loss = 3*nckd_loss + tckd_loss

                loss =loss+0.3*kd_loss
                t2 = time.time()
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()
                scheduler.step()
                t3 = time.time()
                # print("time cost: ", t1 - t0, t1 - t0, t2 - t1, t3 - t2)
                if global_iter % check_iter == 0:
                    if len(loss_list) > 0:
                        logger.info('epoch %3d, iter %5d, loss: %.3f, lr: %.5f' % (
                            epoch, i_iter, np.mean(loss_list), optimizer.param_groups[0]['lr']))
                    else:
                        logger.info('loss error.')
            except Exception as error:
                if exce_counter == 0:
                    logger.info(error)
                exce_counter += 1

            pbar.update(1)
            global_iter += 1
        pbar.close()
        epoch += 1


if __name__ == '__main__':
    arch_config_path = "config/KDMOS-semantickitti.yaml"
    data_config_path = "config/semantic-kitti-MOS.yaml"
    main(arch_config_path, data_config_path)