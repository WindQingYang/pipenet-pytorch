'''

Pytorch code for 'Chalearn Multi-modal Cross-ethnicity Face anti-spoofing Recognition Challenge@CVPR2020'
By Qing Yang, 2020/03/01

MIT License
Copyright (c) 2019

'''

import os
import sys
sys.path.append("..")
import argparse
from data.augmentation import *
from utils.metric import *
from utils.loss import CosineAnnealingLR_with_Restart
from utils.submission import *
from utils.utils import *
import time
# import pretrainedmodels
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

import torch
from models.fusion.PipeNet import PipeNet as Net
# from models.fusion.BlenderXNet import Net


def get_dataset(image_mode):
    if image_mode=='fusion':
        from data.data_fusion import FDDataset
        train_dataset = FDDataset(mode='train', dataset_name=config.dataset_name, modality=config.image_mode,
                                  image_size=config.image_size, fold_index=config.train_fold_index,crop=config.crop,gray=config.gray)
        valid_dataset = FDDataset(mode='val', dataset_name=config.dataset_name, modality=config.image_mode,
                                  image_size=config.image_size, fold_index=config.train_fold_index,crop=config.crop,gray=config.gray)
        test_dataset = FDDataset(mode=config.mode, dataset_name=config.dataset_name, modality=config.image_mode,
                                 image_size=config.image_size,fold_index=config.train_fold_index,crop=config.crop,gray=config.gray)

    return train_dataset,valid_dataset,test_dataset

# def get_augment(image_mode):
#     if image_mode == 'color':
#         augment = color_augumentor
#     elif image_mode == 'depth':
#         augment = depth_augumentor
#     elif image_mode == 'ir':
#         augment = ir_augumentor
#     return augment

def run_train(config):
    out_dir = config.model_dir
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)+\
                        '_'+config.dataset_name
    out_dir = os.path.join(out_dir,config.image_mode+'_'+config.note,config.dataset_name)

    initial_checkpoint = config.pretrained_model
    criterion  = softmax_cross_entropy_criterion

    ## setup  -----------------------------------------------------------------------------
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    log = Logger()
    log.open(os.path.join(out_dir,config.model_name+'.txt'),mode='a')
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')


    train_dataset,valid_dataset,_ = get_dataset(config.image_mode)

    train_loader  = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size  = config.batch_size,
                                drop_last   = True,
                                num_workers = 8)#last4

    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size = config.batch_size // 36,
                                drop_last  = False,
                                num_workers = 8)#last4

    assert(len(train_dataset)>=config.batch_size)
    log.write('batch_size = %d\n'%(config.batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')
    log.write('** net setting **\n')

    num_class = 2
    net = Net(num_class=num_class)
    # print(net)
    net = torch.nn.DataParallel(net)
    net =  net.cuda()

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir +'/',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n'%(type(net)))
    log.write('criterion=%s\n'%criterion)
    log.write('\n')

    iter_smooth = 20
    start_iter = 0
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('                                  |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    log.write('model_name    cycle   lr    iter    epoch     |     loss      acer      acc    |     loss      acc     |  time   \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    iter = 0
    i    = 0

    train_loss = np.zeros(6, np.float32)
    valid_loss = np.zeros(6, np.float32)
    batch_loss = np.zeros(6, np.float32)

    start = timer()
    #-----------------------------------------------
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=config.lr, momentum=0.9, weight_decay=0.0005) #lr=0.1

    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=config.cycle_inter,
                                          T_mult=1,
                                          model=net,
                                          out_dir='../input/',
                                          take_snapshot=False,
                                          eta_min=config.min_lr) #1e-6 #10e-3

    global_min_acer = 1.0
    global_min_vloss=1.0
    global_min_tloss = 1.0

    for cycle_index in range(config.cycle_num):
        print('cycle index: ' + str(cycle_index))
        min_acer = 1.0
        min_vloss= 1.0
        min_tloss= 1.0

        for epoch in range(0, config.cycle_inter):
            sgdr.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr : {:.4f}'.format(lr))

            sum_train_loss = np.zeros(6,np.float32)
            sum = 0
            optimizer.zero_grad()

            for input, truth in train_loader:
                iter = i + start_iter

                # one iteration update  -------------
                net.train()
                input = input.cuda()
                truth = truth.cuda()
                # print("type of truth:",type(truth))

                logit = net.forward(input)

                truth = truth.view(logit.shape[0])

                loss  = criterion(logit, truth)
                precision,_ = metric(logit, truth)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # print statistics  ------------
                batch_loss[:2] = np.array(( loss.item(), precision.item(),))

                sum += 1
                if iter%iter_smooth == 0:
                    train_loss = sum_train_loss/sum
                    sum = 0
                i=i+1

            if epoch >= config.cycle_inter // 2:
                net.eval()
                valid_loss,_ = do_valid_test(net, valid_loader, criterion)
                net.train()

                v0 = "{:.18f}".format(valid_loss[0])
                v1 = "{:.18f}".format(valid_loss[1])
                v2 = "{:.18f}".format(valid_loss[2])
                t0 = "{:.18f}".format(batch_loss[0])
                t1 = "{:.18f}".format(batch_loss[1])

                if valid_loss[1] <= min_acer:

                    if valid_loss[0]<min_vloss :
                        min_acer = valid_loss[1]
                        min_vloss = valid_loss[0]
                        ckpt_name = out_dir + '/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
                        torch.save(net.state_dict(), ckpt_name)
                        log.write('save cycle ' + str(cycle_index) + ' min acer model: acer:' +v1 +' valid_loss:'+ v0+ ' valid_acc:'+ v2 +'\n')

                    if valid_loss[0]== min_vloss and batch_loss[0] < min_tloss:
                        t0 = "{:.18f}".format(batch_loss[0])
                        t1 = "{:.18f}".format(batch_loss[1])
                        min_acer = valid_loss[1]
                        min_tloss = batch_loss[0]
                        ckpt_name = out_dir + '/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
                        torch.save(net.state_dict(), ckpt_name)
                        log.write('save cycle ' + str(cycle_index) + ' min acer model: acer:' +v1 +' train_loss:'+ t0+ ' train_acc:'+ t1 +'\n')

                if valid_loss[1] <= global_min_acer :

                    if valid_loss[0]<global_min_vloss:
                        global_min_acer = valid_loss[1]
                        global_min_vloss= valid_loss[0]
                        ckpt_name = out_dir + '/global_min_acer_model.pth'
                        torch.save(net.state_dict(), ckpt_name)
                        log.write('save global min acer model: acer:' +v1 +' valid_loss:'+ v0+ ' valid_acc:'+ v2 +'\n')

                    if valid_loss[0]== global_min_vloss and batch_loss[0] < global_min_tloss:
                        global_min_acer = valid_loss[1]
                        global_min_tloss = batch_loss[0]
                        ckpt_name = out_dir + '/global_min_acer_model.pth'
                        torch.save(net.state_dict(), ckpt_name)
                        log.write('save global min acer model: acer:' + v1 + ' train_loss:' + t0 + ' train_acc:' + t1 + '\n')

            asterisk = ' '
            log.write(config.model_name+' Cycle %d: %0.4f %5.1f %6.1f | %0.12f  %0.12f  %0.6f %s  | %0.6f  %0.6f |%s \n' % (
                cycle_index, lr, iter, epoch,
                valid_loss[0], valid_loss[1], valid_loss[2], asterisk,
                batch_loss[0], batch_loss[1],
                time_to_str((timer() - start), 'min')))

        ckpt_name = out_dir + '/Cycle_' + str(cycle_index) + '_final_model.pth'
        torch.save(net.state_dict(), ckpt_name)
        log.write('save cycle ' + str(cycle_index) + ' final model \n')

def infer_test( net, test_loader):
    valid_num  = 0
    probs = []

    for i, (input, truth) in enumerate(tqdm(test_loader)):
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)
        input = input.cuda()

        with torch.no_grad():
            logit = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)
            prob = F.softmax(logit, 1)
            # print("prob:",prob)
            # print("len prob:", len(prob))
        valid_num += len(input)
        probs.append(prob.data.cpu().numpy())
    # print("probs:", probs)
    probs = np.concatenate(probs)
    # print("probs[:, 1]:", probs[:, 1])
    return probs[:, 1]

def run_test(config):
    out_dir = config.model_dir
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)+'_'+config.dataset_name
    res_dir = os.path.join(config.model_dir, config.image_mode+'_'+config.note)
    out_dir = os.path.join(out_dir,config.image_mode+'_'+config.note,config.dataset_name)
    initial_checkpoint = config.pretrained_model

    ## net
    num_class = 2
    net = Net(num_class=num_class)

    #save model structure file
    # f=open('./log/'+config.image_mode+'_'+config.model+'_network.txt','w+')
    # f.write(str(net))
    # f.close()

    net = torch.nn.DataParallel(net)
    net =  net.cuda()


    if initial_checkpoint is not None:

        save_dir=res_dir + config.dataset_name
        initial_checkpoint = os.path.join(out_dir +'/',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))


    _,_,test_dataset=get_dataset(config.image_mode)

    test_loader  = DataLoader( test_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers=8)

    net.eval()


    print('infer!!!')
    out = infer_test(net, test_loader)
    print('done!!!')

    print("res_dir:",res_dir)
    add_scores(config.mode,config.dataset_name,out,res_dir)
    frame2video(config.mode,config.dataset_name,res_dir,config.cal_method)

def run_submit(config):
    import time

    time = time.localtime(time.time())
    tm_mark = str(time.tm_year)+'_'+str(time.tm_mon) + '_'+str(time.tm_mday) +'_'+ str(time.tm_hour) +'_'+ str(time.tm_min)

    result_path=os.path.join(config.model_dir,config.image_mode+'_'+config.note)+'/'

    if config.re_cal==True:
        print("re-cal!!!")
        for i in range (1,4):
            if i==1:
                config.cal_method = 'mean'
                print(config.cal_method)
            else:
                config.cal_method = 'medstd'
                print(config.cal_method)
            frame2video('dev', '4@'+str(i), result_path, config.cal_method)
            frame2video('test', '4@'+str(i), result_path, config.cal_method)

    print("combine!!!")
    combine_list_6(result_path + '4@1_dev_video_res.txt',
                result_path + '4@1_test_video_res.txt',
                result_path + '4@2_dev_video_res.txt',
                result_path + '4@2_test_video_res.txt',
                result_path + '4@3_dev_video_res.txt',
                result_path + '4@3_test_video_res.txt',
                result_path + 'A_AP_BP_'+ config.image_mode+'_'+config.note+'-'+tm_mark+'.txt')
    print("saved result in:",'../'+result_path + 'A_AP_BP_'+config.image_mode+'_'+config.note+'-'+tm_mark+'.txt')

def main(config):

    seed=555
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if config.mode == 'train':
        # config.model = 'baseline'
        config.crop=False
        config.gray=False
        run_train(config)

    if config.mode == 'test' or config.mode=='dev':
        config.pretrained_model = r'global_min_acer_model.pth'

        config.cal_method = 'meanstd'

        run_test(config)

    if config.mode == 'submit':
        run_submit(config)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)
    # parser.add_argument('-p','--phase', type=int, default=2, choices=[ 1, 2])
    parser.add_argument('--model', type=str, default='E')
    parser.add_argument('--image_mode', type=str, default='fusion')
    parser.add_argument('--cal_method', type=str, default='meanstd',choices=['mean','medstd','median','part_mean','meanstd'])
    parser.add_argument('--image_size', type=int, default=24)
    parser.add_argument('--re_cal',type=bool, default=False )

    parser.add_argument('--batch_size', type=int, default=256) #last128
    parser.add_argument('--cycle_num', type=int, default=5)#last2

    parser.add_argument('--cycle_inter', type=int, default=50)#last100
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--mode', type=str, default='train', choices=['train','val','dev','test','submit'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--crop',type=bool, default=True )
    parser.add_argument('--gray',type=bool, default=True )
    parser.add_argument('--note', type=str, default='18')
    parser.add_argument('--model_dir', type=str,default='../outputs')
    # parser.add_argument('--result_dir', type=str,default='../outputs')
    parser.add_argument('--dataset_name', type=str, default='4@1', choices=['4@1','4@2','4@3'])


    config = parser.parse_args()
    print(config)
    main(config)
