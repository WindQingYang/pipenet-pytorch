'''

Pytorch code for 'Chalearn Multi-modal Cross-ethnicity Face anti-spoofing Recognition Challenge@CVPR2020'
By Qing Yang, 2020/03/01

MIT License
Copyright (c) 2019

'''
import os
import random
import time
import argparse
from data.data_helper import *
import numpy as np
import math


def add_scores(mode,dataset_name,probs,res_dir):
    f = open(DATA_ROOT[mode] + '/' + dataset_name + '_new_'+mode+'.txt')

    lines = f.readlines()
    f.close()
    lines = [tmp.strip() for tmp in lines]
    frame_res_name=os.path.join(res_dir,dataset_name+'_'+mode+'_frame_res.txt')
    print("frame_res_name:",frame_res_name)
    f = open(frame_res_name, 'w')
    for line, prob in zip(lines, probs):
        out = line + ' ' + str(prob)
        # print("submission_str(prob):",str(prob))
        f.write(out + '\n')
    f.close()


def frame2video(mode,dataset_name,res_dir,cal_method):
    frame_res_name=os.path.join(res_dir,dataset_name+'_'+mode+'_frame_res.txt')
    video_res_name=os.path.join(res_dir,dataset_name+'_'+mode+'_video_res.txt')
    print("video_res_name:",video_res_name)

    write_list(load_list(os.path.join(DATA_ROOT[mode] , dataset_name + '_'+mode+'_res.txt')),
                   cal_list(mode, load_list(frame_res_name),cal_method),video_res_name)

def get_method(data,cal_method):
    if cal_method=='mean':
        return np.mean(data)
    elif cal_method=='part_mean':
        data = sorted(data)
        lsize = len(data)
        # print("size:",lsize)
        aver = 0
        msum = 0
        cut = math.floor(lsize * 0.1)
        for i in range(cut, lsize - cut):
            msum += data[i]
        aver = msum / (lsize - cut * 2)
        return aver

    elif cal_method=='median':
        return np.median(data)
    elif cal_method=='meanstd':
        std = np.std(data)
        mean = np.mean(data)
        b = 3
        lower_limit = mean - b * std
        upper_limit = mean + b * std
        lsize = len(data)
        # print("l", lower_limit)
        # print("u", upper_limit)
        for i in data:
            if i < lower_limit or i > upper_limit:
                data.remove(i)
        # print(data)
        return sum(data) / len(data)
    elif cal_method=='medstd':
        median = np.median(data)
        b = 1.4826
        mad = b * np.median(np.abs(data - median))
        lower_limit = median - (3 * mad)
        upper_limit = median + (3 * mad)
        lsize = len(data)
        # print("l", lower_limit)
        # print("u", upper_limit)
        for i in data:
            if i < lower_limit or i > upper_limit:
                data.remove(i)
        # print(data)
        return sum(data) / len(data)


def cal_list(mode, lines,cal_method):
    list = []
    index = 0
    max_index = 0
    sum_score = 0.0
    aver_score = 0.0
    end_char = 10 if mode == 'dev' else 11
    # end_char=11 if mode=='test'
    previous = int(lines[0].strip().split(' ')[0][7:end_char])

    # print("previous:", previous)
    print("len(lines):", len(lines))
    count = 0

    vlist = []

    for line in lines:
        count += 1
        line = line.strip().split(' ')

        if (previous == int(line[0][7:end_char])):
            index += 1
            vlist.append(float(line[3]))
            sum_score += float(line[3])
            # print("index:", index)
            if (count == len(lines)):
                aver_score = get_method(vlist,cal_method)
                vlist.clear()
                #                 print("aver_score:",aver_score)
                list.append(str(aver_score))
        else:
            aver_score = get_method(vlist,cal_method)
            vlist.clear()

            list.append(str(aver_score))
            index = 1
            sum_score = 0.0
            aver_score = 0.0
            previous = int(line[0][7:end_char])
    print("list:", len(list))
    return list

def combine_list_3(name1, name2, name3, name):
    l1 = load_list(name1)
    l2 = load_list(name2)
    l3 = load_list(name3)
    l = l1 + l2 + l3
    file = open(name, 'w+')
    for line in l:
        file.write(str(line))
    file.close()

def combine_list_6(name1, name2, name3,name4,name5,name6, name):
    l1 = load_list(name1)
    l2 = load_list(name2)
    l3 = load_list(name3)
    l4 = load_list(name4)
    l5 = load_list(name5)
    l6 = load_list(name6)

    l = l1 + l2 + l3 + l4 + l5 + l6
    file = open(name, 'w+')
    for line in l:
        file.write(str(line))
    file.close()

if __name__=='__main__':
    print("submission module!(abondoned)")
    # parser=argparse.ArgumentParser()
    # parser.add_argument('-i',"--image_size",type=int,default=48)
    # parser.add_argument('-p','--phase', type=int, default=2, choices=[ 1, 2])
    # parser.add_argument('--image_mode', type=str, default='fusion', choices=['fusion','color'])
    # parser.add_argument('--result_dir', type=str,default='results')
    #
    # # image_size=64
    # config = parser.parse_args()
    # print("image_mode:",config.image_mode)
    # print("phase:",config.phase)
    #
    # time = time.localtime(time.time())
    # tm_mark = str(time.tm_year)+'_'+str(time.tm_mon) + '_'+str(time.tm_mday) +'_'+ str(time.tm_hour) +'_'+ str(time.tm_min)
    # print("saved result in:", RES_DIR[config.image_mode] + config.image_mode + '_phase' + str(config.phase) + '_' + str(
    #     config.image_size) + '_res-' + tm_mark + '.txt')
    #
    # result_path=os.path.join(config.result_dir,config.image_mode+ '_' +config.model + '_' + str(config.image_size))+'/'
    # #
    # if config.phase==1:
    #     combine_list_3(result_path + '4@1_dev_video_res.txt',
    #                  result_path + '4@2_dev_video_res.txt',
    #                  result_path + '4@3_dev_video_res.txt',
    #                  result_path + config.image_mode+'_phase'+str(config.phase)+'_'+str(config.image_size)+'_res-'+tm_mark+'.txt')
    # elif config.phase==2:
    #     combine_list_6(res_dir + '4@1_dev_video_res.txt',
    #                  res_dir + '4@1_test_video_res.txt',
    #                  res_dir + '4@2_dev_video_res.txt',
    #                  res_dir + '4@2_test_video_res.txt',
    #                  res_dir + '4@3_dev_video_res.txt',
    #                  res_dir + '4@3_test_video_res.txt',
    #                 result_path + config.image_mode+'_phase'+str(config.phase)+'_'+str(config.image_size)+'_'+config.cal_method+'_'+config.note+'-'+tm_mark+'.txt')