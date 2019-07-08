
from glob import glob
import os
import numpy as np
import random


if __name__ == '__main__':
    dataset_root = '/home/cc_lee/Dataset/MIT-BIH_AD/'
    output_dirs = ['Normal/', 'LBBB/', 'RBBB/', 'APC/', 'VPC/', 'PE/', 'VEB/', 'VFW']

    count = 0
    pathes_by_type = {}
    for type in output_dirs:
        dir = os.path.join(dataset_root, type, '*')
        paths = glob(dir)
        pathes_by_type[type] = paths
        count += len(paths)

    train_list = []
    val_list = []
    test_list = []

    for type in output_dirs:
        cur = pathes_by_type[type]
        if len(cur) is 0:
            continue

        random.shuffle(cur)

        for i in range(int(len(cur)*0.6)):
            temp = cur[i].split('/')
            train_list.append('{} {}'.format(os.path.join(temp[-2], temp[-1]), output_dirs.index(type)))
            cur[i] = None

        for i in range(int(len(cur)*0.6), int(len(cur)*0.8)):
            if cur[i] is None:
                continue
            else:
                temp = cur[i].split('/')
                val_list.append('{} {}'.format(os.path.join(temp[-2], temp[-1]), output_dirs.index(type)))
                cur[i] = None

        for i in range(int(len(cur) * 0.8), len(cur)):
            if cur[i] is None:
                continue
            else:
                temp = cur[i].split('/')
                test_list.append('{} {}'.format(os.path.join(temp[-2], temp[-1]), output_dirs.index(type)))
                cur[i] = None

        with open('MIT-BIH_AD_val.txt', 'w') as val:
            for v in val_list:
                val.write(v+'\n')

        with open('MIT-BIH_AD_train.txt', 'w') as train:
            for r in train_list:
                train.write(r+'\n')

        with open('MIT-BIH_AD_test.txt', 'w') as test:
            for t in test_list:
                test.write(t+'\n')

    print('train:{} val:{} test:{} tol:{}'.format(len(train_list), len(val_list), len(test_list), count))