# coding=utf8
from models import proposed_model
from keras.optimizers import Adam
import numpy as np
import cv2
import os
import time
import random
import glob
import os
from keras.utils import np_utils
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

def main():
    # init model
    #class_names = ['Normal', 'LBBB', 'RBBB', 'APC', 'PVC', 'PAB', 'VEB', 'VFW']
    class_names = ['LBBB', 'RBBB', 'APC', 'PVC', 'PAB', 'VEB', 'VFW']
    #class_names = ['Normal', 'Abnormal']

    #PE is PAB
    imageh = 128
    imagew = 128

    inputH = 128
    inputW = 192

    # ---------------------------change file models & weights--------------------

    model = proposed_model()

    lr = 0.0001
    adm = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
    model.summary()

    model.load_weights('result/special_120eps/proposed_model_False.h5', by_name=True)

    # ---------------------------change models & weights--------------------

    test_file = './MIT-BIH_AD_sp_test.txt'
    test_img_path = '/data/MIT-BIH_AD_v3'

    augmentation = False
    output_img = False
    outputdir = os.path.join('./inference_v2/', str(augmentation))
    os.makedirs(outputdir, exist_ok=True)

    os.makedirs(outputdir+'/False', exist_ok=True)
    os.makedirs(outputdir+'/True', exist_ok=True)

    f = open(test_file, 'r')
    lines = f.readlines()
    random.shuffle(lines)
    TP = 0
    count = 0
    total = len(lines)

    # counter = {'Normal': 0, 'LBBB': 0, 'RBBB': 0, 'APC': 0, 'PVC': 0, 'PAB': 0, 'VEB': 0, 'VFW': 0}
    # tp_counter = {'Normal': 0, 'LBBB': 0, 'RBBB': 0, 'APC': 0, 'PVC': 0, 'PAB': 0, 'VEB': 0, 'VFW': 0}

    counter = {'LBBB': 0, 'RBBB': 0, 'APC': 0, 'PVC': 0, 'PAB': 0, 'VEB': 0, 'VFW': 0}
    tp_counter = {'LBBB': 0, 'RBBB': 0, 'APC': 0, 'PVC': 0, 'PAB': 0, 'VEB': 0, 'VFW': 0}

    #counter = {'Normal':0, 'Abnormal':0}
    #tp_counter = {'Normal':0, 'Abnormal':0}

    for line in tqdm(lines):
        path = line.split(' ')[0]
        label = line.split(' ')[-1]

        label = label.strip('\n')
        answer = int(label)
        img = os.path.join(test_img_path, path)

        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if augmentation:
            Hshmean = int(np.round(np.max([0, np.round((imageh - inputH) / 2)])))
            Wshmean = int(np.round(np.max([0, np.round((imagew - inputW) / 2)])))
            image = image[Hshmean:Hshmean + inputH, Wshmean:Wshmean + inputW, :]
            image = cv2.resize(image, (imagew, imageh))
        else:
            pass

        input_data = np.zeros((1, imagew, imageh, 3), dtype='float32')
        input_data[0] = image[:,64:64+128,:]

        pred = model.predict(input_data)
        label = np.argmax(pred[0])

        if label == answer:
            TP += 1
            tp_counter[class_names[label]] += 1

        count += 1
        counter[class_names[answer]] += 1

        if output_img:
            if np.argmax(pred[0]) == 1:
                color_t = (0, 255, 255)
            else:
                color_t = (0, 255, 0)

            image = cv2.resize(image, (128*3, 128*3))

            cv2.putText(image, class_names[answer].split(' ')[-1].strip(), (10, 30),
                        cv2.FONT_ITALIC, 1,
                        color_t, 1)

            cv2.putText(image, class_names[label].split(' ')[-1].strip(), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color_t, 1)
            cv2.putText(image, "prob: %.4f" % pred[0][label], (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color_t, 1)



            cv2.imwrite(os.path.join(outputdir, str(answer==label)) + '/' + '{}_{}'.format(class_names[answer], os.path.split(path)[1][:-4] + '_result.jpg', ), image)


    print('{}/{} Acc: {} Pred:{} Answer: {}'.format(count, total, str(TP / count), class_names[label], class_names[answer] ) )
    '''print('Normal:{}/{}={},\n LBBB:{}/{}={},\n RBBB:{}/{}={},\n APC:{}/{}={},\n PVC:{}/{}={},\n PAB:{}/{}={},\n VEB:{}/{}={},\n VFW:{}/{}={}'.format(
        tp_counter['Normal'], counter['Normal'], (tp_counter['Normal']/counter['Normal']),
        tp_counter['LBBB'], counter['LBBB'], (tp_counter['LBBB'] / counter['LBBB']),
        tp_counter['RBBB'], counter['RBBB'], (tp_counter['RBBB'] / counter['RBBB']),
        tp_counter['APC'], counter['APC'], (tp_counter['APC'] / counter['APC']),
        tp_counter['PVC'], counter['PVC'], (tp_counter['PVC'] / counter['PVC']),
        tp_counter['PAB'], counter['PAB'], (tp_counter['PAB'] / counter['PAB']),
        tp_counter['VEB'], counter['VEB'], (tp_counter['VEB'] / counter['VEB']),
        tp_counter['VFW'], counter['VFW'], (tp_counter['VFW'] / counter['VFW'])

    ))
    '''
    print('LBBB:{}/{}={},\n RBBB:{}/{}={},\n APC:{}/{}={},\n PVC:{}/{}={},\n PAB:{}/{}={},\n VEB:{}/{}={},\n VFW:{}/{}={}'.format(
        tp_counter['LBBB'], counter['LBBB'], (tp_counter['LBBB'] / counter['LBBB']),
        tp_counter['RBBB'], counter['RBBB'], (tp_counter['RBBB'] / counter['RBBB']),
        tp_counter['APC'], counter['APC'], (tp_counter['APC'] / counter['APC']),
        tp_counter['PVC'], counter['PVC'], (tp_counter['PVC'] / counter['PVC']),
        tp_counter['PAB'], counter['PAB'], (tp_counter['PAB'] / counter['PAB']),
        tp_counter['VEB'], counter['VEB'], (tp_counter['VEB'] / counter['VEB']),
        tp_counter['VFW'], counter['VFW'], (tp_counter['VFW'] / counter['VFW'])

    ))
    '''
    print( 'Normal:{}/{}={}, \n Abnormal:{}/{}={}'.format(
        tp_counter['Normal'], counter['Normal'], (tp_counter['Normal'] / counter['Normal']),
        tp_counter['Abnormal'], counter['Abnormal'], (tp_counter['Abnormal'] / counter['Abnormal'])))
    '''
if __name__ == '__main__':
    main()
