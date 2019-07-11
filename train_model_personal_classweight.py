from models import proposed_model
from keras.optimizers import Adam
from keras.utils import np_utils

from callbacks import Step
import numpy as np
import random
import cv2
import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import glob
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()

def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()

def process_batch(lines,img_path,inputH,inputW,train=True):
    imagew = 192
    imageh = 128
    num = len(lines)
    batch = np.zeros((num, inputH, inputW, 3), dtype='float32')


    labels = np.zeros(num, dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]

        label = label.strip('\n')
        label = int(label)

        img = os.path.join(img_path, path)

        if train:
            crop_x = random.randint(0, np.max([0, imagew-inputW]))
            image = cv2.imread(img)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[:, crop_x:crop_x + inputW, :]
            batch[i] = image
            labels[i] = label
        else:
            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[:, 32:32+128, :]
            batch[i] = image
            labels[i] = label

    return batch, labels

def generator_train_batch( train_txt, batch_size, num_classes, img_path, inputH, inputW ):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])

        for i in range(int(num/batch_size)):
            a = i*batch_size
            b = (i+1)*batch_size
            x_train, x_labels = process_batch(new_line[a:b], img_path, inputH, inputW, train=True)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            yield x_train, y

def generator_val_batch(val_txt,batch_size,num_classes,img_path,inputH,inputW):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test,y_labels = process_batch(new_line[a:b],img_path,inputH,inputW,train=False)
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield y_test, y

def main():

    outputdir = 'result/192_128_class_weight_v3_120eps/'
    if os.path.isdir(outputdir):
        print('save in :' + outputdir)
    else:
        os.makedirs(outputdir)

    train_img_path = '/data/MIT-BIH_AD_v3/'
    test_img_path = '/data/MIT-BIH_AD_v3/'
    train_file = './MIT-BIH_AD_train.txt'
    test_file = './MIT-BIH_AD_val.txt'
    num_classes = 8

    f1 = open(train_file, 'r')
    f2 = open(test_file, 'r')
    lines = f1.readlines()
    f1.close()
    train_samples = len(lines)
    lines = f2.readlines()
    f2.close()
    val_samples = len(lines)

    batch_size = 32
    epochs = 120
    input_h = 128
    input_w = 128

    class_weight = {0:(1-(75016/107620))*8, 1:(1-(8072/107620))*8, 2:(1-(7256/107620))*8, 3:(1-(2544/107620))*8,
                    4:(1-(7130/107620))*8, 5:(1-(7024/107620))*8, 6:(1-(106/107620))*8, 7:(1-(472/107620))*8}

    model = proposed_model(nb_classes=num_classes)

    lr = 0.0001
    adam = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    history = model.fit_generator(
        generator_train_batch(train_file, batch_size, num_classes, train_img_path, input_h, input_w),
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        callbacks=[Step()],
        validation_data=generator_val_batch(test_file, batch_size, num_classes, test_img_path, input_h, input_w),
        validation_steps=val_samples // batch_size,
        verbose=1,
        class_weight=class_weight)
    plot_history(history, outputdir)
    save_history(history, outputdir)
    model.save_weights(outputdir+'proposed_model.h5')

if __name__ == '__main__':
    main()
