from keras.callbacks import Callback
import keras.backend as K
import math

class Step(Callback):

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.current_epoch = 1

    def change_lr(self, new_lr):
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose == 1:
            print('Learning rate is %g' %new_lr)

    def on_epoch_begin(self, epoch, logs={}):
        self.current_epoch = epoch
    def on_batch_begin(self, batch, logs={}):
        new_lr = 0.0001 * 0.95 ** math.ceil(self.current_epoch*batch/1000)
        self.change_lr(new_lr)

