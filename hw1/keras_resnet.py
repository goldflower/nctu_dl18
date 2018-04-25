import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec

from keras.models import Sequential, Model, load_model
from keras.layers import add, Dense, Activation, Reshape, Merge, Embedding, BatchNormalization, Dropout, Input, Lambda, TimeDistributed, LSTM
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D, ConvLSTM2D
from keras.layers.merge import concatenate
from keras.engine.topology import Layer
from keras import backend as K
from keras.regularizers import l2

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.utils import plot_model
from keras import initializers
from keras.optimizers import Adamax, Adam, Nadam, SGD
from keras.datasets import cifar10


import pandas as pd
import numpy as np
#np.random.seed(123)

import pickle, sys, csv, os
from datetime import datetime

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


class ResNet():
    
    def __init__(self, nb_blocks=3, img_shape=(32, 32), stages=None, block_type='normal'):
        """ Some global parameters
        Args:
            nb_blocks(int): the total blocks for resnet, default = 3 (resnet-20)
                each stage contains the same blocks
                e.g., 
                    for normal/pre-act resnet, we have 1 + 6n + 1 blocks, each stage contains 2n blocks
                        => for nb_blocks = 18, we have depth = 1 + 18*6 + 1 = 110
                    for bottleneck resnet, we have 1 + 9n + 1 blocks, each stage contains 3n blocks
                        => for nb_blocks = 3, we have depth = 1 + 27 + 1 = 29 
            img_shape(tuple(int, int)): the input image shape, for cifar 10 we have 32x32 images
            stages(dict): we have 3 stages, each contains different kernel size and nb_filters
            block_type(str): the resnet basic block, 'normal', 'bottleneck', 'pre_act', defaults='normal'
        """
        self.nb_blocks = nb_blocks
        self.img_shape = img_shape
        self.block_type = block_type
        if stages is None:
            self.stages = {1:((3, 3), 16), 2:((3, 3), 32), 3:((3, 3), 64)} # key:stage; value:(kernel_size, nb_filters)
        else:
            self.stages = stages
        self.model = self.define_graph()
        
    def res_block(self, inputs,
                        num_filters=16,
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        batch_normalization=True,
                        conv_first=True):
                     
                     
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x
        
    def shortcut_block(self, x, nb_filters, strides):
        
        x = Conv2D(filters=nb_filters, 
                           kernel_size=(3,3), 
                           strides=strides,
                           padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(x)
        return x
        
    def define_graph(self, nb_classes=10):
        inputs = Input(shape=self.img_shape)
        conv_shortcut = self.shortcut_block(inputs, self.stages[1][1], (1,1))
        conv_shortcut = BatchNormalization()(conv_shortcut)
        conv_shortcut = Activation('relu')(conv_shortcut)     
        
        for i in range(1, len(self.stages)+1):
            for j in range(self.nb_blocks):
                strides = (1, 1)
                if i != 1 and j == 0:  # first layer but not first stack
                    strides = (2, 2)  # downsample
                    conv = self.res_block(inputs=conv_shortcut,
                                          num_filters=self.stages[i][1],
                                          strides=strides)
                    conv = self.res_block(inputs=conv,
                                          num_filters=self.stages[i][1],
                                          activation=None)
                    conv_shortcut = self.res_block(inputs=conv_shortcut,
                                       num_filters=self.stages[i][1],
                                       kernel_size=1,
                                       strides=strides,
                                       activation=None,
                                       batch_normalization=False)                    
                else:
                    conv = self.res_block(inputs=conv_shortcut,
                                          num_filters=self.stages[i][1],
                                          strides=strides)
                    conv = self.res_block(inputs=conv,
                                          num_filters=self.stages[i][1],
                                          activation=None)
                conv_shortcut = keras.layers.add([conv_shortcut, conv])
                conv_shortcut = Activation('relu')(conv_shortcut)
                
                #conv_shortcut = self.shortcut_block(conv, self.stages[i][1], (2,2))            
            #print('end loop:', conv.shape)
        out = AveragePooling2D(pool_size=(8,8))(conv_shortcut)
        #print(conv.shape)
        out = Flatten()(out)
        #print(conv.shape)        
        outputs = Dense(nb_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(out)
        model = Model(inputs=inputs, outputs=outputs)
        print(model.summary())
        return model
        
def schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

if __name__ == '__main__':    

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    x_train_mean = np.mean(x_train, axis=0)
    x_train_var = np.std(x_train, axis=0)


    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]
    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train_var = np.std(x_train, axis=0)

    x_train = x_train - x_train_mean
    x_test = x_test - x_train_mean

    x_train = x_train.astype('float32') / x_train_var
    x_test = x_test.astype('float32') / x_train_var
    
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    

    depths = [20, 56, 110]
    depths = [20, 56, 110]
    depths = [110]
    batch_size = 256
    epochs = 300
    ns = [(i - 2) // 6 for i in depths]
    print(ns)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5)    
    optimizer = SGD(0.1, momentum=0.9, decay=1e-4)
    optimizer = Nadam(schedule(0))
    optimizer = Adam(lr=schedule(0))
    for n in ns:
        checkpoint = ModelCheckpoint(filepath='models/model'+str(n)+'.h5',
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True)
        #callbacks = [LearningRateScheduler(schedule), checkpoint]    
        
        resnet = ResNet(nb_blocks = n, img_shape=x_train.shape[1:], block_type='normal')
        model = resnet.model
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        tensorboard = TensorBoard(log_dir='my_board', histogram_freq=1, write_graph=True, write_images=False)
        callbacks = [checkpoint, lr_reducer, tensorboard, LearningRateScheduler(schedule)]
        #callbacks = [lr_reducer, tensorboard, LearningRateScheduler(schedule)]
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=4/32,
                # randomly shift images vertically
                height_shift_range=4/32,
                horizontal_flip=True,
                vertical_flip=False)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        #model.fit(x_train, y_train,
        #          batch_size=batch_size,
        #          epochs=epochs,
        #          validation_data=(x_test, y_test),
        #          shuffle=True,
        #          callbacks=callbacks)
        # Fit the model on the batches generated by datagen.flow().
        
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)