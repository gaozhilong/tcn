from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import utils

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

class ResidualBlock(tf.keras.Model):
    def __init__(self, 
            stage,
            block,
            filters,
            kernel_size,
            dilation,
            padding,
            pre_act=False,
            **kwargs):
        super(ResidualBlock, self).__init__(name='residual_block_'+str(stage), **kwargs)
        assert padding in ['causal', 'same']
        filters1, filters2, filters3 = filters
        self.name_base = name='residual_block_'+str(stage) + '_' + block + str(pre_act) +'_branch'
        conv_name_base = 'res' + str(stage) + block + str(pre_act) +'_branch'
        bn_name_base = 'bn' + str(stage) + block + str(pre_act) + '_branch'

        self.pre_act = pre_act
        # block1
        self.conv1 = layers.Conv1D(filters=filters1, kernel_size=kernel_size,
                            dilation_rate=dilation, padding=padding,
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                            name=conv_name_base + '2a')
        self.batch1 = layers.BatchNormalization(axis=-1,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2a')
        self.ac1 = layers.Activation('relu')

        self.drop_1 = layers.SpatialDropout1D(rate=0.5)
        
        # block2
        self.conv2 = layers.Conv1D(filters=filters2, kernel_size=kernel_size,
                            dilation_rate=dilation, padding=padding,
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                            name=conv_name_base + '2b')
        self.batch2 = layers.BatchNormalization(axis=-1,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2b')
        self.ac2 = layers.Activation('relu')

        self.drop_2 = layers.SpatialDropout1D(rate=0.5)
        
        # 为了防止维度不一致使用 1*1 卷积在channel处进行匹配
        self.downsample = layers.Conv1D(filters=filters3, kernel_size=1,
                                padding='same', kernel_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                                name=conv_name_base + '2c')
        self.batch3 = layers.BatchNormalization(axis=-1,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2c')
        self.ac3 = layers.Activation('relu')

    def call(self, inputs):
        if not self.pre_act:
            x = self.batch1(inputs)
            x = self.ac1(x)
            x = self.drop_1(x)
            x = self.conv1(x)
            x = self.batch2(x)
            x = self.ac2(x)
            x = self.drop_2(x)
            x = self.conv2(x)
            x = self.batch3(x)
            x = self.ac3(x)
            pre_x = self.downsample(inputs)
            x = layers.add([x, pre_x])
        else:
            x = self.conv1(inputs)
            x = self.batch1(x)
            x = self.ac1(x)
            x = self.drop_1(x)
            x = self.conv2(x)
            x = self.batch2(x)
            x = self.ac2(x)
            x = self.drop_2(x)
            pre_x = self.downsample(inputs)
            x = self.batch3(x)
            x = layers.add([x, pre_x])
            x = self.ac3(x)
        return x

class SEResidualBlock(tf.keras.Model):
    def __init__(self, 
            stage,
            block,
            filters,
            kernel_size,
            dilation,
            padding,
            pre_act=False,
            **kwargs):
        super(SEResidualBlock, self).__init__(name='se_residual_block_'+str(stage), **kwargs)
        assert padding in ['causal', 'same']
        filters1, filters2, filters3 = filters
        self.out_dim = filters3
        conv_name_base = 'res' + str(stage) + block + str(pre_act) +'_branch'
        bn_name_base = 'bn' + str(stage) + block + str(pre_act) + '_branch'
        dn_name_base = 'dn' + str(stage) + block + str(pre_act) +'_branch'

        self.pre_act = pre_act
        # block1
        self.conv1 = layers.Conv1D(filters=filters1, kernel_size=kernel_size,
                            dilation_rate=dilation, padding=padding,
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                            name=conv_name_base + '2a')
        self.batch1 = layers.BatchNormalization(axis=-1,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2a')
        self.ac1 = layers.Activation('relu')
        
        # block2
        self.conv2 = layers.Conv1D(filters=filters2, kernel_size=kernel_size,
                            dilation_rate=dilation, padding=padding,
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                            name=conv_name_base + '2b')
        self.batch2 = layers.BatchNormalization(axis=-1,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2b')
        self.ac2 = layers.Activation('relu')

        self.drop_1 = layers.SpatialDropout1D(rate=0.05)
        
        # 为了防止维度不一致使用 1*1 卷积在channel处进行匹配
        self.downsample = layers.Conv1D(filters=filters3, kernel_size=1,
                                padding='same', kernel_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                                name=conv_name_base + '2c')
        self.batch3 = layers.BatchNormalization(axis=-1,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2c')
        self.ac3 = layers.Activation('relu')

        self.drop_2 = layers.SpatialDropout1D(rate=0.05)

        self.gavgpool = layers.GlobalAvgPool1D()

        self.dnn_1 = layers.Dense(3,
            kernel_regularizer=regularizers.l2(0.001),
            activation='relu', 
            name=dn_name_base + '2a')
        self.ac4 = layers.Activation('relu')
        self.dnn_2 = layers.Dense(self.out_dim,
            kernel_regularizer=regularizers.l2(0.001),
            activation='relu', 
            name=dn_name_base + '2b')
        self.ac4 = layers.Activation('relu')

    def call(self, inputs):
        if not self.pre_act:
            x = self.batch1(inputs)
            x = self.ac1(x)
            x = self.conv1(x)
            x = self.batch2(x)
            x = self.ac2(x)
            x = self.drop_1(x)
            x = self.conv2(x)
            x = self.batch3(x)
            x = self.ac3(x)
            x = self.drop_2(x)
            pre_x = self.downsample(inputs)
            squeeze = self.gavgpool(x)
            squeeze = self.dnn_1(squeeze)
            squeeze = self.ac4(squeeze)
            squeeze = self.dnn_2(squeeze)
            squeeze = tf.nn.sigmoid(squeeze)
            squeeze = tf.reshape(squeeze, [-1,1,self.out_dim])
            x = x * squeeze
            x = layers.add([x, pre_x])
        else:
            x = self.conv1(inputs)
            x = self.batch1(x)
            x = self.ac1(x)
            x = self.drop_1(x)
            x = self.conv2(x)
            x = self.batch2(x)
            x = self.ac2(x)
            x = self.drop_2(x)
            pre_x = self.downsample(inputs)
            x = self.batch3(x)
            squeeze = self.gavgpool(x)
            squeeze = self.dnn_1(squeeze)
            squeeze = self.ac4(squeeze)
            squeeze = self.dnn_2(squeeze)
            squeeze = tf.nn.sigmoid(squeeze)
            squeeze = tf.reshape(squeeze, [-1,1,self.out_dim])
            x = x * squeeze
            x = layers.add([x, pre_x])
            x = self.ac3(x)
        return x

class TemporalConvNet(tf.keras.Model):
    def __init__(self, 
            filters,
            kernel_size,
            dilations,
            padding,
            pre_act=False,
            se_block=False,
            **kwargs):
        super(TemporalConvNet, self).__init__(name='tcn_model', **kwargs)
        assert padding in ['causal', 'same']

        # 初始化 model
        self.tcn_net = tf.keras.Sequential()
        num_dilations = len(dilations)
        for i in range(num_dilations):
            if se_block:
                self.tcn_net.add(SEResidualBlock(i,
                    'residual_block',
                    filters,
                    kernel_size,
                    dilations[i],
                    padding,
                    pre_act=pre_act))
            else:
                self.tcn_net.add(ResidualBlock(i,
                    'residual_block',
                    filters,
                    kernel_size,
                    dilations[i],
                    padding,
                    pre_act=pre_act))

    def call(self, inputs):
        return self.tcn_net(inputs)


class TcnClassification(tf.keras.Model):
    def __init__(self,
            class_num, 
            filters,
            kernel_size,
            dilations,
            padding,
            pre_act=False,
            se_block=False,
            **kwargs):
        super(TcnClassification, self).__init__(name='tcn_classification_model', **kwargs)
        assert padding in ['causal', 'same']

        # 初始化 model
        self.tcn_net = TemporalConvNet(
            filters=filters,
            kernel_size=kernel_size,
            dilations=dilations,
            padding=padding,
            pre_act=pre_act,
            se_block=se_block)
        # self.dnn_1 = layers.Dense(81,
        #     kernel_regularizer=regularizers.l2(0.001),
        #     activation='relu', 
        #     name='dnn_1')

        self.dnn_2 = layers.Dense(9,
            kernel_regularizer=regularizers.l2(0.001),
            activation='relu', 
            name='dnn_2')

        self.output_layer = layers.Dense(class_num,
            kernel_regularizer=regularizers.l2(0.001),
            activation='softmax', 
            name='output_layer')

    def call(self, inputs):
        x = self.tcn_net(inputs)
        x = layers.Lambda(lambda tt: tt[:, -1, :])(x)
        #x = self.dnn_1(x)
        x = self.dnn_2(x)
        x = self.output_layer(x)
        return x

class TcnRegression (tf.keras.Model):
    def __init__(self,
            class_num, 
            filters,
            kernel_size,
            dilations,
            padding,
            pre_act=False,
            se_block=False,
            **kwargs):
        super(TcnClassification, self).__init__(name='tcn_classification_model', **kwargs)
        assert padding in ['causal', 'same']

        # 初始化 model
        self.tcn_net = TemporalConvNet(
            filters=filters,
            kernel_size=kernel_size,
            dilations=dilations,
            padding=padding,
            pre_act=pre_act,
            se_block=se_block)
        # self.dnn_1 = layers.Dense(81,
        #     kernel_regularizer=regularizers.l2(0.001),
        #     activation='relu', 
        #     name='dnn_1')

        self.dnn_2 = layers.Dense(9,
            kernel_regularizer=regularizers.l2(0.001),
            activation='relu', 
            name='dnn_2')

        self.output_layer = layers.Dense(1, name='output_layer')

        # layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        # layers.Dense(64, activation='relu'),
        # layers.Dense(1)

    def call(self, inputs):
        x = self.tcn_net(inputs)
        x = layers.Lambda(lambda tt: tt[:, -1, :])(x)
        #x = self.dnn_1(x)
        x = self.dnn_2(x)
        x = self.output_layer(x)
        return x