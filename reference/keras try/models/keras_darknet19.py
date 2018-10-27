"""Darknet19 Model Defined in Keras."""
import sys
import os
import functools
from functools import partial

from keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import glorot_uniform
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from functools import reduce

sys.path.append(os.path.join(os.getcwd(), 'third try'))
# from utils import compose

# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2D, padding='same')


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')



@functools.wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet weight regularizer for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs.update(kwargs)
    return _DarknetConv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def bottleneck_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
    return compose(
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def bottleneck_x2_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
    return compose(
        bottleneck_block(outer_filters, bottleneck_filters),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def darknet_body():
    """Generate first 18 conv layers of Darknet-19."""
    return compose(
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(),
        bottleneck_block(128, 64),
        MaxPooling2D(),
        bottleneck_block(256, 128),
        MaxPooling2D(),
        bottleneck_x2_block(512, 256),
        MaxPooling2D(),
        bottleneck_x2_block(1024, 512))


def darknet19(inputs):
    """Generate Darknet-19 model for Imagenet classification."""
    body = darknet_body()(inputs)
    logits = DarknetConv2D(1000, (1, 1), activation='softmax')(body)
    return Model(inputs=inputs, outputs=logits)


"""
def CNN(inputs):
    body = darknet_body()(inputs)
    X = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', name='avg_pool')(body)
    X = Flatten()(X)
    logits = Dense(units=2, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)
    return Model(inputs=inputs, outputs=logits)


I = Input(shape=(608, 608, 3), dtype='float32')
model = CNN(inputs=I)
# model = darknet19(inputs=I)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ImageGenerator
from keras.preprocessing.image import ImageDataGenerator
path = 'C:/Users/YY/Documents/Data/CCP/Custom_Data'
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        os.path.join(path, 'train'),
        target_size=(608, 608),
        batch_size=2,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        os.path.join(path, 'validation'),
        target_size=(608, 608),
        batch_size=2,
        class_mode='categorical')


# steps_per_epoch: 한 epoch에 사용한 스텝수를 지정함
# 한 epoch이 종료될 때 마다 검증할 때 사용되는 검증 스텝 수를 지정함
model.fit_generator(train_generator, epochs=10, validation_data=validation_generator)
"""
