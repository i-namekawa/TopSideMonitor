import numpy as np

from lasagne import layers
from nolearn.lasagne import NeuralNet
import theano

bboxLength = 96
trained_w_fg = False # indicate that the model is trained on foreground images (median background subtracted)

'''
these functions and the model were taken from Daniel Nouri's excellent 
theano/lasagne tutorial at 

http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

and slightly modified and trained with ~1400 frames.
'''

def float32(k):
    return np.cast['float32'](k)


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


def get_model():
    'danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#dropout'
    net6 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),  # !
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),  # !
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),  # !
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),  # !
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 1, bboxLength, bboxLength),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        dropout1_p=0.1,  # !
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        dropout2_p=0.2,  # !
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        dropout3_p=0.3,  # !
        hidden4_num_units=500,
        dropout4_p=0.5,  # !
        hidden5_num_units=500,
        output_num_units=4, output_nonlinearity=None,

        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float32(0.9)),

        regression=True,

        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            ],
        max_epochs=1350,
        verbose=1,
        )

    # the path is relative to multitrack.py which will import this module
    best_weights = np.load('models/lasagne/best_weights_nouriNet6.pickle')['best_weights']
    net6.load_params_from(best_weights)

    return net6
