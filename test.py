import time
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
from sklearn.metrics import confusion_matrix
from datetime import timedelta
from skimage.transform import resize
import cv2;
# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 128

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['laptops', 'phones','clothes','drinks']
num_classes = len(classes)

# batch size
#batch_size = 32

# validation split
#validation_size = .16

# how long to wait after validation loss stops improving before terminating training
#early_stopping = None  # use None if you don't want to implement early stoping

#train_path = 'train/'
#test_path = 'test1/'
#checkpoint_dir = "models/"

#data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
#test_images, test_ids = dataset.read_test_set(test_path, img_size)

#print("Size of:")
#print("- Training-set:\t\t{}".format(len(data.train.labels)))
#print("- Test-set:\t\t{}".format(len(test_images)))
#print("- Validation-set:\t{}".format(len(data.valid.labels)))



# Get some random images and their labels from the train set.

#images, cls_true  = data.train.images, data.train.cls

# Plot the images and labels using our helper-function above.
#plot_images(images=images, cls_true=cls_true)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

#layer_conv1

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

#layer_conv2

layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)

#layer_conv3
layer_flat, num_features = flatten_layer(layer_conv3)
#layer_flat
#num_
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)


y_pred = tf.nn.softmax(layer_fc2)
global y_pred_cls 
y_pred_cls = tf.argmax(y_pred, dimension=1)
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
#cost = tf.reduce_mean(cross_entropy)
#optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#correct_prediction = tf.equal(y_pred_cls, y_true_cls)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#tf.reset_default_graph()
session = tf.Session()

#session.run(tf.initialize_all_variables())
#train_batch_size = batch_size




# print_validation_accuracy()
#optimize(num_iterations=99)  # We already performed 1 iteration above.
# print_validation_accuracy()
#optimize(num_iterations=900)  # We performed 100 iterations above.


# print_validation_accuracy(show_example_errors=True)
#optimize(num_iterations=9000) # We performed 1000 iterations above.

# print_validation_accuracy(show_example_errors=True, show_confusion_matrix=True)
saver = tf.train.Saver()
savee_path = saver.restore(session, "ckpt/model.ckpt")
print ("restore path: ")
plt.axis('off')

test_cat1 = cv2.imread('phone.jpg')
test_cat2 = cv2.imread('phone1.jpg')
test_cat3 = cv2.imread('phone2.jpg')
test_cat4 = cv2.imread('phone3.jpg')
test_cat5 = cv2.imread('phone4.jpg')
test_cat6 = cv2.imread('phone5.jpg')
test_cat7 = cv2.imread('phone6.jpg')
test_cat8 = cv2.imread('phone7.jpg')



test_dog = cv2.imread('laptop.jpg')
test_dog2 = cv2.imread('laptop1.jpg')
test_dog3 = cv2.imread('laptop2.jpg')
test_dog4 = cv2.imread('laptop3.jpg')
test_dog5 = cv2.imread('laptop4.jpg')
test_dog6 = cv2.imread('laptop5.jpg')
test_dog7 = cv2.imread('laptop6.jpg')
test_dog8 = cv2.imread('laptop7.jpg')




def sample_prediction(test_im):
    
    feed_dict_test = {
        x: test_im.reshape(1, img_size_flat),
        y_true: np.array([[0,1,2,3]])
    }

    test_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    return classes[test_pred[0]]


test_cat1 = resize(test_cat1, (128, 128))
test_cat2 = resize(test_cat2, (128, 128))
test_cat3 = resize(test_cat3, (128, 128))
test_cat4 = resize(test_cat4, (128, 128))
test_cat5 = resize(test_cat5, (128, 128))
test_cat6 = resize(test_cat6, (128, 128))
test_cat7 = resize(test_cat7, (128, 128))
test_cat8 = resize(test_cat8, (128, 128))


test_dog = resize(test_dog, (128, 128))
test_dog2 = resize(test_dog2, (128, 128))
test_dog3 = resize(test_dog3, (128, 128))
test_dog4 = resize(test_dog4, (128, 128))
test_dog5 = resize(test_dog5,(128, 128))
test_dog6 = resize(test_dog6, (128, 128))
test_dog7 = resize(test_dog7, (128, 128))
test_dog8 = resize(test_dog8, (256, 256))


print("Predicted class for test_phone: {}".format(sample_prediction(test_cat1)))
print("Predicted class for test_phone: {}".format(sample_prediction(test_cat2)))
print("Predicted class for test_phone: {}".format(sample_prediction(test_cat3)))
print("Predicted class for test_phone: {}".format(sample_prediction(test_cat4)))
print("Predicted class for test_phone: {}".format(sample_prediction(test_cat5)))
print("Predicted class for test_phone: {}".format(sample_prediction(test_cat6)))
print("Predicted class for test_phone: {}".format(sample_prediction(test_cat7)))
print("Predicted class for test_phone: {}".format(sample_prediction(test_cat8)))


print("Predicted class for laptop: {}".format(sample_prediction(test_dog)))
print("Predicted class for laptop: {}".format(sample_prediction(test_dog2)))
print("Predicted class for laptop: {}".format(sample_prediction(test_dog3)))
print("Predicted class for laptop: {}".format(sample_prediction(test_dog4)))
print("Predicted class for laptop: {}".format(sample_prediction(test_dog5)))
print("Predicted class for laptop: {}".format(sample_prediction(test_dog6)))
print("Predicted class for laptop: {}".format(sample_prediction(test_dog7)))
print("Predicted class for laptop: {}".format(sample_prediction(test_dog8)))

