#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 19:36:02 2018

@author: pankaj
"""

import tensorflow as tf
import numpy as np
from PIL import Image as PImage
#import os 
from glob import glob
import tflearn


train_data=[]
train_labels= []
test_data =[]
test_labels = []
size=(128,128)
height = 128
width = 128

# function to prepare training set
def train_set():
    folder_list=glob("All_61326/train_61326/*/");
    print folder_list

    count =0;
    for folder in folder_list:
        count = count+1
        image_list = glob(folder+'*.jpg')
        for i,image in enumerate(image_list):
            print image
            img = PImage.open(image)
            train_data.append(img.getdata().resize(size))
            train_labels.append(count)

# function to prepare test set
# folder structure under test directory is same as train directory
# Because of same structure, it is easy to provide the labels
            
def test_set():
    folder_list=glob("All_61326/test_61326/*/")
    count =0;
    for folder in folder_list:
        count = count+1
        image_list = glob(folder+'*.jpg')
        for i,image in enumerate(image_list):
            print image
            img = PImage.open(image)
            test_data.append(img.getdata().resize(size))
            test_labels.append(count)

def cnn_model_ic(features, labels, mode):
        """Model function for CNN."""
        
        # Real-time data preprocessing
        print("Doing preprocessing...")
        img_prep = tflearn.ImagePreprocessing()
        img_prep.add_featurewise_zero_center(per_channel=True, mean=[0.573364,0.44924123,0.39455055])
            
            # Real-time data augmentation
        print("Building augmentation...")
        img_aug = tflearn.ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_crop([128, 128], padding=4)
        
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        # 61326 images are 4160x3120 pixels, i resize to 128*128 and have three color channel
        input_layer = tflearn.input_data(shape=[-1, 128, 128, 3],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug)
        input_layer = tf.reshape(features["x"], [-1, 128, 128, 3])
        # Convolutional Layer #1
        # Computes 32 features using a 5x5 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 128, 128, 1]
        # Output Tensor Shape: [batch_size, 128, 128, 32]
        conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
        print "Conv1 is completed"
        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 128, 128, 32]
        # Output Tensor Shape: [batch_size, 64, 64, 32]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 64, 64, 32]
        # Output Tensor Shape: [batch_size, 64, 64, 64]
        print "Conv2 is in progress"
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 64, 64, 64]
        # Output Tensor Shape: [batch_size, 32, 32, 64]
        print "Conv2 is completed"
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 32, 32, 64]
        # Output Tensor Shape: [batch_size, 32 * 32 * 64]
        print "pool2 is in processing"
        pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])
        print "pool2 is completed"
        # Dense Layer
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 32 * 32 * 64]
        # Output Tensor Shape: [batch_size, 1024]
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout(
                inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        # Logits layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, 6]
        logits = tf.layers.dense(inputs=dropout, units=7)
        
        predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    # calling train_set to prepare training and test set
    train_set()
    #train_data_set & lables_set as numpy array
    train_data_set = np.asarray(train_data,dtype=np.float32)
    train_data_set = np.reshape(train_data_set,(len(train_data),height,width,3))
    train_labels_set = np.asarray(train_labels,dtype=np.int32)
    
    test_set()
    # test_data_set & labels as numpy array
    test_data_set = np.asarray(test_data,dtype=np.float32)
    train_data_set = np.reshape(train_data_set,(len(train_data),height,width,3))
    test_labels_set = np.asarray(test_labels,dtype=np.int32) 
       
    print train_data_set.shape,train_labels_set.shape
    print test_data_set.shape,test_labels_set
    
    print "stage_1 completed"
    #cnn classifier starts now    
    image_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_ic, model_dir="model/image_classification_model")
    
    
    print "stage_2 completed"
    
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data_set},
            y=train_labels_set,
            batch_size=20,
            num_epochs=None,
            shuffle=True)
    
    print "Stage 3 complete"
    
    image_classifier.train(
            input_fn=train_input_fn,
            steps=1000,
            hooks=[logging_hook])
    
    # Evaluate the model and print results
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_data_set},
            y=test_labels_set,
            num_epochs=1,
            shuffle=False)
    
    test_results = image_classifier.evaluate(input_fn=test_input_fn)
    print(test_results)
    
if __name__ == '__main__':
     main()        
    













   
#folder_list= os.listdir('All_61326/test_61326/train_61326/')
#folder_list = glob.glob('All_61326/test_61326/train_61326/*/')