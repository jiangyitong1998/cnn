
# coding: utf-8

# In[2]:


import os
import glob
import pickle
import copy
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd


# In[3]:


size = 192284


# In[5]:


print("Reading data...")
ipt = pd.read_csv("train_input.csv", index_col=0) # path incorrect    
ddi_pairs = list(ipt.index)
X = ipt.values
X = np.delete(X, 0, axis = 1)
X = X.astype(float)
print("Almost there...")
opt = pd.read_csv("train_label.csv", index_col=0)    
ddi_pairs = list(opt.index)
Y = opt.values
Y = np.delete(Y, 0, axis = 1)
Y = Y.astype(int)
print("Done!")


# In[40]:


index = list()
for i in range(size):
    index.append(i)
np.random.shuffle(index)


# In[50]:


print("Spliting data...")
train_size = int(size * 0.6)
# 20% for test
test_size = int(size * 0.8) - train_size
# 20% for validation
validation_size = size - int(size * 0.8)


train_input = X[index[:train_size]]
train_label = Y[index[:train_size]]

test_input = X[index[train_size:train_size + test_size]]
test_label = Y[index[train_size:train_size + test_size]]

validation_start = 153827
#validation_end = 
validation_input = X[index[int(size * 0.8):]]
validation_label = Y[index[int(size * 0.8):]]

validation_size = len(validation_input)

print("Done!")

# In[80]:

def translater(inputs, label, in_size):
        input_temp = inputs
        input_temp = np.asarray(input_temp)

        outputs_raw = label
        outputs_raw = np.asarray(outputs_raw).T

        # change outputs into onehots form
        outputs = np.zeros((in_size,86))
        outputs[np.arange(in_size), outputs_raw] = 1
        return input_temp, outputs

print("Translating data...")
train_inputs, train_outputs = translater(train_input, train_label, train_size)
test_inputs, test_outputs = translater(test_input, test_label, test_size)
validation_inputs, validation_outputs = translater(validation_input, validation_label, validation_size)
print("Done!")

# In[ ]:


def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,weights)+biases   
    outputs = activation_function(Wx_plus_b)
    return outputs

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def next_batch(num, data, label):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    label_shuffle = label[idx]
    
    return np.asarray(data_shuffle), np.asarray(label_shuffle)



featuremap_size = int(32)

xs = tf.placeholder(tf.float32, [None, 100])
ys = tf.placeholder(tf.float32, [None, 86])
keep_prob = tf.placeholder(tf.float32)

x_pca = tf.reshape(xs, [-1, 10, 10, 1])

#first conv1 layer
W_conv1 = weight_variable([5, 5, 1, featuremap_size])
b_conv1 = bias_variable([featuremap_size])
h_conv1 = tf.nn.relu(conv2d(x_pca,W_conv1) + b_conv1) #output size 10*10*size
h_pool1 = max_pool_2x2(h_conv1) #5*5*size

#second convl layer
W_conv2 = weight_variable([5, 5, featuremap_size, featuremap_size * 2])
b_conv2 = bias_variable([featuremap_size * 2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #5*5*size*2
h_pool2 = max_pool_2x2(h_conv2)

shapes = 1 * 3 * 3 * featuremap_size * 2

#fully connecter layer
W_fc1 = weight_variable([shapes, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, shapes])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#Readout layer
W_fc2 = weight_variable([1024, 86])
b_fc2 = bias_variable([86])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = ys, logits = prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    y_pre[y_pre<0.47]=0
    y_pre[y_pre>=0.47]=1
    correct_prediction = tf.equal(y_pre,v_ys)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[ ]:


print("=================================Train start!=================================")
print("=================================Train start!=================================")
print("=================================Train start!=================================")
print("=================================Train start!=================================")

for i in range(10000):
    # training
    train_batch_xs,train_batch_ys = next_batch(50, train_inputs, train_outputs)
    train, train_cross = sess.run([train_step, cross_entropy], feed_dict={xs: train_batch_xs, ys: train_batch_ys, keep_prob: 1})
    #sess.run(train_step, feed_dict={xs: inputs, ys: outputs})
    if i % 100 == 0:
        # to see the step improvement
        accuracy = compute_accuracy(test_inputs, test_outputs)
        test_cross = sess.run(cross_entropy, feed_dict={xs: test_inputs, ys: test_outputs, keep_prob: 1})
        print(i/100, "% finished.", "Accurancy:", accuracy, "TrainCross:", train_cross, "TestCross", test_cross)

print("=================================Finished!=================================")
print("=================================Finished!=================================")
print("=================================Finished!=================================")
print("=================================Finished!=================================")

