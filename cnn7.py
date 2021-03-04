# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 23:36:47 2020

@author: 30790
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:35:48 2020

@author: 30790
"""
import os
import cv2
import tensorflow as tf
import tensorflow.contrib as contrib 
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import random
#from fashionmnist import Network
#tmp4 三层网络 第一层输入为28*28的图片，通道数为1，卷积核为5*5，通道数16
#第二层输入为12*12 通道数为16 卷积核为3*3 通道数为32
#第三层输入为5*5 通道数为64

IMAGE_SIZE=64
NUM_CHANNELS=1
CONV1_1SIZE=3
CONV1_1KERNEL_NUM=32
CONV1_2SIZE=3
CONV1_2KERNEL_NUM=32

CONV2_1SIZE=3
CONV2_1KERNEL_NUM=64
CONV2_2SIZE=3
CONV2_2KERNEL_NUM=64

CONV3_1SIZE=3
CONV3_1KERNEL_NUM=128
CONV3_2SIZE=3
CONV3_2KERNEL_NUM=128
CONV3_3SIZE=2
CONV3_3KERNEL_NUM=128

FC1_SIZE=512
FC2_SIZE=256
OUTPUT_NODE=10 #10类样本
IMAGE_SIZE_flat=IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS
REGULARIZER=0.001
LEARNING_RATE_BASE=1e-4
BATCH_SIZE=100
BATCH_NUM =20
root="E:/task1_Chinese_character_recognition/DATA/HWDB1"

class DataLoader():
    def __init__(self, txt_path, num_class):
        images = []
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('/')[-2]) >= num_class:  # just get images of the first #num_class
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('/')[-2]))
        self.images = images
        self.labels = labels

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, len(self.images), batch_size)
        data_batch = []
        label_batch = []
        for i in range(batch_size):
            # image = Image.open(self.images[index[i]]).convert('RGB')
            image = cv2.imread(self.images[index[i]])
            image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
            image = cv2.resize(src=image, dsize=(IMAGE_SIZE, IMAGE_SIZE))
            image = np.expand_dims(image, axis=-1)
            image = image/255
            # or
            # image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
            # image = tf.image.resize(image, (args.image_size, args.image_size))
            # image = tf.image.rgb_to_grayscale(image)
            data_batch.append(image)

            label = self.labels[index[i]]
            label_batch.append(label)
        return np.array(data_batch), np.array(label_batch)

    def get_test_batch(self, start, end):
        # 从数据集中随机取出batch_size个元素并返回
        data_batch = []
        label_batch = []
        for i in range(start, end):
            # image = Image.open(self.images[index[i]]).convert('RGB')
            image = cv2.imread(self.images[i])
            image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
            image = cv2.resize(src=image, dsize=(IMAGE_SIZE, IMAGE_SIZE))
            image = np.expand_dims(image, axis=-1)
            # or
            # image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
            # image = tf.image.resize(image, (args.image_size, args.image_size))
            # image = tf.image.rgb_to_grayscale(image)
            data_batch.append(image)

            label = self.labels[i]
            label_batch.append(label)
        return np.array(data_batch), np.array(label_batch)

def random_int_list(start, stop, length):    #生成随机数列
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_number=random.randint(start, stop)
        while(random_number in random_list):
            random_number=random.randint(start, stop)
        random_list.append(random_number)   
    return random_list
#生成权重 输入
def get_weigth(shape,regularizer):       
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w)) #对权重进行L2正则化，防止过拟合
    return w

#添加偏置项
def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b

#卷积运算
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME') #步长为1

#最大值池化
def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#前向网络
def forward(x,train,regularizer):
    #第一层
    conv1_1w = get_weigth([CONV1_1SIZE,CONV1_1SIZE,NUM_CHANNELS,CONV1_1KERNEL_NUM],regularizer)
    conv1_1b = get_bias([CONV1_1KERNEL_NUM])
    conv1_1 = conv2d(x,conv1_1w)
    relu1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1,conv1_1b))
    conv1_2w = get_weigth([CONV1_2SIZE,CONV1_2SIZE,CONV1_1KERNEL_NUM,CONV1_2KERNEL_NUM],regularizer)
    conv1_2b = get_bias([CONV1_2KERNEL_NUM])
    conv1_2 = conv2d(relu1_1,conv1_2w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1_2,conv1_2b))
    pool1 = max_pool_2(relu1)
    #第二层
    conv2_1w = get_weigth([CONV2_1SIZE,CONV2_1SIZE,CONV1_2KERNEL_NUM,CONV2_1KERNEL_NUM],regularizer)
    conv2_1b = get_bias([CONV2_1KERNEL_NUM])
    conv2_1 = conv2d(pool1,conv2_1w)
    relu2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1,conv2_1b))
    conv2_2w = get_weigth([CONV2_2SIZE,CONV2_2SIZE,CONV2_1KERNEL_NUM,CONV2_2KERNEL_NUM],regularizer)
    conv2_2b = get_bias([CONV2_1KERNEL_NUM])
    conv2_2 = conv2d(relu2_1,conv2_2w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2_2,conv2_2b))
    pool2 = max_pool_2(relu2)
    #第三层
    conv3_1w = get_weigth([CONV3_1SIZE,CONV3_1SIZE,CONV2_2KERNEL_NUM,CONV3_1KERNEL_NUM],regularizer)
    conv3_1b = get_bias([CONV3_1KERNEL_NUM])
    conv3_1 = conv2d(pool2,conv3_1w)
    relu3_1 = tf.nn.relu(tf.nn.bias_add(conv3_1,conv3_1b))
    
    conv3_2w = get_weigth([CONV3_2SIZE,CONV3_2SIZE,CONV3_1KERNEL_NUM,CONV3_2KERNEL_NUM],regularizer)
    conv3_2b = get_bias([CONV3_2KERNEL_NUM])
    conv3_2 = conv2d(relu3_1,conv3_2w)
    relu3_2 = tf.nn.relu(tf.nn.bias_add(conv3_2,conv3_2b))
    
    conv3_3w = get_weigth([CONV3_3SIZE,CONV3_3SIZE,CONV3_2KERNEL_NUM,CONV3_3KERNEL_NUM],regularizer)
    conv3_3b = get_bias([CONV3_3KERNEL_NUM])
    conv3_3 = conv2d(relu3_2,conv3_3w)
    relu3_3 = tf.nn.relu(tf.nn.bias_add(conv3_3,conv3_3b))
    pool3 = max_pool_2(relu3_3)
    
    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    nodes_reshaped=tf.layers.flatten(pool3)

    #全连接层1
    fc1_w = get_weigth([nodes,FC1_SIZE],regularizer)
    fc1_b = get_bias([FC1_SIZE])
    fc1 = tf.nn.relu(tf.matmul(nodes_reshaped,fc1_w)+fc1_b)
    if train:
        fc1 = tf.nn.dropout(fc1,0.5) #dropout
    
    #全连接层2
    fc2_w = get_weigth([FC1_SIZE,FC2_SIZE],regularizer)
    fc2_b = get_bias([FC2_SIZE])
    fc2 = tf.nn.relu(tf.matmul(fc1,fc2_w)+fc2_b)
    if train:
        fc2 = tf.nn.dropout(fc2,0.5)
    
    #输出层
    final_w = get_weigth([FC2_SIZE,OUTPUT_NODE],regularizer)
    final_b = get_bias([OUTPUT_NODE])
    y=tf.matmul(fc2,final_w)+final_b
    return y

def one_hot(w, h, arr):
    z = np.zeros([w, h])

    for i in range(w):  #
        j = int(arr[i])  # 拿到数组里面的数字
        # print(j)
        z[i][j] = 1
    return z

def trans(image_arr,size):
    num=np.shape(image_arr)[0]
    image_trans=np.zeros((num,size))
    for i in range(num):
        image_trans[i,:]=np.resize(image_arr[i,:,:],(1,size))
    return image_trans
                    
#重置，建立变量
tf.reset_default_graph()
x = tf.placeholder(tf.float32,shape=[None,IMAGE_SIZE_flat],name='x')
x_image = tf.reshape(x,[-1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])
y_true = tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name='y_true')
y_true_cls = tf.argmax(y_true,dimension=1)
#建立网络
y=forward(x_image,True,REGULARIZER)
y_pred = tf.nn.softmax(y)
y_pred_cls = tf.argmax(y_pred,dimension=1)
#计算交叉熵
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=y,
    labels=y_true)
loss = tf.reduce_mean(cross_entropy)
#adam优化器 训练
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_BASE).minimize(loss)
correct_prediction = tf.equal(y_pred_cls,y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #
new_saver = tf.train.Saver()
#meta_graph=tf.train.import_meta_graph(".tmp/model.ckpt.meta")


#开启session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#    classes_txt(root + '/train', root + '/train1.txt', num_class=OUTPUT_NODE)
#    classes_txt(root + '/test', root + '/test1.txt', num_class=OUTPUT_NODE)
    train_path = 'E:/task1_Chinese_character_recognition/DATA/HWDB1/train1.txt'
    test_path = 'E:/task1_Chinese_character_recognition/DATA/HWDB1/test1.txt'
    train_loader = DataLoader(train_path, OUTPUT_NODE)
    test_loader = DataLoader(test_path, OUTPUT_NODE)
#    new_saver.restore(sess,"./tmp5/model2.ckpt")
#    test_list=random_int_list(0,10000,2000) #随机抽取测试样本
    for epoch in range(50):
        for i in range(BATCH_NUM):
            batch_x, batch_y = train_loader.get_batch(BATCH_SIZE)    
            batch_y_onehot = one_hot(len(batch_y.T),OUTPUT_NODE,batch_y.T)
            batch_x_trans = trans(batch_x,IMAGE_SIZE*IMAGE_SIZE)
            sess.run(optimizer, feed_dict={x: batch_x_trans, y_true:batch_y_onehot}) 
            cost=sess.run(loss,feed_dict={x: batch_x_trans, y_true:batch_y_onehot})  
        X_test,y_test=test_loader.get_batch(500)
        X_test_trans = trans(X_test,IMAGE_SIZE*IMAGE_SIZE)
        y_test_onehot=one_hot(len(y_test.T),OUTPUT_NODE,y_test.T)
        acc= sess.run(accuracy,feed_dict={x:X_test_trans, y_true:y_test_onehot})
        print('Epoch: '+str(epoch)+',acc:'+str(acc)+',loss: '+str(cost))   
    new_saver.save(sess,"./tmp/model2.ckpt")#已达到90
    