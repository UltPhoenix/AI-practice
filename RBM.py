# -*- coding: utf-8 -*-

"""受限玻尔兹曼机"""
import os
import timeit
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import tile_raster_images
import input_data

class RBM(object):
    """受限玻尔兹曼机类"""
    def __init__(self, inpt=None, n_visiable=784, n_hidden=500, W=None,
                 hbias=None, vbias=None):
        """
        --------------参数---------------

        : inpt: Tensor, 输入的tensor [None, n_visiable];
        : n_visiable: int, 可见层单元的数目;
        : n_hidden:   int, 隐含层单元的数目;
        : W, hbias, vbias: Tensor, RBM的参数 (tf.Variable);
        """
        # 设置参数;
        self.n_visiable = n_visiable
        self.n_hidden = n_hidden
        # 初始化输入input(如未给出);
        if inpt is None:
            inpt = tf.placeholder(dtype=tf.float32, shape=[None, self.n_visiable])
        self.input = inpt
        # 初始化未给出的参数;
        if W is None:
            # 设置随机分布的边界;
            bounds = -4.0 * np.sqrt(6.0 / (self.n_visiable + self.n_hidden))
            # 服从正态分布的随机变量;
            W = tf.Variable(tf.random_uniform([self.n_visiable, self.n_hidden], minval=-bounds,
                                              maxval=bounds), dtype=tf.float32)
        # 隐含层偏置;
        if hbias is None:
            hbias = tf.Variable(tf.zeros([self.n_hidden,]), dtype=tf.float32)
        # 可见层偏置;
        if vbias is None:
            vbias = tf.Variable(tf.zeros([self.n_visiable,]), dtype=tf.float32)
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        # 这一步是保证训练DBN时参数及时更新;
        self.params = [self.W, self.hbias, self.vbias]

    def propup(self, v):
        """计算从可见层到隐含层的Sigmoid()函数的激活值"""
        return tf.nn.sigmoid(tf.matmul(v, self.W) + self.hbias)

    def propdown(self, h):
        """计算从隐含层到可见层的Sigmoid()函数的激活值"""
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.vbias)

    def sample_prob(self, prob):
        """根据给出的概率进行采样 (you can use binomial in Theano)"""
        return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))

    def sample_h_given_v(self, v0_sample):
        """由可见层采样隐含层"""
        h1_mean = self.propup(v0_sample)
        h1_sample = self.sample_prob(h1_mean)
        return (h1_mean, h1_sample)

    def sample_v_given_h(self, h0_sample):
        """由隐含层采样可见层"""
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.sample_prob(v1_mean)
        return (v1_mean, v1_sample)

    def gibbs_vhv(self, v0_sample):
        """由可见状态对隐含层、可见层做一轮Gibbs采样"""
        """Implement one step of Gibbs sampling from the visiable state"""
        h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return (h1_mean, h1_sample, v1_mean, v1_sample)

    def gibbs_hvh(self, h0_sample):
        """由隐含状态对隐含层、可见层做一轮Gibbs采样"""
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return (v1_mean, v1_sample, h1_mean, h1_sample)

    def free_energy(self, v_sample):
        """计算RBM的自由能"""
        wx_b = tf.matmul(v_sample, self.W) + self.hbias
        vbias_term = tf.matmul(v_sample, tf.expand_dims(self.vbias, axis=1))
        hidden_term = tf.reduce_sum(tf.log(1.0 + tf.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def get_train_ops(self, learning_rate=0.1, k=1, persistent=None):
        """
        -----由CD-k算法生成训练操作-----

        : learning_rate: float;
        : k: int, Gibbs采样的步数 (注意 k=1 被证明很有效);
        : persistent: Tensor, PCD-k (TO DO:);
        """
        # 计算positive phase(正相);
        ph_mean, ph_sample = self.sample_h_given_v(self.input)
        # 采样链的旧状态;本次chain_start存在persistent变量;
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # 使用tf.while_loop进行CD-k;
        # lambda表达式的循环体(尚未执行);
        # i<k时计算nv_mean, nv_sample, nh_mean, nh_sample;
        cond = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: i < k
        # i+1并对nv_mean, nv_sample, nh_mean, nh_sample做一轮Gibbs采样;
        body = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: (i+1, ) + self.gibbs_hvh(nh_sample)
        # tf.while_loop()执行CD-k采样;即loop参数先传入cond判断条件是否成立,成立之后,把loop参数传入body执行操作,然后返回操作后的loop参数;
        i, nv_mean, nv_sample, nh_mean, nh_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0), tf.zeros(tf.shape(self.input)), 
                                                            tf.zeros(tf.shape(self.input)), tf.zeros(tf.shape(chain_start)), chain_start])

        """
        # 计算更新每个参数;
        update_W = self.W + learning_rate * (tf.matmul(tf.transpose(self.input), ph_mean) - 
                                tf.matmul(tf.transpose(nv_sample), nh_mean)) / tf.to_float(tf.shape(self.input)[0])  # use probability
        update_vbias = self.vbias + learning_rate * (tf.reduce_mean(self.input - nv_sample, axis=0))   # use binary value
        update_hbias = self.hbias + learning_rate * (tf.reduce_mean(ph_mean - nh_mean, axis=0))       # use probability
        # 为参数分配新的值;
        new_W = tf.assign(self.W, update_W)
        new_vbias = tf.assign(self.vbias, update_vbias)
        new_hbias = tf.assign(self.hbias, update_hbias)
        """

        # 结束计算梯度;
        chain_end = tf.stop_gradient(nv_sample)   
        # 计算代价cost;
        cost = tf.reduce_mean(self.free_energy(self.input)) - tf.reduce_mean(self.free_energy(chain_end))
        # 计算梯度d(ys)/d(xs);
        gparams = tf.gradients(ys=[cost], xs=self.params)
        # 参数更新;tf.assign(A, new_number):这个函数的功能主要是把A的值变为new_number;
        new_params = []
        for gparam, param in zip(gparams, self.params):
            new_params.append(tf.assign(param, param - gparam*learning_rate))

        # 更新persistent;persistent存储的是最新的nh_sample(hidden layer采样值);
        if persistent is not None:
            new_persistent = [tf.assign(persistent, nh_sample)]
        else:
            new_persistent = []
        # 用于训练;
        return new_params + new_persistent  

    def get_reconstruction_cost(self):
        """计算原始输入和计算v -> h -> v后之间的cross-entropy"""
        # v -> h;
        activation_h = self.propup(self.input)
        # h -> v;
        activation_v = self.propdown(activation_h)
        # 这一步是为了避免Nan(使下界clip_value_min=1e-30);
        # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
        activation_v_clip = tf.clip_by_value(activation_v, clip_value_min=1e-30, clip_value_max=1.0)
        # 计算1.0 - activation_v;
        reduce_activation_v_clip = tf.clip_by_value(1.0 - activation_v, clip_value_min=1e-30, clip_value_max=1.0)
        # 计算cross_entropy;
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.input*(tf.log(activation_v_clip)) + 
                                    (1.0 - self.input)*(tf.log(reduce_activation_v_clip)), axis=1))
        return cross_entropy   

    def reconstruct(self, v):
        """对RBM做一次计算v -> h -> v"""
        h = self.propup(v)
        return self.propdown(h) 

if __name__ == "__main__":
    # 读取mnist数据集;
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # 定义输入;
    x = tf.placeholder(tf.float32, shape=[None, 784])
    # 设置random_seed(用于随机数);
    tf.set_random_seed(seed=99999)
    np.random.seed(123)
    # rbm模型;
    n_visiable, n_hidden = 784, 500
    rbm = RBM(x, n_visiable=n_visiable, n_hidden=n_hidden)

    # 学习参数
    learning_rate = 0.1
    batch_size = 20
    # 计算cost
    cost = rbm.get_reconstruction_cost()
    # 定义persistent变量;
    # persistent_chain ∈ [batch_size, n_hidden];
    persistent_chain = tf.Variable(tf.zeros([batch_size, n_hidden]), dtype=tf.float32)
    # 获取cd-15的训练方式;
    train_ops = rbm.get_train_ops(learning_rate=learning_rate, k=15, persistent=persistent_chain)
    # 初始化变量;
    init = tf.global_variables_initializer()

    # 结果输出文件夹;
    output_folder = "rbm_plots"
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # 训练参数;
    training_epochs = 15
    display_step = 1
    print("Start training...")

    # 开始跑;
    with tf.Session() as sess:
        # 记录开始时间;
        start_time = timeit.default_timer()
        # 执行变量初始化;
        sess.run(init)
        # 执行训练;
        for epoch in range(training_epochs):
            # 设置参数;
            avg_cost = 0.0
            batch_num = int(mnist.train.num_examples / batch_size)
            # 开始batch;
            for i in range(batch_num):
                # 获取下一个batch的训练数据;
                x_batch, _ = mnist.train.next_batch(batch_size)
                # 执行训练;
                sess.run(train_ops, feed_dict={x: x_batch})
                # 计算cost;
                avg_cost += sess.run(cost, feed_dict={x: x_batch,}) / batch_num
            # 输出;
            if epoch % display_step == 0:
                print("Epoch {0} cost: {1}".format(epoch, avg_cost))
            # 用图像存储该批次训练结果;
            image = Image.fromarray(
            tile_raster_images(
                X=sess.run(tf.transpose(rbm.W)),
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)))
            image.save("new_filters_at_epoch_{0}.png".format(epoch))

        # 记录训练结束时间;
        end_time = timeit.default_timer()
        # 计算执行时间;
        training_time = end_time - start_time
        print("Finished!")
        print("  The training ran for {0} minutes.".format(training_time/60,))
        # 通过RBM采样来生成图像;
        print("...Sampling from the RBM")
        # 采样参数;
        n_chains = 20
        n_samples = 10
        number_test_examples = mnist.test.num_examples
        # 随机选择 'n_chains' examples
        test_indexs = np.random.randint(number_test_examples - n_chains)
        test_samples = mnist.test.images[test_indexs:test_indexs+n_chains]
        # 构造persistent变量存储可见层状态;
        persistent_v_chain = tf.Variable(tf.to_float(test_samples), dtype=tf.float32)
        # Gibbs采样步数;
        step_every = 1000
        # 实现Gibbs采样;(用tf.while_loop);
        cond = lambda i, h_mean, h_sample, v_mean, v_sample: i < step_every
        body = lambda i, h_mean, h_sample, v_mean, v_sample: (i+1, ) + rbm.gibbs_vhv(v_sample)
        i, h_mean, h_sample, v_mean, v_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0), tf.zeros([n_chains, n_hidden]), 
                                                            tf.zeros([n_chains, n_hidden]), tf.zeros(tf.shape(persistent_v_chain)), persistent_v_chain])
        # 更新persistent_v_chain;
        new_persistent_v_chain = tf.assign(persistent_v_chain, v_sample)
        # 存储采样得到的image;
        # 先构造包容所有图像的空图;
        image_data = np.zeros((29*(n_samples+1)+1, 29*(n_chains)-1),
                          dtype="uint8")
        # 添加原有的图像;(来自test_samples);(每一行第一个图是"真图");
        image_data[0:28,:] = tile_raster_images(X=test_samples,
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
        # 初始化变量;
        sess.run(tf.variables_initializer(var_list=[persistent_v_chain]))
        # 执行采样并存储到image_data对应位置;
        for idx in range(1, n_samples+1):
            sample = sess.run(v_mean)
            sess.run(new_persistent_v_chain)
            print("...plotting sample", idx)
            image_data[idx*29:idx*29+28,:] = tile_raster_images(X=sample,
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
        # 转化为图像格式并保存为png;
        image = Image.fromarray(image_data)
        image.save("new_original_and_{0}samples.png".format(n_samples))