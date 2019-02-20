import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

# 反向传播过程实现利用训练数据集对神经网络模型训练，通过降低损失函数值，实现网络模型参数的优化
# 从而得到准确率高且泛化能力强的神经网络模型

BATCH_SIZE = 200           # 每轮喂入神经网络的图片数
LEARNING_RATE_BASE = 0.1    # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
REGULARIZER = 0.0001        # 正则化系数
STEPTS = 50000              # 训练轮数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率
MODEL_SAVE_PATH = "./model/"# 模型保存路径
MODEL_NAME = "mnist_model"  # 模型保存名称

def backward(mnist):
    # 用placeholder给训练数据x和标签_y占位，调用mnist_forward文件中的前向传播过程forward()函数，
    # 并设置正则化，计算训练数据集上的预测结果y，并给当前计算轮数计数器赋值，设定为不可训练模型
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='xInput')#一次喂多组数据，784个特征
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_')#已知答案
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    # 调用包含所有参数正则化损失的损失函数loss，并设定指数衰减学习率learning_rate
    # 然后使用梯度衰减算法对模型优化，减低损失函数，并定义参数的滑动平均
    # 表征两个概率分布之间的距离
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))# ce=Cross Entropy交叉熵-一种减少loss的方法
    cem = tf.reduce_mean(ce)# 交叉熵
    loss = cem + tf.add_n(tf.get_collection('losses'))# 总损失函数=交叉熵+sum(losses)

    # 指数衰减学习率。动态更新
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,# 学习率初始值
        global_step,# 计数器
        mnist.train.num_examples / BATCH_SIZE,# 多少轮更新一次学习率 = 总样本/BATCH_SIZE
        LEARNING_RATE_DECAY,# 学习率衰减率
        staircase=True)# 学习率梯形衰减

    # 使用梯度衰减算法对模型优化，减低损失函数，并定义参数的滑动平均
    # global step refers to the number of batches seen by the graph
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 滑动平均：记录了一段时间内模型中所有参数w和b各自的平均值。利用滑动平均值可以增强模型泛化能力
    ema_op = ema.apply(tf.trainable_variables())# 所有待优化参数求滑动平均
    # 所有待优化的参数求滑动平均，滑动平均和训练过程同步运行
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    # 在with结构中，实现所有参数初始化，每次喂入batch_size组（200组）训练数据和对应标签，循环迭代steps轮
    # 每隔1000轮打印出一次损失函数值信息，并将当前会话下载到指定路径
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPTS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training steps, loss on training batch is %g"%(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

# 加载指定路径下的训练数据集，并调用规定的backward（）函数训练模型
def main():
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    backward(mnist)

if __name__ == '__main__':
    main()