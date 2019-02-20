import tensorflow as tf

# 在前向传播过程中，需要定义网络模型输入层个数、隐藏层节点数、输出层个数，
# 定义网络参数w，偏置b，定义由输入到输出的神经网络构架

INPUT_NODE = 784 # 输入层数个数，每张图片的像素个数
OUTPUT_NODE = 10 # 输出层个数，隐藏层节点
LAYER1_NODE = 500 # 隐藏层节点数

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))#随机生成，并去掉偏离2个标准差的点（偏离点）的正态分布

    #参数满足截断正态分布，并使用正则化，将每个参数的正则化损失加到总损失中
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x, regularizer):
    #由输入层到隐藏层的参数w1形状为[784,500],
    #由输入层到隐藏层的偏置b1形状为长度为500的一维数组，初始化值全为0
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    #由隐藏层到输出层的参数w2形状为[500,10]
    #由隐藏层到输出层的偏置b2形状为长度为10的一维数组，初始化值全为0
    #由于输出y要经过softmax函数，使其符合概率分布，故输出y不经过relu函数
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y
