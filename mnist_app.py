from Tools.scripts.treesync import raw_input
import tensorflow as tf
import mnist_forward
import mnist_backward
from PIL import Image
import numpy as np

def application():
    testPicArr = pre_pic('./pic/9.jpg')# 对手写图片做预处理
    preValue = restore_model(testPicArr)# 将符合神经网络输入要求的图片喂给复现的神经网络模型，输出预测值
    print("the prediction num is :", preValue)

def restore_model(testPicArr):
    #创建一个默认图在该图中执行以下操作
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)#得到概率最大的预测值

        # 实现滑动平均模型，参数MIOVING_AVERAGE_DECAY用于控制模型更新的速度，训练过程中会对每一个变量
        # 维护一个影子变量。这个影子变量的初始值
        # 就是相应变量的初始值，没次变量更新时，影子变量就会随之更新
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_store = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_store)

        with tf.Session() as sess:
            # 通过checkPoint文件定位到最新保存的模型
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1

#预处理函数,包括resize，转变灰度图，二值化操作
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28,28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50     #设定合理阈值
    # 二值化处理
    for i in range(28):
        for j in range(28):
            #im_arr[i][j] = 255 - im_arr[i][j]
            if(im_arr[i][j] < threshold ):
                im_arr[i][j] = 0
            else: im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready

application()