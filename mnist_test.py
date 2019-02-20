#encoding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
TEST_INTERVAL_SECS = 5 #规定循环五秒间隔时间

def test(mnist):
    with tf.Graph().as_default() as g:
        # 利用tf.Graph()复现之前定义的计算图，利用placeholder给训练数据x和标签y_占位
        # 调用mnist_forward文件中的前向传播过程forward()函数，计算训练数据集上的预测结果y
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        # 实例化具有滑动平均的saver对象，从而在会话被加载时模型中的所有参数被赋值为各自的滑动平均值，
        # 增强模型稳定性，然后计算模型在测试集上的准确率
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)#滑动平均
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        #准确率计算方法
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #加载指定路径下的ckpt，而不必加载参数，若模型存在，则加载出模型到当前会话，若不存在，打出不存在的提示
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]# [-1]的含义：索引为-1的元素
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})
                    print("After %s training test accuracy = %g"%(global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)
def main():
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()





