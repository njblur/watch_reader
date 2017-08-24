import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import IPython

def create_watch(hand_minute,hand_hour,base,h,m,scale=1.0):
    shape = hand_minute.shape
    width = shape[1]
    height = shape[0]
    center_x = width/2
    center_y = width/2
    angle = ((60-(m-15))%60)*360/60
    matrix = cv2.getRotationMatrix2D((center_x,center_y),angle,1.0)
    minute_rotate = cv2.warpAffine(hand_minute,matrix,(width,height))

    shape = hand_hour.shape
    width = shape[1]
    height = shape[0]
    center_x = width/2
    center_y = width/2
    angle = ((12-(h*1.0-3))%12)*360/12-m/60.0*30
    matrix = cv2.getRotationMatrix2D((center_x,center_y),angle,1.0)
    hour_rotate = cv2.warpAffine(hand_hour,matrix,(width,height))

    base = cv2.add(base,hour_rotate)
    base = cv2.add(base,minute_rotate)

    return base
def generate_data(hand_minute,hand_hour,base,data_size,scale=1.0):
    minutes = np.random.randint(0,60,size=data_size)
    hours = np.random.randint(0,12,size=data_size)
    train_data = [create_watch(hand_minute,hand_hour,base,hour,minute) for minute,hour in zip(minutes,hours)]
    target_data = [[minute*1.0/60,hour*1.0/12+minute*1.0/60/12] for minute,hour in zip(minutes,hours)]
    return np.array(train_data),np.array(target_data)
def main(train):
    if train:
        batch_size = 50
    else:
        batch_size = 10
    hand_h = cv2.imread('watch/hour.png')
    hand_m = cv2.imread('watch/minute.png')
    base = cv2.imread('watch/base.png')
    hand_h = cv2.bitwise_not(hand_h)
    hand_m = cv2.bitwise_not(hand_m)
    base = cv2.bitwise_not(base)
    input = tf.placeholder(shape=[batch_size,224,224,3],dtype=tf.float32)
    target = tf.placeholder(dtype=tf.float32,shape=[batch_size,2])
    filter1_weights = tf.Variable(tf.truncated_normal(shape=[5,5,3,32],stddev=0.01))
    filter1_bias = tf.Variable(tf.zeros(shape=[32]))
    filter2_weights = tf.Variable(tf.truncated_normal(shape=[3,3,32,64],stddev=0.01))
    filter2_bias = tf.Variable(tf.zeros(shape=[64]))

    conv = tf.nn.conv2d(input,filter1_weights,strides=[1,2,2,1],padding="SAME")
    conv = tf.nn.bias_add(conv,filter1_bias)
    conv = tf.nn.relu(conv)
    conv = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    conv = tf.nn.conv2d(conv,filter2_weights,strides=[1,2,2,1],padding="SAME")
    conv = tf.nn.bias_add(conv,filter2_bias)
    conv = tf.nn.relu(conv)
    conv = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    shape = conv.get_shape().as_list()

    batch = shape[0]
    size = shape[1]*shape[2]*shape[3]

    flat = tf.reshape(conv,[batch,size])

    fc1_weights = tf.Variable(tf.truncated_normal(shape=[size,128],stddev=0.01))
    fc1_bias = tf.Variable(tf.zeros(dtype=tf.float32,shape=[128]))

    fc1 = tf.matmul(flat,fc1_weights) + fc1_bias

    fc2_weights = tf.Variable(tf.truncated_normal(shape=[128,2],stddev=0.01))
    fc2_bias = tf.Variable(tf.zeros(dtype=tf.float32,shape=[2]))

    fc2 = tf.matmul(fc1,fc2_weights) + fc2_bias

    loss = tf.nn.l2_loss((fc2-target))/batch_size

    trainer = tf.train.GradientDescentOptimizer(0.01)

    step = trainer.minimize(loss)

    train = False

    epoch = 200
   
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        if(train):
            train_data_g, target_data_g = generate_data(hand_m,hand_h,base,batch_size*100)
            for i in range(epoch):
                for j in range(100):
                    train_data = train_data_g[j*batch_size:j*batch_size+batch_size]
                    target_data = target_data_g[j*batch_size:j*batch_size+batch_size]
                    [l,s] = sess.run([loss,step],feed_dict={input:train_data,target:target_data})
                    print "loss is " + str(l)
            saver.save(sess,'model')
        else:
            saver.restore(sess,'model')
            v,l = generate_data(hand_m,hand_h,base,batch_size)
            [o] = sess.run([fc2],feed_dict={input:v})
            for d,p in zip(v,o):
                plot_time(d,p)
            IPython.embed()
def plot_time(image,pred):
    minute = pred[0]
    hour = pred[1]
    minute = int(minute*60+0.5) % 60
    hour = int(hour*12)
    if hour == 0:
        hour = 12
    ax = plt.subplot(111)
    ax.set_title("predicted time "+str(hour)+" : "+str(minute))
    plt.imshow(image)
    plt.show()
if __name__ == '__main__':
    if(not os.path.exists('model.meta')):
        main(True)
    else:
        main(False)
 

    # IPython.embed()

