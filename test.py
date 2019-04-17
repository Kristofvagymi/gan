import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os

print(tf.__version__)
if tf.test.is_built_with_cuda():
    print("The installed version of TensorFlow includes GPU support.")
else:
    print("The installed version of TensorFlow does not include GPU support.")

img_dim = 28*28
batch_size = 64
z_dim = 100
mnist_data = input_data.read_data_sets('MNIST_data')

keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')


def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.1))


def plot(samples, i_ex):
    fig = plt.figure(figsize=(4, 4))
    samples = (samples / 2) + 0.5
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples[:9]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    if not os.path.exists('plots/'):
        os.makedirs('plots/')
    plt.savefig('plots/{}.png'.format(str(int(i_ex / 100)).zfill(3)), bbox_inches='tight')
    plt.close(fig)


def generator(z):
    activation = lrelu
    momentum = 0.8
    with tf.variable_scope("generator", reuse=None):
        x = z
        d1 = 4
        d2 = 1
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[7, 7])
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.tanh)
    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
    return x, theta_g


def discriminator(img_in, reuse=False):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(img_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
    theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
    return x, theta_d


def latent_z(batch=batch_size, z_dim = z_dim):
    return np.random.uniform(-1.0, 1.0, size=[batch, z_dim])


if __name__ == '__main__':
    X = tf.placeholder(tf.float32, shape=[None, img_dim])
    Z = tf.placeholder(tf.float32, shape=[None, z_dim])

    g_z, trainable_g = generator(Z)

    d_real, trainable_d = discriminator(X)
    d_fake, _ = discriminator(g_z, reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))

    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))

    d_optimizer = tf.train.AdamOptimizer().minimize(d_loss, var_list=trainable_d)
    g_optimizer = tf.train.AdamOptimizer().minimize(g_loss, var_list=trainable_g)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    keep_prob_train = 0.25

    for i in range(100000):
        x_batch, labels= mnist_data.train.next_batch(batch_size)
        x_batch = (x_batch-0.5)*2
        z_batch = latent_z()
        _, _ = sess.run([d_optimizer, g_optimizer], feed_dict={X: x_batch, Z: z_batch, keep_prob: keep_prob_train, is_training: True})

        if i % 100 == 0:
            generated_imgs = sess.run(g_z, feed_dict={Z: latent_z(16), keep_prob: keep_prob_train, is_training: True})
            plot(generated_imgs, i)



def generator(z):
    activation = lrelu
    with tf.variable_scope("generator", reuse=None):
        net = tf.keras.layers.Dense(1568, activation=activation)(z)
        net = tf.keras.layers.Reshape([7, 7, 32])(net)
        net = tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same', activation=activation)(net)
        net = tf.keras.layers.Dropout(0.2)(net)
        x = tf.keras.layers.Conv2DTranspose(1, 3, 2, padding='same', activation=tf.nn.tanh)(net)
    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
    return x, theta_g