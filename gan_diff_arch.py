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
input_size = 784
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


def generator(z, out_dim, n_units=128, reuse=False,  alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        h1 = tf.layers.dense(z, n_units, activation=None)
        h1 = tf.maximum(h1, alpha * h1)
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.nn.tanh(logits)
    theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
    return out, theta_d


def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_units, activation=None)
        h1 = tf.maximum(h1, alpha * h1)
        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.nn.sigmoid(logits)
    theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
    return out,logits, theta_d


def latent_z(batch=batch_size, z_dim = z_dim):
    return np.random.uniform(-1.0, 1.0, size=[batch, z_dim])


if __name__ == '__main__':
    X = tf.placeholder(tf.float32, shape=[None, img_dim])
    Z = tf.placeholder(tf.float32, shape=[None, z_dim])

    g_z, trainable_g = generator(Z,input_size)

    d_real,d_logits_real, trainable_d = discriminator(X)
    d_fake,d_logits_fake,  _ = discriminator(g_z, reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_fake)))

    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_fake)))

    d_optimizer = tf.train.AdamOptimizer().minimize(d_loss, var_list=trainable_d)
    g_optimizer = tf.train.AdamOptimizer().minimize(g_loss, var_list=trainable_g)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        x_batch, labels = mnist_data.train.next_batch(batch_size)
        x_batch = (x_batch-0.5)*2  #-1 1 koze normalizalas
        batch_images = x_batch.reshape((batch_size, 784))
        z_size=100
        z_batch = np.random.uniform(-1, 1, size=(batch_size, z_size))

        _, __, d_real_, d_fake_,d_logits_real_, d_logits_fake_ = sess.run([d_optimizer, g_optimizer,d_real,d_fake, d_logits_real, d_logits_fake],feed_dict={X: x_batch, Z: z_batch})

        print( d_logits_real_, d_logits_fake_)

        if i % 100 == 0:
            generated_imgs = sess.run(g_z, feed_dict={Z: latent_z(16)})
            plot(generated_imgs, i);