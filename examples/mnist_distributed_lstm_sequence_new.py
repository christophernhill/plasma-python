from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from tflearn.layers.core import fully_connected

from mpi4py import MPI
comm = MPI.COMM_WORLD
task_index = comm.Get_rank()
num_tasks = comm.Get_size()
NUM_GPUS = 4
MY_GPU = task_index % NUM_GPUS
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(MY_GPU)

from plasma.utils.mpi_launch_tensorflow import get_worker_host_list,get_ps_host_list,get_host_list,get_my_host_id


flags = tf.app.flags
flags.DEFINE_string("data_dir", "./mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")

FLAGS = flags.FLAGS
num_epochs = 1000

IMAGE_PIXELS = 28
model_length = 28
n_inputs = 28
n_classes = 10
rnn_size = 100
batch_size = 128
num_epochs = 20000
n_outputs = 28
#FIXME for test
#n_outputs = rnn_size

def tfMakeCluster(num_tasks_per_host,num_tasks,num_ps_hosts):
    worker_hosts = get_worker_host_list(2222,num_tasks_per_host)
    print ("worker_hosts {}".format(worker_hosts))
    ps_hosts = get_ps_host_list(2322,num_ps_hosts)
    print ("ps_hosts {}".format(ps_hosts))

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    return cluster


def main(unused_argv):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  num_hosts = len(get_host_list())
  num_ps_hosts = len(get_host_list())
  ps_task_index = get_my_host_id()
  cluster = tfMakeCluster(NUM_GPUS,num_tasks,num_ps_hosts)

  if (task_index+1)%(NUM_GPUS+1) == 0:
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name="ps",
                           task_index=ps_task_index)
    server.join()
  else:
    # Create and start a server for the local task.
    worker_task_index = task_index - ps_task_index
    server = tf.train.Server(cluster,
                           job_name="worker",
                           task_index=worker_task_index)

    is_chief = (task_index == 0)
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % worker_task_index,
        cluster=cluster)):

        global_step = tf.Variable(0, name="global_step", trainable=False)

        # tf Graph input
        img = tf.placeholder(tf.float32, [batch_size, model_length, n_inputs])
        labels = tf.placeholder(tf.float32, [batch_size, model_length, n_outputs]) #rnn_size])

        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([rnn_size, n_outputs]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_outputs]))
        }

        num_layers = 5
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True),output_keep_prob=0.5)
                for _ in range(num_layers)
               ])
        outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm, img, dtype=tf.float32, time_major=False)

        x = tf.unstack(outputs, axis=1)
        #x = [fully_connected(x[i],n_outputs,activation='linear') for i in range(model_length)]
        x = [tf.matmul(x[i], weights['out']) + biases['out'] for i in range(model_length)]
        x = tf.stack(x)
        pred = tf.transpose(x, [1, 0, 2])
        print(pred.get_shape())

        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

        replicas_to_aggregate = num_tasks

        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=replicas_to_aggregate,
            total_num_replicas=num_tasks,
            name="mnist_sync_replicas")

        train_step = opt.minimize(loss, global_step=global_step)

        local_init_op = opt.local_step_init_op
        if is_chief:
            local_init_op = opt.chief_init_op

        ready_for_local_init_op = opt.ready_for_local_init_op

        # Initial token and chief queue runners required by the sync_replicas mode
        chief_queue_runner = opt.get_chief_queue_runner()
        sync_init_op = opt.get_init_tokens_op()

        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()

        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init_op,
            local_init_op=local_init_op,
            ready_for_local_init_op=ready_for_local_init_op,
            recovery_wait_secs=1,
            global_step=global_step)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps", "/job:worker/task:%d" % task_index])

        # The chief worker (task_index==0) session will prepare the session,
        #while the remaining workers will wait for the preparation to complete.
        if is_chief:
            print("Worker %d: Initializing session..." % task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                task_index)

        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        print("Worker %d: Session initialization complete." % task_index)

        if is_chief:
            # Chief worker will start the chief queue runner and call the init op.
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        # Perform training
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)

        local_step = 0
        while True:
            # Training feed
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #FIXME for test
            batch_xs = batch_xs.reshape([batch_size, model_length, n_inputs])
            train_feed = {img: batch_xs, labels: batch_xs}

            _, step = sess.run([train_step, global_step], feed_dict=train_feed)
            local_step += 1

            now = time.time()
            print("%f: Worker %d: training step %d done (global step: %d)" %
                 (now, task_index, local_step, step))

            if step >= num_epochs:
                break

        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        ## Validation feed
        #val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        #val_xent = sess.run(loss, feed_dict=val_feed)
        #print("After %d training step(s), validation cross entropy = %g" %
        #      (num_epochs, val_xent))

if __name__ == "__main__":
    tf.app.run()
