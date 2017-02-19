import os
import tensorflow as tf

checkpoint_dir = os.path.join(os.path.dirname(__file__), 'ckpt')
checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')

a = tf.Variable([0, 1, 2, 3, 4, 5], tf.float32, name='a')
b = tf.Variable([0, 1, 2, 3, 4, 5], tf.float32, name='b')
c = tf.add(a, b)
update_op = tf.assign(a, c)

saver = tf.train.Saver()

with tf.Session() as sess:
    if tf.train.get_checkpoint_state(checkpoint_dir):
        saver.restore(sess, checkpoint_path)
    else:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    for step in range(100):
        print('a: ', sess.run(a))
        print('update_op: ', sess.run(update_op))

        if step % 9 == 0:
            print('step: ', step)
            saver.save(sess, checkpoint_path, global_step=step)
