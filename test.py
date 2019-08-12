import os
import shutil
import numpy as np

import tensorflow as tf

from src.character_level_cnn import Char_level_cnn
from src.utils import get_num_classes, create_dataset

tf.flags.DEFINE_string("alphabet", """abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""",
                       "Valid characters used for model")
tf.flags.DEFINE_string("test_set", "data/test.csv", "Path to the test set")
tf.flags.DEFINE_integer("max_length", 1014, "Maximum length of input")
tf.flags.DEFINE_integer("batch_size", 128, "Minibatch size")
tf.flags.DEFINE_string("saved_path", "trained_models", "path to store trained model")


FLAGS = tf.flags.FLAGS


def test():
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(FLAGS.saved_path,'char_level_cnn.meta'))
        g = tf.get_default_graph()
	        
        # input	
       	handle = g.get_tensor_by_name('Placeholder:0') 
       	keep_prob = g.get_tensor_by_name('dropout_prob:0') 
        
        test_set, num_test_iters = create_dataset(FLAGS.test_set, FLAGS.alphabet, FLAGS.max_length, FLAGS.batch_size, False)
        test_iterator = test_set.make_initializable_iterator()
        test_handle = sess.run(test_iterator.string_handle())
        
        iterator = tf.data.Iterator.from_string_handle(test_handle, test_set.output_types, test_set.output_shapes)
        texts, labels = iterator.get_next()
        
        num_classes = get_num_classes(FLAGS.test_set)        
	
	
        output = g.get_tensor_by_name('fc3/dense:0')
        acc, acc_op = tf.metrics.accuracy(labels=tf.cast(labels, tf.int64), predictions=tf.argmax(output, 1))
        
	#accuracy = g.get_tensor_by_name('accuracy:0')
        sess.run(test_iterator.initializer)
        sess.run(tf.local_variables_initializer())
        #sess.run(tf.global_variables_initializer())
        
        saver.restore(sess, os.path.join(FLAGS.saved_path, 'char_level_cnn'))
	# run graph output fc3/dense
        counter = 0
        while True:
            counter += 1
            try:
                print('calculating accuracy for batch {}/{}...'.format( counter, int(60000/FLAGS.batch_size)))
                out = sess.run([acc_op], feed_dict={handle: test_handle, keep_prob: 1.0})
                print('accuracy for batch {:.2f}', out)	
            except (tf.errors.OutOfRangeError, StopIteration):
                break

        print('Accuracy of model: {:.2f}%'.format(sess.run(acc) * 100))




        exit()






        

if __name__ == "__main__":
    test()
