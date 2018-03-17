from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import NiftiDataset
import os
import datetime

# tensorflow app flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir','./data/evaluate',
    """Directory of evaluation data""")
tf.app.flags.DEFINE_string('model_path','./tmp/ckpt/checkpoint-9784.meta',
    """Path to saved models""")
tf.app.flags.DEFINE_string('checkpoint_dir','./tmp/ckpt',
    """Directory of saved checkpoints""")
tf.app.flags.DEFINE_integer('patch_size',128,
    """Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',128,
    """Number of layers in data patch""")

def evaluate():
    """evaluate the vnet model by stepwise moving along the 3D image"""
    # restore model grpah
    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph(FLAGS.model_path)
    
    # ops to load data
    # support multiple image input, but here only use single channel, label file should be a single file with different classes
    image_filename = 'img.nii.gz'

    # Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
    with tf.device('/cpu:0'):
        # create transformations to image and labels
        transforms = [
            NiftiDataset.Normalization(),
            NiftiDataset.Resample(0.4356),
            NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer))      
            ]

        Dataset = NiftiDataset.NiftiDataset(
            data_dir=FLAGS.data_dir,
            image_filename=image_filename,
            transforms=transforms,
            train=False
            )
        
        dataset = Dataset.get_dataset()
        dataset = dataset.batch(1)
            
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:  
        print("{}: Start evaluation...".format(datetime.datetime.now()))

        imported_meta.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir,latest_filename="checkpoint-latest"))
        print("{}: Restore checkpoint success".format(datetime.datetime.now()))
        
        [image, label] = sess.run(next_element)
        print(image.shape, label.shape)


        # image = image[:,:,:,:,np.newaxis]
        # label = label[:,:,:,:,np.newaxis]
                        
        # loss, summary = sess.run([loss_op, summary_op], feed_dict={images_placeholder: image, labels_placeholder: label})
        # test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))

        # h_est2 = sess.run('hor_estimate:0')
        # v_est2 = sess.run('ver_estimate:0')
        # print("h_est: %.2f, v_est: %.2f" % (h_est2, v_est2))
        
        


def main(argv=None):
    evaluate()

if __name__=='__main__':
    tf.app.run()