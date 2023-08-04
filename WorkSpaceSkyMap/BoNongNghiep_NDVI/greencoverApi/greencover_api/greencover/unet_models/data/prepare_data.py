import os
import glob
import fiona
import rasterio
import numpy as np
import rasterio.mask
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tensorflow.keras import utils

import warnings
warnings.filterwarnings("ignore")

def remove_empty(shp_path):
    for _ in glob.glob(shp_path):
        with fiona.open(_, "r") as shapefile:
            features = [f["geometry"] for f in shapefile]
        if all(x is None for x in features):
            try:
                os.remove(_)
                os.remove(_.replace('shp', 'dbf'))
                os.remove(_.replace('shp', 'prj'))
                os.remove(_.replace('shp', 'shx'))
            except:
                pass
            try:
                extend = _.split('/')[-2].replace('crop_shape','crop')
                os.remove(_.replace('shp', 'tif').replace(_.split('/')[-2], extend))
            except:
                pass
        shapefile.close

def random_rotate(image, label):
    rot = np.random.uniform(-20*np.pi/180, 20*np.pi/180)
    modified = tfa.image.rotate(image, rot)
    m_label = tfa.image.rotate(label, rot)
    return modified, m_label

def random_rot_flip(image, label, width, height):
    m_label = tf.reshape(label, (width, height, 1))
    axis = np.random.randint(0, 2)
    if axis == 1:
        # vertical flip
        modified = tf.image.flip_left_right(image=image)
        m_label = tf.image.flip_left_right(image=m_label)
    else:
        # horizontal flip
        modified = tf.image.flip_up_down(image=image)
        m_label = tf.image.flip_up_down(image=m_label)
    # rot 90
    k_90 = np.random.randint(4)
    modified = tf.image.rot90(image=modified, k=k_90)
    m_label = tf.image.rot90(image=m_label, k=k_90)

    m_label = tf.reshape(m_label, (width, height))
    return modified, m_label

def data_augment(image, label, N_CLASSES):
    rand1, rand2 = np.random.uniform(size=(2, 1))
    w,h = image.shape[0], image.shape[1]
    if rand1 > 0.25:
        modified, m_label = random_rot_flip(image, label, w, h)
    elif rand2 > 0.25:
        modified, m_label = random_rotate(image, label)
    else:
        modified, m_label = image, label
    m_label = tf.cast(m_label, tf.uint8)
    m_label = tf.one_hot(m_label, depth=N_CLASSES)
    return modified, m_label


class generator_train(utils.Sequence):
    def __init__(self, file_names, batch_size, N_CLASSES, numband):
        self.file_names = file_names
        self.batch_size = batch_size
        self.N_CLASSES = N_CLASSES
        self.numband = numband

    def __len__(self):
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        path_mask = self.file_names[index]
        path_img = path_mask.replace('mask_cut_img_crop','img_cut_img_crop')
        mask = rasterio.open(path_mask).read().swapaxes(0,1).swapaxes(1,2)/255
        image = rasterio.open(path_img).read().swapaxes(0,1).swapaxes(1,2)
        image = image[:,:,:self.numband]
        image = tf.cast(image, tf.float32)
        image, mask = data_augment(image, mask[:,:,0], self.N_CLASSES)
        return image, mask

    def on_epoch_end(self):
        np.random.shuffle(self.file_names)

class generator_valid(utils.Sequence):
    def __init__(self, file_names, batch_size, N_CLASSES, numband):
        self.file_names = file_names
        self.batch_size = batch_size
        self.N_CLASSES = N_CLASSES
        self.numband = numband

    def __len__(self):
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        path_mask = self.file_names[index]
        path_img = path_mask.replace('mask_cut_img_crop','img_cut_img_crop')
        mask = rasterio.open(path_mask).read().swapaxes(0,1).swapaxes(1,2)/255
        image = rasterio.open(path_img).read().swapaxes(0,1).swapaxes(1,2)
        image = image[:,:,:self.numband]
        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.uint8)
        mask = tf.one_hot(mask[:,:,0], depth=self.N_CLASSES)
        return image, mask

    def on_epoch_end(self):
        np.random.shuffle(self.file_names)
        
def input_fn(file_names, image_size, batch_size, N_CLASSES, numband ,train=True):
    def generator_fn():
        if train:
            generator = utils.OrderedEnqueuer(generator_train(file_names, batch_size, N_CLASSES, numband), False)
        else:
            generator = utils.OrderedEnqueuer(generator_valid(file_names, batch_size, N_CLASSES, numband), False)
        generator.start()
        
        n = 0
        while n<int(len(file_names)):
            image, mask = generator.get().__next__()
            yield image, mask
            n+=1

    output_types = (tf.float32, tf.float32)
    output_shapes = ((image_size, image_size, numband),
                     (image_size, image_size, N_CLASSES))

    dataset = tf.data.Dataset.from_generator(generator=generator_fn,
                                             output_types=output_types,
                                             output_shapes=output_shapes)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def input_fn_multi(file_names, image_size, batch_size, N_CLASSES, numband ,train=True):
    def generator_fn():
        if train:
            generator = utils.OrderedEnqueuer(generator_train(file_names, batch_size, N_CLASSES, numband), False)
        else:
            generator = utils.OrderedEnqueuer(generator_valid(file_names, batch_size, N_CLASSES, numband), False)
        generator.start()
        
        n = 0
        while n<int(len(file_names)):
            image, mask = generator.get().__next__()
            yield image, (mask[::2,::2,:], mask[::4,::4,:], mask[::8,::8,:], mask[::16,::16,:], mask)
            n+=1

    output_types = (tf.float32, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
    output_shapes = ((image_size, image_size, numband),
                     ((int(image_size/2), int(image_size/2), N_CLASSES),
                     (int(image_size/4), int(image_size/4), N_CLASSES),
                     (int(image_size/8), int(image_size/8), N_CLASSES),
                     (int(image_size/16), int(image_size/16), N_CLASSES),
                     (image_size, image_size, N_CLASSES),))

    dataset = tf.data.Dataset.from_generator(generator=generator_fn,
                                             output_types=output_types,
                                             output_shapes=output_shapes)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def data_gen(mask_path, img_size, batch_size, N_CLASSES, numband, split_ratios, test_data=False, multi=False):
    mask_img = sorted(glob.glob(mask_path))
    np.random.shuffle(mask_img)
    L = len(mask_img)
    if multi:
        if test_data:
            L_train = int(split_ratios*L)
            L_valid = int((1-split_ratios)/2*L)
            L_test = L - L_train - L_valid
            train_dataset = input_fn_multi(mask_img[:L_train], img_size, batch_size, N_CLASSES, numband)
            valid_dataset = input_fn_multi(mask_img[L_train:L_train+L_valid], img_size, batch_size, N_CLASSES, numband, False)
            test_dataset = input_fn_multi(mask_img[L_train+L_valid:L], img_size, batch_size, N_CLASSES, numband, False)
            print("Training:validation:testing = {}:{}:{}".format(L_train, L_valid, L_test))
            return train_dataset, valid_dataset, test_dataset, L_train, L_valid, L_test
        else:
            L_train = int(split_ratios*L)
            L_valid = int(L-L_train)
            train_dataset = input_fn_multi(mask_img[:L_train], img_size, batch_size, N_CLASSES, numband)
            valid_dataset = input_fn_multi(mask_img[L_train:L], img_size, batch_size, N_CLASSES, numband, False)
            print("Training:validation = {}:{}".format(L_train, L_valid))
        return train_dataset, valid_dataset, L_train, L_valid
    else:
        if test_data:
            L_train = int(split_ratios*L)
            L_valid = int((1-split_ratios)/2*L)
            L_test = L - L_train - L_valid
            train_dataset = input_fn(mask_img[:L_train], img_size, batch_size, N_CLASSES, numband)
            valid_dataset = input_fn(mask_img[L_train:L_train+L_valid], img_size, batch_size, N_CLASSES, numband, False)
            test_dataset = input_fn(mask_img[L_train+L_valid:L], img_size, batch_size, N_CLASSES, numband, False)
            print("Training:validation:testing = {}:{}:{}".format(L_train, L_valid, L_test))
            return train_dataset, valid_dataset, test_dataset, L_train, L_valid, L_test
        else:
            L_train = int(split_ratios*L)
            L_valid = int(L-L_train)
            train_dataset = input_fn(mask_img[:L_train], img_size, batch_size, N_CLASSES, numband)
            valid_dataset = input_fn(mask_img[L_train:L], img_size, batch_size, N_CLASSES, numband, False)
            print("Training:validation = {}:{}".format(L_train, L_valid))
        return train_dataset, valid_dataset, L_train, L_valid