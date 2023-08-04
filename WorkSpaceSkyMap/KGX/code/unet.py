import os
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import concatenate as merge_l

from keras.optimizers import Adam, Nadam, Adadelta,SGD
from keras.layers import (
    Input, Convolution2D, MaxPooling2D, UpSampling2D,
    Reshape, core, Dropout, Flatten,
    Activation, BatchNormalization, Lambda, Dense, Conv2D, Conv2DTranspose, concatenate,Permute,Cropping2D,Add)
from keras.losses import binary_crossentropy
from keras.callbacks import (ModelCheckpoint, TensorBoard, CSVLogger, History, EarlyStopping, LambdaCallback,ReduceLROnPlateau)
from data_defores_generator import data_gen, create_list_id, split_black_image

def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def unet(num_channel,size):
    conv_params = dict(activation='relu', border_mode='same')
    merge_params = dict(axis=-1)
    inputs1 = Input((size, size,int(num_channel)))
    conv1 = Convolution2D(32, (3,3), **conv_params)(inputs1)
    conv1 = Convolution2D(32, (3,3), **conv_params)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, (3,3), **conv_params)(pool1)
    conv2 = Convolution2D(64, (3,3), **conv_params)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, (3,3), **conv_params)(pool2)
    conv3 = Convolution2D(128, (3,3), **conv_params)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, (3,3), **conv_params)(pool3)
    conv4 = Convolution2D(256, (3,3), **conv_params)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, (3,3), **conv_params)(pool4)
    conv5 = Convolution2D(512, (3,3), **conv_params)(conv5)

    up6 = merge_l([UpSampling2D(size=(2, 2))(conv5), conv4], **merge_params)
    conv6 = Convolution2D(256, (3,3), **conv_params)(up6)
    conv6 = Convolution2D(256, (3,3), **conv_params)(conv6)

    up7 = merge_l([UpSampling2D(size=(2, 2))(conv6), conv3], **merge_params)
    conv7 = Convolution2D(128, (3,3), **conv_params)(up7)
    conv7 = Convolution2D(128, (3,3), **conv_params)(conv7)

    up8 = merge_l([UpSampling2D(size=(2, 2))(conv7), conv2], **merge_params)
    conv8 = Convolution2D(64, (3,3), **conv_params)(up8)
    conv8 = Convolution2D(64, (3,3), **conv_params)(conv8)

    up9 = merge_l([UpSampling2D(size=(2, 2))(conv8), conv1], **merge_params)
    conv9 = Convolution2D(32, (3,3), **conv_params)(up9)
    conv9 = Convolution2D(32, (3,3), **conv_params)(conv9)

    conv10 = Convolution2D(1, (1, 1), activation='sigmoid')(conv9)
    optimizer=SGD(lr=1e-4, decay=1e-8, momentum=0.9, nesterov=True)
    model = Model(input=inputs1, output=conv10)
    model.compile(optimizer=optimizer,
                loss=binary_crossentropy,
                metrics=['accuracy', jaccard_coef, jaccard_coef_int])
    return model

unet_model = unet(4, 512)
unet_model.summary()

from datetime import datetime
CURRENT_DATE = datetime.now().strftime("Ngay%dThang%mNam%Y_%Hh_%Mp_%Ss")
BATCH_SIZE=1
NUM_CHANNEL=4
NUM_CLASS=1
INPUT_SIZE=512
EARLY=70
SPLIT_RATIO=0.9
MODEL_NAME="KGX_Unet_512_Nuoc"
IMAGE_PATH=r"C:\Users\SkyMap\Desktop\Test\Test\KGX_Unet_512_Nuoc\TrainingDataset\img_crop"
MASK_PATH=r"C:\Users\SkyMap\Desktop\Test\Test\KGX_Unet_512_Nuoc\TrainingDataset\img_mask_crop"
MODEL_DIR=r"C:\Users\SkyMap\Desktop\Test\Test\{}\{}".format(MODEL_NAME, CURRENT_DATE)


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
model_checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, "{}_val_weights.h5".format(CURRENT_DATE + "_{epoch:04d}")),
        verbose=1,
        save_best_only=True,
        save_weights_only=False)
model_earlystop = EarlyStopping(
        patience=EARLY,
        verbose=0,
        )
model_history = History()
model_board = TensorBoard(
    log_dir=os.path.join(MODEL_DIR, 'logs'),
    histogram_freq=0,
    write_graph=True,
    embeddings_freq=0)
lr_reducer = ReduceLROnPlateau(
    factor=np.sqrt(0.1),
    verbose=1,
    patience=10,
    min_lr=0.5e-7)
image_list= create_list_id(IMAGE_PATH)
np.random.shuffle(image_list)
count = len(image_list)    
cut_idx = int(round(count*SPLIT_RATIO))    
train_list = image_list[0:cut_idx]
val_list = [id_image for id_image in image_list if id_image not in train_list]

thres_neg = 1/100.0
scale_neg = 20/100.0
print("Train samples: {}".format(len(train_list)))
print("Validation samples: {}".format(len(val_list)))   
print("===========================")

pos_train,neg_train = split_black_image(image_list,MASK_PATH,thres_neg)
pos_val,neg_val = split_black_image(image_list,MASK_PATH,thres_neg)
step1 = round(len(pos_train)*(1+scale_neg)/BATCH_SIZE)
step2 = round(len(pos_val)*(1+scale_neg)/BATCH_SIZE)
num_chanel=NUM_CHANNEL
print(step2)
unet_model.fit_generator(
    generator= data_gen(pos_train,neg_train, IMAGE_PATH, MASK_PATH, BATCH_SIZE, scale_neg,num_chanel,NUM_CLASS,augment = True),
    validation_data = data_gen(pos_val,neg_val, IMAGE_PATH, MASK_PATH, BATCH_SIZE, scale_neg,num_chanel,NUM_CLASS),
    validation_steps = 100,
    steps_per_epoch = 500, 
    epochs=500, 
    verbose=1, 
    callbacks=[model_checkpoint,model_earlystop, model_history, model_board,lr_reducer])
unet_model.save_weights(os.path.join(MODEL_DIR, "{}_val_weights_last.h5".format(CURRENT_DATE)))