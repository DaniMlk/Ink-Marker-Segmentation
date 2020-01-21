import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
import segmentation_models as sm
from dataset import Dataset, Dataloder
from utility import *
from model import network
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# DATA_DIR = './data/Marker'
DATA_DIR = './data/Data/'
k_fold = 4
# x_train_dir = os.path.join(DATA_DIR, 'train')
# y_train_dir = os.path.join(DATA_DIR, 'train_result_with_model')

# x_valid_dir = os.path.join(DATA_DIR, 'val')
# y_valid_dir = os.path.join(DATA_DIR, 'val_result_with_model')

# x_test_dir = os.path.join(DATA_DIR, 'test')
# y_test_dir = os.path.join(DATA_DIR, 'test_result_with_model')
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--backbone', default='mobilenet',
                    choices=['efficientnetb0', 'efficientnetb3', 'resnet18'])
parser.add_argument('--arch', default='Unet', choices=['Unet', 'FPN'])
args = parser.parse_args()
print(args)
# define hyper parametere of the network
BACKBONE = args.backbone
BATCH_SIZE = 6
CLASSES = ['marker']
LR = 0.00001
EPOCHS = 40


# Dataset for train images
iou_list = []
loss_list = []
iou_val_list = []
loss_val_list = []
for i in range(k_fold):

    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define network parameters
    model, n_classes = network(CLASSES, BACKBONE, args.arch)
    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss(
    ) if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5),
               sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)
    x_train_dir = os.path.join(DATA_DIR, 'trainfolder/' + str((i + 1) % 4))
    y_train_dir = os.path.join(
        DATA_DIR, 'trainfolder_annot/' + str((i + 1) % 4))

    x_train_dir_2 = os.path.join(DATA_DIR, 'trainfolder/' + str((i + 2) % 4))
    y_train_dir_2 = os.path.join(
        DATA_DIR, 'trainfolder_annot/' + str((i + 2) % 4))

    x_train_dir_3 = os.path.join(DATA_DIR, 'trainfolder/' + str((i - 1) % 4))
    y_train_dir_3 = os.path.join(
        DATA_DIR, 'trainfolder_annot/' + str((i - 1) % 4))
    x_test_dir_2 = None
    y_test_dir_2 = None
    x_test_dir_3 = None
    y_test_dir_3 = None

    x_valid_dir = os.path.join(DATA_DIR, 'trainfolder/' + str(i))
    y_valid_dir = os.path.join(DATA_DIR, 'trainfolder_annot/' + str(i))
    x_valid_dir_2 = None
    y_valid_dir_2 = None
    x_valid_dir_3 = None
    y_valid_dir_3 = None
    train_dataset = Dataset(
        x_train_dir,
        x_train_dir_2,
        x_train_dir_3,
        y_train_dir,
        y_train_dir_2,
        y_train_dir_3,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir,
        x_valid_dir_2,
        x_valid_dir_3,
        y_valid_dir,
        y_valid_dir_2,
        y_valid_dir_3,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),)

    # test_dataset = Dataset(
    #     x_test_dir,
    #     x_test_dir_2,
    #     x_test_dir_3,
    #     y_test_dir,
    #     y_test_dir_2,
    #     y_test_dir_3,
    #     classes=CLASSES,
    #     augmentation=get_validation_augmentation(),
    #     preprocessing=get_preprocessing(preprocess_input),
    # )
    # import pdb
    # # pdb.set_trace()
    # test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    train_dataloader = Dataloder(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    # import pdb
    # pdb.set_trace()
    # assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
    # assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint('./npy/'+BACKBONE+'_'+args.arch+'_'+str(
            i)+'.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    # train model
    history = model.fit_generator(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )
    iou_list.append(history.history['iou_score'])
    iou_val_list.append(history.history['val_iou_score'])
    loss_list.append(history.history['loss'])
    loss_val_list.append(history.history['val_loss'])
    # load best weights
    # model.load_weights('./npy/'+BACKBONE+'_'+args.arch+'_'+str(i)+'.h5')

    # scores = model.evaluate_generator(test_dataloader)

    # print("Loss: {:.5}".format(scores[0]))
    # for metric, value in zip(metrics, scores[1:]):
    #     print("mean {}: {:.5}".format(metric.__name__, value))


np.save('./npy/iou_list_'+BACKBONE+'_'+args.arch+'.npy', np.asarray(iou_list))
np.save('./npy/iou_val_list_'+BACKBONE+'_'+args.arch +
        '.npy', np.asarray(iou_val_list))
np.save('./npy/loss_list_'+BACKBONE+'_' +
        args.arch+'.npy', np.asarray(loss_list))
np.save('./npy/loss_val_list_'+BACKBONE+'_'+args.arch +
        '.npy', np.asarray(loss_val_list))
# # Plot training & validation iou_score values
# plt.figure(figsize=(30, 5))
# plt.subplot(121)
# plt.plot(history.history['iou_score'])
# plt.plot(history.history['val_iou_score'])
# plt.title('Model iou_score')
# plt.ylabel('iou_score')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# np.save('./iou_'+BACKBONE+'_'+args.arch+'.npy',history.history['iou_score'])
# np.save('./iou_val_'+BACKBONE+'_'+args.arch+'.npy',history.history['val_iou_score'])
# # Plot training & validation loss values
# plt.subplot(122)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('./iou_score.png')
# np.save('./loss_'+BACKBONE+'_'+args.arch+'.npy',history.history['loss'])
# np.save('./loss_val_'+BACKBONE+'_'+args.arch+'.npy',history.history['val_loss'])
