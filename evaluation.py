import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
from glob import glob
import cv2
import openslide
import keras
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from dataset import Dataset, Dataloder
from utility import *
from model import network
import segmentation_models as sm
import argparse

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--backbone', default='mobilenet',
                    choices=['efficientnetb0', 'efficientnetb3', 'resnet18'])
parser.add_argument('--arch', default='Unet', choices=['Unet', 'FPN'])
args = parser.parse_args()
print(args)

DATA_DIR = './data/Marker'
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

x_temp_dir = os.path.join(DATA_DIR, 'temp')

# define hyper parametere of the network
BACKBONE = 'efficientnetb2'
BATCH_SIZE = 4
CLASSES = ['marker']
LR = 0.00001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)
# define network parameters
model, n_classes = network(CLASSES, BACKBONE, args.arch)
# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    classes=CLASSES, 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
# load best weights
model.load_weights('./saved_model/resnet18_Unet.h5')
scores = model.evaluate_generator(test_dataloader)

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))


# n = 5
# ids = np.random.choice(np.arange(len(test_dataset)), size=n)

# for i in ids:
    
#     image, gt_mask = test_dataset[i]
#     image = np.expand_dims(image, axis=0)
#     import pdb
#     pdb.set_trace()
#     pr_mask = model.predict(image).round()
    
#     visualize(
#         image=denormalize(image.squeeze()),
#         gt_mask=gt_mask[..., 0].squeeze(),
#         pr_mask=pr_mask[..., 0].squeeze(),
#     )

# img_paths = glob('/media/dani/DATA/GCD/Test/WSIs/*.svs')
# crop_size = (384,480)
# # for i in range(len(img_paths)):
# list = []

# for i in range(5):
#     counter = 0
# #     pdb.set_trace()
#     img = openslide.OpenSlide(img_paths[i])
#     max_lvl = (img.level_dimensions[-1])
#     pil_img = img.read_region((0,0),len(img.level_dimensions)-1,(max_lvl[0],max_lvl[1]))
#     pil_img = pil_img.convert('RGB')
#     npy_img = np.asarray(pil_img)
#     mask = np.zeros((npy_img.shape[0],npy_img.shape[1]))
#     #org_img = np.zeros((npy_img.shape[0],npy_img.shape[1],npy_img.shape[2]))
#     for j in range(0, npy_img.shape[0], crop_size[0]):
#         for k in range(0,npy_img.shape[1], crop_size[1]):
#             x = 0
#             y = 0
#             if j + crop_size[0] > npy_img.shape[0]:
#                 x = j - npy_img.shape[0] + crop_size[0]
#             if k + crop_size[1] > npy_img.shape[1]:
#                 y = k - npy_img.shape[1] + crop_size[1]
#             clip_img = npy_img[j-x:j-x+crop_size[0], k-y:k-y+crop_size[1]]
#             # import pdb
#             # pdb.set_trace()
#             io.imsave('./Masks/temp/temp.png', clip_img)
#             test_dataset = Dataset(
#                 x_temp_dir, 
#                 x_temp_dir, 
#                 classes=CLASSES, 
#                 augmentation=get_validation_augmentation(),
#                 preprocessing=get_preprocessing(preprocess_input),
#             )       
#             inp = np.expand_dims(test_dataset[0][0], axis=0)
#             pr_mask = model.predict(inp).round()

#             mask[j-x:j-x+crop_size[0], k-y:k-y+crop_size[1]] = pr_mask[...,0]
#             #org_img[j-x:j-x+crop_size[0], k-y:k-y+crop_size[1]] = clip_img

#     io.imsave('./Masks/' + img_paths[i].split('/') [-1][:-4] + '.png', mask)
#     if (np.sum(mask)/(npy_img.shape[0]*npy_img.shape[1]) > 0.01):
#         list.append(img_paths[i].split('/') [-1][:-4])

# np_list = np.asarray(list)
# np.save('./list.npy', np_list)
#     #io.imsave('./Masks/' + img_paths[i].split('/') [-1][:-4] + 'M.png', org_img)
