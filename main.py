"""Imports"""

import torch
import torchvision.models as models
import hiddenlayer as hl
import segmentation_models_pytorch as smp
from easydict import EasyDict as edict

from dataloader import MyDataset
from plots import VisdomLinePlotter
from train import run_epoch
import torch.optim as optim
from torch.optim import lr_scheduler

import os
import time

"""Parameters"""
params = edict({'batch_size':3,
                'width': 1920,
                'height': 1184,
                'numclasses': 2,
                'num_epochs': 1000,
                'steps_per_epoch': 20,
                'optimizer': 'SGD',
                'train_identifier': 'DEBUG'})

device = 'cuda:0'


"""Visdom stuff"""
from visdom import Visdom
visdom_log_path = os.path.join("/tmp")
#visdom_log_path = outdir
print("Saving visdom logs to", visdom_log_path)
viz = Visdom(port=6065, log_to_filename=visdom_log_path)
# for env in viz.get_env_list():
viz.delete_env(params.train_identifier)
viz.log_to_filename = os.path.join(visdom_log_path,params.train_identifier+".visdom")
plotter  = VisdomLinePlotter(env_name=params.train_identifier, plot_path=visdom_log_path)


""" Load the model and make sure it works """
#inputs = torch.zeros([2, 3, 1920, 1184])
model = smp.Unet('resnet34', classes=params.numclasses+2, encoder_weights='imagenet')
model = model.to(device)
#output = model.forward(inputs)
#print("output-shape:", output.shape)

preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name = 'resnet34', pretrained='imagenet')



"""dataloader"""
image_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/original_image/'
segvecs_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/center_vectors/'
classes_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/pixelwise_annotation_xml'


dataset_train = MyDataset(phase='train', image_dir = image_dir, classes_dir = classes_dir, segvecs_dir = segvecs_dir)
dataset_val = MyDataset(phase='val',   image_dir = image_dir, classes_dir = classes_dir, segvecs_dir = segvecs_dir)


#print(x.shape)


train_sampler = None
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=params.batch_size, shuffle=(train_sampler is None),
    num_workers=20, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=params.batch_size, shuffle=False,
    num_workers=20, pin_memory=True)

dataloader = {'train':train_loader, 'val': val_loader}

print("Created train loader")


if params.optimizer == 'SGD':
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9)
elif params.optimizer == 'Adam':
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.1)
else:
    raise ValueError("Requested optimizer not recognized:", params.optimizer)


scheduler = lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.5)

time_train_start = time.time()
for epoch in range(params.num_epochs):
    for phase in ['train', 'val']:
        epoch_loss = run_epoch(
            params=params,
            phase=phase,
            num_batches=params.steps_per_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader[phase],
            device=device)

        dates = time.localtime()
        datestring = "-".join([str(dates.tm_year), str(dates.tm_mon), str(dates.tm_mday)])
        timestring = ":".join([str(dates.tm_hour), str(dates.tm_min), str(dates.tm_min)])

        plotter.plot(var_name='loss over epoch',
                     split_name=phase,
                     title_name='loss over epoch',
                     x=epoch,
                     y=epoch_loss)

        min_since_train_start = round((time.time() - time_train_start) / 60,1)
        plotter.plot(var_name='loss over time',
                     split_name=phase,
                     title_name='loss over time',
                     x=min_since_train_start,
                     y=epoch_loss
                     xlabel = 'Minutes')

        # save the model
        if phase == 'val':# and epoch_acc > best_acc:
            outdir = 'checkpoints'
            #best_model_wts = copy.deepcopy(model.state_dict())
            # save the currently best model to disk
            torch.save(model.state_dict(), outdir+'/'+str(epoch)+params.train_identifier+'.pth')
            print("Saved new best checkpoint to:\n"+outdir+'/'+str(epoch)+'_'+params.train_identifier+'.pth')


print("Training finished")