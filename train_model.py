# This file is copyright (C) 2023 Quantitative Imaging Lab, Charité Universitätsmedizin Berlin, All rights reserved.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from batchgenerators.transforms import CenterCropTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading import SlimDataLoaderBase, MultiThreadedAugmenter, DataLoaderFromDataset, Dataset
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms import NumpyToTensor, Compose, RandomCropTransform, CenterCropTransform

import pickle
import functools

'''
The data is prepared as follows:
    imageids_and_targets: a pickled python dict containing the image ids and targets 
    for training and test set. It structured as:
        {'train': [{'id':<string>, 'target':<0 or 1>, 'target_onehot':<one hot vector>}],
        'test': [{'id':<string>, 'target':<0 or 1>, 'target_onehot':<one hot vector>'}]}
    imagePatches: npy files: numpy array of normalized t2, adc, dwi patches with a shape of: (1, 3, 96, 96, 16)
    name by the image ids
'''

class DirOfImagesDataset(Dataset):
    def __init__(self, id_and_target):
        self._data = id_and_target

        
    def __getitem__(self, item):
        imageid = self._data[item]['id']
        with open(f'./data/image_patches/{imageid}.npy', 'rb') as f:
            img = np.load(f)
        return {"name":imageid,"idx":item, "data":img, "target":self._data[item]['target'], "target_onehot":self._data[item]['target_onehot']}
    
    def __len__(self):
        return len(self._data)

def make_batch_generators(imageids_and_targets, augment_train = True):
    print("Making batch generators")
    def make_batch_generator(ids_and_targets, shuffle, transforms = None):
        n_threads = 16
        dataset =  DirOfImagesDataset(ids_and_targets)
        
        data_loader = DataLoaderFromDataset(data = dataset, batch_size = BATCH_SIZE, num_threads_in_multithreaded = n_threads, return_incomplete = False, shuffle = shuffle)
        mta = MultiThreadedAugmenter(data_loader,transforms,n_threads,1,None)
        mta.length = len(ids_and_targets)
        mta.n_batches = np.ceil(mta.length/BATCH_SIZE)
        return mta

    train_augmentations = Compose([
        SpatialTransform(IMG_SIZE, np.array(IMG_SIZE) // 2, 
                     do_elastic_deform=False,
                     do_rotation=True, angle_z=(0, np.pi/5), angle_x = (0,0), angle_y = (0,0), 
                     do_scale=False, scale=(0.95, 1.05), 
                     border_mode_data='constant', border_cval_data=0, order_data=1,
                     random_crop=False),
        RandomCropTransform(crop_size = IMAGE_SIZE_FOR_NET, margins = [8,8,3])
    ])
    
    center_crop = Compose([

        CenterCropTransform(crop_size = IMAGE_SIZE_FOR_NET)
    ])
    return {'train':make_batch_generator(imageids_and_targets['train'], shuffle=True, transforms = train_augmentations if augment_train else center_crop), 
            'test':make_batch_generator(imageids_and_targets['test'], shuffle=False,transforms = center_crop),}

def cudat(numpyarray):
    v =  Variable(torch.from_numpy(numpyarray).cuda()).float()
    return v

def train(model, generator, path_model_output, criterion):
    print("Beginning Training")
    earlyStoppingCounter = 0
    smallestLost = 1
    optimizer = optim.Adam(model.parameters())#, lr = 0.00001)
    for epoch in range(N_EPOCHS):
        avg_loss = 0
        for batch_idx, batch in enumerate(generator):
            optimizer.zero_grad()
            x, target = cudat(batch['data']), cudat(batch['target']).long()
            out, activations = model(x)
            loss = criterion(out, target)
            avg_loss = avg_loss*0.8 + loss.item()*0.2
            loss.backward()
            optimizer.step()
            if (batch_idx+1) == 27:
                print(f"Epoch : {epoch}, batch index : {batch_idx+1}, train loss : {avg_loss}")
        if epoch > 20:
            if avg_loss < smallestLost:
                smallestLost = avg_loss
                earlyStoppingCounter = 0
                torch.save(model, os.path.join(path_model_output,'model_best_training.pt'))
            else:
                earlyStoppingCounter = earlyStoppingCounter + 1
        
        if epoch !=0 and epoch%10== 0:
            torch.save(model, os.path.join(path_model_output,'model_8_training.pt'))
            print(f"Saving to {os.path.join(path_model_output,'model_8_training.pt')}")
        
        if earlyStoppingCounter == 30:
            print(f'Sopped early after {epoch-30} epochs with, train loss : {smallestLost}')
            return torch.load(os.path.join(path_model_output,'model_best_training.pt'))
    return model

def run_experiment():        
    model_name = f"model"

    # GENERATE DATA GENERATORS AND MODEL
    with open('./data/imageids_and_targets.pickle', 'rb') as handle:
        imageids_and_targets = pickle.load(handle)
    generators = make_batch_generators(imageids_and_targets)
    model = Network(in_channels = 3, n_targets = 2).cuda()

    # TRAIN MODEL
    criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(imageids_and_targets['train']))
    train(model, generators['train'], './output', criterion)
    torch.save(model, os.path.join('./output/model_0.pt'))
    
class Network(nn.Module):
    def __init__(self, in_channels=3, n_targets=2):
        super(Network,self).__init__()
        self.conv11 = nn.Sequential(
            nn.Conv3d(in_channels,8,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(8, affine=False),
        )
        
        self.conv12 = nn.Sequential(
            nn.Conv3d(8,8,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(8, affine=False),            
        )
        self.conv13 = nn.Sequential(
            nn.Conv3d(8,8,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(8, affine=False),
        )
        
        self.conv21 = nn.Sequential(           
            nn.MaxPool3d(2),            
            nn.Conv3d(8,64,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(64, affine=False),
        )
        self.conv22 = nn.Sequential(          
            nn.Conv3d(64,64,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(64, affine=False),
        )
        self.conv23 = nn.Sequential(            
            nn.Conv3d(64,64,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(64, affine=False),
        )
        self.conv31 = nn.Sequential(                        
            nn.MaxPool3d(2),            
            nn.Conv3d(64,128,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(128, affine=False),
        )
        self.conv32 = nn.Sequential(            
            nn.Conv3d(128,128,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(128, affine=False),
        )
        self.conv33 = nn.Sequential(            
            nn.Conv3d(128,128,[3,3,1], padding = [1,1,0]), 
            nn.ReLU(),
            nn.BatchNorm3d(128, affine=False),
        )
            
        self.final_conv_depth = 128
        n_maxpools = 2
        size = [i//(2**n_maxpools) for i in IMAGE_SIZE_FOR_NET]
        print(size)
        self.flattened_size = int(functools.reduce(lambda x, y:x*y, size)) * self.final_conv_depth
        print(self.flattened_size)
        
        self.lin0 = nn.Sequential(
            nn.Linear(self.flattened_size,2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048, affine=False),
        )
        
        self.lin1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, affine=False),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, affine=False),
        )
        self.classifier_to_end = nn.Sequential(
            nn.Linear(64,n_targets),
        )
    def forward(self, x):
        activations = []
        x = self.conv11(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv12(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv13(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv21(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv22(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv23(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv31(x)
        activations.append(x.view(-1, np.prod(np.floor_divide(IMAGE_SIZE_FOR_NET,4))*128))
        
        x = self.conv32(x)
        activations.append(x.view(-1, np.prod(np.floor_divide(IMAGE_SIZE_FOR_NET,4))*128))
        
        x = self.conv33(x)
        activations.append(x.view(-1, np.prod(np.floor_divide(IMAGE_SIZE_FOR_NET,4))*128))
        
        x = x.view(-1, self.flattened_size)
        
        x = self.lin0(x)
        activations.append(x.view(-1, 2048))
        
        x = self.lin1(x)
        activations.append(x.view(-1, 1024))
        
        x = self.lin2(x)
        activations.append(x.view(-1, 64))
        
        return self.classifier_to_end(x), activations
    
def calculate_class_weights(imageids_and_targets):
    target_list = [id_and_target['target'] for id_and_target in imageids_and_targets]
    num_tumorlesions = np.sum(target_list)
    num_non_tumorlesion =  len(target_list) - num_tumorlesions
    class_frequency = [num_non_tumorlesion/len(target_list), num_tumorlesions/len(target_list)]
    class_weights = torch.FloatTensor(class_frequency).cuda()
    return class_weights

if __name__ == "__main__":
    IMG_SIZE = [96,96,16]
    IMAGE_SIZE_FOR_NET = [64,64,10] 
    BATCH_SIZE = 64

    N_EPOCHS = 200

    generators = run_experiment()
