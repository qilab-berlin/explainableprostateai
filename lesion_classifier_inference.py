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
import numpy as np
from skimage.transform import resize

BATCH_SIZE = 1
IMAGE_SIZE_FOR_NET = [64,64,10]
x_scale = 4
y_scale = 4
z_scale = 1

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
        activations.append(x.view(BATCH_SIZE, -1))
        
        x = self.conv12(x)
        activations.append(x.view(BATCH_SIZE, -1))
        
        x = self.conv13(x)
        activations.append(x.view(BATCH_SIZE, -1))
        
        x = self.conv21(x)
        activations.append(x.view(BATCH_SIZE, -1))
        
        x = self.conv22(x)
        activations.append(x.view(BATCH_SIZE, -1))
        
        x = self.conv23(x)
        activations.append(x.view(BATCH_SIZE, -1))
        
        x = self.conv31(x)
        activations.append(x.view(BATCH_SIZE, -1))
        
        x = self.conv32(x)
        activations.append(x.view(BATCH_SIZE, -1))
        
        x = self.conv33(x)
        activations.append(x.view(BATCH_SIZE, -1))
        
        x = x.view(-1, self.flattened_size)
        
        x = self.lin0(x)
        activations.append(x.view(BATCH_SIZE, -1))
        
        x = self.lin1(x)
        activations.append(x.view(BATCH_SIZE, -1))
        
        x = self.lin2(x)
        activations.append(x.view(BATCH_SIZE, -1))
        
        return self.classifier_to_end(x), activations

class ModelRunner:    
    def load_model(self):
        PATH_MODEL = './output/model_0.pt'
        print("saved model path : ",PATH_MODEL)
        self.model = torch.load(PATH_MODEL)
        self.model.eval()

    def run_model_with_input(self, x):
        x = cudat(x)
        out, activations = self.model(x)
        out = out.cpu().detach().numpy()
        activations = torch.cat(activations, 1).cpu().detach().numpy()
        index = np.argmax(out,axis=1)
        return {'outputs':out, 
            'index':index, 
            'activations': activations,
            }
    def run_model_with_input_and_mask(self, input_array, mask_array):
        self.x = cudat(input_array)
        heat_array = np.zeros([int(mask_array.shape[0]/x_scale), int(mask_array.shape[1]/y_scale), int(mask_array.shape[2]/z_scale)])
        for x in range(32, input_array.shape[2]-(32-1), x_scale):
            for y in range(32, input_array.shape[3]-(32-1), y_scale):
                for z in range(5, input_array.shape[4]-(5-1), z_scale):
                    if(mask_array[x][y][z] != 0):
                        try:
                            proba, _ = self.model(self.x[:,:,x-32:x+32,y-32:y+32,z-5:z+5])
                            proba = proba[0].cpu().detach().numpy()[1]
                            if proba > 0:
                                heat_array[int(x/x_scale)][int(y/y_scale)][int(z/z_scale)] = proba
                        except Exception as e:
                            print(f'x: {x}, y: {y}, z:{z}')
                            print(e)
        heat_array = resize(heat_array, mask_array.shape)
        return heat_array


    
def cudat(numpyarray):
    v =  Variable(torch.from_numpy(numpyarray).cuda()).float()
    return v

