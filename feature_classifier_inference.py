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

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

featureNames = [...]

'''featureNames is filled with the feature names from the feature classifier training'''

IMAGE_SIZE_FOR_NET = [64,64,10]

SIZE_LAYER_1_AND_2 = np.prod(IMAGE_SIZE_FOR_NET) * 8
OFFSET_LAYER_3 = 6 * SIZE_LAYER_1_AND_2
SIZE_LAYER_3 = np.prod(np.floor_divide(IMAGE_SIZE_FOR_NET,4))*128

offsets = []
for i in range(6):
    offsets.append(i * SIZE_LAYER_1_AND_2)
for i in range(4): 
    offsets.append(OFFSET_LAYER_3 + i * SIZE_LAYER_3)
offsets.append(offsets[9]+2048)
offsets.append(offsets[10]+1024)
offsets.append(offsets[11]+64)

class FeatureClassifier:
    def load_model(self):
        with open('./output/classifiers.pickle', 'rb') as handle:
            self.dict_classifiers = pickle.load(handle)
    def evaluate_activations(self, activations):
        dict_features = {}    
        for name in featureNames:
            probabilities = []
            layer = self.dict_classifiers[name]['layer']
            for i in range(5):
                probabilities.append(self.dict_classifiers[name]['classifiers'][i].predict_proba(activations[:,offsets[layer]:offsets[layer+1]])[:,1])
            dict_features[name] = np.mean(probabilities)
        return dict_features
