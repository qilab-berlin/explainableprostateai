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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.import sys

from feature_classifier_inference import *
from lesion_classifier_inference import *
import numpy as np

inference_type = 'heatmapGeneration' #'lesionClassification'

model_runner = ModelRunner()
model_runner.load_model()
feature_classifier = FeatureClassifier()
feature_classifier.load_model()

inference_cases = [...]
'''Names of the images used for inference. For lesion inference there must be .npy files with
a numpy array of the shape (1, 3, 64, 64, 10) in the ./data/inferenceImages folder. Heapmap generation 
can intake images of alternating x, y and z size and needs a array of the mask of the prostate.'''


def classify_lesion(idx):
    try:
        with open(f'./data/inference_images/{idx}.npy', 'rb') as f:
            image_array = np.load(f)
        output = model_runner.run_model_with_input(image_array)
        output['features'] = feature_classifier.evaluate_activations(output['activations'])
        del output['activations']

        benign_value = output['outputs'][0][0]
        malignant_value = output['outputs'][0][1]
        print(f'{idx}:\nOutput: bengin: {benign_value:.3f} malignant: {malignant_value:.2f}\nFeatures:')
        for feature in output['features'].keys():
            feature_value = output['features'][feature] * 100
            print(f'{feature}: {feature_value:.1f}%')
    except Exception as e:
        print(e)


def generate_heatmap(idx):
    try:
        with open(f'./data/inference_images/{idx}.npy', 'rb') as f:
            image_array = np.load(f)
        with open(f'./data/inference_images/{idx}_mask.npy', 'rb') as f:
            mask_array = np.load(f)
        output = model_runner.run_model_with_input_and_mask(image_array, mask_array)
        with open(f'./output/{idx}_heatmap.npy', 'wb') as f:
            np.save(f, output)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    if inference_type == 'lesionClassification':
        for idx in inference_cases:
            classify_lesion(idx)

    if inference_type == 'heatmapGeneration':
        for idx in inference_cases:
            generate_heatmap(idx)
