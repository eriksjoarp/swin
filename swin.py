# conda activate swin
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
# conda install -c conda-forge opencv -y
# conda install -c huggingface transformers -y
# conda install -c conda-forge huggingface_hub -y
# conda install -c huggingface -c conda-forge datasets -y
# conda install -c conda-forge pytorch-model-summary -y
# conda install -c intel scikit-learn -y
# conda install -c anaconda seaborn -y

import sys
import os

############################        IMPORT HELPER MODULES       ############################
sys.path.append(os.getcwd() + '/..')
import python_imports
for path in python_imports.dirs_to_import(): sys.path.insert(1, path)
############################################################################################

from transformers import AutoFeatureExtractor, SwinForImageClassification
import torchvision
import transformers

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch

import constants_dataset
import ml_helper_training

import ml_helper_proj_params
import constants_ai_h as c
import erik_functions_files
import ml_helper_visualization
import dataset_load_helper
import constants_dataset
import torchvision.transforms as transforms

# turn off gradients during inference for memory effieciency
def get_all_preds(network, dataloader):
    """function to return the number of correct predictions across data set"""
    all_preds = torch.tensor([])
    model = network
    for batch in dataloader:
        images, labels = batch
        preds = model(images)  # get preds
        all_preds = torch.cat((all_preds, preds), dim=0)  # join along existing axis

    return all_preds


p = ml_helper_proj_params.params()


transform_to_tensor = transforms.Compose([transforms.ToTensor()])

dataset = load_dataset("huggingface/cats-image")

DIR_MODEL_CACHE = constants_dataset.DIR_MODEL_CACHE
DIR_DATASET_HUGGINGFACE = constants_dataset.DIR_DATASET_HUGGINGFACE
DIR_DATASET = os.path.join(constants_dataset.BASE_DIR_DATASET, 'ImageNet', 'test')

#dataset = load_dataset("huggingface/cats-image", cache_dir=DIR_DATASET_HUGGINGFACE)
#test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=p[c.NUM_WORKERS], pin_memory=p[c.PIN_MEMORY])


#### Test load images
train_dataset = torchvision.datasets.ImageFolder(root=DIR_DATASET, transform=transform_to_tensor)
train_loader = DataLoader(train_dataset, batch_size=32)


### Load images
images_path = r'C:\ai\datasets\ImageNet\test\cat1'
images = erik_functions_files.load_images_from_folder(images_path)
images_nr = len(images)

#def show_image_grid(images, grid_size_x, grid_size_y = False, labels = False, gray = False ):
#ml_helper_visualization.show_image_grid(images, 3, permutate=False)

#image = dataset["test"]["image"][0]

### Prepare swin model
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", cache_dir=DIR_MODEL_CACHE)

torch.no_grad()

#inputs = feature_extractor(image, return_tensors="pt")
#with torch.no_grad():
#    logits = model(**inputs).logits

#test_dl = DataLoader(dataset['test'], batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
#predictions = get_all_preds(model, test_dl)

pred_labels = []
pred_labels_name = []

label2id, id2label = dataset_load_helper.label_to_id(constants_dataset.FILE_LABELS_IMAGENET1K)

with torch.no_grad():
    for i in range(images_nr):
        inputs = feature_extractor(images[i], return_tensors="pt")
        logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        pred_labels.append(predicted_label)
        pred_labels_name.append(id2label[predicted_label])


print(type(images[0]))

inputs = []
for image in images:
    #input = torch.from_numpy(images)
    inputs.append(input)

#inputs = [torch.from_numpy(item).float for item in images]


ml_helper_visualization.show_image_grid(images, 4, permutate=False, labels=pred_labels_name)
#ml_helper_visualization.show_image(inputs, pred_labels_name)



'''
with torch.no_grad():
    for i in range(len(dataset['test']['image'])):
        logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        pred_labels.append(predicted_label)



for i in range(images_nr):
    print(model.config.id2label[predicted_label] + '  :  ' + str(predicted_label))
'''