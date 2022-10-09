import os, sys

from ai_helper import vision_classifying
from ai_helper import ml_helper_proj_params
from ai_helper import constants_dataset
from ai_helper import torch_help_functions
from helper import erik_functions_images

from huggingface_hub import notebook_login

torch_help_functions.is_cuda_available()

#notebook_login()

#from huggingface_hub import hf_hub_download
#hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json")


p = ml_helper_proj_params.params()

# swin v1
model_name = r'microsoft/swin-tiny-patch4-window7-224'
model_name = r'microsoft/swin-small-patch4-window7-224'

model_name = r'microsoft/swin-base-patch4-window7-224'
model_name = r'microsoft/swin-base-patch4-window12-384'
model_name = r'microsoft/swin-base-patch4-window7-224-in22k'

model_name = r'microsoft/swin-large-patch4-window7-224'
model_name = r'microsoft/swin-large-patch4-window12-384'
model_name = r'microsoft/swin-large-patch4-window7-224-in22k'
model_name = r'microsoft/swin-large-patch4-window12-384-in22k'

# swin v2
#model_name = r'microsoft/swinv2-tiny-patch4-window8-256'
#model_name = r'microsoft/swinv2-tiny-patch4-window16-256'

#model_name = r'microsoft/swinv2-small-patch4-window8-256'
#model_name = r'microsoft/swinv2-small-patch4-window16-256'pip

model_name = r'microsoft/swinv2-base-patch4-window8-256'
model_name = r'microsoft/swinv2-base-patch4-window12-192-22k'

#model_name = r'microsoft/swinv2-large-patch4-window12-192-22k'

model_name = r'erikejw/swin-tiny-patch4-window7-224-finetuned-eurosat'
model_name = r'erikejw/swin-base-patch4-window7-224-in22k-finetuned-eurosat'

model_name = r'nielsr/swin-tiny-patch4-window7-224-finetuned-eurosat'

#r'C:\ai_experiments\swin\swinv2-base-patch4-window12-192-22k-finetuned-eurosat_3'

#dataset_path = r'C:\ai\datasets\ImageNet\test\cat1'

model_name = r'nielsr/swin-tiny-patch4-window7-224-finetuned-eurosat'
dataset_path = r'C:\ai\datasets\gis\linkoping_small'
path_label_classes = constants_dataset.FILE_LABELS_EUROSAT

#train_dataset = torchvision.datasets.ImageFolder(root=DIR_DATASET, transform=transform_to_tensor)
#train_loader = DataLoader(train_dataset, batch_size=32)

#path_class_labels = constants_dataset.FILE_LABELS_IMAGENET22K

#def classify_swin(p, dataset, model_name, cache_dir = constants_dataset.DIR_MODEL_CACHE, dataset_path = False, show_grid = False, correct_labels = False , type_dataset = 'numpy'):
labels, label_names, path_images = vision_classifying.classify_swin(p, dataset=dataset_path , model_name=model_name, dataset_path=dataset_path, show_grid=True, path_label_classes=path_label_classes)

for i in range(len(labels)):
    print(labels[i], path_images[i])

# group label by category in directorys
dir_save = dataset_path

#def group_images_by_category_into_directorys(labels, label_names, path_images, dir_save):
erik_functions_images.group_images_by_category_into_directorys(labels, label_names, path_images, dir_save)







