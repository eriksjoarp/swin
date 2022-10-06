from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import os, sys

from ..ai_helper import ml_helper_visualization
from ..ai_helper import erik_functions_files


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

paths_images = erik_functions_files.get_filenames_in_dir(r'C:\ai\datasets\object_detection\test', full_path=True)
#paths_images = erik_functions_files.get_filenames_in_dir(r'C:\ai\datasets\object_detection', full_path=True)
paths_images = erik_functions_files.get_filenames_in_dir(r'C:\ai\datasets\object_detection\mil', full_path=True)

print(paths_images)

images = []
for path in paths_images:
    images.append(Image.open(path))


feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


for image in images:
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

    confidences = []
    labels = []
    boxes = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # let's only keep detections with score > 0.9

        if score > 0.8:
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
            confidences.append(score.item())
            labels.append(model.config.id2label[label.item()])
            boxes.append(box)


    #def display_bounding_boxes(image, boxes, labels, confidences = 'None', path_save_to_disk = False, show = True):
    ml_helper_visualization.display_bounding_boxes(image=image, boxes=boxes, labels=labels, confidences=confidences)
    print()










