import torch
import numpy as np
import albumentations as albu

from datasouls_antispoof.pre_trained_models import create_model
from retinaface.pre_trained_models import get_model
from albumentations.pytorch.transforms import ToTensorV2
from io import BytesIO
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Face Detector Model
model_detector = get_model("resnet50_2020-07-20", max_size=2048, device = device)
model_detector.eval()
# Spoofing Model
model_spoof = create_model("tf_efficientnet_b3_ns")
model_spoof.to(device)
model_spoof.eval();

transform = albu.Compose([albu.PadIfNeeded(min_height=400, min_width=400),
                          #albu.CenterCrop(height=400, width=400),
                          albu.Normalize(p=1),
                          albu.pytorch.ToTensorV2(p=1)], p=1)

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def check_edges_of_bbox(xy_min,xy_max,bbox_size,xy_border):
    if xy_max - xy_min < bbox_size:
        margin = int((bbox_size-(xy_max-xy_min))/2)
        xy_min = xy_min-margin if xy_min - margin >= 0 else 0
        xy_max = xy_max + margin if xy_max + margin <= xy_border else xy_border

    return xy_min, xy_max

def resize_bbox(bbox,image_size):
    bbox_size = 380
    bbox[0], bbox[2] = check_edges_of_bbox(bbox[0], bbox[2],bbox_size,image_size[0])
    bbox[1], bbox[3] = check_edges_of_bbox(bbox[1], bbox[3],bbox_size,image_size[1])

    return bbox

def crop_image(bbox,image):
    x_min, y_min, x_max, y_max = bbox
    x_min = np.clip(x_min, 0, x_max)
    y_min = np.clip(y_min, 0, y_max)

    crop = image[y_min:y_max, x_min:x_max]
    im = Image.fromarray(crop)
    #im.save('test.png') # save cropped image
    return crop

def predict(image: Image.Image, threshold):
    image_size = image.size
    image = np.asarray(image)[..., :3]
    result = model_detector.predict_jsons(image)
    if result[0]['score'] < 0:
        return "No Face found in this IMAGE"

    bbox = resize_bbox(result[0]['bbox'], image_size)
    cropped_image = crop_image(bbox,image)
    with torch.no_grad():
        prediction = model_spoof((torch.unsqueeze(transform(image=cropped_image)['image'], 0)).to(device)).cpu().numpy()[0]

    return {'image_id': 1, 'real': format(prediction[0],".2f"), 'bbox': bbox}
