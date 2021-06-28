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
                          albu.CenterCrop(height=400, width=400),
                          albu.Normalize(p=1),
                          albu.pytorch.ToTensorV2(p=1)], p=1)

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def crop_image(annotation,image):
    x_min, y_min, x_max, y_max = annotation[0]['bbox']

    x_min = np.clip(x_min, 0, x_max)
    y_min = np.clip(y_min, 0, y_max)
    crop = image[y_min:y_max, x_min:x_max]
    im = Image.fromarray(crop)
    # im.save('test.png') // save cropped image
    return crop

def predict(image: Image.Image, threshold):
    image = np.asarray(image)[..., :3]
    result = model_detector.predict_jsons(image)
    if result[0]['score'] < 0:
        return "No Face found in this IMAGE"

    cropped_image = crop_image(result,image)
    with torch.no_grad():
        prediction = model_spoof((torch.unsqueeze(transform(image=cropped_image)['image'], 0)).to(device)).cpu().numpy()[0]

    return {'image_id': 1, 'real': format(prediction[0],".2f"), 'bbox': result[0]['bbox']}
