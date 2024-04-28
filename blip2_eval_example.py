import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
display(raw_image.resize((596, 437)))

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
)

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

model.generate({"image": image})

# due to the non-determinstic nature of necleus sampling, you may get different captions.
model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)

model.generate({"image": image, "prompt": "Question: which city is this? Answer:"})

model.generate({
    "image": image,
    "prompt": "Question: which city is this? Answer: singapore. Question: why?"})