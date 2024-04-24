import lightning.fabric
import torch
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from loguru import logger
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    from PIL import Image
import requests
import zipfile
from io import BytesIO

PRETRAINED_MODEL_URL = "https://files.webis.de/teaching/machine-learning-le-ws23/christmas-exercise/models.zip"
"""
!rm -f data.zip
!rm -f models.zip
!wget -q https://files.webis.de/teaching/machine-learning-le-ws23/christmas-exercise/data.zip
!wget -q 
!unzip -o -q data.zip
!unzip -o -q models.zip
"""


def download_pretrained_model():
    response = requests.get(PRETRAINED_MODEL_URL)
    if response.status_code == 200:
        # Extract the contents of the zip file
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            # Specify the directory where you want to extract the contents
            extraction_path = "/tmp/"
            zip_ref.extractall(extraction_path)
            logger.info(f"Zip archive extracted to {extraction_path}")
    else:
        logger.error(
            f"Failed to download zip archive. Status code: {response.status_code}"
        )


class ChristmasPredictor(object):
    def __init__(self, fabric: lightning.fabric.Fabric):
        download_pretrained_model()
        self.encoder, self.preprocess = clip.load("ViT-L/14", jit=False)
        self.n_px = self.encoder.visual.input_resolution
        self.special_preprocessor = Compose(
            [
                Resize(self.n_px, interpolation=BICUBIC),
                CenterCrop(self.n_px),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.pretrained_model = Pretrained()
        self.pretrained_model = fabric.setup(self.pretrained_model)
        self.dtype = torch.float16
        #if self.device == "cpu":
        #    self.dtype = torch.float

    def get_score(self, torch_image: torch.Tensor):
        features = self.encoder.encode_image(
            self.special_preprocessor(torch_image).unsqueeze(0)[0]
        )
        features = features / torch.linalg.norm(features)
        prediction = self.pretrained_model.y(features)
        score = prediction[0][0]
        score = (
                score * 20
        )  # Multiply by 20 to get same value range as aesthetics predictor
        logger.info(f"Christmas Present Score: {score.detach().cpu().numpy().item()}")
        return score


class Pretrained(torch.nn.Module):
    def __init__(self):
        super(Pretrained, self).__init__()
        self.fc1 = torch.nn.Linear(768, 1)
        self.load_state_dict(torch.load("/tmp/models/model.pt"), strict=True)
        self.fc1 = self.fc1.to(torch.float16)

    def y(self, x):
        return self.fc1(x)

    def forward(self):
        c = self.y(self.inp / torch.linalg.norm(self.inp))
        return c
