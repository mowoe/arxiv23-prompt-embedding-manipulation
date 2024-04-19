import sys
import comet_ml
import platform

if platform.system() == "Linux":
    sys.path.append("/workspace")

from ldm.stable_diffusion import StableDiffusion
import torch
from aesthetic_predictor.simple_inference import AestheticPredictor
from torchvision.transforms import CenterCrop, Resize, Normalize, InterpolationMode
from optimizer.adam_on_lion import AdamOnLion
from best_christmas_present_predictor.present_predictor import ChristmasPredictor
from torch.nn import functional as F
import os
from tqdm.auto import tqdm
from accelerate import Accelerator

seed = 61582
dim = 512

# device = "cpu"
accelerator = Accelerator()
device = accelerator.device

ldm = StableDiffusion(device=device)
aesthetic_predictor = AestheticPredictor(device=device)

christmas_predictor = ChristmasPredictor(device=device)


def compute_blurriness(image):
    # Convert the image to grayscale
    gray_image = (
        0.2989 * image[:, 0, :, :]
        + 0.5870 * image[:, 1, :, :]
        + 0.1140 * image[:, 2, :, :]
    )

    # Compute the Laplacian filter
    # laplacian_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_filter = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    laplacian_filter = laplacian_filter.to(gray_image.device).to(gray_image.dtype)

    # Apply the Laplacian filter to the grayscale image
    filtered_image = F.conv2d(
        gray_image.unsqueeze(0), laplacian_filter.unsqueeze(0).unsqueeze(0)
    )

    # Compute the variance of the Laplacian filter response
    variance = torch.var(filtered_image)

    return variance


def preprocess(rgb):
    rgb = Resize(
        size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None
    )(rgb)
    rgb = CenterCrop(size=(224, 224))(rgb)
    rgb = Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )(rgb)
    return rgb


def laion_aesthetic(image):
    image = preprocess(image)
    image_embedding = aesthetic_predictor.clip.encode_image(image).float()
    image_embedding = aesthetic_predictor.get_features(
        image_embedding, image_input=False
    )
    return aesthetic_predictor.mlp(image_embedding).squeeze()


def get_shifted_embedding(text_embedding, default_std, default_mean):
    shifted_text_embedding = text_embedding / (torch.std(text_embedding)) * default_std
    shifted_text_embedding = (
        shifted_text_embedding - torch.mean(shifted_text_embedding) + default_mean
    )
    return shifted_text_embedding


class GradientDescent(torch.nn.Module):
    def __init__(self, condition):
        super().__init__()
        self.condition_row = condition[:, 0, :]
        self.condition = torch.nn.Parameter(condition[:, 1:, :])
        self.uncondition = torch.nn.Parameter(
            ldm.text_enc([""], condition.shape[1])[:, 1:, :]
        )
        self.condition.requires_grad = True
        self.uncondition.requires_grad = True
        self.latents = None
        self.default_cond_std = torch.std(condition[:, 1:, :])
        self.default_cond_mean = torch.mean(condition[:, 1:, :])
        self.default_uncond_std = torch.std(self.uncondition)
        self.default_uncond_mean = torch.mean(self.uncondition)

    def get_text_embedding(self):
        cond = torch.cat((self.condition_row.unsqueeze(dim=1), self.condition), dim=1)
        uncond = torch.cat(
            (self.condition_row.unsqueeze(dim=1), self.uncondition), dim=1
        )
        return torch.cat([uncond, cond])

    def forward(self, metric, loss_scale, seed=61582, g=7.5, steps=70):
        self.latents = ldm.embedding_2_img(
            self.get_text_embedding(),
            dim=dim,
            seed=seed,
            keep_init_latents=False,
            return_pil=False,
            g=g,
            steps=steps,
            return_latents_step=69,
        )

        image = ldm.latents_to_image(self.latents, return_pil=False)
        if metric == "LAION-Aesthetics V2":
            score = laion_aesthetic(image)
        elif metric == "Christmas Present":
            score = christmas_predictor.get_score(image)
        else:
            score = compute_blurriness(image)

        return score * loss_scale

    def get_optimizer(self, eta):
        # return AdamOnLion(
        #    params=self.parameters(),
        #    lr=eta,
        #    eps=0.001,
        # )
        return torch.optim.AdamW(self.parameters(), lr=eta, eps=0.001)

    def shift_embedding(self, eta):
        self.condition = torch.nn.Parameter(
            get_shifted_embedding(
                self.condition, self.default_cond_std, self.default_cond_mean
            )
        )
        self.uncondition = torch.nn.Parameter(
            get_shifted_embedding(
                self.uncondition, self.default_uncond_std, self.default_uncond_mean
            )
        )
        self.condition.requires_grad = True
        self.uncondition.requires_grad = True
        return self.get_optimizer(eta)


def get_image(seed, iterations, prompt, metric,loss_scale = None):
    if (
        metric == "Sharpness"
        or metric == "LAION-Aesthetics V2"
        or metric == "Christmas Present"
    ):
        optimized_score = -1000.0
        optimized_latents = None
        if loss_scale is None:
            loss_scale = 50
    else:
        optimized_score = 1000.0
        optimized_latents = None
        if loss_scale is None:
            loss_scale = 20

    gradient_descent = GradientDescent(ldm.text_enc([prompt]))
    optimizer = gradient_descent.get_optimizer(0.001)
    os.makedirs(
        f"./output/metric_optimization/{metric}/{prompt[0:45].strip()}/embeddings"
    )
    os.makedirs(f"./output/metric_optimization/{metric}/{prompt[0:45].strip()}/images")
    score_list = list()
    print("iterations:", iterations)
    experiment = comet_ml.Experiment(
      api_key=os.environ.get('COMET_ML_KEY'), project_name="christmas_present", workspace="mowoe"
    )
    experiment.log_parameters({
        "n_iterations":iterations,
        "metric":metric
    })

    gradient_descent, optimizer = accelerator.prepare(gradient_descent, optimizer)
    

    for i in tqdm(range(int(iterations))):
        optimizer.zero_grad()
        score = gradient_descent.forward(metric, loss_scale, seed=int(seed), steps=70)

        score_list.append(score.item())
        torch.save(
            gradient_descent.get_text_embedding(),
            f"./output/metric_optimization/{metric}/{prompt[0:45].strip()}/embeddings/{i}_{prompt[0:45].strip()}.pt",
        )
        pil_image = ldm.latents_to_image(gradient_descent.latents)[0]
        pil_image.save(
            f"output/metric_optimization/{metric}/{prompt[0:45].strip()}/images/{i}_{prompt[0:45].strip()}_{round(score.item(), 4)}.jpg"
        )
        experiment.log_image(pil_image, step=i)
        experiment.log_metric("score",score.item())

        if metric == "Blurriness" and score < optimized_score:
            optimized_score = round(score.item(), 4)
            optimized_latents = torch.clone(gradient_descent.latents)
        elif score > optimized_score:
            optimized_score = round(score.item(), 4)
            optimized_latents = torch.clone(gradient_descent.latents)
        loss = -score

        if metric == "Blurriness":
            loss = score

        if i == 0:
            pil_image = ldm.latents_to_image(gradient_descent.latents)[0]
            pil_image.save(
                f"output/metric_optimization/{metric}/{prompt[0:45].strip()}/initial_{prompt[0:45].strip()}_{round(optimized_score, 4)}.jpg"
            )
        # loss.backward(retain_graph=True)
        accelerator.backward(loss, retain_graph=True)
        optimizer.step()

        if metric == "LAION-Aesthetics V2" or metric == "Christmas Present":
            if (i + 1) % 150:
                optimizer = gradient_descent.shift_embedding(0.005)

    with open(
        f"./output/metric_optimization/{metric}/{prompt[0:45].strip()}/{round(optimized_score, 4)}_output.txt",
        "w",
    ) as file:
        for item in score_list:
            file.write(str(item) + "\n")

    gradient_descent.latents = optimized_latents
    pil_image = ldm.latents_to_image(gradient_descent.latents)[0]
    pil_image.save(
        f"output/metric_optimization/{metric}/{prompt[0:45].strip()}/{prompt[0:45].strip()}_{round(optimized_score, 4)}.jpg"
    )


def increase_blurriness():
    prompt = "a coffee cup filled with magma, digital art, highly detailed, sparks in the background, out of focus background"
    get_image(61582, 7, prompt, "Blurriness", loss_scale = 10)


def increase_sharpness():
    prompt = "a coffee cup filled with magma, digital art, highly detailed, sparks in the background, out of focus background"
    get_image(61582, 7, prompt, "Sharpness", loss_scale = 10)


def increase_aesthetic_score():
    with open(
        "./metric_based_optimization/dataset/LAION-Aesthetics-V2_prompts.txt",
        "r",
        encoding="utf-8",
    ) as file:
        prompts = file.readlines()
        prompts = [
            line.strip() for line in prompts
        ]  # Remove leading/trailing whitespace and newlines
    for prompt in prompts:
        get_image(61582, 7, prompt, "LAION-Aesthetics V2", loss_scale = 1)


def increase_christmas_present_score(n_iterations=500):
    with open(
        "./metric_based_optimization/dataset/christmas-present_prompts.txt",
        "r",
        encoding="utf-8",
    ) as file:
        prompts = file.readlines()
        prompts = [
            line.strip() for line in prompts
        ]  # Remove leading/trailing whitespace and newlines
    for prompt in prompts:
        get_image(61582, n_iterations, prompt, "Christmas Present")


if __name__ == "__main__":
    # increase_sharpness()
    # increase_blurriness()
    # increase_aesthetic_score()
    increase_christmas_present_score(n_iterations=500)
    # Please increase number of iterations from 7 (to, e.g., 400 for the aesthetic score or 50 for the blurriness and sharpness) to get reasonable results.
