# Stable Diffusion China Painting Benchmark

This repository provides a benchmark for image generation using diffusion models, specifically designed for **Chinese painting generation**. The benchmark includes multiple diffusion-based models like Flux, SD3, SDXL and others. It offers various generation methods, such as console-based generation, bash script generation, and config file-based generation. The repository also includes full **inference and training code**, allowing users to customize parameters and model architectures for tailored results. Additionally, we provide a series of **evaluation metrics** specifically designed to assess the performance of models in generating Chinese paintings.

## Features

- **Multiple Models**: Includes popular diffusion models like Flux, SD3, and other variants for Chinese painting generation.
- **Flexible Generation Methods**:
    - Console-based image generation.
    - Bash script-based image generation.
    - Config file-based image generation.
- **Complete Inference and Training Code**: Includes all necessary code for both inference and training, with clear documentation to guide users.
- **Evaluation Metrics**: A set of evaluation methods tailored to assess the generation quality of Chinese paintings.
- **Customizability**: Allows users to modify parameters and adjust model architectures to suit specific needs.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.2.0+
- Dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:
    ``` bash
    git clone https://github.com/chaoerlie/sdbench.git
    cd sdbench
    ```
    
2. Install the required dependencies:
    ``` bash
    pip install -r requirements.txt
    ```
## Inference

1. **Console-based generation**: Use the provided scripts for generating images from the command line:

     ``` bash
    python modules/stable/gen_img_diffusers.py --ckpt stable-diffusion-v1-5/stable-diffusion-v1-5 --outdir outputs --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a --steps 32 --batch_size 4 --images_per_prompt 64 --prompt "Chinese_painting"
    ```
    
1. **Bash script-based generation**: You can also use the provided bash scripts to automate the generation process:
    
    ``` bash
    bash scripts/flux_inf.sh
    ```
    
2. **Config file-based generation**: Modify the `configs/inference.toml` file  to adjust model parameters and settings, and run:
    
    ``` bash
    python inference.py
    ```

    It will automatically read the parameter information from `configs/inference.toml` file and run the inference code

## Training

### Preparing Training Data

Prepare the training image files in any folder (multiple folders are also supported). Supported file formats include `.png`, `.jpg`, `.jpeg`, `.webp`, and `.bmp`. Usually, no preprocessing like resizing is needed.

**Notes:**

- Do not use extremely small images. If the image dimensions are smaller than the training resolution, it is recommended to enlarge them using super-resolution AI.
- Avoid using very large images (e.g., images over 3000 x 3000 pixels), as this may cause errors. It is recommended to resize the images before training.

When training, you need to organize and specify the image data to be used for model training. Depending on the number of training data, training objectives, and whether image captions are provided, you can choose from several ways to specify the training data.

### Dataset Configuration File

Create a text file and change its extension to `.toml`. For example, you can describe the dataset configuration file as follows:

```toml
[general]
enable_bucket = true                        # Whether to use Aspect Ratio Bucketing
# shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = 512                            # Training resolution
batch_size = 1                               # Batch size

  [[datasets.subsets]]
  image_dir = 'train/datasets/chinese'      # Specify the folder containing training images
  caption_extension = '.txt'                # If using .txt files, modify this
  class_tokens = 'chinese painting'         # Specify class tokens
  num_repeats = 5                           # Number of repetitions for training images
#  resolution = 1024
```

**Configuration File Explanation:**
1. **Training Resolution**  
    Specify a single number for a square resolution (e.g., `512` for 512x512), or use two numbers separated by a comma to specify width x height (e.g., `[512,768]` for 512x768). In SD1.x series, the default training resolution is 512. In SD2.x 768 series, the resolution is 768.
2. **Batch Size**  
    Specifies the number of samples to train at once. This depends on the GPU's VRAM size and the training resolution. For tasks like `fine-tuning`, `DreamBooth`, or `LoRA`, batch size will vary, so please refer to the script documentation for adjustments.
3. **Folder Specification**  
    Specify the folder that contains the training data and, optionally, regularization images (if used).
4. **Class Tokens**  
    Used to assign labels to the images for classification. The `class_tokens` defines the category labels for training.
5. **Repetition Count**  
    Specifies how many times each image (and regularization image, if used) should be repeated in training. Detailed logic will be explained below.

**About Repetition Count**
Repetition count is used to adjust the ratio of regularization images to training images. Since there are usually more regularization images than training images, the training image repetition count should be adjusted to ensure a 1:1 ratio of training images to regularization images.
**Formula:**
`Training Image Repeats × Number of Training Images ≥ Regularization Image Repeats × Number of Regularization Images
One epoch of training data is the sum of "Training Image Repeats × Number of Training Images". If there are more regularization images than this, the excess regularization images will not be used.
###  Training

Use the `train.py` script for training.
Run with the following command:
`python train.py`
The parameters will be read from the `train_lora.toml` file.
Below is an example for the `flux` section in the `train_lora.toml` file:
``` toml
[flux]
pretrained_model_name_or_path = "models/flux/flux1-dev.sft"
clip_l = "models/flux/clip_l.safetensors"
t5xxl = "models/flux/t5xxl_fp16.safetensors"
ae = "models/flux/ae.sft"
cache_latents_to_disk = true
save_model_as = "safetensors"
sdpa = true
persistent_data_loader_workers = true
max_data_loader_n_workers = 2
seed = 42
gradient_checkpointing = true
mixed_precision = "bf16"
save_precision = "bf16"
network_module = "networks.lora_flux"
network_dim = 4
optimizer_type = "adamw8bit"
sample_prompts = "chinese painting"
sample_every_n_steps = "500"
learning_rate = 8e-4
cache_text_encoder_outputs = true
cache_text_encoder_outputs_to_disk = true
fp8_base = true
highvram = true
max_train_epochs = 30
save_every_n_epochs = 5
dataset_config = "configs/datasets.toml"
output_dir = "train/output/flux"
output_name = "chinese_painting"
timestep_sampling = "shift"
discrete_flow_shift = 3.1582
model_prediction_type = "raw"
guidance_scale = 1
loss_type = "l2"
logging_dir = "train/logs"

```
For specific training methods, the parameters might differ. Please refer to `configs/train_lora.toml` for more details.

## Models Included

- **Flux**: A model based on state-of-the-art diffusion techniques, optimized for generating high-quality Chinese paintings.
- **SD3**: A custom diffusion model built specifically for the generation of stylized images.

## Customization

- Adjust the model architecture and parameters by modifying the `configs/config.toml` or the training script.
- Customize the generation pipeline by modify the source code


## Contributing

Feel free to open issues or submit pull requests. We welcome any contributions to improve the models or the benchmark!

## License

This project is licensed under the MIT License.
