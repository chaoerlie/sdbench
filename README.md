# Stable Diffusion Chinese Painting Benchmark 111

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
## Usage

### Inference

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

### Training（TODO）

For training a model on your own dataset:

1. Prepare your dataset (make sure the images are formatted correctly).
2. Edit the `config.toml` file to specify the dataset and training parameters.
3. Run the training script:
    

    `python train.py --config config.yaml`
    

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