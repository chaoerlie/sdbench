import argparse
import subprocess
import modules.stable.gen_img as gen_img
# import modules.stable.gen_img_diffusers as gen_img2
import modules.dev.flux_minimal_inference as flux_gen
import modules.stable.sdxl_gen_img as sdxl_gen
import sys,os
import toml # type: ignore


os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

def gen_parser():
    parser = gen_img.setup_parser()
    parser.add_argument(
        "--modelType", action="store_true", help="The base model for inference: flux/sd3/sd",default='sd'
    )
    return parser


def simulated_SD_argv(toml_config):
    # Extract task and version information from the configuration
    task = toml_config.get('task')
    version = toml_config.get('version')

    # Initialize the simulated command-line input, starting with the script name
    simulated_input = ["inference.py"]

    # Select relevant configuration based on the task type
    selected_config = {}
    selected_config.update(toml_config.get('general', {}))  # Always include 'general' configuration

    if task == "i2i":
        selected_config.update(toml_config.get('img2img', {}))  # img2img settings
    elif task == "inpainting":
        selected_config.update(toml_config.get('text2img', {}))  # Text-to-image settings for inpainting

    # Add additional configurations if present
    if toml_config.get('has_lora'):
        selected_config.update(toml_config.get('lora', {}))
    if toml_config.get('has_controlnet'):
        selected_config.update(toml_config.get('controlnet', {}))

    # Extract prompt and negative prompt separately
    prompt = selected_config.pop('prompt', "")
    negative_prompt = selected_config.pop('negative_prompt', "")
    simulated_input.extend(["--prompt", prompt + " --n " + negative_prompt])

    # Append version information
    simulated_input.append(f"--{version}")

    # Add other configuration options as command-line arguments
    for key, value in selected_config.items():
        if isinstance(value, bool) and value:  # Boolean flags (e.g., --xformers)
            simulated_input.append(f"--{key}")
        elif isinstance(value, (str, int, float)):  # String, integer, or float values
            simulated_input.extend([f"--{key}", str(value)])
        elif isinstance(value, list):  # List values (each item as separate argument)
            simulated_input.extend([f"--{key}"] + [str(v) for v in value])

    # Simulate the command-line input by assigning it to sys.argv
    sys.argv = simulated_input

    # Optionally print for debugging
    print(simulated_input)


def simulated_SDXL_argv(toml_config):
    # Initialize the command to run the inference script
    command = ["python", "modules/stable/sdxl_minimal_inference.py"]

    # Get SDXL-related parameters from the config
    model_params = toml_config.get('SDXL', {})

    # Handle lora parameters (model path and weight)
    lora_model_paths = model_params.pop('lora_model_path', [])
    lora_weights = model_params.pop('lora_weight', [1.0])

    if lora_model_paths:
        # Ensure lora_model_paths and lora_weights are of the same length
        lora_weights = lora_weights[:len(lora_model_paths)]
        lora_params = [f"{path};{weight}" for path, weight in zip(lora_model_paths, lora_weights)]
        command.extend(["--lora_weights"] + lora_params)

    # Dynamically add other parameters to the command
    for key, value in model_params.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")  # Add flag for True boolean values
        else:
            command.append(f"--{key}")
            command.append(str(value))  # Add key-value pairs for other types

    # Print the final command for debugging
    print(command)

    # Run the command using subprocess
    subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)


def simulated_flux_argv(toml_config):
    # Initialize the command with the script name
    simulated_input = ["inference.py"]

    # Select the configuration from the 'flux' section in the toml_config
    selected_config = {}
    selected_config.update(toml_config.get('flux', {}))

    # Handle lora parameters (lora_model_path and lora_weight)
    lora_model_paths = selected_config.pop('lora_model_path', [])
    lora_weights = selected_config.pop('lora_weight', [1.0])

    if lora_model_paths:
        # Ensure that lora_model_paths and lora_weights have the same length
        lora_weights = lora_weights[:len(lora_model_paths)]  # Trim lora_weights to match the length of lora_model_paths
        lora_params = [f"{path};{weight}" for path, weight in zip(lora_model_paths, lora_weights)]
        simulated_input.extend(["--lora_weights"] + lora_params)  # Add lora weights to the command

    # Dynamically add other parameters from the selected configuration
    for key, value in selected_config.items():
        if isinstance(value, bool):
            if value:
                simulated_input.append(f"--{key}")  # Add flag for True boolean values
        elif isinstance(value, (str, int, float)):  # Handle string, integer, and float values
            simulated_input.extend([f"--{key}", str(value)])
        elif isinstance(value, list):  # Handle list values by adding each item as a separate argument
            simulated_input.extend([f"--{key}"] + [str(v) for v in value])

    # Print the final command (for debugging purposes)
    print("Args:  ------------------------  ", simulated_input)

    # Simulate the command-line input by assigning it to sys.argv
    sys.argv = simulated_input


def simulated_SD3_argv(toml_config):
    # Initialize the command with the script name
    command = ["python", "modules/dev/sd3_minimal_inference.py"]
    # Select the configuration from the 'flux' section in the toml_config
    model_params = {}
    # Get SD3-related model parameters from the config
    model_params.update(toml_config.get('SD3', {}))

    # Handle lora parameters: lora_model_path and lora_weight
    lora_model_paths = model_params.pop('lora_model_path', [])
    lora_weights = model_params.pop('lora_weight', [1.0])

    if lora_model_paths:
        # Ensure lora_model_paths and lora_weights have the same length
        lora_weights = lora_weights[:len(lora_model_paths)]
        lora_params = [f"{path};{weight}" for path, weight in zip(lora_model_paths, lora_weights)]
        command.extend(["--lora_weights"] + lora_params)  # Add lora weights to the command

    # Dynamically add other parameters to the command
    for key, value in model_params.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")  # Add flag for True boolean values
        else:
            command.append(f"--{key}")
            command.append(str(value))  # Add key-value pairs for other types

    # Print the final command for debugging
    print(command)

    # Run the command using subprocess
    subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)


def gen_img_SD():
    # Setup the argument parser using gen_img's setup_parser method
    parser = gen_img.setup_parser()
    # Parse command-line arguments
    args = parser.parse_args()
    # Call the main function from gen_img with the parsed arguments
    gen_img.main(args)

if __name__ == "__main__":
    # Load the configuration file
    config = toml.load("configs/inference.toml")
    model_type = config.get('model_type')

    # Check the model type and call the corresponding function
    if model_type is None or model_type == "SD":
        simulated_SD_argv(config)  # Simulate SD model argument setup
        print("model_type:   ", model_type)
        gen_img_SD()  # Generate image using SD model
        print("SD:DONE!!!!!!!!!!!!!!!!!")

    elif model_type == "flux":
        print(1231231232)
        simulated_flux_argv(config)  # Simulate Flux model argument setup
        flux_gen.gen_img()  # Generate image using Flux model
        print("Flux:DONE!!!!!!!!!!!!!!!!!")

    elif model_type == "SD3":
        simulated_SD3_argv(config)  # Simulate SD3 model argument setup & Generating image
        print("SD3:DONE!!!!!!!!!!!!!!!!!")

