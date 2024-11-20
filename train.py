import argparse

import modules.stable.train_network as train_network
import sys,os
import toml
import subprocess

def gen_parser():
    parser = train_network.setup_parser()
    parser.add_argument(
        "--modelType", action="store_true", help="The base model for inference: flux/sd3/sd",default='sd'
    )
    return parser


def simulated_argv(toml_config):
    # Initialize the command with the script name
    simulated_input = ["train.py"]

    # Select the configuration for SD model from the TOML file
    selected_config = {}
    selected_config.update(toml_config.get('SD', {}))

    # Dynamically add command-line arguments based on the selected configuration
    for key, value in selected_config.items():
        if isinstance(value, bool):  # Special handling for boolean values
            if value:
                simulated_input.append(f"--{key}")  # Add flag for True boolean values
        elif isinstance(value, (str, int, float)):  # Handle string, integer, or float values
            simulated_input.extend([f"--{key}", str(value)])
        elif isinstance(value, list):  # Handle list values by adding each item as a separate argument
            simulated_input.extend([f"--{key}"] + [str(v) for v in value])

    # Simulate command-line input by assigning the constructed arguments to sys.argv
    sys.argv = simulated_input

def simulated_argv_SDXL(toml_config):

    # Initialize the command with accelerate launch and basic parameters for SDXL model
    command = ["python", "modules/stable/sdxl_train_network.py"]

    # Get SDXL-related model parameters from the configuration
    model_params = toml_config.get('SDXL', {})

    # Dynamically add parameters to the command based on the configuration
    for key, value in model_params.items():
        if isinstance(value, bool):  # Special handling for boolean values
            if value:
                command.append(f"--{key}")  # Add the flag only for True boolean values
        else:
            # For other types (string, int, float), add key-value pairs to the command
            command.append(f"--{key}")
            command.append(str(value))

    # Print the final command for debugging purposes
    print(command)

    # Execute the command using subprocess
    subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)

def simulated_argv_flux(toml_config):
    # Initialize the command with accelerate launch and basic parameters
    command = ["accelerate", "launch", "--mixed_precision", "bf16", "--num_cpu_threads_per_process", "1",
               "modules/dev/flux_train_network.py"]

    # Get flux-related model parameters from the configuration
    model_params = toml_config.get('flux', {})

    # Dynamically add parameters to the command based on the configuration
    for key, value in model_params.items():
        if isinstance(value, bool):  # Special handling for boolean values
            if value:
                command.append(f"--{key}")  # Only add the flag for True boolean values
        else:
            # For other types (string, int, float), add key-value pairs to the command
            command.append(f"--{key}")
            command.append(str(value))

    # Print the final command for debugging purposes
    print(command)

    # Execute the command using subprocess
    subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)



def simulated_argv_SD3(toml_config):
    # Initialize the command with accelerate launch and basic parameters for SD3 model
    command = ["accelerate", "launch", "--mixed_precision", "bf16", "--num_cpu_threads_per_process", "1",
               "modules/dev/sd3_train_network.py"]

    # Get SD3-related model parameters from the configuration
    model_params = toml_config.get('SD3', {})

    # Dynamically add parameters to the command based on the configuration
    for key, value in model_params.items():
        if isinstance(value, bool):  # Special handling for boolean values
            if value:
                command.append(f"--{key}")  # Add the flag only for True boolean values
        else:
            # For other types (string, int, float), add key-value pairs to the command
            command.append(f"--{key}")
            command.append(str(value))

    # Print the final command for debugging purposes
    print(command)

    # Execute the command using subprocess
    subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)


def train_lora_SD():
    # Set up the argument parser using gen_parser
    parser = gen_parser()
    # Parse the command-line arguments
    args = parser.parse_args()
    # Verify the parsed command-line arguments for training
    train_network.train_util.verify_command_line_training_args(args)
    # Read the training configuration from the specified file
    args = train_network.train_util.read_config_from_file(args, parser)
    # Initialize the trainer and start the training process
    trainer = train_network.NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":

    # Load the configuration file for training
    config = toml.load("configs/train_lora.toml")

    # Get the model type from the configuration
    model_type = config.get('model_type')

    # Check the model type and call the corresponding function for training
    if model_type == "SD":
        # Simulate command-line arguments for SD model and start training
        simulated_argv(config)
        train_lora_SD()
        print("model_type:   ", model_type)

    elif model_type == "SDXL":
        # Simulate command-line arguments for SDXL model and print status
        simulated_argv_SDXL(config)
        print(213)
        print("SD3:DONE!!!!!!!!!!!!!!!!!")

    elif model_type == "flux":
        # Simulate command-line arguments for Flux model and print status
        simulated_argv_flux(config)
        print(1231231232)
        print("Flux:DONE!!!!!!!!!!!!!!!!!")

    elif model_type == "SD3":
        # Simulate command-line arguments for SD3 model and print status
        simulated_argv_SD3(config)
        print(213)
        print("SD3:DONE!!!!!!!!!!!!!!!!!")









