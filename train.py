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
    # 构造模拟的命令行输入参数
    simulated_input = ["train.py"]  # 第一个参数是脚本名称
    selected_config = {}
    selected_config.update(toml_config.get('SD', {}))
    # 动态添加命令行参数，避免使用 if 判断
    for key, value in selected_config.items():
        # 如果是布尔值且为 True，直接添加相应的命令行选项
        if isinstance(value, bool):
            if value:
                simulated_input.append(f"--{key}")  # 比如 --xformers
            # 其他类型参数以 `--<key>` 的形式添加
        elif isinstance(value, (str, int, float)):  # 支持字符串、整数、浮动值
            simulated_input.extend([f"--{key}", str(value)])
        elif isinstance(value, list):  # 如果是列表类型，将每个元素都加入到命令行参数中
            simulated_input.extend([f"--{key}"] + [str(v) for v in value])

    # 模拟命令行输入：赋值给 sys.argv
    sys.argv = simulated_input

def simulated_argv_flux(toml_config):
    # 基本命令
    command = ["accelerate", "launch","--mixed_precision", "bf16", "--num_cpu_threads_per_process", "1", "modules/dev/flux_train_network.py"]
    model_params = config.get('flux',{})
    # 动态添加参数
    for key, value in model_params.items():
        # 对布尔值进行特殊处理，只在值为 True 时添加参数
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            # 对其他类型的值进行普通处理
            command.append(f"--{key}")
            command.append(str(value))
    print(command)
    subprocess.run(command,stdout=sys.stdout, stderr=sys.stderr)

def train_lora_SD():
    parser = gen_parser()
    args = parser.parse_args()

    train_network.train_util.verify_command_line_training_args(args)
    args = train_network.train_util.read_config_from_file(args, parser)

    trainer = train_network.NetworkTrainer()
    trainer.train(args)

if __name__ == "__main__":

    config = toml.load("configs/train_lora.toml")

    model_type = config.get('model_type')

    if model_type == "SD":
        simulated_argv(config)
        train_lora_SD()
        print("model_type:   ",model_type)

    elif model_type == "flux":

        simulated_argv_flux(config)

        print(1231231232)
        print("Flux:DONE!!!!!!!!!!!!!!!!!")





