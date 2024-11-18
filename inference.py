import argparse
import subprocess


import modules.stable.gen_img as gen_img
# import modules.stable.gen_img_diffusers as gen_img2
import modules.dev.flux_minimal_inference as flux_gen
import modules.stable.sdxl_gen_img as sdxl_gen
import sys,os
print(sys.path)
import toml
import evaluate.evaluate_uitils as eva

def gen_parser():
    parser = gen_img.setup_parser()
    parser.add_argument(
        "--modelType", action="store_true", help="The base model for inference: flux/sd3/sd",default='sd'
    )
    return parser

def simulated_SD_argv(toml_config):
    task = toml_config.get('task')
    version = toml_config.get('version')
    # 构造模拟的命令行输入参数
    simulated_input = ["inference.py"]  # 第一个参数是脚本名称
    selected_config = {}
    selected_config.update(toml_config.get('general', {}))
    if task == "i2i":
        # 对于 img2img 任务，读取与图像处理相关的配置
        selected_config.update(toml_config.get('img2img', {}))
    if task == "inpainting":
        # 对于 inpainting 任务，读取与文本生成相关的配置
        selected_config.update(toml_config.get('text2img', {}))  #
    if toml_config.get('has_lora'):
        selected_config.update(toml_config.get('lora', {}))
    if toml_config.get('has_controlnet'):
        selected_config.update(toml_config.get('controlnet', {}))

    prompt = selected_config.pop('prompt', "")  # prompt
    negative_prompt = selected_config.pop('negative_prompt', "")  # 同样处理 negative prompt
    simulated_input.extend(["--prompt",prompt+" --n "+negative_prompt])

    simulated_input.append(f"--{version}")
    print(simulated_input)

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
    print(simulated_input)

def simulated_SDXL_argv(toml_config):
    command = ["python", "modules/stable/sdxl_minimal_inference.py"]
    model_params = toml_config.get('SDXL', {})

    # 处理 lora 参数：lora_model_path 和 lora_weight
    lora_model_paths = model_params.pop('lora_model_path', [])  # 使用 pop 从字典中删除，防止被加入到命令行
    lora_weights = model_params.pop('lora_weight', [1.0])  # 同样处理 lora_weight

    if lora_model_paths:
        # 确保 lora_model_paths 和 lora_weights 长度一致
        lora_weights = lora_weights[:len(lora_model_paths)]  # 截取 lora_weights 的前部分以确保长度一致
        lora_params = [f"{path};{weight}" for path, weight in zip(lora_model_paths, lora_weights)]
        model_params.extend(["--lora_weights"] + lora_params)

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
    subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)

def simulated_flux_argv(toml_config):
    simulated_input = ["inference.py"]  # 第一个参数是脚本名称
    selected_config = {}
    selected_config.update(toml_config.get('flux', {}))

    # 处理 lora 参数：lora_model_path 和 lora_weight
    lora_model_paths = selected_config.pop('lora_model_path', [])  # 使用 pop 从字典中删除，防止被加入到命令行
    lora_weights = selected_config.pop('lora_weight', [1.0])  # 同样处理 lora_weight

    # if lora_model_path:
    #     lora_param = f"{lora_model_path};{lora_weight}"
    #     simulated_input.extend(["--lora", lora_param])

    if lora_model_paths:
        # 确保 lora_model_paths 和 lora_weights 长度一致
        lora_weights = lora_weights[:len(lora_model_paths)]  # 截取 lora_weights 的前部分以确保长度一致
        lora_params = [f"{path};{weight}" for path, weight in zip(lora_model_paths, lora_weights)]
        simulated_input.extend(["--lora_weights"] + lora_params)

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


    print("Args:  ------------------------  ",simulated_input)
    # 模拟命令行输入：赋值给 sys.argv
    sys.argv = simulated_input


def simulated_SD3_argv(toml_config):

    command = ["python", "modules/dev/sd3_minimal_inference.py"]
    model_params = toml_config.get('SD3', {})

    # 处理 lora 参数：lora_model_path 和 lora_weight
    lora_model_paths = model_params.pop('lora_model_path', [])  # 使用 pop 从字典中删除，防止被加入到命令行
    lora_weights = model_params.pop('lora_weight', [1.0])  # 同样处理 lora_weight


    if lora_model_paths:
        # 确保 lora_model_paths 和 lora_weights 长度一致
        lora_weights = lora_weights[:len(lora_model_paths)]  # 截取 lora_weights 的前部分以确保长度一致
        lora_params = [f"{path};{weight}" for path, weight in zip(lora_model_paths, lora_weights)]
        model_params.extend(["--lora_weights"] + lora_params)

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
    subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)


def gen_img_SD():

    # parser = gen_parser()
    # args = parser.parse_args()
    # # 处理 negative prompt
    # args.prompt = args.prompt + " --n " + args.negative_prompt
    # gen_img.setup_logging(args, reset=True)
    # # 图片生成
    # gen_img.main(args)


    # 处理 negative prompt
    # args.prompt = args.prompt + " --n " + args.negative_prompt

    parser = gen_img.setup_parser()

    args = parser.parse_args()
    gen_img.main(args)

if __name__ == "__main__":
    config = toml.load("configs/inference.toml")

    model_type = config.get('model_type')


    if model_type is None or model_type == "SD":
        simulated_SD_argv(config)
        print("model_type:   ",model_type)
        gen_img_SD()
        # print(gen_img.ImageName)
        #
        # print(eva.evaluate(gen_img.ImageName))

    # elif model_type == "SDXL":
    #     simulated_SDXL_argv(config)

    elif model_type == "flux":
        print(1231231232)
        simulated_flux_argv(config)
        flux_gen.gen_img()
        print("Flux:DONE!!!!!!!!!!!!!!!!!")

    elif model_type == "SD3":

        print(331133)
        simulated_SD3_argv(config)
        print("SD3:DONE!!!!!!!!!!!!!!!!!")

