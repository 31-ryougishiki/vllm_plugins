from setuptools import setup, find_packages
import os
import subprocess
import shutil


# NOTE: [lqf] This is tmp solution to patch rotarty_embedding related func
def get_package_location(package_name):
    result = subprocess.run(
        ['python3', '-c', f'import {package_name}; print({package_name}.__file__)'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def replace_file_content(new_path, old_path):
    # 备份原文件
    backup_path = old_path + '.bak'
    print(f"backup_path={backup_path}")
    if os.path.exists(old_path) and not os.path.exists(backup_path):
        shutil.copy2(old_path, backup_path)
        print(f"Backed up {old_path} to {backup_path}")

    # 替换文件
    if os.path.exists(new_path):
        shutil.copy2(new_path, old_path)
        print(f"Replaced {old_path}")


# 在 setup() 之后检查并执行替换（保留原文件备份）
# 找到 vllm_ascend & vllm 路径
env = os.environ.copy()
env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

# 尝试多种方式查找 vllm_ascend 路径
vllm_ascend_path = None

# 方式1: 通过 pip show 查找（不依赖 import，构建时更可靠）
# 优先使用 Editable project location（editable mode），否则使用 Location
try:
    pip_show_result = subprocess.run(
        ['pip', 'show', 'vllm-ascend'],
        capture_output=True, text=True
    )
    if pip_show_result.returncode == 0:
        editable_location = None
        regular_location = None
        for line in pip_show_result.stdout.split('\n'):
            if line.startswith('Editable project location:'):
                editable_location = line.split(':', 1)[1].strip()
            elif line.startswith('Location:'):
                regular_location = line.split(':', 1)[1].strip()

        # 优先使用 editable location，需要加上 vllm_ascend 子目录
        if editable_location:
            vllm_ascend_path = os.path.join(editable_location, 'vllm_ascend')
            print(f"Found vllm_ascend via pip show (editable): {vllm_ascend_path}")
        elif regular_location:
            vllm_ascend_path = os.path.join(regular_location, 'vllm_ascend')
            print(f"Found vllm_ascend via pip show (regular): {vllm_ascend_path}")
except Exception as e:
    print(f"pip show vllm-ascend failed: {e}")

# 通过 pip show 查找 vllm 路径
vllm_path = None

try:
    pip_show_vllm_result = subprocess.run(
        ['pip', 'show', 'vllm'],
        capture_output=True, text=True
    )
    if pip_show_vllm_result.returncode == 0:
        editable_location = None
        regular_location = None
        for line in pip_show_vllm_result.stdout.split('\n'):
            if line.startswith('Editable project location:'):
                editable_location = line.split(':', 1)[1].strip()
            elif line.startswith('Location:'):
                regular_location = line.split(':', 1)[1].strip()

        # 优先使用 editable location，需要加上 vllm 子目录
        if editable_location:
            vllm_path = os.path.join(editable_location, 'vllm')
            print(f"Found vllm via pip show (editable): {vllm_path}")
        elif regular_location:
            vllm_path = os.path.join(regular_location, 'vllm')
            print(f"Found vllm via pip show (regular): {vllm_path}")
except Exception as e:
    print(f"pip show vllm failed: {e}")

print("executing vllm_custom_plugins setup.py")
# 替换算子相关patch文件
print("start replacing vllm_ascend's rotary_embedding.py")

if vllm_ascend_path and os.path.exists(vllm_ascend_path):
    print(f"Start replacing rotary_embedding.py...")
    rotary_emb_path = os.path.join(vllm_ascend_path, 'ops', 'rotary_embedding.py')
    src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins', 'plugins', 'vllm_ascend', 'ops',
                            'triton', 'rotary_embedding.py')
    print(f"src_path={src_path}, rotary_emb_path={rotary_emb_path}")
    replace_file_content(src_path, rotary_emb_path)

    # 替换vllm_ascend中的文件
    print(f"Start replacing parallel_state.py...")
    parallel_state_dest_path = os.path.join(vllm_ascend_path, 'distributed', 'parallel_state.py')
    parallel_state_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins', 'plugins',
                                           'zero_interrupt', 'vllm_ascend', 'distributed', 'parallel_state.py')
    replace_file_content(parallel_state_src_path, parallel_state_dest_path)

    print(f"Start replacing worker.py...")
    worker_dest_path = os.path.join(vllm_ascend_path, 'worker', 'worker.py')
    worker_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins',
                                   'plugins', 'zero_interrupt', 'vllm_ascend', 'worker', 'worker.py')
    replace_file_content(worker_src_path, worker_dest_path)

    print(f"Start replacing patch_qwen3_5.py in vllm_ascend...")
    patch_qwen_3_5_dest_path = os.path.join(vllm_ascend_path, 'patch', 'worker', 'patch_qwen3_5.py')
    patch_qwen_3_5_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins', 'plugins',
                                           'zero_interrupt', 'vllm_ascend', 'patch', 'worker', 'patch_qwen3_5.py')
    replace_file_content(patch_qwen_3_5_src_path, patch_qwen_3_5_dest_path)
else:
    print(f"Fail to find vllm_ascend path")

# 替换vllm中的文件
if vllm_path and os.path.exists(vllm_path):
    print(f"Start replacing config.py...")
    config_dest_path = os.path.join(vllm_path, 'model_executor', 'layers', 'fused_moe',
                                    'config.py')
    config_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins',
                                   'plugins', 'zero_interrupt', 'vllm', 'model_executor', 'layers', 'fused_moe',
                                   'config.py')
    replace_file_content(config_src_path, config_dest_path)
    print(f"Start replacing parallel.py...")
    parallel_dest_path = os.path.join(vllm_path, 'config', 'parallel.py')
    parallel_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins',
                                     'plugins', 'zero_interrupt', 'vllm', 'config', 'parallel.py')
    replace_file_content(parallel_src_path, parallel_dest_path)
    print(f"Start replacing parallel_state.py...")
    parallel_state_dest_path = os.path.join(vllm_path, 'distributed', 'parallel_state.py')
    parallel_state_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins',
                                           'plugins', 'zero_interrupt', 'vllm', 'distributed', 'parallel_state.py')
    replace_file_content(parallel_state_src_path, parallel_state_dest_path)

    # print(f"Start run_batch.py...")
    # run_batch_dest_path = os.path.join(os.path.dirname(vllm_path), 'entrypoints', 'openai', 'run_batch.py')
    # run_batch_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins',
    #                                         'plugins', 'security_patch', 'vllm', 'entrypoints', 'openai', 'run_batch.py')
    # replace_file_content(run_batch_src_path, run_batch_dest_path)
    #
    # print(f"Start video.py...")
    # video_dest_path = os.path.join(os.path.dirname(vllm_path), 'multimodal', 'media', 'video.py')
    # video_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins',
    #                                         'plugins', 'security_patch', 'vllm', 'multimodal', 'media', 'video.py')
    # replace_file_content(video_src_path, video_dest_path)
    #
    # print(f"Start extract_hidden_states.py...")
    # extract_hidden_states_dest_path = os.path.join(os.path.dirname(vllm_path), 'v1', 'spec_decode', 'extract_hidden_states.py')
    # extract_hidden_states_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins',
    #                                         'plugins', 'security_patch', 'vllm', 'v1', 'spec_decode', 'extract_hidden_states.py')
    # replace_file_content(extract_hidden_states_src_path, extract_hidden_states_dest_path)
    #
    # print(f"Start envs.py...")
    # envs_dest_path = os.path.join(os.path.dirname(vllm_path), 'envs.py')
    # envs_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins',
    #                                         'plugins', 'security_patch', 'vllm', 'envs.py')
    # replace_file_content(envs_src_path, envs_dest_path)
    #
    # print(f"Start sampling_params.py...")
    # sampling_params_dest_path = os.path.join(os.path.dirname(vllm_path), 'sampling_params.py')
    # sampling_params_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins',
    #                                         'plugins', 'security_patch', 'vllm', 'sampling_params.py')
    # replace_file_content(sampling_params_src_path, sampling_params_dest_path)
    #
    print(f"Start replacing kv_cache_utils.py...")
    kv_cache_utils_dest_path = os.path.join(vllm_path, 'v1', 'core', 'kv_cache_utils.py')
    kv_cache_utils_src_path = os.path.join(os.path.dirname(__file__), 'vllm_custom_plugins', 'plugins',
                                           'zero_interrupt', 'vllm', 'v1', 'core', 'patch_kv_cache_utils.py')
    replace_file_content(kv_cache_utils_src_path, kv_cache_utils_dest_path)

else:
    print(f"Fail to find vllm path")
# end replacing

import sys

if len(sys.argv) == 1:
    # 无命令行参数时，只执行文件替换，不调用 setup()
    print("setup.py file replacement completed.")
    sys.exit(0)

setup(
    name='hw-modelmate-vllm-custom-plugins',
    version='0.2.3',
    description='A vLLM plugin system for surgical runtime patches',
    packages=find_packages(),
    package_data={
        'vllm_custom_plugins': ['py.typed', 'setup.py'],
    },
    install_requires=[
        'vllm>=0.9.1',
        'packaging>=20.0',
    ],
    # Register with vLLM plugin system
    entry_points={
        'vllm.general_plugins': [
            'custom_patches = vllm_custom_plugins:register_patches'
        ]
    },
    python_requires='>=3.11',
)
