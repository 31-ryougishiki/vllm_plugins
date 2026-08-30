from setuptools import setup, find_packages
import os
import subprocess
import shutil


# NOTE: [lqf] This is tmp solution to patch rotary_embedding related func
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
    else:
        print(f"WARNING: source patch file not found, skip: {new_path}")


def find_package_dir(pip_name, import_name, package_subdir, import_env=None):
    """查找已安装包路径。

    [merge-0829] 合并两个仓库的查找方式：
    1. 优先使用 0829 分支的 ``pip show`` 方式（不依赖 import，构建时更可靠，
       并优先使用 Editable project location）。
    2. 失败时回退到 vllm_plugins 分支的 ``import <pkg>`` 方式。
    """
    package_path = None

    # 方式1: pip show
    try:
        pip_show_result = subprocess.run(
            ['pip', 'show', pip_name],
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

            if editable_location:
                package_path = os.path.join(editable_location, package_subdir)
                print(f"Found {package_subdir} via pip show (editable): {package_path}")
            elif regular_location:
                package_path = os.path.join(regular_location, package_subdir)
                print(f"Found {package_subdir} via pip show (regular): {package_path}")
    except Exception as e:
        print(f"pip show {pip_name} failed: {e}")

    if package_path and os.path.exists(package_path):
        return package_path

    # 方式2: import fallback
    try:
        result = subprocess.run(
            ['python3', '-c', f'import {import_name}; print({import_name}.__file__)'],
            capture_output=True, text=True, env=import_env
        )
        if result.returncode == 0:
            package_path = os.path.dirname(result.stdout.strip())
            print(f"Found {package_subdir} via import fallback: {package_path}")
    except Exception as e:
        print(f"import {import_name} fallback failed: {e}")

    return package_path


# =============================================================================
# 整文件替换源目录。
#
# 合并后的仓库把 A/B 两个场景的实现统一到了主目录的单个替换文件里；
# 运行期由 VLLM_ITS_DEEPSEEK_V4（以及配置形态）在文件内部选择分支，
# 因此 setup.py 安装阶段不再依赖该环境变量。
# =============================================================================
print("vllm_custom_plugins setup.py: unified runtime-dispatched replacements")

_env = os.environ.copy()
_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

vllm_ascend_path = find_package_dir("vllm-ascend", "vllm_ascend", "vllm_ascend")
vllm_path = find_package_dir("vllm", "vllm", "vllm", import_env=_env)

# 统一后的替换源都在 zero_interrupt 主目录。
zero_interrupt_root = os.path.join(
    os.path.dirname(__file__),
    'vllm_custom_plugins',
    'plugins',
    'zero_interrupt',
)

print("executing vllm_custom_plugins setup.py")
# 替换算子相关patch文件
print("start replacing vllm_ascend's rotary_embedding.py")

if vllm_ascend_path and os.path.exists(vllm_ascend_path):
    print(f"Start replacing rotary_embedding.py...")
    rotary_emb_path = os.path.join(vllm_ascend_path, 'ops', 'rotary_embedding.py')
    src_path = os.path.join(zero_interrupt_root, 'vllm_ascend', 'ops',
                            'triton', 'rotary_embedding.py')
    print(f"src_path={src_path}, rotary_emb_path={rotary_emb_path}")
    replace_file_content(src_path, rotary_emb_path)

    # 替换vllm_ascend中的文件
    print(f"Start replacing parallel_state.py...")
    parallel_state_dest_path = os.path.join(vllm_ascend_path, 'distributed', 'parallel_state.py')
    parallel_state_src_path = os.path.join(zero_interrupt_root, 'vllm_ascend',
                                           'distributed', 'parallel_state.py')
    replace_file_content(parallel_state_src_path, parallel_state_dest_path)

    print(f"Start replacing worker.py...")
    worker_dest_path = os.path.join(vllm_ascend_path, 'worker', 'worker.py')
    worker_src_path = os.path.join(zero_interrupt_root, 'vllm_ascend',
                                   'worker', 'worker.py')
    replace_file_content(worker_src_path, worker_dest_path)

    print(f"Start replacing patch_qwen3_5.py in vllm_ascend...")
    patch_qwen_3_5_src_root = os.path.join(
        zero_interrupt_root, 'vllm_ascend', 'patch', 'worker'
    )
    patch_qwen_3_5_dest_dir = os.path.join(
        vllm_ascend_path, 'patch', 'worker'
    )
    # 安装运行时分发器 + 两份实现，import 时按环境变量选择。
    for src_name, dest_name in (
        ("patch_qwen3_5.py", "patch_qwen3_5.py"),
        (
            "patch_qwen3_5_deepseek_v4.py",
            "patch_qwen3_5_deepseek_v4.py",
        ),
        ("patch_qwen3_5_0829.py", "patch_qwen3_5_0829.py"),
    ):
        replace_file_content(
            os.path.join(patch_qwen_3_5_src_root, src_name),
            os.path.join(patch_qwen_3_5_dest_dir, dest_name),
        )
else:
    print(f"Fail to find vllm_ascend path")

# 替换vllm中的文件
if vllm_path and os.path.exists(vllm_path):
    print(f"Start replacing config.py...")
    config_dest_path = os.path.join(vllm_path, 'model_executor', 'layers', 'fused_moe',
                                    'config.py')
    config_src_path = os.path.join(zero_interrupt_root, 'vllm',
                                    'model_executor', 'layers', 'fused_moe', 'config.py')
    replace_file_content(config_src_path, config_dest_path)

    print(f"Start replacing parallel.py...")
    parallel_dest_path = os.path.join(vllm_path, 'config', 'parallel.py')
    parallel_src_path = os.path.join(zero_interrupt_root, 'vllm', 'config', 'parallel.py')
    replace_file_content(parallel_src_path, parallel_dest_path)

    print(f"Start replacing parallel_state.py...")
    parallel_state_dest_path = os.path.join(vllm_path, 'distributed', 'parallel_state.py')
    parallel_state_src_path = os.path.join(zero_interrupt_root, 'vllm',
                                           'distributed', 'parallel_state.py')
    replace_file_content(parallel_state_src_path, parallel_state_dest_path)

    # 安全 patch 文件为 v0.23.0 同源版本，两个分流下均可安装；
    # 对 origin_0.23.0 而言是等价替换，不会改变默认行为。
    security_root = os.path.join(
        os.path.dirname(__file__),
        'vllm_custom_plugins',
        'plugins',
        'security_patch',
        'vllm',
    )
    print(f"Start run_batch.py...")
    run_batch_dest_path = os.path.join(vllm_path, 'entrypoints', 'openai', 'run_batch.py')
    run_batch_src_path = os.path.join(security_root, 'entrypoints', 'openai', 'run_batch.py')
    replace_file_content(run_batch_src_path, run_batch_dest_path)

    print(f"Start video.py...")
    video_dest_path = os.path.join(vllm_path, 'multimodal', 'media', 'video.py')
    video_src_path = os.path.join(security_root, 'multimodal', 'media', 'video.py')
    replace_file_content(video_src_path, video_dest_path)

    print(f"Start extract_hidden_states.py...")
    extract_hidden_states_dest_path = os.path.join(vllm_path, 'v1', 'spec_decode', 'extract_hidden_states.py')
    extract_hidden_states_src_path = os.path.join(security_root, 'v1', 'spec_decode', 'extract_hidden_states.py')
    replace_file_content(extract_hidden_states_src_path, extract_hidden_states_dest_path)

    print(f"Start envs.py...")
    envs_dest_path = os.path.join(vllm_path, 'envs.py')
    envs_src_path = os.path.join(security_root, 'envs.py')
    replace_file_content(envs_src_path, envs_dest_path)

    print(f"Start sampling_params.py...")
    sampling_params_dest_path = os.path.join(vllm_path, 'sampling_params.py')
    sampling_params_src_path = os.path.join(security_root, 'sampling_params.py')
    replace_file_content(sampling_params_src_path, sampling_params_dest_path)

    print(f"Start replacing kv_cache_utils.py...")
    kv_cache_utils_dest_path = os.path.join(vllm_path, 'v1', 'core', 'kv_cache_utils.py')
    kv_cache_utils_src_path = os.path.join(zero_interrupt_root, 'vllm', 'v1',
                                           'core', 'patch_kv_cache_utils.py')
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
