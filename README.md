# vLLM Custom Plugins

A **vLLM plugin system** for applying clean, surgical patches to vLLM classes at runtime without modifying upstream source code. Uses vLLM's plugin entry point system to auto-register patches on vLLM startup.

> 本仓为 `vllm_plugins` 与 `vllm_plugins_0829` 的合并结果，模块级合并说明见
> [`MERGE_0829.md`](MERGE_0829.md)。


## Installation

```bash
# Development mode
pip install -e .

# Production mode
pip install .
```

## Architecture

```
vLLM Startup
    ↓
register_patches() [entry point]
    ↓
Plugin.register(manager) → imports and registers each plugin
    ↓
PatchManager.apply_from_env() → applies patches from VLLM_CUSTOM_PATCHES env
```

## Directory Structure

```
vllm_custom_plugins/
├── setup.py                              # Package setup & entry point registration
├── requirements.txt                      # Dependencies
├── vllm_custom_plugins/                  # Main package
│   ├── __init__.py                       # register_patches() entry point
│   ├── core.py                           # VLLMPatch base class
│   └── plugins/                          # Plugin directory
│       ├── __init__.py                   # PLUGINS list
│       └── priority_scheduler/           # Priority-based scheduler patch
│           ├── __init__.py               # register(manager) function
│           └── patch.py                  # Patch implementation
```

## Creating a New Plugin

1. Create `vllm_custom_plugins/plugins/<name>/__init__.py`:

```python
def register(manager):
    """Register this plugin with the patch manager."""
    from .patch import MyPatch
    manager.register('MyPatch', MyPatch)
```

2. Create `vllm_custom_plugins/plugins/<name>/patch.py`:

```python
from vllm_custom_plugins.core import VLLMPatch, min_vllm_version
from vllm.module.to.patch import TargetClass

@min_vllm_version("0.9.1")
class MyPatch(VLLMPatch[TargetClass]):
    def new_method(self):
        return "patched behavior"
```

3. Register plugin in `vllm_custom_plugins/plugins/__init__.py`:

```python
PLUGINS = [
    'priority_scheduler',
    'my_new_plugin',  # Add your plugin here
]
```

4. Apply via environment variable:

```bash
export VLLM_CUSTOM_PATCHES="MyPatch"
```

## Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `VLLM_CUSTOM_PATCHES` | Comma-separated patch names to apply (e.g., `"PriorityScheduler"`) |
| `VLLM_PATCH_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR). Default: DEBUG |
| `VLLM_ITS_DEEPSEEK_V4` | `1` selects the DeepSeek-V4 runtime patch family (`plugins/zero_interrupt/deepseekv4/`); default `0` uses the main 0829 family. Installation no longer depends on this variable |

## Available Plugins

| Plugin | Description |
|--------|-------------|
| `priority_scheduler` | Priority-based request scheduling |

## Building Distribution

```bash
python -m build
```