# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A vLLM plugin system for applying surgical, runtime patches to vLLM classes without modifying upstream source. Uses vLLM's entry point system (`vllm.general_plugins`) to auto-register patches on vLLM startup.

## Development Commands

```bash
# Install in development mode
pip install -e .

# Build distribution package
python -m build
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
├── setup.py                    # Package setup & entry point registration
├── build.sh                    # Build script
├── CLAUDE.md                   # This file
├── README.md                   # Documentation
├── vllm_custom_plugins/        # Core framework
│   ├── __init__.py             # register_patches() entry point
│   ├── core.py                 # VLLMPatch base class
│   └── plugins/                # Plugin directory
│       ├── __init__.py         # PLUGINS list
│       ├── ops/                # Ascend ops patch (silu_custom)
│       ├── zero_interrupt/     # Zero-interruption inference
│       ├── async_kv_offload/   # Async KV offload
│       ├── security_patch/     # Security patches
│       └── license_verify/     # License verification
└── ascend_custom_ops/          # Ascend custom operators (separate package)
```

## Core Components

| File | Purpose |
|------|---------|
| `vllm_custom_plugins/__init__.py` | `register_patches()` entry point; `PatchManager` class |
| `vllm_custom_plugins/core.py` | `VLLMPatch` base class with `apply()` method |

## Plugin Development Pattern

### plugins/<name>/__init__.py
```python
def register(manager):
    from .patch import MyPatch
    manager.register('MyPatch', MyPatch)
```

### plugins/<name>/patch.py
```python
from vllm_custom_plugins.core import VLLMPatch, min_vllm_version
from vllm.module.to.patch import TargetClass

@min_vllm_version("0.11.0")
class MyPatch(VLLMPatch[TargetClass]):
    def patched_method(self):
        original = getattr(self, '_original_method')
        return original() + " (patched)"
```

## Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `VLLM_CUSTOM_PATCHES` | Comma-separated patch names to apply |
| `VLLM_PATCH_LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR (default: DEBUG) |
| `VLLM_ITS_DEEPSEEK_V4` | `1` selects the DeepSeek-V4 patch family in `plugins/zero_interrupt/deepseekv4/`; default `0` uses the main 0829 implementation |

### Dual patch-family layout (merge-0829)

`plugins/zero_interrupt/` contains two self-contained runtime families:

- Main directory: the `vllm_plugins_0829` implementation (default when
  `VLLM_ITS_DEEPSEEK_V4` is unset/0).
- `deepseekv4/` subdirectory: the `vllm_plugins` DeepSeek-V4 heterogeneous-TP
  implementation (used when `VLLM_ITS_DEEPSEEK_V4=1`).

`zero_interrupt/patch.py` selects the runtime patch family. Whole-file
replacement sources (parallel.py, parallel_state.py, fused_moe config,
kv_cache_utils, ascend worker/parallel_state, rotary, patch_qwen3_5) are
unified files installed by `setup.py` without consulting the environment;
they branch internally at runtime. `patch_qwen3_5.py` is installed as a
runtime dispatcher plus `*_deepseek_v4.py` / `*_0829.py` implementations.



## Adding a New Plugin

1. Create `plugins/<name>/__init__.py` with `register(manager)` function
2. Create `plugins/<name>/patch.py` with patch class
3. Add plugin name to `plugins/__init__.py` PLUGINS list

## Available Plugins

| Plugin | Purpose |
|--------|---------|
| `ops` | Replace SiLU with Ascend ACLNN `silu_custom` operator |
| `zero_interrupt` | Zero-interruption inference with NPU fault handling |
| `async_kv_offload` | Async KV offload for memory optimization |
| `security_patch` | Security patches |
| `license_verify` | License verification |

## Dependencies

- `vllm>=0.9.1`
- `packaging>=20.0`
- `python>=3.11`