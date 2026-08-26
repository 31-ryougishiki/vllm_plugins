import logging
from types import MethodType, ModuleType
from typing import Type, Union
from packaging import version
import vllm

logger = logging.getLogger("vllm_custom_plugins")

PatchTarget = Union[Type, ModuleType]

class VLLMPatch:
    """
    Base class for creating clean, surgical vLLM class patches.

    Usage:
        class MyPatch(VLLMPatch[TargetClass]):
            def new_method(self):
                return "patched behavior"

        MyPatch.apply()
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_patch_target'):
            raise TypeError(
                f"{cls.__name__} must be defined as VLLMPatch[Target]"
            )

    @classmethod
    def __class_getitem__(cls, target: PatchTarget) -> Type:
        if not isinstance(target, (type, ModuleType)):
            raise TypeError(f"Can only patch class or module, not {type(target)}")

        return type(
            f"{cls.__name__}[{target.__name__}]",
            (cls,),
            {'_patch_target': target}
        )

    @classmethod
    def apply(cls):
        """Apply this patch to the target class/module."""
        if cls is VLLMPatch:
            raise TypeError("Cannot directly apply base VLLMPatch class")

        target = cls._patch_target

        # Track applied patches
        if not hasattr(target, '_applied_patches'):
            target._applied_patches = {}

        for name, attr in cls.__dict__.items():
            if name.startswith('_') or name in ('apply',):
                continue

            if name in target._applied_patches:
                existing = target._applied_patches[name]
                raise ValueError(
                    f"{target.__name__}.{name} already patched by {existing}"
                )

            target._applied_patches[name] = cls.__name__

            # Handle class methods
            if isinstance(attr, MethodType):
                attr = MethodType(attr.__func__, target)

            # Save original method to _original_<name> (single underscore to avoid name mangling)
            if hasattr(target, name):
                original_method = getattr(target, name)
                if isinstance(original_method, MethodType):
                    original_method = original_method.__func__
                setattr(target, f'_original_{name}', original_method)

            setattr(target, name, attr)
            action = "Replaced" if hasattr(target, name) else "Added"
            logger.info(f"[x] {cls.__name__} {action} {target.__name__}.{name}")

def min_vllm_version(version_str: str):
    """
    Decorator to specify the minimum vLLM version required for a patch.

    Usage:
        @min_vllm_version("0.9.1")
        class MyPatch(VLLMPatch[SomeClass]):
            pass
    """
    def decorator(cls):
        original_apply = cls.apply

        @classmethod
        def checked_apply(cls):
            # NOTE: [lqf] in build-env, version is dev, and it's invalid
            # provide a fake version
            if vllm.__version__ == "dev":
                current = version.parse("1.0.0")
            else:
                current = version.parse(vllm.__version__)
            minimum = version.parse(version_str)

            if current < minimum:
                logger.warning(
                    f"Skipping {cls.__name__}: requires vLLM >= {version_str}, "
                    f"but current version is {vllm.__version__}"
                )
                return

            original_apply()

        cls.apply = checked_apply
        cls._min_version = version_str
        return cls

    return decorator
