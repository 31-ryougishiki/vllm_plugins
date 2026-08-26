import logging


def get_tp_asymmetric_shardings(zero_interrupt_config):
    engine_parallel_config_list = zero_interrupt_config.get('engine_parallel_config', None)
    if not engine_parallel_config_list:
        return []

    executor_id = zero_interrupt_config.get('executor_id', '0')

    config = None
    for engine_parallel_config in engine_parallel_config_list:
        if executor_id == engine_parallel_config.get('executor_id', None):
            config = engine_parallel_config
            break

    if config is None:
        return []

    ori_tp = config["tp"]
    asym_tp = config["new_tp"]


    # 尽量均分, 这里假设head_num >= world_size, head_num 被 world_size 整除
    base = ori_tp // asym_tp
    remainder = ori_tp % asym_tp
    tp_asymmetric_shardings = [base] * asym_tp
    for i in range(remainder):
        tp_asymmetric_shardings[asym_tp - 1 - i] += 1

    return tp_asymmetric_shardings