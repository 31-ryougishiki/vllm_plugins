#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set -e

ACTION="${1:-install}"

case "$ACTION" in
    install)
        # 构建 wheel
        export PIP_NO_INDEX=1
        python3 -m pip wheel --verbose --no-deps --no-build-isolation . -w dist/
        # 安装 wheel 包
        pip3 install dist/*.whl --no-deps
        ;;
    whl)
        # 仅构建 wheel
        export PIP_NO_INDEX=1
        python3 -m pip wheel --verbose --no-deps --no-build-isolation . -w dist/
        ;;
    *)
        echo "Usage: $0 [install|whl]"
        exit 1
        ;;
esac
