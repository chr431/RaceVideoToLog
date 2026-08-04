"""tensorrt → tensorrt_bindings 兼容 shim。

tensorrt_cu13_bindings（thin bindings，~1MB）只提供 ``tensorrt_bindings``
顶层模块名，而 rapidocr 的 TRT 引擎与 gpu_setup 的检测都 ``import tensorrt``。
此 shim 把 ``tensorrt`` 解析到同一套 CUDA 13 编译的绑定，满足两者。
（tensorrt 元包被有意排除：会拉入 ~2.2GB 的 tensorrt_libs。）
"""
from tensorrt_bindings import *  # noqa: F401,F403
from tensorrt_bindings import __version__  # noqa: F401
