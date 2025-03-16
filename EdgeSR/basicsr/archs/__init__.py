import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'basicsr.archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net


def ditill_build_network(opt):
    """Build two networks (e.g., teacher and student) from options.

    Args:
        opt (dict): Configuration. It must contain:
            teacher_network (dict): Configuration for the teacher network.
            student_network (dict): Configuration for the student network.
    """
    opt = deepcopy(opt)

    # 构建教师网络
    teacher_network_type = opt['teacher_network']['name']
    teacher_network_params = opt['teacher_network']['params']
    teacher_network = ARCH_REGISTRY.get(teacher_network_type)(**teacher_network_params)

    # 构建学生网络
    student_network_type = opt['student_network']['name']
    student_network_params = opt['student_network']['params']
    student_network = ARCH_REGISTRY.get(student_network_type)(**student_network_params)

    logger = get_root_logger()
    logger.info(f'Teacher Network [{teacher_network.__class__.__name__}] is created.')
    logger.info(f'Student Network [{student_network.__class__.__name__}] is created.')

    return teacher_network, student_network

