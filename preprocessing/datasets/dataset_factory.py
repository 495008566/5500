from .casia_b import CASIAB
from .oumvlp import OUMVLP
from .gait3d import Gait3D


def get_dataset(name, config_path=None):
    """Get dataset by name.

    Args:
        name: Name of the dataset.
        config_path: Path to the dataset configuration file.

    Returns:
        Dataset instance.
    """
    if name.lower() == 'casia-b':
        return CASIAB(config_path) if config_path else CASIAB()
    elif name.lower() == 'oumvlp':
        return OUMVLP(config_path) if config_path else OUMVLP()
    elif name.lower() == 'gait3d':
        return Gait3D(config_path) if config_path else Gait3D()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_all_datasets(config_paths=None):
    """Get all available datasets.

    Args:
        config_paths: Dictionary mapping dataset names to configuration paths.

    Returns:
        Dictionary of dataset instances.
    """
    config_paths = config_paths or {}
    datasets = {
        'casia-b': CASIAB(config_paths.get('casia-b'))
        if 'casia-b' in config_paths else CASIAB(),
        'oumvlp': OUMVLP(config_paths.get('oumvlp'))
        if 'oumvlp' in config_paths else OUMVLP(),
        'gait3d': Gait3D(config_paths.get('gait3d'))
        if 'gait3d' in config_paths else Gait3D()
    }
    return datasets
