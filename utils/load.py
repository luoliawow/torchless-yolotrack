import re, yaml
from pathlib import Path
import sys
SimpleNamespace = type(sys.implementation)

class IterableSimpleNamespace(SimpleNamespace):
    """
    An iterable SimpleNamespace class that provides enhanced functionality for attribute access and iteration.

    This class extends the SimpleNamespace class with additional methods for iteration, string representation,
    and attribute access. It is designed to be used as a convenient container for storing and accessing
    configuration parameters.

    Methods:
        __iter__: Returns an iterator of key-value pairs from the namespace's attributes.
        __str__: Returns a human-readable string representation of the object.
        __getattr__: Provides a custom attribute access error message with helpful information.
        get: Retrieves the value of a specified key, or a default value if the key doesn't exist.

    Examples:
        >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
        >>> for k, v in cfg:
        ...     print(f"{k}: {v}")
        a: 1
        b: 2
        c: 3
        >>> print(cfg)
        a=1
        b=2
        c=3
        >>> cfg.get("b")
        2
        >>> cfg.get("d", "default")
        'default'

    Notes:
        This class is particularly useful for storing configuration parameters in a more accessible
        and iterable format compared to a standard dictionary.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            DEFAULT_CFG_PATH with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)

def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data

def load_track_cfg(cfg):
    return IterableSimpleNamespace(**yaml_load(cfg))