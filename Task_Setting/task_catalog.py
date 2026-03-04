"""
Task catalog for PES multi-task classification.
Defines all 7 evaluation tasks and supports arbitrary subset selection.
"""

# Full catalog of 7 PES evaluation tasks
# Keys are internal English identifiers, values contain Chinese column names (for label file) and display names
TASK_CATALOG = {
    'mesial_papilla':    {'label_col': '近中牙龈乳头', 'display': 'Mesial Papilla',       'num_classes': 3},
    'distal_papilla':    {'label_col': '远中牙龈乳头', 'display': 'Distal Papilla',       'num_classes': 3},
    'gingival_margin':   {'label_col': '龈缘水平',     'display': 'Gingival Margin Level', 'num_classes': 3},
    'soft_tissue':       {'label_col': '软组织形态',   'display': 'Soft Tissue Contour',   'num_classes': 3},
    'alveolar_defect':   {'label_col': '牙槽突缺损',   'display': 'Alveolar Ridge Defect', 'num_classes': 3},
    'mucosal_color':     {'label_col': '粘膜颜色',     'display': 'Mucosal Color',         'num_classes': 3},
    'mucosal_texture':   {'label_col': '黏膜质地',     'display': 'Mucosal Texture',       'num_classes': 3},
}

# Default 4-task subset (used by most legacy experiments)
DEFAULT_4_TASKS = ['mesial_papilla', 'distal_papilla', 'soft_tissue', 'mucosal_color']

# Full 7-task set
ALL_TASKS = list(TASK_CATALOG.keys())


def resolve_tasks(task_keys):
    """
    Resolve task keys to task info dicts.
    Args:
        task_keys: list of task key strings (e.g. ['mesial_papilla', 'soft_tissue'])
    Returns:
        OrderedDict of {task_key: task_info}
    Raises:
        ValueError if any key is not in TASK_CATALOG
    """
    from collections import OrderedDict
    resolved = OrderedDict()
    for key in task_keys:
        if key not in TASK_CATALOG:
            raise ValueError(f"Unknown task key '{key}'. Available: {list(TASK_CATALOG.keys())}")
        resolved[key] = TASK_CATALOG[key]
    return resolved


def get_label_columns(task_keys):
    """Return list of Chinese column names for the given task keys, in order."""
    tasks = resolve_tasks(task_keys)
    return [info['label_col'] for info in tasks.values()]


def get_display_names(task_keys):
    """Return list of English display names for the given task keys, in order."""
    tasks = resolve_tasks(task_keys)
    return [info['display'] for info in tasks.values()]
