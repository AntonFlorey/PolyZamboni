import bpy
import importlib
import pkgutil
import os

translations_dict = {}

# Auto-discover and load all language modules in this package
package_dir = os.path.dirname(__file__)
for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    if module_name.startswith("_"):
        continue
    module = importlib.import_module(f".{module_name}", package=__package__)
    if hasattr(module, " translations"):
        translations_dict.update(module.translations)


def register():
    bpy.app.translations.register(__name__, translations_dict)


def unregister():
    bpy.app.translations.unregister(__name__)
