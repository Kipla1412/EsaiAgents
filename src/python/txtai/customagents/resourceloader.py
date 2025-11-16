from .configloader import ConfigLoader
import importlib.resources as resources

class ConfigResourceLoader:
    @staticmethod
    def load(package, filename):
        with resources.open_text(package, filename) as f:
            return ConfigLoader.load(f.name)
