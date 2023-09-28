from io import TextIOWrapper
import json
import os
from .utils import logdebug

module_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

DEFAULT_CONFIG = {
    'resources_dir': f'{module_parent_dir}/resources',
    'graphs_dir': f'{module_parent_dir}/resources/graph',
    'ele_method': 'opentopodata',
    'ele_api_key': '',
    'routing_method': 'graphhopper',
    'routing_api_key': '',
}

CFG_FNAME = 'route2vel.json'

cfg = DEFAULT_CONFIG.copy()

_loaded = False

def load_config(path = module_parent_dir, file:TextIOWrapper=None, force=False, create_if_missing=True):
    global cfg
    global _loaded

    if not _loaded or force:
        full_path = os.path.join(path, CFG_FNAME)
        if os.path.isfile(full_path) or file:
            cfg.clear()
            newcfg = None
            if file:
                newcfg = json.load(f)
            else:
                with open(full_path) as f:
                    newcfg = json.load(f)

            for key in newcfg:
                cfg[key] = newcfg[key]

            logdebug("Config file loaded:", cfg)

            for key in DEFAULT_CONFIG:
                if key not in cfg:
                    cfg[key] = DEFAULT_CONFIG[key]
                    logdebug(f"Filled missing config key {key}")
        elif not os.path.exists(full_path):
            logdebug("Config file missing, using default...")
            cfg.clear()
            for key in DEFAULT_CONFIG:
                cfg[key] = DEFAULT_CONFIG[key]
            if create_if_missing:
                save_config(path)
        else:
            raise Exception(f"Cfg path {full_path} is a directory!")
        _loaded = True

def save_config(path = ".", file:TextIOWrapper=None):
    global cfg
    full_path = os.path.join(path, CFG_FNAME)

    if file:
        json.dump(cfg, file, indent=2)
        print("Saved config in stream")
    else:
        with open(full_path, 'w') as f:
            json.dump(cfg, f, indent=2)
            print("Saved config at", full_path)