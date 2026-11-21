"""Small versioned simulator inputs. No network, native import or hardware setup."""
import hashlib
import json
from importlib.resources import files


def _load(name):
    return json.loads(files(__package__).joinpath('data', name).read_text())


def reference_context():
    """Recorded FR3/tool/reference pose, with non-deployable network metadata."""
    home = _load('fr3-reference.json')
    for key in ('home', 'controller_config', 'hardware_settings', 'urdf'):
        value = home[key]
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, allow_nan=False)
        if hashlib.sha256(text.encode()).hexdigest() != home['sha256'][key]:
            raise ValueError(f'Changed reference context: {key}')
    return home


def identified_plant():
    from ..twin_model import PlantParameters
    return PlantParameters(**_load('fr3-local-identified-20260923.json'))


def wrench_observer_config():
    return _load('fr3-wrench-observer.approx-20260923.json')


def nearstatic_noise_config():
    return _load('fr3-kinematic-nearstatic.approx-20260925.json')
