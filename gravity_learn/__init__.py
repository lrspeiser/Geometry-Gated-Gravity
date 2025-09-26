# Alias top-level package to inner implementation directory so
# imports like `from gravity_learn.physics import ...` resolve.
# This keeps all code physically under gravity_learn/gravity_learn/
# while exposing it as `gravity_learn` at import time.
import os as _os
__path__ = [_os.path.join(_os.path.dirname(__file__), 'gravity_learn')]
