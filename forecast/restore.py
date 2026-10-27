"""Restore must succeed before any deployment can replace the public archive."""
import os
from .data import restore

if __name__=='__main__':
    state=restore(os.environ.get('SNAPSHOT_URL','https://huayuansong.github.io/open-current/'))
    print(f'Restored forecast history: {len(state["forecasts"])} issued outlooks.')
