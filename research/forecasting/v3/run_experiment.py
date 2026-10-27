"""Reproducible local orchestration; independent studies use separate processes."""
from research.forecasting.v3.common import CACHE,ROOT
import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

def execute(module,*args):
    name=module.rsplit('.',1)[-1]+'-'+('-'.join(args) or 'run')
    path=CACHE/f'{name}.log'
    with path.open('w') as log:
        subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Completed:',name,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-fetch',action='store_true');args=parser.parse_args()
    prefix='research.forecasting.v3.'
    if not args.skip_fetch:
        execute(prefix+'fleet');execute(prefix+'weather')
    execute(prefix+'prepare')
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks=[pool.submit(execute,prefix+'train_weather',study,'select') for study in ['fleet','evening','regional']]
        for task in tasks:task.result()
    # Every weather setting is now frozen before any new October errors.
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks=[pool.submit(execute,prefix+'train_weather',study,'evaluate') for study in ['fleet','evening','regional']]
        for task in tasks:task.result()
    execute(prefix+'baseline_evening')
    execute(prefix+'adapt_experiment','select');execute(prefix+'adapt_experiment','evaluate')
    execute(prefix+'report');execute(prefix+'verify')

if __name__=='__main__':main()
