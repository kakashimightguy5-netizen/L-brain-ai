"""Same as train_L1.py but with finer spike counts (-8..8 instead of -4..4): tests whether spike precision limits larger L."""
import os, sys, runpy
os.environ['SPIKEK'] = '8'; sys.argv = ['train_L1.py'] + sys.argv[1:]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_L1.py'), run_name='__main__')
