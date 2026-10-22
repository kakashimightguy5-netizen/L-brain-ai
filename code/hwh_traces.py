"""Laptop wrapper: collect activity traces for one checkpoint and run the trace-based analyses H1 H2 H3 H4(trace) H5 H8 H9.
Usage: python hwh_traces.py CKPT TAG     (traces -> LOUT/hw_traces/ [never committed], CSV/PNG -> LOUT/hw_hypotheses_<TAG>/)"""
import os, sys, runpy, time
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
ck, tag = sys.argv[1], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ['HWH_TRACES'] = os.path.join(OUT, 'hw_traces'); os.environ['HWH_OUT'] = os.path.join(OUT, f'hw_hypotheses_{tag}')
os.makedirs(os.environ['HWH_TRACES'], exist_ok=True)
sys.argv = ['collect_traces.py', ck, tag]; runpy.run_path(os.path.join(HERE, 'collect_traces.py'), run_name='__main__')
for s in ('h1_energy_proportionality', 'h2_prefetch', 'h3_hot_neurons', 'h4_groups_trace', 'h5_spike_queue', 'h8_multistream', 'h9_clock_gating'):
    print('==', s, flush=True); sys.argv = [s + '.py', tag]; runpy.run_path(os.path.join(HERE, s + '.py'), run_name='__main__')
print('done', flush=True)
