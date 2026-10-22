"""Part 1 - energy measurement you can trust (GPU via NVML; CPU via RAPL where the OS exposes it).
Meter() measures one block of work:
  * NVML total-energy counter (nvmlDeviceGetTotalEnergyConsumption, mJ, if supported)  -> J_counter
  * integrated power samples (nvmlDeviceGetPowerUsage) at the fastest practical rate (~5 ms target) -> J_samples
    (the two are cross-checked: rel_diff)
  * temperature, SM/memory clocks and throttle reasons sampled every ~50 ms (start / end / max, union of reasons)
  * above-idle energy = J - idle_W x seconds (idle measured by idle_watts() with the same sampler)
  * CPU package energy via Linux RAPL (/sys/class/powercap/intel-rapl:0/energy_uj) if readable; on Windows: 'not measured'
  * AC/battery state via psutil (if installed)  - measurements should be on mains power
Whole-laptop energy needs a plug-in wattmeter: see bench/energy/WATTMETER.md and wattmeter_template.csv.
All numbers this module returns are MEASURED on the machine it runs on."""
import os, time, threading, sys
try:
    import pynvml
except ImportError:
    import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'nvidia-ml-py']); import pynvml
pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
THROTTLE = {getattr(pynvml, n): n.replace('nvmlClocksThrottleReason', '').replace('nvmlClocksEventReason', '') for n in dir(pynvml)
            if (n.startswith('nvmlClocksThrottleReason') or n.startswith('nvmlClocksEventReason')) and n not in ('nvmlClocksThrottleReasonAll', 'nvmlClocksEventReasonAll')
            and isinstance(getattr(pynvml, n), int) and getattr(pynvml, n) > 0}
def _counter():
    try: return pynvml.nvmlDeviceGetTotalEnergyConsumption(H) / 1000.0
    except Exception: return None
def _rapl():
    p = '/sys/class/powercap/intel-rapl:0/energy_uj'
    try: return int(open(p).read()) / 1e6
    except Exception: return None
def _throttle():
    try: return pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(H)
    except Exception: return 0
def on_mains():
    try:
        import psutil; b = psutil.sensors_battery(); return None if b is None else bool(b.power_plugged)
    except Exception: return None
def gpu_info():
    r = {'name': pynvml.nvmlDeviceGetName(H)}
    for k, f in (('power_limit_W', pynvml.nvmlDeviceGetPowerManagementLimit), ('default_power_limit_W', pynvml.nvmlDeviceGetPowerManagementDefaultLimit)):
        try: r[k] = f(H) / 1000
        except Exception: r[k] = None
    try: mn, mx = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(H); r['power_limit_range_W'] = [mn / 1000, mx / 1000]
    except Exception: r['power_limit_range_W'] = None
    for k, c in (('sm_clock_max_MHz', pynvml.NVML_CLOCK_SM), ('mem_clock_max_MHz', pynvml.NVML_CLOCK_MEM)):
        try: r[k] = pynvml.nvmlDeviceGetMaxClockInfo(H, c)
        except Exception: r[k] = None
    r['energy_counter_supported'] = _counter() is not None; r['rapl_readable'] = _rapl() is not None; r['on_mains'] = on_mains()
    return r
class Meter:
    def __init__(s, idle_W=None, sync=None): s.idle_W, s.sync = idle_W, sync
    def _loop(s):
        last_slow = 0
        while s.run:
            t = time.perf_counter(); s.p.append((t, pynvml.nvmlDeviceGetPowerUsage(H) / 1000.0))
            if t - last_slow > 0.05:
                last_slow = t; s.temp.append(pynvml.nvmlDeviceGetTemperature(H, pynvml.NVML_TEMPERATURE_GPU))
                s.sm.append(pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM)); s.thr |= _throttle()
            time.sleep(0.004)
    def __enter__(s):
        if s.sync: s.sync()
        s.p, s.temp, s.sm, s.thr, s.run = [], [], [], 0, True
        s.t0c, s.t0r, s.t0 = _counter(), _rapl(), time.perf_counter(); s.th = threading.Thread(target=s._loop, daemon=True); s.th.start(); return s
    def __exit__(s, *a):
        if s.sync: s.sync()
        s.dt = time.perf_counter() - s.t0; s.run = False; s.th.join(); s.t1c, s.t1r = _counter(), _rapl()
    def result(s, units=None, unit_name='token'):
        p = s.p; Js = sum((p[i + 1][0] - p[i][0]) * (p[i][1] + p[i + 1][1]) / 2 for i in range(len(p) - 1))
        Jc = s.t1c - s.t0c if s.t0c is not None and s.t1c is not None else None; J = Jc if Jc is not None else Js
        r = {'seconds': round(s.dt, 4), 'J_counter': None if Jc is None else round(Jc, 3), 'J_samples': round(Js, 3),
             'counter_vs_samples_rel_diff': None if Jc is None or Js == 0 else round((Jc - Js) / Js, 4), 'n_power_samples': len(p),
             'avg_W': round(J / s.dt, 2), 'peak_W': round(max(x[1] for x in p), 2) if p else None,
             'temp_start_C': s.temp[0] if s.temp else None, 'temp_end_C': s.temp[-1] if s.temp else None, 'temp_max_C': max(s.temp) if s.temp else None,
             'sm_clock_mean_MHz': round(sum(s.sm) / len(s.sm)) if s.sm else None, 'throttle_reasons': sorted(v for k, v in THROTTLE.items() if s.thr & k),
             'cpu_package_J': None if s.t0r is None or s.t1r is None else round(s.t1r - s.t0r, 3)}
        if s.idle_W is not None: r['J_above_idle'] = round(J - s.idle_W * s.dt, 3)
        if units:
            r[f'J_per_{unit_name}'] = J / units; r[f'mJ_per_{unit_name}'] = round(J / units * 1000, 4)
            if s.idle_W is not None: r[f'mJ_per_{unit_name}_above_idle'] = round(r['J_above_idle'] / units * 1000, 4)
        return r
def idle_watts(sec=8, settle=3):
    time.sleep(settle); m = Meter()
    with m: time.sleep(sec)
    r = m.result(); return r['avg_W'], r
def repeat(fn, n_rep, units, idle_W, sync, unit_name='token'):
    """run fn() n_rep times, each measured separately -> list of results + mean/std summary"""
    import statistics as st
    rs = []
    for _ in range(n_rep):
        m = Meter(idle_W, sync)
        with m: fn()
        rs.append(m.result(units, unit_name))
    key = f'mJ_per_{unit_name}'; keyi = key + '_above_idle'
    summ = {'n': n_rep, key + '_mean': round(st.mean(r[key] for r in rs), 4), key + '_std': round(st.stdev(r[key] for r in rs), 4) if n_rep > 1 else 0,
            keyi + '_mean': round(st.mean(r[keyi] for r in rs), 4), keyi + '_std': round(st.stdev(r[keyi] for r in rs), 4) if n_rep > 1 else 0,
            'seconds_mean': round(st.mean(r['seconds'] for r in rs), 4), 'avg_W_mean': round(st.mean(r['avg_W'] for r in rs), 2),
            'peak_W_max': max(r['peak_W'] or 0 for r in rs), 'temp_max_C': max(r['temp_max_C'] or 0 for r in rs),
            'temp_rise_C': (rs[-1]['temp_end_C'] or 0) - (rs[0]['temp_start_C'] or 0),
            'counter_vs_samples_rel_diff_max': max((abs(r['counter_vs_samples_rel_diff']) for r in rs if r['counter_vs_samples_rel_diff'] is not None), default=None),
            'throttle_reasons': sorted({t for r in rs for t in r['throttle_reasons']})}
    return summ, rs
