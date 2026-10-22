import json, os, time, torch
R = {'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}
if torch.cuda.is_available():
    R['device'] = torch.cuda.get_device_name(0); R['vram_GB'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    a = torch.randn(2048, 2048, device='cuda'); torch.cuda.synchronize(); t = time.time()
    for _ in range(50): a @ a
    torch.cuda.synchronize(); R['matmul_TFLOPS_fp32'] = round(50 * 2 * 2048 ** 3 / (time.time() - t) / 1e12, 2)
R['cpu_threads'] = torch.get_num_threads()
print(json.dumps(R)); json.dump(R, open(os.path.join(os.environ.get('LOUT', '.'), 'gpu_check.json'), 'w'), indent=1)
