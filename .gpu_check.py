import json, sys
out = {}
try:
    import torch
    import torch.backends.cudnn as cudnn
    out['torch_version'] = torch.__version__
    out['cuda_available'] = torch.cuda.is_available()
    out['gpu_count'] = torch.cuda.device_count() if out['cuda_available'] else 0
    if out['cuda_available'] and out['gpu_count'] > 0:
        i = torch.cuda.current_device()
        out['device_index'] = i
        out['device_name'] = torch.cuda.get_device_name(i)
        out['capability'] = tuple(torch.cuda.get_device_capability(i))
        out['cudnn_available'] = cudnn.is_available()
        out['cudnn_version'] = getattr(cudnn, 'version', lambda: None)()
        x = torch.ones(1, device='cuda')
        out['tiny_tensor'] = float(x.item())
    print(json.dumps(out))
except Exception as e:
    print(json.dumps({'error': repr(e)}))
    sys.exit(1)