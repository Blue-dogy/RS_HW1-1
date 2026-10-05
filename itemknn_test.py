import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

from recbole.quick_start import run_recbole

METHOD = 'item'   # 'item' 또는 'user'
K = 5            # 1, 5, 10, 100, 500

r = run_recbole(
    model='ItemKNN',
    dataset='ml-100k',
    config_dict={'k': K, 'knn_method': METHOD, 'show_progress': False},
)

line = f"{METHOD},{K},{dict(r['test_result'])}"
print("RESULT", line)
with open('results.txt', 'a', encoding='utf-8') as f:
    f.write(line + "\n")