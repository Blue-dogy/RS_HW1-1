import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

from recbole.quick_start import run_recbole

MODEL = 'SLIMElastic'    # 'EASE' 또는 'SLIMElastic'

r = run_recbole(
    model=MODEL,
    dataset='ml-100k',
    config_dict={'show_progress': False},
)
line = f"{MODEL},{dict(r['test_result'])}"
print("RESULT", line)
with open('results_item2item.txt', 'a', encoding='utf-8') as f:
    f.write(line + "\n")