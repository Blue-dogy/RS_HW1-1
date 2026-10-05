import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

from recbole.quick_start import run_recbole

EMB = 64     # 32, 64, 128, 256
REG = 0.5     # 0.01, 0.1, 0.5

r = run_recbole(
    model='BPR',
    dataset='ml-100k',
    config_dict={'embedding_size': EMB, 'reg_lambda': REG,
                 'valid_metric': 'NDCG@10', 'show_progress': False},
)
line = f"BPR,{EMB},{REG},{dict(r['test_result'])}"
print("RESULT", line)
with open('results_bpr.txt', 'a', encoding='utf-8') as f:
    f.write(line + "\n")