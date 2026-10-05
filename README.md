# RS_HW1-1

재현 방법 (itemknn의 경우)

# 1) 의존 패키지 (RecBole의 pip 설치본이 아니라 clone 코드를 썻습니다.)
pip install recbole "ray[tune]" kmeans-pytorch
pip install "numpy<2" "scipy<1.14" "scikit-learn<1.6" "pandas<2.3" // 최신 버전으로는 실행이 안되었습니다.

# 2) RecBole clone
git clone https://github.com/RUCAIBox/RecBole.git

# 3) 이 저장소의 실행 파일을 RecBole 폴더 안에 복사하고, 그 폴더에서 실행하였습니다.
cd RecBole
python itemknn_test.py

# clone 폴더 안에서 실행한 이유

- pip로 설치한 RecBole(1.2.0)은 knn_method 설정을 무시하고 항상 item 기반으로 동작해서, knn_method=user로 돌려도 ItemKNN과 같은 결과가 나왔습니다.
- clone 폴더(1.2.1)에서 실행하면 Python이 그 폴더의 recbole을 먼저 불러와서 knn_method=user가 정상 동작합니다. (pip install -e .는 ray 버전 제약 때문에 설치되지 않아 쓰지 않았습니다.)

# 4) BPR 수정 사항
- 과제는 reg_lambda를 바꿔가며 비교하라고 하지만, RecBole의 기본 BPR에는 이 파라미터가 없습니다. 그래서 recbole/model/general_recommender/bpr.py에 아래 변경을 넣었습니다 (배치에 쓰인 임베딩에 RecBole의 EmbLoss 기반 정규화 항을 더하는 방식)

- import
  from recbole.model.loss import BPRLoss, EmbLoss
  
- __init__ 안, self.loss = BPRLoss() 바로 아래
  self.reg_lambda = config["reg_lambda"] if "reg_lambda" in config else 0.0
  self.reg_loss = EmbLoss()
  
- calculate_loss의 끝
  loss = self.loss(pos_item_score, neg_item_score)
  loss = loss + self.reg_lambda * self.reg_loss(user_e, pos_e, neg_e)
  return loss
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
공통 패치 (모든 실행 파일 맨 위) PyTorch 2.6 이상에서는 RecBole이 저장한 체크포인트를 불러올 때 weights_only 오류가 나서, 아래 3줄을 넣었습니다.

import torch

_orig_load = torch.load

torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
