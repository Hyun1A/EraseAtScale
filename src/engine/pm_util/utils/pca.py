import torch
from typing import Optional, Tuple

def pca_reduce(
    X: torch.Tensor,
    *,
    v: Optional[float] = None,
    k: Optional[int] = None,
    center: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    N×d 텐서 X를 PCA로 축소합니다.
    (v 또는 k 중 정확히 하나만 지정해야 합니다)

    Args:
        X: shape (N, d) 실수 텐서.
        v: 설명가능 분산 비율(target explained variance ratio) in (0, 1]. 누적 비율이 v 이상이 되도록 최소 개수의 성분을 선택.
        k: 가장 큰 고유값 기준 상위 k개의 성분 선택(1 ≤ k ≤ d).
        center: True면 평균을 빼고 진행. 복원 시 같은 평균을 더해 사용.

    Returns:
        Z: 축소된 표현, shape (N, m)
        W: 선택한 고유벡터(주성분)들, shape (d, m), 열 벡터가 성분
        mean: 데이터 평균, shape (d,) (center=False면 0 벡터)
        explained_ratio: 선택한 성분들의 누적 설명가능분산비율, shape (m,)
                         (각 성분까지의 누적 비율; 마지막 값이 최종 누적 비율)
    """
    if (v is None) == (k is None):
        raise ValueError("v 또는 k 중 정확히 하나만 지정하세요.")
    if v is not None and not (0.0 < v <= 1.0):
        raise ValueError("v는 (0, 1] 범위여야 합니다.")
    if k is not None and not (1 <= k <= X.shape[1]):
        raise ValueError("k는 1 이상 d 이하이어야 합니다.")

    X = X.detach()
    device, dtype = X.device, X.dtype
    N, d = X.shape

    # 평균 제거
    if center:
        mean = X.mean(dim=0)
    else:
        mean = torch.zeros(d, device=device, dtype=dtype)
    Xc = X - mean

    # 공분산 행렬 (표본 공분산, N-1로 나눔)
    # d×d 대칭 양의반정치이므로 eigh 사용
    cov = (Xc.t() @ Xc) / max(N - 1, 1)

    # 고유분해: 오름차순으로 나옴
    evals, evecs = torch.linalg.eigh(cov)  # evals: (d,), evecs: (d, d)
    # 내림차순 정렬
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]

    total_var = torch.clamp((evals ** 2).sum(), min=torch.finfo(evals.dtype).eps)
    ratios = evals ** 2 / total_var                                   # 각 성분의 분산 비율
    cum_ratios = torch.cumsum(ratios, dim=0)                     # 누적 비율

    if v is not None:
        m = int((cum_ratios >= v).nonzero(as_tuple=True)[0][0].item() + 1)
    else:
        m = int(k)

    # 최소 1 보장
    m = max(1, min(m, d))

    W = evecs[:, :m]                      # d×m
    Z = Xc @ W                            # N×m
    explained_ratio = cum_ratios[m - 1]      # m×

    return Z, W, mean, explained_ratio

def pca_reconstruct(
    Z: torch.Tensor,
    W: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    축소 표현 Z로부터 원공간으로 복원합니다.

    Args:
        Z: shape (N, m), pca_reduce가 반환한 Z
        W: shape (d, m), pca_reduce가 반환한 고유벡터(주성분) 행렬
        mean: shape (d,), pca_reduce가 반환한 mean (없으면 0으로 간주)

    Returns:
        X_hat: 복원된 데이터, shape (N, d)
    """
    if mean is None:
        mean = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
    # 복원: X̂ = Z Wᵀ + mean
    return Z @ W.T + mean