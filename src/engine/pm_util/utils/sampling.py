import torch
from typing import Tuple
from .tmm import safe_cholesky

@torch.no_grad()
def build_loglik_reference(gmm, n_ref: int = 100_000, batch_size: int = 8192):
    """
    모델로부터 참조 샘플을 생성해 log p(Y) 분포의 CDF를 근사.
    반환: ref_ll_sorted (오름차순, CPU 텐서)
    """
    device = gmm.means_.device
    ref_ll = []
    remain = n_ref
    while remain > 0:
        b = min(batch_size, remain)
        Y = gmm.sample(b)
        ref_ll.append(gmm.score_samples(Y).to("cpu"))
        remain -= b
    ref_ll = torch.cat(ref_ll, dim=0)
    ref_ll_sorted, _ = torch.sort(ref_ll)
    return ref_ll_sorted  # (n_ref,)

@torch.no_grad()
def loglik_to_percentile(gmm, X, ref_ll_sorted: torch.Tensor = None,
                         n_ref: int = 100_000, batch_size: int = 8192):
    """
    각 x에 대해: P = P_Y[ log p(Y) <= log p(x) ]  (모델 기준 경험적 확률)
    반환: percentiles in [0,1]  (값이 클수록 "고밀도 쪽")
    """
    if ref_ll_sorted is None:
        ref_ll_sorted = build_loglik_reference(gmm, n_ref=n_ref, batch_size=batch_size)
    ll = gmm.score_samples(X).to("cpu")  # (N,)
    # 이진탐색으로 순위 계산
    idx = torch.searchsorted(ref_ll_sorted, ll, right=True)
    P = idx.to(torch.float32) / ref_ll_sorted.numel()
    return P  # (N,)

@torch.no_grad()
def responsibility_weighted_chi2_tail(gmm, X):
    """
    각 샘플 x에 대해, 컴포넌트 k의 Mahalanobis^2 = (x-μ_k)^T Σ_k^{-1} (x-μ_k) ~ χ^2_d
    tail_k = 1 - CDF_χ2(d)(m2_k)
    근사 p-value ≈ sum_k γ_k(x) * tail_k
    (정확한 mixture tail은 아님; 해석 가능한 근사)
    """
    from torch.distributions import Chi2

    assert gmm.means_ is not None
    X = X.detach()
    N, d = X.shape
    K = gmm.n_components
    device = X.device

    # responsibilities
    gamma = gmm.predict_proba(X)  # (N,K)

    # Mahalanobis^2 per component
    m2 = torch.empty((N, K), device=device)
    for k in range(K):
        mu = gmm.means_[k]
        cov = gmm.covariances_[0 if getattr(gmm, "tied_covariance", False) else k]
        # 안전한 역행렬 (Cholesky 기반)
        L = torch.linalg.cholesky(cov)
        diff = (X - mu)  # (N,d)
        sol = torch.cholesky_solve(diff.T, L)  # (d,N)
        m2[:, k] = (diff.T * sol).sum(dim=0)

    chi2 = Chi2(df=d)
    # tail = 1 - CDF
    tail = 1.0 - chi2.cdf(m2.clamp_min(0))
    # resp-가중 평균
    pval = (gamma * tail).sum(dim=1)  # (N,)
    return pval  # (N,), 값이 작을수록 더 "이상치"에 가깝다.

@torch.no_grad()
def _logp_and_grad(gmm, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    assert gmm.means_ is not None
    X = X.detach()
    N, d = X.shape
    K = gmm.n_components
    device = X.device

    # log p(x), responsibilities γ
    logp = gmm.score_samples(X)                 # (N,)
    gamma = gmm.predict_proba(X)                # (N,K)

    # Cholesky per component (재사용 가능)
    Ls = []
    covs = gmm.covariances_
    for k in range(K):
        cov_k = covs[0 if getattr(gmm, "tied_covariance", False) else k]
        Ls.append(safe_cholesky(cov_k, gmm.chol_jitter_init, gmm.chol_jitter_max))

    grad = torch.zeros_like(X)
    for k in range(K):
        mu_k = gmm.means_[k]
        Lk   = Ls[k]
        diff = (X - mu_k)                       # (N,d)
        # Σ^{-1}(x-μ) via cholesky_solve
        sol  = torch.cholesky_solve(diff.T, Lk).T   # (N,d)
        grad -= gamma[:, k:k+1] * sol               # 누적
    return logp, grad

@torch.no_grad()
def _project_to_isodensity(gmm, X: torch.Tensor, target_ll: float, n_steps: int = 2) -> torch.Tensor:
    """
    뉴턴 한두 번으로 log p(x)=target_ll 등밀도 곡선에 투영.
    """
    x = X.clone()
    for _ in range(max(1, n_steps)):
        ll, g = _logp_and_grad(gmm, x)
        denom = (g * g).sum(dim=1).clamp_min(1e-12)  # ||∇L||^2
        step  = (ll - target_ll) / denom             # 스칼라
        x = x - step.unsqueeze(1) * g
    return x

@torch.no_grad()
def _hpd_threshold_ll(gmm, conf: float, *, n_ref: int = 500_000, batch_size: int = 16384) -> Tuple[float, torch.Tensor]:
    """
    HPD 경계 임계값 t = F_L^{-1}(1-conf) (L=log p(Y)).
    큰 conf일수록 1-conf가 작으니 n_ref를 크게 잡으세요.
    """
    ref_ll_sorted = build_loglik_reference(gmm, n_ref=n_ref, batch_size=batch_size)  # 오름차순
    q = max(0.0, min(1.0, 1.0 - conf))
    idx = int(q * (ref_ll_sorted.numel() - 1))
    target_ll = ref_ll_sorted[idx].item()
    return target_ll, ref_ll_sorted


@torch.no_grad()
def sample_on_hpd_boundary(
    gmm,
    n: int,
    conf: float = 0.9999,
    *,
    n_ref: int = 500_000,
    helper_batch: int = 32768,
    max_batches: int = 200,
    band_prob: float = 5e-4,   # 경계 주변 '확률구간 폭' (적을수록 더 얇은 띠)
    project_steps: int = 2
) -> torch.Tensor:
    """
    mixture 전체의 HPD(conf) 경계(등밀도) 근처 샘플 n개를 빠르게 생성.
    1) target_ll = F^{-1}(1-conf) 계산
    2) |logp - target_ll|가 매우 작은 샘플만 채택 (band_prob로 밴드 두께 제어)
    3) 선택 샘플을 등밀도로 투영 (뉴턴 1~2회)
    """
    target_ll, ref_sorted = _hpd_threshold_ll(gmm, conf, n_ref=n_ref, batch_size=helper_batch)
    Nref = ref_sorted.numel()
    # target 인덱스 주변의 로그우도 간격으로 밴드 두께 설정 (적응형)
    center_idx = int((1.0 - conf) * (Nref - 1))
    d_idx = max(1, int(band_prob * Nref))
    lo = ref_sorted[max(0, center_idx - d_idx)].item()
    hi = ref_sorted[min(Nref - 1, center_idx + d_idx)].item()

    out = []
    need = n
    for _ in range(max_batches):
        Y = gmm.sample(helper_batch)             # (B,d)
        ll = gmm.score_samples(Y)                # (B,)
        # 얇은 띠에서만 채택
        mask = (ll >= lo) & (ll <= hi)
        if mask.any():
            cand = Y[mask]
            # 등밀도에 투영
            cand = _project_to_isodensity(gmm, cand, target_ll, n_steps=project_steps)
            if cand.shape[0] >= need:
                out.append(cand[:need])
                break
            else:
                out.append(cand)
                need -= cand.shape[0]

    if not out:
        return torch.empty((0, gmm.means_.shape[1]), device=gmm.means_.device)
    return torch.cat(out, dim=0)

@torch.no_grad()
def sample_in_conf_interval(
    gmm,
    n: int,
    conf_low: float,
    conf_high: float,
    *,
    n_ref: int = 500_000,
    helper_batch: int = 32768,
    max_batches: int = 200,
    pad_ll: float = 0.0,   # 경계 여유(로그우도 단위, 선택)
):
    """
    HPD(conf_high)와 HPD(conf_low)의 차집합(밴드)에서 n개 샘플을 수집.
      - conf_low < conf_high (필요시 자동 스왑)
      - 로그우도 임계값: t(c) = F_L^{-1}(1-c)
      - 최종 조건: t_high <= log p(x) <= t_low

    반환: X (n,d), info(dict: thr_low, thr_high, conf_low, conf_high)
    """
    assert 0.0 < conf_low < 1.0 and 0.0 < conf_high < 1.0, "conf_* must be in (0,1)"
    if conf_low > conf_high:
        conf_low, conf_high = conf_high, conf_low

    # 1) 참조 로그우도 분포
    ref_sorted = build_loglik_reference(gmm, n_ref=n_ref, batch_size=helper_batch)  # 오름차순
    Nref = ref_sorted.numel()

    # 2) 두 임계값 (conf_high가 더 바깥쪽 경계 -> 더 낮은 log p)
    idx_lo  = int((1.0 - conf_low)  * (Nref - 1))
    idx_hi  = int((1.0 - conf_high) * (Nref - 1))
    thr_low  = ref_sorted[idx_lo].item()   # inner boundary: 밀도 높음 (log p 큼)
    thr_high = ref_sorted[idx_hi].item()   # outer boundary: 밀도 낮음 (log p 작음)

    # 선택: 경계 완충
    thr_low  = float(thr_low  + pad_ll)
    thr_high = float(thr_high - pad_ll)

    # 3) 거절샘플링
    need, out = n, []
    for _ in range(max_batches):
        Y  = gmm.sample(helper_batch)          # (B,d)
        ll = gmm.score_samples(Y)              # (B,)
        m  = (ll >= thr_high) & (ll <= thr_low)
        if m.any():
            take = Y[m]
            # 약간 섞어서 편향 방지
            if take.shape[0] >= need:
                perm = torch.randperm(take.shape[0], device=take.device)
                out.append(take[perm[:need]])
                need = 0
                break
            else:
                out.append(take)
                need -= take.shape[0]

    if not out:
        return torch.empty((0, gmm.means_.shape[1]), device=gmm.means_.device), {
            "thr_low": thr_low, "thr_high": thr_high,
            "conf_low": conf_low, "conf_high": conf_high,
            "note": "밴드가 너무 얇거나 tail이 얇습니다. helper_batch/max_batches↑ 또는 (conf_low,conf_high) 간격↑/n_ref↑ 권장."
        }

    X = torch.cat(out, dim=0)
    info = {"thr_low": thr_low, "thr_high": thr_high,
            "conf_low": conf_low, "conf_high": conf_high}
    return X, info
