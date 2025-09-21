import os
from transformers import CLIPTextModel, CLIPTokenizerFast
from diffusers import (
    UNet2DConditionModel,
    SchedulerMixin,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AltDiffusionPipeline,
    DiffusionPipeline,
)


from typing import List, Sequence, Optional, Tuple
import re
from tqdm import tqdm
import numpy as np
from scipy.stats import chi2, norm
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
import torch
from torch.special import gammaln, digamma, polygamma

import utils
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# 1) 차원 정규화 지표: per-dim log-likelihood, bits-per-dim(BPD)
@torch.no_grad()
def loglik_per_dim(gmm, X, base: str = "e"):
    """
    반환:
      per_sample: (N,) 각 샘플의 [log p(x) / d]  (base=e)
      mean: 스칼라 평균
      bpd: (N,)  bits-per-dim = -log2 p(x) / d   (base=2)
      bpd_mean: 스칼라 평균
    """
    ll = gmm.score_samples(X)                  # (N,) natural log
    d = X.shape[1]
    per = ll / d
    if base == "2":
        per = per / math.log(2.0)

    bpd = -(ll / d) / math.log(2.0)
    return {
        "per_sample": per,
        "mean": per.mean(),
        "bpd": bpd,
        "bpd_mean": bpd.mean(),
    }


# 2) 모델 캘리브레이션: log-likelihood -> [0,1] 확률척도(경험적 CDF 퍼센타일)
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


# 3) "얼마나 멀리 있나?"를 mixture 관점에서 근사 p-value로
#    responsibility-weighted Chi-square 꼬리확률 근사
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
def sample_by_percentile(
    gmm,
    n: int,
    pmin: float = 0.0,
    pmax: float = 0.90,
    mode: str = "between",  # between | outside | above | below
    n_ref: int = 200_000,
    batch_size: int = 8192,
    helper_batch: int = 2048,
    max_batches: int = 1000,
):
    """
    1) 모델로 ref log-likelihood 분포를 만들고,
    2) 주어진 퍼센타일 구간의 log-우도 임계값(thr_low, thr_high)을 구한 뒤,
    3) 해당 구간 조건을 만족하도록 rejection sampling.

    mode:
      - "between": thr_low <= ll <= thr_high
      - "outside": ll <= thr_low 또는 ll >= thr_high
      - "above":   ll >= thr_high (pmax 기준)
      - "below":   ll <= thr_low  (pmin 기준)
    """
    if not (0.0 <= pmin <= 1.0 and 0.0 <= pmax <= 1.0):
        raise ValueError("pmin, pmax must be in [0, 1]")
    if pmin > pmax:
        # 사용 편의상 자동 스왑 (원하면 여기서 에러로 바꿔도 됩니다)
        pmin, pmax = pmax, pmin

    # 1) 참조 분포 (log-likelihood) 구축
    ref_ll = build_loglik_reference(gmm, n_ref=n_ref, batch_size=batch_size)

    # 2) 임계값 계산
    q = torch.tensor([pmin, pmax], device=ref_ll.device, dtype=ref_ll.dtype)
    thr_low, thr_high = torch.quantile(ref_ll, q).tolist()

    out = []
    need = n

    # 3) 샘플링 루프
    for _ in range(max_batches):
        Y = gmm.sample(helper_batch)          # (helper_batch, dim)
        ll = gmm.score_samples(Y)             # (helper_batch,)

        if mode == "between":
            mask = (ll >= thr_low) & (ll <= thr_high)
        elif mode == "outside":
            mask = (ll <= thr_low) | (ll >= thr_high)
        elif mode == "above":
            mask = (ll >= thr_high)
        elif mode == "below":
            mask = (ll <= thr_low)
        else:
            raise ValueError(f"알 수 없는 mode: {mode}")

        if mask.any():
            take = Y[mask]
            if take.shape[0] >= need:
                out.append(take[:need])
                break
            else:
                out.append(take)
                need -= take.shape[0]

    if out:
        return torch.cat(out, dim=0)
    # 비어있을 때의 안전 반환 (장치/차원 일치)
    return torch.empty((0, gmm.means_.shape[1]), device=gmm.means_.device)



def _safe_cholesky(A: torch.Tensor, jitter_init: float = 1e-6, jitter_max: float = 1e-1) -> torch.Tensor:
    """Cholesky가 실패하면 대각선에 지터를 점증적으로 더해 성공시킴."""
    d = A.shape[-1]
    I = torch.eye(d, device=A.device, dtype=A.dtype)
    jitter = jitter_init
    for _ in range(8):
        try:
            return torch.linalg.cholesky(A + jitter * I)
        except RuntimeError:
            jitter *= 10
            if jitter > jitter_max:
                break
    # 최종 안전장치: 고유값 바닥 후 Cholesky
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=jitter_init)
    A_fixed = (evecs * evals) @ evecs.T
    return torch.linalg.cholesky(A_fixed)


def _log_student_t_full_safe(X: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor,
                             nu: float, jitter_init: float, jitter_max: float) -> Tuple[torch.Tensor, torch.Tensor]:

    d = X.shape[1]
    L = _safe_cholesky(scale, jitter_init=jitter_init, jitter_max=jitter_max)  # (d,d)
    diff = X - mean
    sol = torch.cholesky_solve(diff.T, L).T
    m2 = (diff * sol).sum(dim=1).clamp_min(1e-12)           # (N,)
    logdet = 2.0 * torch.log(torch.diag(L)).sum()           # scalar tensor

    # --- 텐서 캐스팅(장치/정밀도 일치) ---
    nu_t   = torch.as_tensor(nu, dtype=X.dtype, device=X.device)
    d_t    = torch.as_tensor(float(d), dtype=X.dtype, device=X.device)
    pi_t   = torch.as_tensor(math.pi, dtype=X.dtype, device=X.device)
    half_t = torch.as_tensor(0.5, dtype=X.dtype, device=X.device)

    c = torch.special.gammaln((nu_t + d_t) * half_t) \
        - torch.special.gammaln(nu_t * half_t) \
        - half_t * (d_t * torch.log(nu_t * pi_t) + logdet)

    logpdf = c - ((nu_t + d_t) * half_t) * torch.log1p(m2 / nu_t)  # (N,)
    return logpdf, m2



def _floor_cov_eig(cov: torch.Tensor, floor: float) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=floor)
    return (evecs * evals) @ evecs.T


def _to_diag(cov: torch.Tensor) -> torch.Tensor:
    return torch.diag(torch.diag(cov))


def _shrinkage(cov: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """(1-alpha)*cov + alpha*target"""
    if alpha <= 0.0:
        return cov
    return (1 - alpha) * cov + alpha * target


def _weighted_cov(X_centered: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # X_centered: (N,d), weights: (N,1)
    return (X_centered.T @ (X_centered * weights)) / (weights.sum() + 1e-12)






@dataclass
class TMMTorch:
    n_components: int
    reg_scale: float = 1e-6         # scale 행렬(Σ) 안정화
    max_iters: int = 200
    tol: float = 1e-4
    verbose: bool = False
    seed: Optional[int] = None

    covariance_type: str = "full"   # "full" | "diag"  (여기서는 'scale' 행렬)
    tied_covariance: bool = False
    shrinkage_alpha: float = 0.0
    scale_floor_scale: float = 1e-3
    min_cluster_weight: float = 1e-3
    max_condition_number: float = 1e6
    chol_jitter_init: float = 1e-6
    chol_jitter_max: float = 1e-1

    # t 분포 자유도 설정
    learn_dof: bool = False
    dof_init: float = 8.0
    min_dof: float = 2.1            # ν>2면 공분산 존재
    max_dof: float = 200.0
    dof_newton_iters: int = 10
    dof_newton_tol: float = 1e-3

    # 학습된 파라미터 (GMM과 호환되는 이름을 사용)
    means_: Optional[torch.Tensor] = None        # (K,d)
    covariances_: Optional[torch.Tensor] = None  # (K,d,d) or (1,d,d) if tied (scale matrix)
    weights_: Optional[torch.Tensor] = None      # (K,)
    dofs_: Optional[torch.Tensor] = None         # (K,)

    # ---------- 초기화 ----------
    def _init_params(self, X: torch.Tensor):
        N, d = X.shape
        g = torch.Generator(device=X.device) if self.seed is not None else None
        if g is not None:
            g.manual_seed(self.seed)
        idx = torch.randperm(N, generator=g, device=X.device)[: self.n_components]
        means = X[idx].clone()  # (K,d)

        diff = X - X.mean(dim=0, keepdim=True)
        global_var = (diff * diff).mean(dim=0) + 1e-3
        base_scale = torch.diag(global_var)      # 초기 scale
        scales = torch.stack([base_scale.clone() for _ in range(self.n_components)], dim=0)  # (K,d,d)
        weights = torch.full((self.n_components,), 1.0 / self.n_components, device=X.device)
        dofs = torch.full((self.n_components,), float(self.dof_init), device=X.device)
        return means, scales, weights, dofs, global_var

    # ---------- 내부 log prob ----------
    def _log_prob_matrix(self, X: torch.Tensor, means: torch.Tensor, scales: torch.Tensor,
                         weights: torch.Tensor, dofs: torch.Tensor) -> torch.Tensor:
        N, d = X.shape
        K = means.shape[0]
        logs = []
        if self.tied_covariance:
            scale = scales[0]
            for k in range(K):
                lp, _ = _log_student_t_full_safe(
                    X.to(scale.device), means[k], scale, float(dofs[k]),
                    self.chol_jitter_init, self.chol_jitter_max
                )
                logs.append(lp + torch.log(weights[k] + 1e-12))
        else:
            for k in range(K):
                lp, _ = _log_student_t_full_safe(
                    X.to(scales[k].device), means[k], scales[k], float(dofs[k]),
                    self.chol_jitter_init, self.chol_jitter_max
                )
                logs.append(lp + torch.log(weights[k] + 1e-12))
        return torch.stack(logs, dim=1)  # (N,K)

    # ---------- 재초기화 ----------
    def _reinit_component(self, X: torch.Tensor, k: int, global_var: torch.Tensor):
        N, d = X.shape
        ridx = torch.randint(0, N, (1,), device=X.device)
        self.means_[k] = X[ridx].squeeze(0)
        base = torch.diag(global_var)
        if self.covariance_type == "diag":
            base = torch.diag(torch.clamp(torch.diag(base), min=1e-6))
        self.covariances_[k] = base
        self.dofs_[k] = max(self.min_dof, float(self.dof_init))

    # ---------- 적합 ----------
    def fit(self, X: torch.Tensor):
        X = X.detach()
        N, d = X.shape
        K = self.n_components

        means, scales, weights, dofs, global_var = self._init_params(X)
        I = torch.eye(d, device=X.device, dtype=X.dtype)
        prev_ll = -torch.inf

        scale_floor = self.scale_floor_scale * (X.var(dim=0, unbiased=False).mean() + 1e-12)

        for it in range(self.max_iters):
            # E-step: resp + u = (ν+d)/(ν+m2)
            log_prob = self._log_prob_matrix(X, means, scales if not self.tied_covariance else scales[:1], weights, dofs)
            max_lp, _ = torch.max(log_prob, dim=1, keepdim=True)
            lse = max_lp + torch.log(torch.exp(log_prob - max_lp).sum(dim=1, keepdim=True))
            log_resp = log_prob - lse
            resp = torch.exp(log_resp)  # (N,K)

            # per-component m2, u, z=E[log W|x] (ν 업데이트용)
            u_list, z_list = [], []
            if self.tied_covariance:
                L = _safe_cholesky(scales[0], self.chol_jitter_init, self.chol_jitter_max)
            for k in range(K):
                scale_k = scales[0] if self.tied_covariance else scales[k]
                if not self.tied_covariance:
                    L = _safe_cholesky(scale_k, self.chol_jitter_init, self.chol_jitter_max)
                diff = X - means[k]                 # (N,d)
                sol  = torch.cholesky_solve(diff.T, L).T
                m2   = (diff * sol).sum(dim=1).clamp_min(1e-12)  # (N,)
                nu_k = dofs[k]
                u_k  = (nu_k + d) / (nu_k + m2)                  # (N,)
                # z_k = E[log W | x] = ψ((ν+d)/2) - log((ν+m2)/2)
                z_k  = digamma((nu_k + d) * 0.5) - torch.log((nu_k + m2) * 0.5)
                u_list.append(u_k)
                z_list.append(z_k)
            U = torch.stack(u_list, dim=1)  # (N,K)
            Z = torch.stack(z_list, dim=1)  # (N,K)

            # M-step
            Nk = resp.sum(dim=0) + 1e-12                 # (K,)
            weights = (Nk / N).clamp_min(1e-12)

            # means
            numer_mu = (resp * U).T @ X                  # (K,d)
            denom_mu = (resp * U).sum(dim=0)[:, None]    # (K,1)
            means = numer_mu / denom_mu.clamp_min(1e-12)

            # scales (scale matrix; diag/full/tied)
            scales_new = []
            for k in range(K):
                Xc = X - means[k]                        # (N,d)
                Wk = (resp[:, k:k+1] * U[:, k:k+1])      # (N,1)
                S_k = (Xc.T @ (Xc * Wk)) / Nk[k]         # (d,d)
                S_k = S_k + self.reg_scale * I

                if self.covariance_type == "diag":
                    S_k = _to_diag(S_k)

                # shrinkage toward diag(global_var)
                target = torch.diag(global_var)
                if self.covariance_type == "diag":
                    target = torch.diag(torch.diag(target))
                S_k = _shrinkage(S_k, target, self.shrinkage_alpha)

                # eigenvalue floor
                S_k = _floor_cov_eig(S_k, float(scale_floor))
                scales_new.append(S_k)
            scales = torch.stack(scales_new, dim=0)      # (K,d,d)

            if self.tied_covariance:
                # Nk 가중합(분자엔 resp*U가 들어가 있으므로 여기선 Nk로)
                S = torch.zeros((d, d), device=X.device, dtype=X.dtype)
                for k in range(K):
                    S = S + (Nk[k] / N) * scales[k]
                S = _shrinkage(S, torch.diag(global_var), self.shrinkage_alpha)
                S = _floor_cov_eig(S, float(scale_floor))
                if self.covariance_type == "diag":
                    S = _to_diag(S)
                scales = S.unsqueeze(0).repeat(K, 1, 1)

            # ν 업데이트(선택)
            if self.learn_dof:
                for k in range(K):
                    nu = float(dofs[k].item())
                    # S_k = 평균(u - z) (Peel & McLachlan 2000의 ECM 업데이트 방정식)
                    Sk = ((resp[:, k] * (U[:, k] - Z[:, k])).sum() / Nk[k]).item()
                    for _ in range(self.dof_newton_iters):
                        # f(nu) = log(nu/2) - ψ(nu/2) + 1 - Sk = 0
                        f  = math.log(nu * 0.5) - digamma(torch.tensor(nu * 0.5)).item() + 1.0 - Sk
                        df = 1.0/nu - 0.5 * polygamma(1, torch.tensor(nu * 0.5)).item()
                        step = f / (df + 1e-12)
                        nu_new = max(self.min_dof, min(self.max_dof, nu - step))
                        if abs(nu_new - nu) < self.dof_newton_tol: 
                            nu = nu_new; break
                        nu = nu_new
                    dofs[k] = torch.tensor(nu, device=X.device, dtype=X.dtype)

            # 붕괴/저랭크 진단
            for k in range(K):
                evals = torch.linalg.eigvalsh(scales[k])
                cond = (evals.max() / (evals.min() + 1e-24)).item()
                too_small = evals.min().item() < float(scale_floor) * 0.5
                too_light = (weights[k].item() < self.min_cluster_weight)
                too_ill = cond > self.max_condition_number
                if too_small or too_light or too_ill:
                    if self.verbose:
                        print(f"[Iter {it+1}] Reinit comp {k}: "
                              f"minEig={evals.min().item():.3e}, w={weights[k].item():.3e}, cond={cond:.2e}")
                    self.means_, self.covariances_, self.weights_, self.dofs_ = means.clone(), scales.clone(), weights.clone(), dofs.clone()
                    self._reinit_component(X, k, global_var)
                    self.weights_[k] = max(self.min_cluster_weight, float(self.weights_[k]))
                    self.weights_ = self.weights_ / self.weights_.sum()
                    means, scales, weights, dofs = self.means_, self.covariances_, self.weights_, self.dofs_

            # 수렴 체크
            curr_ll = lse.mean()
            if self.verbose and (it % 5 == 0 or it == self.max_iters - 1):
                if self.tied_covariance:
                    evals = torch.linalg.eigvalsh(scales[0])
                    msg = (f"[Iter {it+1}] avg ll={curr_ll.item():.6f}, "
                           f"tied minEig={evals.min().item():.3e}, minW={weights.min().item():.3e}")
                else:
                    mins = [torch.linalg.eigvalsh(scales[k]).min().item() for k in range(K)]
                    msg = (f"[Iter {it+1}] avg ll={curr_ll.item():.6f}, "
                           f"minEig={min(mins):.3e}, minW={weights.min().item():.3e}")
                print(msg)

            if torch.abs(curr_ll - prev_ll) < self.tol:
                if self.verbose:
                    print("Converged.")
                break
            prev_ll = curr_ll

        self.means_ = means
        self.covariances_ = scales if not self.tied_covariance else scales.clone()
        self.weights_ = weights
        self.dofs_ = dofs
        return self

    # ---------- 예측/점수 ----------
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        assert self.means_ is not None, "Call fit() first."
        X = X.detach()
        log_prob = self._log_prob_matrix(
            X, self.means_, self.covariances_ if not self.tied_covariance else self.covariances_[:1],
            self.weights_, self.dofs_
        )
        max_lp, _ = torch.max(log_prob, dim=1, keepdim=True)
        lse = max_lp + torch.log(torch.exp(log_prob - max_lp).sum(dim=1, keepdim=True))
        return torch.exp(log_prob - lse)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_proba(X), dim=1)

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        assert self.means_ is not None, "Call fit() first."
        X = X.detach()
        N, d = X.shape
        K = self.n_components
        logs = []
        if self.tied_covariance:
            scale = self.covariances_[0]
            for k in range(K):
                lp, _ = _log_student_t_full_safe(
                    X.to(scale.device), self.means_[k], scale, float(self.dofs_[k]),
                    self.chol_jitter_init, self.chol_jitter_max
                )
                logs.append(lp + torch.log(self.weights_[k] + 1e-12))
        else:
            for k in range(K):
                lp, _ = _log_student_t_full_safe(
                    X.to(self.covariances_[k].device), self.means_[k], self.covariances_[k], float(self.dofs_[k]),
                    self.chol_jitter_init, self.chol_jitter_max
                )
                logs.append(lp + torch.log(self.weights_[k] + 1e-12))
        log_prob = torch.stack(logs, dim=1)  # (N,K)
        max_lp, _ = torch.max(log_prob, dim=1, keepdim=True)
        lse = max_lp + torch.log(torch.exp(log_prob - max_lp).sum(dim=1, keepdim=True))
        return lse.squeeze(1)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """
        x = μ + (L @ z) / sqrt(g),   z~N(0,I),  g~Gamma(ν/2, rate=ν/2)
        (scale 행렬의 촐레스키 L)
        """
        assert self.means_ is not None, "Call fit() first."
        d = self.means_.shape[1]
        comp_idx = torch.multinomial(self.weights_, num_samples=n, replacement=True)
        X = torch.zeros((n, d), device=self.means_.device, dtype=self.means_.dtype)
        for k in range(self.n_components):
            mask = (comp_idx == k)
            m = int(mask.sum().item())
            if m == 0:
                continue
            S_k = self.covariances_[k if not self.tied_covariance else 0]
            L = _safe_cholesky(S_k, jitter_init=self.chol_jitter_init, jitter_max=self.chol_jitter_max)
            z = torch.randn((m, d), device=self.means_.device, dtype=self.means_.dtype)
            g = torch.distributions.Gamma(self.dofs_[k]*0.5, self.dofs_[k]*0.5).sample((m,)).to(self.means_.device)  # rate=ν/2
            X[mask] = self.means_[k] + (z @ L.T) / torch.sqrt(g).unsqueeze(1)
        return X

    # ---------- 진단 ----------
    def diagnostics(self) -> dict:
        assert self.means_ is not None, "Call fit() first."
        K = self.n_components
        mins, conds = [], []
        for k in range(K):
            evals = torch.linalg.eigvalsh(self.covariances_[0 if self.tied_covariance else k])
            mins.append(evals.min().item())
            conds.append((evals.max() / (evals.min() + 1e-24)).item())
        return {
            "min_eigenvalue": float(min(mins)),
            "max_condition_number": float(max(conds)),
            "min_weight": float(self.weights_.min().item()),
            "min_dof": float(self.dofs_.min().item()),
        }




def plot_tmm_2d(
    X: torch.Tensor,
    gmm: TMMTorch,
    ax: Optional[plt.Axes] = None,
    grid_size: int = 200,
    levels: int = 10,
    point_alpha: float = 0.6,
    ell_scale: float = 2.0,
):
    """
    시각화 (d=2 전용):
      - 데이터 산점도 (클러스터 색은 책임도 argmax)
      - 혼합분포 log-density 등고선
      - 각 가우시안의 공분산 타원(고유값/고유벡터 기반, ell_scale 표준편차 배수)
    """
    assert X.shape[1] == 2, "plot_gmm_2d는 d=2에서만 사용 가능합니다."
    assert gmm.means_ is not None, "먼저 gmm.fit(X)를 호출하세요."

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    device = X.device
    X_cpu = X.detach().to("cpu")
    means = gmm.means_.detach().to("cpu")
    covs = gmm.covariances_.detach().to("cpu")
    weights = gmm.weights_.detach().to("cpu")

    # 산점도 (하드 클러스터)
    hard = gmm.predict(X).to("cpu")
    ax.scatter(X_cpu[:, 0].numpy(), X_cpu[:, 1].numpy(), c=hard.numpy(), s=12, alpha=point_alpha)

    # 등고선용 grid
    x_min, x_max = X_cpu[:, 0].min().item(), X_cpu[:, 0].max().item()
    y_min, y_max = X_cpu[:, 1].min().item(), X_cpu[:, 1].max().item()
    x_pad = 0.1 * (x_max - x_min + 1e-6)
    y_pad = 0.1 * (y_max - y_min + 1e-6)
    x = torch.linspace(x_min - x_pad, x_max + x_pad, grid_size)
    y = torch.linspace(y_min - y_pad, y_max + y_pad, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(device)

    # log p(x) on grid
    with torch.no_grad():
        logp = gmm.score_samples(grid).to("cpu").reshape(grid_size, grid_size)
    cs = ax.contour(xx.numpy(), yy.numpy(), logp.numpy(), levels=levels, linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=8)

    # 공분산 타원
    for k in range(gmm.n_components):
        cov = covs[k].numpy()
        mu = means[k].numpy()
        # 고유분해
        vals, vecs = torch.linalg.eigh(covs[k])
        vals = vals.numpy()
        vecs = vecs.numpy()
        # 가장 큰 고유값이 첫 번째가 되도록 정렬
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        # 타원 파라미터
        angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))  # 첫 번째 고유벡터 각도
        width, height = 2 * ell_scale * (vals ** 0.5)            # 표준편차 * 스케일 * 2 (지름)
        ell = Ellipse(xy=(mu[0], mu[1]), width=width, height=height, angle=angle, fill=False, lw=2)
        ax.add_patch(ell)
        ax.scatter([mu[0]], [mu[1]], marker="x", s=80)

    ax.set_title("TMM (scatter + log-density contours + covariance ellipses)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_xticks(range(-45, 46, 15))
    ax.set_yticks(range(-45, 46, 15))

    plt.tight_layout()
    return ax






# @torch.no_grad()
# def _take_cov_for_index(covs: torch.Tensor, idx: torch.Tensor, tied: bool) -> torch.Tensor:
#     if tied:
#         return covs[0].unsqueeze(0).expand(idx.shape[0], -1, -1)
#     else:
#         return covs[idx]


# @torch.no_grad()
# def move_samples_to_confidence(
#     X: torch.Tensor,
#     gmm,
#     *,
#     conf: Optional[float] = 0.95, 
#     kappa: Optional[float] = None,
#     eps: float = 1e-8
# ) -> Tuple[torch.Tensor, dict]:

#     assert gmm.means_ is not None
#     device = X.device
#     means = gmm.means_
#     covs = gmm.covariances_
#     tied = bool(getattr(gmm, "tied_covariance", False))

#     gamma = gmm.predict_proba(X)       # (N,K)
#     k_idx = torch.argmax(gamma, dim=1) # (N,)

#     mu = means[k_idx]                               # (N,d)
#     cov = _take_cov_for_index(covs, k_idx, tied)    # (N,d,d)

#     v = X - mu                                      # (N,d)
#     v_norm = v.norm(dim=1, keepdim=True)            # (N,1)

#     u = torch.empty_like(v)
#     nonzero = (v_norm.squeeze(1) > eps)
#     if nonzero.any():
#         u[nonzero] = v[nonzero] / v_norm[nonzero]

#     if (~nonzero).any():
#         nz_idx = (~nonzero).nonzero(as_tuple=False).squeeze(1)
#         for i in nz_idx.tolist():
#             evals, evecs = torch.linalg.eigh(cov[i])
#             j = torch.argmax(evals)
#             u[i] = evecs[:, j]

#     Sigma_u = torch.bmm(cov, u.unsqueeze(-1)).squeeze(-1) # (N,d)
#     sigma = torch.sqrt(torch.clamp((u * Sigma_u).sum(dim=1), min=1e-12))  # (N,)

#     if kappa is not None:
#         z = float(kappa)
#     else:
#         assert conf is not None and 0 < conf < 1.0, "conf in (0,1)"
#         z = float(norm.ppf(0.5 * (1.0 + conf)))

#     r = z * sigma  # (N,)

#     X_new = mu + (r.unsqueeze(1) * u)

#     info = {
#         "k": k_idx,
#         "sigma": sigma,
#         "z": z,
#         "u": u,
#     }
#     return X_new, info














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
        Ls.append(_safe_cholesky(cov_k, gmm.chol_jitter_init, gmm.chol_jitter_max))

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
def sample_outside_kappa(
    gmm,
    n: int,
    kappa: float,
    *,
    n_ref: int = 500_000,       # κ가 클수록 크게
    helper_batch: int = 32768,  # 속도 ↑
    max_batches: int = 200,
):
    """
    log p 분포 L의 z-score가 κ 이상(= L <= μ_L - κσ_L)인 샘플을 n개 수집.
    혼합분포 전체 기준이라 멀티모달에서도 일관됨.
    반환: X (n,d), info(dict: target_ll, tail_prob, mu_L, sigma_L, kappa)
    """
    # 1) 참조 분포로 μ_L, σ_L 추정
    ref_sorted = build_loglik_reference(gmm, n_ref=n_ref, batch_size=helper_batch)  # 오름차순
    mu_L = ref_sorted.mean().item()
    sigma_L = ref_sorted.std(unbiased=False).item()
    target_ll = mu_L - float(kappa) * sigma_L

    # 경험적 꼬리확률 P(L <= t) (= 'κ 이상'의 전체 질량)
    idx = torch.searchsorted(ref_sorted, torch.tensor(target_ll, device=ref_sorted.device), right=True).item()
    tail_prob = idx / ref_sorted.numel()

    # 2) 거절 샘플링: log p(x) <= target_ll 인 것만 모두 채택
    need = n
    out = []
    for _ in range(max_batches):
        Y  = gmm.sample(helper_batch)
        ll = gmm.score_samples(Y)
        m  = (ll <= target_ll)  # 'κ 이상' 조건을 만족하는 모든 점
        if m.any():
            take = Y[m]
            if take.shape[0] >= need:
                out.append(take[:need]); break
            else:
                out.append(take); need -= take.shape[0]

    if not out:
        return torch.empty((0, gmm.means_.shape[1]), device=gmm.means_.device), {
            "target_ll": target_ll, "tail_prob": tail_prob,
            "mu_L": mu_L, "sigma_L": sigma_L, "kappa": kappa,
            "note": "채택률이 0입니다. helper_batch/max_batches↑ 또는 kappa↓/n_ref↑ 권장."
        }

    X = torch.cat(out, dim=0)
    info = {"target_ll": target_ll, "tail_prob": tail_prob,
            "mu_L": mu_L, "sigma_L": sigma_L, "kappa": kappa}
    return X, info


































def _seq_find_all(seq: Sequence[int], pattern: Sequence[int]) -> List[List[int]]:
    """seq 안에서 pattern 부분수열이 등장하는 모든 시작 위치를 [연속 인덱스 리스트]로 반환"""
    hits = []
    n = len(pattern)
    if n == 0:
        return hits
    for i in range(len(seq) - n + 1):
        if seq[i:i+n] == list(pattern):
            hits.append(list(range(i, i+n)))
    return hits

def _with_special_index_map(tokenizer, prompt: str):
    """무특수 토큰 인덱스 -> 특수 토큰 포함 인덱스 매핑 테이블 생성"""
    enc_with = tokenizer(prompt, add_special_tokens=True, return_special_tokens_mask=True)
    mask = enc_with["special_tokens_mask"]
    # special_tokens_mask==0 인 위치들이 '무특수' 시퀀스의 각 위치에 대응
    no_to_with = [i for i, m in enumerate(mask) if m == 0]
    return enc_with["input_ids"], no_to_with

def find_word_token_spans(
    tokenizer,
    prompt: str,
    word: str,
    *,
    include_special_tokens: bool = False,
    case_sensitive: bool = True,
) -> List[List[int]]:
    """
    prompt 를 tokenizer 로 토큰화했을 때, word 가 차지하는 모든 토큰 인덱스 구간을 반환.
    - 반환값: [ [i0, i1, ...], [j0, j1, ...], ... ]  (각 리스트가 한 번의 등장)
    - include_special_tokens=True 면, 반환하는 인덱스는 특수 토큰을 포함한 시퀀스 기준.
    """
    # 1) Fast 토크나이저면 offset 기반으로 가장 정확하게 처리
    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(
            prompt,
            add_special_tokens=include_special_tokens,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]        # (start, end) 문자 오프셋
        ids = enc["input_ids"]

        haystack = prompt if case_sensitive else prompt.lower()
        needle = word if case_sensitive else word.lower()

        spans: List[List[int]] = []
        for m in re.finditer(re.escape(needle), haystack):
            start, end = m.span()
            token_idxs = []
            for ti, (s, e) in enumerate(offsets):
                # 특수 토큰은 보통 (0,0) 또는 s==e 로 나옴 -> 스킵
                if e <= s:
                    continue
                # 토큰과 단어가 하나라도 겹치면 포함
                if not (e <= start or end <= s):
                    token_idxs.append(ti)
            if token_idxs:
                # 연속 구간만 남기고 분할(이례적이지만 중간에 겹치지 않는 토큰이 끼면 나눔)
                cur = [token_idxs[0]]
                for a, b in zip(token_idxs, token_idxs[1:]):
                    if b == a + 1:
                        cur.append(b)
                    else:
                        spans.append(cur)
                        cur = [b]
                spans.append(cur)
        return spans

    # 2) Slow 토크나이저(예: CLIPTokenizer)면 부분수열 탐색으로 처리
    #    (공백/개행이 단어 토큰에 같이 붙는 BPE 특성을 고려해 여러 패턴 시도)
    ids_no = tokenizer.encode(prompt, add_special_tokens=False)
    patterns = set()

    # 단어 그 자체
    p0 = tokenizer.encode(word, add_special_tokens=False)
    if p0:
        patterns.add(tuple(p0))
    # 앞에 공백/개행/탭이 붙어서 하나의 토큰으로 합쳐지는 경우
    for lead in (" " , "\n", "\t"):
        p = tokenizer.encode(lead + word, add_special_tokens=False)
        if p:
            patterns.add(tuple(p))

    # 찾기
    raw_hits_no_special: List[List[int]] = []
    seen = set()
    for pat in patterns:
        for hit in _seq_find_all(ids_no, list(pat)):
            key = tuple(hit)
            if key not in seen:
                raw_hits_no_special.append(hit)
                seen.add(key)

    if not include_special_tokens:
        return raw_hits_no_special

    # 특수 토큰 포함 시퀀스 기준 인덱스로 변환
    _, no_to_with = _with_special_index_map(tokenizer, prompt)
    spans_with = []
    for span in raw_hits_no_special:
        spans_with.append([no_to_with[i] for i in span])
    return spans_with

def encode_prompt(
    prompt,
    device,
    text_encoder,
    tokenizer
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        lora_scale (`float`, *optional*):
            A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
    """
    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]

    prompt_embeds_dtype = text_encoder.dtype

    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

    return prompt_embeds


def stack_embeds(prompts, text_model, tokenizer, word):
    full_embeds = []
    embeds = []
    spans = []

    for prompt in tqdm(prompts):
        prompt = prompt.format(word)
        span = find_word_token_spans(tokenizer, prompt, word, include_special_tokens=True, case_sensitive=False)[0]
        spans.append(span)

        with torch.no_grad():
            full_prompt_embeds = encode_prompt(prompt, "cuda", text_model, tokenizer).squeeze()
            prompt_embeds = full_prompt_embeds[span,:]
            full_embeds.append(full_prompt_embeds)
            embeds.append(prompt_embeds)

    full_embeds = torch.stack(full_embeds, dim=0)
    embeds = torch.stack(embeds, dim=0)
    embeds = embeds.reshape(embeds.shape[0], -1)

    return full_embeds, embeds, spans




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













if __name__ == '__main__':
    text_model = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", variant='fp16').to("cuda")
    tokenizer =  CLIPTokenizerFast.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")


    word = " Brad Pitt"
    # word = " a person"
    word_surr = " a person"

    with open("templates.txt", "r", encoding="utf-8") as f:
        templates = f.readlines()
    

    prompts = [prompt.rstrip("\n").format(word) for prompt in templates]
    full_embeds, embeds, spans = stack_embeds(prompts, text_model, tokenizer, word)

    pca = PCA(2)
    # fit on data
    pca.fit(embeds.cpu().numpy())
    reduced = pca.transform(embeds.cpu().numpy())
    reduced_pt = torch.tensor(reduced).to("cuda")

    principal = torch.tensor(pca.components_).to("cuda")

    dof_init = 2.0
    tmm = TMMTorch(
            n_components=5,
            covariance_type="diag",
            tied_covariance=False,
            shrinkage_alpha=0.1,
            scale_floor_scale=1e-3,
            min_cluster_weight=1e-3,
            learn_dof=False,    
            dof_init=dof_init,
            verbose=True,
            seed=42
        ).fit(reduced_pt)

    os.makedirs("./scatters_by_scale_t_dist_band", exist_ok=True)
    plt.figure()
    plot_tmm_2d(reduced_pt, tmm)  
    plt.show()
    plt.savefig(f"./scatters_by_scale_t_dist_band/scatter_org.png")

    n_rand = 1500

    for conf_level in range(100):
        conf_low, conf_high = 0.89999+conf_level*0.001, 0.89999+(conf_level+1)*0.001
        X_ring, info = sample_in_conf_interval(
            tmm,
            n=n_rand,
            conf_low=conf_low,
            conf_high=conf_high,
            n_ref=800_000,     
            helper_batch=65536,
            max_batches=300,
            pad_ll=0.0          
        )

        plt.figure()
        plot_tmm_2d(X_ring, tmm)
        plt.show()
        plt.savefig(f"./scatters_by_scale_t_dist_band/scatter_{conf_low}_{conf_high}.png")