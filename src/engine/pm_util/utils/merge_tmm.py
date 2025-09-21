from typing import Optional, Sequence
import torch
from .tmm import TMMTorch
from typing import Optional, Sequence, Tuple
import torch

# ---------- 최소 직교 기반 U (SVD 기본, Gram-Schmidt 옵션) ----------
def orthonormal_union_basis(
    Ws: Sequence[torch.Tensor],
    *,
    tol: float = 1e-8,
    use_svd: bool = True,
) -> torch.Tensor:
    assert len(Ws) > 0
    device, dtype = Ws[0].device, Ws[0].dtype
    A = torch.cat(Ws, dim=1).to(device=device, dtype=dtype)  # d × (Σ m_i)

    if use_svd:
        U, S, _ = torch.linalg.svd(A, full_matrices=False)
        thr = (S.max() if S.numel() else torch.tensor(0.0, device=device, dtype=dtype)) * tol
        r = int((S > thr).sum().item()) if S.numel() else 1
        r = max(r, 1)
        return U[:, :r]
    else:
        Qcols = []
        for j in range(A.shape[1]):
            v = A[:, j].clone()
            for q in Qcols:
                v -= (q @ v) * q
            n = torch.norm(v)
            if n > tol:
                Qcols.append(v / n)
        if not Qcols:
            e1 = torch.zeros(A.shape[0], device=device, dtype=dtype); e1[0] = 1.0
            Qcols = [e1]
        return torch.stack(Qcols, dim=1)

# ---------- (보조) 한 모델의 "혼합 평균" (원공간) ----------
def _model_mixture_mean_in_x_space(model: "TMMTorch") -> torch.Tensor:
    """
    m_i + W_i ( sum_k w_{ik} * mu_{ik} )
    """
    W = model.basis
    mean_i = model.mean
    w = model.weights_ / model.weights_.sum()
    mu_bar = (w[:, None] * model.means_).sum(dim=0)      # (m_i,)
    return mean_i + W @ mu_bar                           # (d,)

# ---------- 한 모델의 모든 컴포넌트를 U-좌표계로 변환 ----------
def _model_to_U_params(
    model: "TMMTorch",
    U: torch.Tensor,              # (d, r)
    m0: torch.Tensor,             # (d,) -> merged global mean
    *,
    eig_floor: float = 1e-9,            
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    반환: means_U (K,r), covs_U (K,r,r), dofs (K,), weights (K,)
    (model 내부 basis/mean 사용)                                  
    """
    assert model.means_ is not None and model.covariances_ is not None
    assert model.weights_ is not None and model.dofs_ is not None
    assert hasattr(model, "basis") and hasattr(model, "mean")

    device, dtype = U.device, U.dtype
    W = model.basis.to(device=device, dtype=dtype)     # (d, m_i)
    mean_i = model.mean.to(device=device, dtype=dtype) # (d,)

    K, m_i = model.means_.shape
    r = U.shape[1]

    G = U.T @ W                                        # (r, m_i)

    means_U = torch.empty((K, r), device=device, dtype=dtype)
    covs_U  = torch.empty((K, r, r), device=device, dtype=dtype)
    dofs    = model.dofs_.to(device=device, dtype=dtype).clone()
    weights = model.weights_.to(device=device, dtype=dtype).clone()

    # covariances_: (K,m_i,m_i) 또는 tied이면 (1,m_i,m_i)
    tied = getattr(model, "tied_covariance", False)
    for k in range(K):
        mu_red  = model.means_[k].to(device=device, dtype=dtype)             # (m_i,)
        if tied:
            Sig_red = model.covariances_[0].to(device=device, dtype=dtype)   # (m_i,m_i)
        else:
            Sig_red = model.covariances_[k].to(device=device, dtype=dtype)

        # 위치: μ_u = U^T( mean_i + W_i μ_k - m0 )
        x_loc = mean_i + W @ mu_red                   # (d,)
        means_U[k] = U.T @ (x_loc - m0)

        # 스케일: Σ_u = (U^T W_i) Σ_k (U^T W_i)^T
        Sig_u = G @ Sig_red @ G.T
        Sig_u = 0.5 * (Sig_u + Sig_u.T)              # 대칭화
        evals, evecs = torch.linalg.eigh(Sig_u)
        # 수치 바닥(상대/절대 혼합)
        local_floor = max(eig_floor, 1e-12 * float(evals.abs().max().item()) if evals.numel() else eig_floor)
        evals = torch.clamp(evals, min=local_floor)
        covs_U[k] = (evecs * evals) @ evecs.T

    return means_U, covs_U, dofs, weights

# ---------- 여러 모델을 하나로 병합 (basis/mean은 모델 내부에서 읽기) ----------
def merge_tmm_models(
    models: Sequence["TMMTorch"],
    *,
    model_priors: Optional[Sequence[float]] = None,  # None -> 균등
    tol: float = 1e-8,
    eig_floor: float = 1e-9,
    use_svd_basis: bool = True,
) -> "TMMTorch":
    """
    내부에 (basis, mean)을 가진 여러 TMMTorch를 하나로 합쳐 단일 TMMTorch를 반환.
    반환된 모델은 자신만의 (merged.mean, merged.basis=U)를 보유하고,
    입력 X에 대해 Z = (X - merged.mean) @ merged.basis 로 사용.
    """
    assert len(models) > 0

    # 1) 공통 최소 직교기반 U
    Ws = [mdl.basis for mdl in models]
    U = orthonormal_union_basis(Ws, tol=tol, use_svd=use_svd_basis)
    d, r = U.shape
    device, dtype = U.device, U.dtype

    # 2) 모델 프라이어 정규화
    if model_priors is None:
        model_priors = [1.0 / len(models)] * len(models)
    else:
        s = float(sum(model_priors))
        if s <= 0:
            raise ValueError("model_priors 합은 양수여야 합니다.")
        model_priors = [float(p) / s for p in model_priors]  # 합=1

    # 3) 원공간에서 혼합의 전역 평균 m0 계산
    #    m0 = Σ_i π_i * ( mean_i + W_i Σ_k w_{ik} μ_{ik} )
    mix_means_x = [ _model_mixture_mean_in_x_space(m) for m in models ]   # (d,) list
    m0 = torch.zeros(d, device=device, dtype=dtype)
    for pi, mm in zip(model_priors, mix_means_x):
        m0 += float(pi) * mm.to(device=device, dtype=dtype)

    # 4) 각 모델 컴포넌트를 U-좌표계로 변환 & 쌓기 (m0 기준 센터링)
    all_means, all_covs, all_dofs, all_weights = [], [], [], []
    for i, mdl in enumerate(models):
        muU, SigU, dofU, wU = _model_to_U_params(mdl, U, m0, eig_floor=eig_floor)
        all_means.append(muU)
        all_covs.append(SigU)
        all_dofs.append(dofU)
        all_weights.append(wU * float(model_priors[i]))

    means_U  = torch.cat(all_means, dim=0)           # (K_tot, r)
    covs_U   = torch.cat(all_covs, dim=0)            # (K_tot, r, r)
    dofs_U   = torch.cat(all_dofs, dim=0)            # (K_tot,)
    weightsU = torch.cat(all_weights, dim=0)         # (K_tot,)
    weightsU = torch.clamp(weightsU, min=1e-12)
    weightsU = weightsU / weightsU.sum()

    K_tot = means_U.shape[0]

    # 5) 최종 TMMTorch 구성 (U-좌표계)
    merged = TMMTorch(
        n_components=K_tot,
        covariance_type="full",
        tied_covariance=False,   # 변환 후 일반적으로 full
        learn_dof=False,         # 필요시 True로
    )
    merged.means_       = means_U
    merged.covariances_ = covs_U
    merged.weights_     = weightsU
    merged.dofs_        = dofs_U

    # 포인트: merged는 자신의 기준 좌표계를 명시
    merged.basis = U                                  # (d, r)
    merged.mean  = m0                                 # (d,)

    return merged