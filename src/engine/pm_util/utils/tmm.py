from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch.special import digamma, polygamma
import math

def safe_cholesky(A: torch.Tensor, jitter_init: float = 1e-6, jitter_max: float = 1e-1) -> torch.Tensor:
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
    # 수정됨: 올바른 행렬 재구성 방식 V @ diag(L) @ V.T
    A_fixed = evecs @ torch.diag(evals) @ evecs.T
    return torch.linalg.cholesky(A_fixed)

def _log_student_t_full_safe(X: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor,
                             nu: torch.Tensor, jitter_init: float, jitter_max: float) -> Tuple[torch.Tensor, torch.Tensor]: # nu 타입을 텐서로 받음

    d = X.shape[1]
    L = safe_cholesky(scale, jitter_init=jitter_init, jitter_max=jitter_max)  # (d,d)
    diff = X - mean
    sol = torch.cholesky_solve(diff.T, L).T
    m2 = (diff * sol).sum(dim=1).clamp_min(1e-12)           # (N,)
    logdet = 2.0 * torch.log(torch.diag(L)).sum()           # scalar tensor

    # --- 텐서 캐스팅(장치/정밀도 일치) ---
    d_t    = torch.as_tensor(float(d), dtype=X.dtype, device=X.device)
    pi_t   = torch.as_tensor(math.pi, dtype=X.dtype, device=X.device)
    half_t = torch.as_tensor(0.5, dtype=X.dtype, device=X.device)

    c = torch.special.gammaln((nu + d_t) * half_t) \
        - torch.special.gammaln(nu * half_t) \
        - half_t * (d_t * torch.log(nu * pi_t) + logdet)

    logpdf = c - ((nu + d_t) * half_t) * torch.log1p(m2 / nu)  # (N,)
    return logpdf, m2

def _floor_cov_eig(cov: torch.Tensor, floor: float) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=floor)
    # 수정됨: 올바른 행렬 재구성 방식 V @ diag(L) @ V.T
    return evecs @ torch.diag(evals) @ evecs.T


def _to_diag(cov: torch.Tensor) -> torch.Tensor:
    return torch.diag(torch.diag(cov))

def _shrinkage(cov: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """(1-alpha)*cov + alpha*target"""
    if alpha <= 0.0:
        return cov
    return (1 - alpha) * cov + alpha * target

@dataclass
class TMMTorch:
    n_components: int
    reg_scale: float = 1e-6
    max_iters: int = 200
    tol: float = 1e-4
    verbose: bool = False
    seed: Optional[int] = None

    covariance_type: str = "full"
    tied_covariance: bool = False
    shrinkage_alpha: float = 0.0
    scale_floor_scale: float = 1e-3
    min_cluster_weight: float = 1e-3
    max_condition_number: float = 1e6
    chol_jitter_init: float = 1e-6
    chol_jitter_max: float = 1e-1

    learn_dof: bool = False
    dof_init: float = 8.0
    min_dof: float = 2.1
    max_dof: float = 200.0
    dof_newton_iters: int = 10
    dof_newton_tol: float = 1e-3

    basis: torch.Tensor = None
    mean: torch.Tensor = None

    means_: Optional[torch.Tensor] = None
    covariances_: Optional[torch.Tensor] = None
    weights_: Optional[torch.Tensor] = None
    dofs_: Optional[torch.Tensor] = None

    def _init_params(self, X: torch.Tensor):
        N, d = X.shape
        g = torch.Generator(device=X.device) if self.seed is not None else None
        if g is not None:
            g.manual_seed(self.seed)
        idx = torch.randperm(N, generator=g, device=X.device)[: self.n_components]
        means = X[idx].clone()

        diff = X - X.mean(dim=0, keepdim=True)
        global_var = (diff * diff).mean(dim=0) + 1e-3
        base_scale = torch.diag(global_var)
        scales = torch.stack([base_scale.clone() for _ in range(self.n_components)], dim=0)
        weights = torch.full((self.n_components,), 1.0 / self.n_components, device=X.device)
        dofs = torch.full((self.n_components,), float(self.dof_init), device=X.device)
        return means, scales, weights, dofs, global_var

    def _log_prob_matrix(self, X: torch.Tensor, means: torch.Tensor, scales: torch.Tensor,
                         weights: torch.Tensor, dofs: torch.Tensor) -> torch.Tensor:
        N, d = X.shape
        K = means.shape[0]
        logs = []
        is_tied = scales.shape[0] == 1 # CHANGED: Check shape to determine if tied
        
        for k in range(K):
            scale_k = scales[0] if is_tied else scales[k]
            # Revised: float() casting removal
            lp, _ = _log_student_t_full_safe(
                X.to(scale_k.device), means[k], scale_k, dofs[k],
                self.chol_jitter_init, self.chol_jitter_max
            )
            logs.append(lp + torch.log(weights[k] + 1e-12))
        return torch.stack(logs, dim=1)

    def _reinit_component(self, X: torch.Tensor, k: int, global_var: torch.Tensor):
        N, d = X.shape
        ridx = torch.randint(0, N, (1,), device=X.device)
        self.means_[k] = X[ridx].squeeze(0)
        base = torch.diag(global_var)
        if self.covariance_type == "diag":
            base = torch.diag(torch.clamp(torch.diag(base), min=1e-6))
        
        # CHANGED: Handle tied covariance case during re-initialization
        if self.tied_covariance:
            # For tied, re-initialization might be tricky. Resetting just one component's
            # contribution to the tied covariance is complex. Here, we just reset its mean.
            # The covariance will adjust in the next M-step.
            pass
        else:
            self.covariances_[k] = base

        self.dofs_[k] = torch.tensor(max(self.min_dof, self.dof_init), device=X.device, dtype=X.dtype)

    def fit(self, X: torch.Tensor):
        X = X.detach()
        N, d = X.shape
        K = self.n_components

        means, scales, weights, dofs, global_var = self._init_params(X)
        
        if self.tied_covariance:
            # Start with a single, averaged scale matrix
            scales = scales.mean(dim=0, keepdim=True)

        I = torch.eye(d, device=X.device, dtype=X.dtype)
        prev_ll = -torch.inf
        scale_floor = self.scale_floor_scale * (X.var(dim=0, unbiased=False).mean() + 1e-12)

        for it in range(self.max_iters):
            # E-step: resp + u
            log_prob = self._log_prob_matrix(X, means, scales, weights, dofs)
            max_lp, _ = torch.max(log_prob, dim=1, keepdim=True)
            lse = max_lp + torch.log(torch.exp(log_prob - max_lp).sum(dim=1, keepdim=True))
            log_resp = log_prob - lse
            resp = torch.exp(log_resp)

            u_list, z_list = [], []
            for k in range(K):
                scale_k = scales[0] if self.tied_covariance else scales[k]
                L = safe_cholesky(scale_k, self.chol_jitter_init, self.chol_jitter_max)
                diff = X - means[k]
                sol  = torch.cholesky_solve(diff.T, L).T
                m2   = (diff * sol).sum(dim=1).clamp_min(1e-12)
                nu_k = dofs[k]
                u_k  = (nu_k + d) / (nu_k + m2)
                z_k  = digamma((nu_k + d) * 0.5) - torch.log((nu_k + m2) * 0.5)
                u_list.append(u_k)
                z_list.append(z_k)
            U = torch.stack(u_list, dim=1)
            Z = torch.stack(z_list, dim=1)

            # M-step
            Nk = resp.sum(dim=0) + 1e-12
            weights = (Nk / N).clamp_min(1e-12)

            numer_mu = (resp * U).T @ X
            denom_mu = (resp * U).sum(dim=0)[:, None]
            means = numer_mu / denom_mu.clamp_min(1e-12)

            scales_new = []
            for k in range(K):
                Xc = X - means[k]
                Wk = (resp[:, k:k+1] * U[:, k:k+1])
                S_k = (Xc.T @ (Xc * Wk)) / Nk[k]
                S_k = S_k + self.reg_scale * I
                if self.covariance_type == "diag":
                    S_k = _to_diag(S_k)
                target = torch.diag(global_var)
                if self.covariance_type == "diag":
                    target = _to_diag(target)
                S_k = _shrinkage(S_k, target, self.shrinkage_alpha)
                S_k = _floor_cov_eig(S_k, float(scale_floor))
                scales_new.append(S_k)
            
            if self.tied_covariance:
                # 수정됨: Nk로 가중 평균을 내어 단일 공분산 행렬 업데이트
                S = torch.sum(torch.stack(scales_new, dim=0) * (Nk / Nk.sum()).view(K, 1, 1), dim=0)
                scales = S.unsqueeze(0)
            else:
                scales = torch.stack(scales_new, dim=0)

            if self.learn_dof:
                # 수정됨: 루프를 더 깔끔하고 효율적으로 변경
                new_dofs = []
                for k in range(K):
                    nu = dofs[k].item()
                    Sk = ((resp[:, k] * (U[:, k] - Z[:, k])).sum() / Nk[k]).item()
                    
                    for _ in range(self.dof_newton_iters):
                        try:
                            f = math.log(nu * 0.5) - math.digamma(nu * 0.5) + 1.0 - Sk
                            df = 1.0/nu - 0.5 * polygamma(1, torch.tensor(nu * 0.5, device=X.device)).item() # polygamma는 torch 사용
                            step = f / (df + 1e-12)
                            nu_new = nu - step
                            if abs(nu_new - nu) < self.dof_newton_tol:
                                nu = nu_new
                                break
                            nu = nu_new
                        except ValueError: # math.log 또는 math.digamma에서 nu가 너무 작을 때 발생
                            nu = self.min_dof
                            break
                    
                    nu = max(self.min_dof, min(self.max_dof, nu))
                    new_dofs.append(torch.tensor(nu, device=X.device, dtype=X.dtype))
                dofs = torch.stack(new_dofs)

            for k in range(K):
                scale_k = scales[0] if self.tied_covariance else scales[k]
                evals = torch.linalg.eigvalsh(scale_k)
                cond = (evals.max() / (evals.min() + 1e-24)).item()
                if (weights[k] < self.min_cluster_weight) or (cond > self.max_condition_number):
                    if self.verbose:
                        print(f"[Iter {it+1}] Reinit comp {k}: w={weights[k].item():.3e}, cond={cond:.2e}")
                    self.means_, self.covariances_, self.weights_, self.dofs_ = means.clone(), scales.clone(), weights.clone(), dofs.clone()
                    self._reinit_component(X, k, global_var)
                    self.weights_[k] = max(self.min_cluster_weight, float(self.weights_[k]))
                    self.weights_ = self.weights_ / self.weights_.sum()
                    means, scales, weights, dofs = self.means_, self.covariances_, self.weights_, self.dofs_


            curr_ll = lse.mean()
            if self.verbose and (it % 5 == 0 or it == self.max_iters - 1):
                evals = torch.linalg.eigvalsh(scales[0] if self.tied_covariance else scales)
                msg = (f"[Iter {it+1}] avg ll={curr_ll.item():.6f}, "
                       f"minEig={evals.min().item():.3e}, minW={weights.min().item():.3e}")
                print(msg)

            if torch.abs(curr_ll - prev_ll) < self.tol:
                if self.verbose: print("Converged.")
                break
            prev_ll = curr_ll

        self.means_ = means
        # 수정됨: tied covariance일 때 (1, d, d) 형태로 저장
        self.covariances_ = scales if not self.tied_covariance else scales[:1]
        self.weights_ = weights
        self.dofs_ = dofs
        return self

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        assert self.means_ is not None, "Call fit() first."
        X = X.detach()
        # CHANGED: self.covariances_ 는 이미 올바른 shape (K,d,d) or (1,d,d)를 가짐
        log_prob = self._log_prob_matrix(X, self.means_, self.covariances_, self.weights_, self.dofs_)
        max_lp, _ = torch.max(log_prob, dim=1, keepdim=True)
        lse = max_lp + torch.log(torch.exp(log_prob - max_lp).sum(dim=1, keepdim=True))
        return torch.exp(log_prob - lse)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_proba(X), dim=1)

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        assert self.means_ is not None, "Call fit() first."
        X = X.detach()
        # 수정됨: 코드 중복 제거, _log_prob_matrix 재사용
        log_prob = self._log_prob_matrix(X, self.means_, self.covariances_, self.weights_, self.dofs_)
        max_lp, _ = torch.max(log_prob, dim=1, keepdim=True)
        lse = max_lp + torch.log(torch.exp(log_prob - max_lp).sum(dim=1, keepdim=True))
        return lse.squeeze(1)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        assert self.means_ is not None, "Call fit() first."
        d = self.means_.shape[1]
        comp_idx = torch.multinomial(self.weights_, num_samples=n, replacement=True)
        X = torch.zeros((n, d), device=self.means_.device, dtype=self.means_.dtype)
        
        is_tied = self.covariances_.shape[0] == 1

        for k in range(self.n_components):
            mask = (comp_idx == k)
            m = int(mask.sum().item())
            if m == 0:
                continue
            S_k = self.covariances_[0 if is_tied else k]
            L = safe_cholesky(S_k, jitter_init=self.chol_jitter_init, jitter_max=self.chol_jitter_max)
            z = torch.randn((m, d), device=self.means_.device, dtype=self.means_.dtype)
            # rate=β=ν/2, shape=α=ν/2
            gamma_dist = torch.distributions.Gamma(self.dofs_[k]*0.5, self.dofs_[k]*0.5)
            g = gamma_dist.sample((m,)).to(self.means_.device)
            X[mask] = self.means_[k] + (z @ L.T) / torch.sqrt(g).unsqueeze(1)
        return X

    def diagnostics(self) -> dict:
        assert self.means_ is not None, "Call fit() first."
        K = self.n_components
        is_tied = self.covariances_.shape[0] == 1
        
        evals_list = [torch.linalg.eigvalsh(self.covariances_[0 if is_tied else k]) for k in range(K)]
        mins = [evals.min().item() for evals in evals_list]
        conds = [(evals.max() / (evals.min() + 1e-24)).item() for evals in evals_list]

        return {
            "min_eigenvalue": float(min(mins)),
            "max_condition_number": float(max(conds)),
            "min_weight": float(self.weights_.min().item()),
            "min_dof": float(self.dofs_.min().item()),
        }