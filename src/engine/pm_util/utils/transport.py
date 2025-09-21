import torch
from typing import Optional, Tuple, Union

# ---- 주어진 유틸들 사용 (그대로 붙여넣기) ----
def _sym(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.transpose(-1, -2))

def _sqrtm_psd(M: torch.Tensor, jitter: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    M = _sym(M)
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=jitter)
    sqrtD = torch.diag(evals.sqrt())
    invsqrtD = torch.diag(evals.rsqrt())
    Msqrt = evecs @ sqrtD @ evecs.T
    Minvsqrt = evecs @ invsqrtD @ evecs.T
    return _sym(Msqrt), _sym(Minvsqrt)

def _gaussian_w2_map_matrix(S1: torch.Tensor, S2: torch.Tensor, jitter: float = 1e-9) -> torch.Tensor:
    S2sqrt, _ = _sqrtm_psd(S2, jitter)
    M = _sym(S2sqrt @ S1 @ S2sqrt)
    _, Minvsqrt = _sqrtm_psd(M, jitter)
    A = _sym(S2sqrt @ Minvsqrt @ S2sqrt)
    return A

def _gaussian_w2_sq(mu1: torch.Tensor, S1: torch.Tensor,
                    mu2: torch.Tensor, S2: torch.Tensor, jitter: float = 1e-9) -> torch.Tensor:
    diff = mu1 - mu2
    S1sqrt, _ = _sqrtm_psd(S1, jitter)
    M = _sym(S1sqrt @ S2 @ S1sqrt)
    Msqrt, _ = _sqrtm_psd(M, jitter)
    tr_term = torch.trace(S1 + S2 - 2.0 * Msqrt)
    return (diff @ diff) + tr_term

def _sinkhorn(w: torch.Tensor, v: torch.Tensor, C: torch.Tensor,
              eps: float = 0.05, max_iters: int = 500, tol: float = 1e-9) -> torch.Tensor:
    w = w / (w.sum() + 1e-12)
    v = v / (v.sum() + 1e-12)
    Kmat = torch.exp(-C / eps).clamp_min(1e-300)
    u = torch.ones_like(w)
    a = w
    b = v
    for _ in range(max_iters):
        u_prev = u
        u = a / (Kmat @ (b / (Kmat.T @ u + 1e-300) + 1e-300) + 1e-300)
        if torch.max(torch.abs(u - u_prev)).item() < tol:
            break
    v_scal = b / (Kmat.T @ u + 1e-300)
    Pi = torch.diag(u) @ Kmat @ torch.diag(v_scal)
    Pi = Pi / (Pi.sum() + 1e-12)
    return Pi
# ---------------------------------------------

def _expand_covariances_to_K(model: "TMMTorch", device, dtype) -> torch.Tensor:
    """model.covariances_ -> (K, m, m)으로 확장."""
    if model.tied_covariance:
        return model.covariances_[0:1].repeat(model.n_components, 1, 1).to(device=device, dtype=dtype)
    return model.covariances_.to(device=device, dtype=dtype)

def _lift_to_full(mu_red: torch.Tensor, S_red: torch.Tensor,
                  W: torch.Tensor, mean_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (m,) / (m,m) -> (d,) / (d,d),  x_full = mean + z @ W^T  기준
    mean_full: (d,), W: (d,m)
    """
    # means
    mu_full = mean_full + (mu_red @ W.T)
    # covariances
    S_full = W @ S_red @ W.T
    return mu_full, _sym(S_full)

@torch.no_grad()
def optimal_transport(
    tmm_src, tmm_tgt, X_src_red: torch.Tensor,
    *,
    sinkhorn_eps: Optional[float] = None,
    sinkhorn_max_iters: int = 500,
    sinkhorn_tol: float = 1e-9,
    jitter: float = 1e-8,
    return_plan: bool = False,
    return_full: bool = False,
) -> Union[
    Tuple[torch.Tensor, Optional[torch.Tensor]],
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
]:
    """
    (PCA 저차원에서 학습된) source TMM의 샘플 X_src_red(N,m1)을
    원공간으로 복원해 OT 바리센터 맵으로 target TMM에 보낸 뒤,
    다시 target의 PCA 좌표계(N,m2)로 투영해 반환합니다.

    Returns
    -------
    Z_tgt : (N, m2)  # target TMM의 PCA 좌표
    [Y_full] : (N, d_full)  # return_full=True일 때만 함께 반환
    [Pi] : (K1, K2)  # return_plan=True일 때만 함께 반환
    """
    assert tmm_src.means_ is not None and tmm_tgt.means_ is not None, "두 모델 모두 fit()이 필요합니다."
    assert tmm_src.basis is not None and tmm_src.mean is not None, "source TMM에 basis/mean이 필요합니다."
    assert tmm_tgt.basis is not None and tmm_tgt.mean is not None, "target TMM에 basis/mean이 필요합니다."

    device = X_src_red.device
    dtype = X_src_red.dtype

    # --- (0) 각 모델의 PCA 정보 (원공간 복원/투영용) ---
    W1 = tmm_src.basis.to(device=device, dtype=dtype)     # (d_full, m1)
    m1_full = tmm_src.mean.to(device=device, dtype=dtype) # (d_full,)
    W2 = tmm_tgt.basis.to(device=device, dtype=dtype)     # (d_full, m2)
    m2_full = tmm_tgt.mean.to(device=device, dtype=dtype) # (d_full,)

    K1, m1 = tmm_src.means_.shape
    K2, m2 = tmm_tgt.means_.shape
    d_full = W1.shape[0]

    # --- (1) source 샘플을 원공간으로 복원 ---
    # z @ W^T + mean
    X_full = X_src_red.to(device=device, dtype=dtype) @ W1.T + m1_full  # (N, d_full)

    # --- (2) 컴포넌트 파라미터들을 원공간으로 승격(lift) ---
    mu1_red = tmm_src.means_.to(device=device, dtype=dtype)         # (K1, m1)
    S1_red = _expand_covariances_to_K(tmm_src, device, dtype)       # (K1, m1, m1)
    w = tmm_src.weights_.to(device=device, dtype=dtype)             # (K1,)

    mu2_red = tmm_tgt.means_.to(device=device, dtype=dtype)         # (K2, m2)
    S2_red = _expand_covariances_to_K(tmm_tgt, device, dtype)       # (K2, m2, m2)
    v = tmm_tgt.weights_.to(device=device, dtype=dtype)             # (K2,)

    # lift
    mu1_full = torch.empty((K1, d_full), device=device, dtype=dtype)
    S1_full  = torch.empty((K1, d_full, d_full), device=device, dtype=dtype)
    for i in range(K1):
        mu1_full[i], S1_full[i] = _lift_to_full(mu1_red[i], S1_red[i], W1, m1_full)

    mu2_full = torch.empty((K2, d_full), device=device, dtype=dtype)
    S2_full  = torch.empty((K2, d_full, d_full), device=device, dtype=dtype)
    for j in range(K2):
        mu2_full[j], S2_full[j] = _lift_to_full(mu2_red[j], S2_red[j], W2, m2_full)

    # --- (3) 컴포넌트 간 비용행렬 C (K1,K2): 가우시안 W2^2 (원공간에서) ---
    C = torch.empty((K1, K2), device=device, dtype=dtype)
    for i in range(K1):
        for j in range(K2):
            C[i, j] = _gaussian_w2_sq(mu1_full[i], S1_full[i], mu2_full[j], S2_full[j], jitter=jitter)

    # Sinkhorn epsilon 기본값 자동 설정
    if sinkhorn_eps is None:
        sinkhorn_eps = float(torch.median(C).item() * 0.1 + 1e-12)

    # --- (4) Sinkhorn으로 이산 OT 결합 π (K1,K2) ---
    Pi = _sinkhorn(w, v, C, eps=sinkhorn_eps, max_iters=sinkhorn_max_iters, tol=sinkhorn_tol)

    # 조건부 p(j|i)
    p_j_given_i = Pi / (w[:, None] + 1e-12)  # (K1,K2)

    # --- (5) 각 i→j의 선형 사상 (원공간에서) A_ij, b_ij ---
    A_i = [None] * K1
    b_i = [None] * K1
    for i in range(K1):
        A_acc = torch.zeros((d_full, d_full), device=device, dtype=dtype)
        b_acc = torch.zeros((d_full,), device=device, dtype=dtype)
        for j in range(K2):
            A_ij = _gaussian_w2_map_matrix(S1_full[i], S2_full[j], jitter=jitter)  # (d,d)
            b_ij = mu2_full[j] - (A_ij @ mu1_full[i])                               # (d,)
            pij  = p_j_given_i[i, j]
            A_acc = A_acc + pij * A_ij
            b_acc = b_acc + pij * b_ij
        A_i[i] = _sym(A_acc)
        b_i[i] = b_acc

    # --- (6) r_{ni} = p(i|x_n) (source TMM posterior, 저차원에서 계산) ---
    R = tmm_src.predict_proba(X_src_red.to(device=device, dtype=dtype))  # (N,K1)

    # --- (7) 최종 바리센터 맵 (원공간): Y_full_n = Σ_i r_{ni} (A_i X_full_n + b_i) ---
    N = X_src_red.shape[0]
    AX = torch.stack([ (X_full @ A_i[i].T) + b_i[i] for i in range(K1) ], dim=1)  # (N,K1,d_full)
    Y_full = torch.einsum('nk,nkd->nd', R, AX)  # (N,d_full)

    # --- (8) 타깃 PCA 좌표로 투영: Z_tgt = (Y_full - m2) @ W2 ---
    Z_tgt = (Y_full - m2_full) @ W2  # (N,m2)

    if return_full and return_plan:
        return Z_tgt, Y_full, Pi
    if return_full and not return_plan:
        return Z_tgt, Y_full, None
    if not return_full and return_plan:
        return Z_tgt, Pi
    return Z_tgt, None

def tmm_emd(
    tmm1,
    tmm2,
    *,
    epsilon: float = 0.05,
    sinkhorn_max_iters: int = 200,
    sinkhorn_tol: float = 1e-7,
    jitter: float = 1e-12,
) -> float:
    """
    두 개의 TMMTorch 모델 사이의 earth-mover distance(1-Wasserstein, EMD)를 계산.
    - 서로 다른 PCA 기저(W)와 데이터 평균(mean)을 가진 경우를 안전히 처리
    - 성분 수가 달라도 됨
    - 비용행렬은 T-분포 성분을 '가우시안(모멘트 일치)'으로 근사한 뒤의 2-Wasserstein 거리(= Bures + 평균 차) 사용
    - 운송은 Sinkhorn(엔트로피 정규화)으로 효율적으로 근사

    Args:
        tmm1, tmm2: TMMTorch 인스턴스 (각각 basis: (D,m), mean: (D,), means_: (K,m), covariances_: (K|1,m,m), dofs_: (K,))
        epsilon: Sinkhorn 엔트로피 정규화 강도(클수록 빠르고 매끄럽게, 작을수록 정확하지만 느림)
        sinkhorn_max_iters: Sinkhorn 반복 최대 횟수
        sinkhorn_tol: Sinkhorn 수렴 허용오차(스케일 벡터 변화 기준)
        jitter: 수치 안정화를 위한 PSD 행렬 대각선 보정

    Returns:
        emd (float)
    """
    # --- 안전검사 & 준비 ---
    if tmm1.means_ is None or tmm2.means_ is None:
        raise ValueError("각 모델에 대해 fit()이 완료되어 있어야 합니다.")

    device = torch.device("cpu")  # 수치 안정성/호환성 위해 CPU, float64로 계산
    dtype = torch.float64

    def to_cpu64(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if x is None else x.detach().to(device=device, dtype=dtype)

    W1 = to_cpu64(tmm1.basis)
    W2 = to_cpu64(tmm2.basis)
    m1 = to_cpu64(tmm1.mean)
    m2 = to_cpu64(tmm2.mean)
    if W1 is None or W2 is None or m1 is None or m2 is None:
        raise ValueError("각 TMM에는 basis와 mean이 반드시 있어야 합니다.")

    if W1.shape[0] != W2.shape[0] or m1.shape[0] != m2.shape[0] or W1.shape[0] != m1.shape[0]:
        raise ValueError("두 모델의 원공간 차원 D가 일치해야 합니다.")

    # reduced-space params
    M1 = to_cpu64(tmm1.means_)                 # (K1, m1)
    M2 = to_cpu64(tmm2.means_)                 # (K2, m2)
    S1 = to_cpu64(tmm1.covariances_)           # (K1|1, m1, m1)
    S2 = to_cpu64(tmm2.covariances_)           # (K2|1, m2, m2)
    nu1 = to_cpu64(tmm1.dofs_)                 # (K1,)
    nu2 = to_cpu64(tmm2.dofs_)                 # (K2,)
    w1 = to_cpu64(tmm1.weights_)               # (K1,)
    w2 = to_cpu64(tmm2.weights_)               # (K2,)

    # 성분 수
    K1, m1_dim = M1.shape
    K2, m2_dim = M2.shape

    # 가중치 정규화 & 미미한 성분 제거
    def _clean_weights(w, thr=1e-12):
        w = torch.clamp(w, min=0)
        s = w.sum()
        if s <= 0:
            raise ValueError("모든 가중치가 0 입니다.")
        w = w / s
        mask = w > thr
        return w[mask], mask

    w1, mask1 = _clean_weights(w1)
    w2, mask2 = _clean_weights(w2)
    M1, nu1 = M1[mask1], nu1[mask1]
    M2, nu2 = M2[mask2], nu2[mask2]
    if S1.shape[0] != 1:
        S1 = S1[mask1]
    if S2.shape[0] != 1:
        S2 = S2[mask2]
    K1, m1_dim = M1.shape
    K2, m2_dim = M2.shape

    # --- 공통 부분공간 U 구성 (W1, W2, 평균 오프셋까지 포함) ---
    D = W1.shape[0]
    # 평균 오프셋 벡터(두 모델의 원공간 평균 차이)도 포함하여 U가 평균 차까지 설명하도록 함
    delta = (m1 - m2).reshape(D, 1)
    # 정규화가 너무 작은 경우 제외
    if torch.norm(delta) < 1e-15:
        concat = torch.cat([W1, W2], dim=1)  # D × (m1+m2)
    else:
        concat = torch.cat([W1, W2, delta / (torch.norm(delta) + 1e-30)], dim=1)

    # QR로 직교기저 생성
    # (W1, W2는 원래 직교지만 수치적으로 안전하게 한번 더 직교화)
    U, _ = torch.linalg.qr(concat, mode="reduced")  # D × r, r ≤ m1+m2+1
    r = U.shape[1]

    # U^T W (저차 투영 행렬) 및 U^T m (투영 평균) 사전계산
    Ut = U.T
    A1 = Ut @ W1  # r × m1
    A2 = Ut @ W2  # r × m2
    mu1_base = Ut @ m1  # r
    mu2_base = Ut @ m2  # r

    # --- 성분별 (U-공간에서의) 평균/공분산 계산 ---
    # T 분포의 모멘트 일치 가우시안 근사: Cov = (nu/(nu-2)) * (W Σ W^T)   (nu>2)
    I_r = torch.eye(r, dtype=dtype, device=device)

    def _proj_gaussian_params(M, S, nu, A, mu_base):
        """
        M: (K, m), S: (K|1, m, m), nu: (K,), A: (r, m), mu_base: (r,)
        반환: means_proj (K, r), covs_proj (K, r, r)
        """
        K, m = M.shape
        tied = (S.shape[0] == 1)
        means_proj = torch.empty((K, r), dtype=dtype, device=device)
        covs_proj  = torch.empty((K, r, r), dtype=dtype, device=device)
        for k in range(K):
            mk = M[k]                                  # (m,)
            Sk = S[0] if tied else S[k]               # (m,m)
            nuk = float(nu[k].item())
            # 평균
            means_proj[k] = mu_base + A @ mk
            # 공분산 (U-공간)
            cov_scale = A @ (Sk + jitter*torch.eye(m, dtype=dtype, device=device)) @ A.T  # r×r
            # T -> Gaussian approx
            factor = nuk / max(nuk - 2.0, 1e-12)
            Ck = factor * cov_scale
            # 수치 안정화
            Ck = 0.5 * (Ck + Ck.T) + jitter * I_r
            covs_proj[k] = Ck
        return means_proj, covs_proj

    mu1s, C1s = _proj_gaussian_params(M1, S1, nu1, A1, mu1_base)  # (K1,r), (K1,r,r)
    mu2s, C2s = _proj_gaussian_params(M2, S2, nu2, A2, mu2_base)  # (K2,r), (K2,r,r)

    # --- 가우시안 간 2-Wasserstein 거리 (Bures + 평균차) ---
    def _sqrtm_psd(A: torch.Tensor) -> torch.Tensor:
        # A: (r,r) 대칭 PSD 가정
        A = 0.5 * (A + A.T)
        evals, vecs = torch.linalg.eigh(A)
        evals = torch.clamp(evals, min=0.0)
        sqrt_e = torch.sqrt(evals)
        # vecs @ diag(sqrt_e) @ vecs^T : 열스케일을 이용한 빠른 곱
        return (vecs * sqrt_e) @ vecs.T

    def _bures_squared(Ca: torch.Tensor, Cb: torch.Tensor) -> float:
        SqrtCb = _sqrtm_psd(Cb)
        M = SqrtCb @ Ca @ SqrtCb
        SqrtM = _sqrtm_psd(M)
        val = torch.trace(Ca) + torch.trace(Cb) - 2.0 * torch.trace(SqrtM)
        return float(max(val.item(), 0.0))

    # 비용행렬 C[i,j] = W2( N(mu1_i, C1_i), N(mu2_j, C2_j) )
    # = sqrt( ||mu1-mu2||^2 + Bures^2(C1,C2) )
    C = torch.empty((K1, K2), dtype=dtype, device=device)
    for i in range(K1):
        mui = mu1s[i]
        Ci  = C1s[i]
        for j in range(K2):
            muj = mu2s[j]
            Cj  = C2s[j]
            dm2 = torch.sum((mui - muj) ** 2).item()
            b2  = _bures_squared(Ci, Cj)
            C[i, j] = torch.sqrt(torch.as_tensor(dm2 + b2, dtype=dtype, device=device) + 1e-24)

    # --- Sinkhorn (엔트로피 정규화 OT)로 EMD 근사 ---
    # p in R^{K1}, q in R^{K2}, C in R^{K1×K2}
    p = w1
    q = w2

    # 수치적으로 안정적인 커널
    eps = max(float(epsilon), 1e-8)
    Kmat = torch.exp(-C / eps)  # (K1,K2)
    # 0 회피
    Kmat = torch.clamp(Kmat, min=1e-300)

    u = torch.ones_like(p)
    v = torch.ones_like(q)

    for _ in range(sinkhorn_max_iters):
        u_prev = u
        Kv = Kmat @ v + 1e-300
        u = p / Kv
        KTu = Kmat.T @ u + 1e-300
        v = q / KTu
        # 수렴 체크
        if torch.max(torch.abs(u - u_prev)).item() < sinkhorn_tol:
            break

    T = (u.unsqueeze(1) * Kmat) * v.unsqueeze(0)  # 최종 운송계획
    emd = torch.sum(T * C).item()
    return float(emd)