import torch
from .tmm import TMMTorch
from typing import Optional
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Ellipse

def plot_tmm_2d(
    X: torch.Tensor,
    tmm: TMMTorch,
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
    assert X.shape[1] == 2, "plot_tmm_2d는 d=2에서만 사용 가능합니다."
    assert tmm.means_ is not None, "먼저 tmm.fit(X)를 호출하세요."

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    device = X.device
    X_cpu = X.detach().to("cpu")
    means = tmm.means_.detach().to("cpu")
    covs = tmm.covariances_.detach().to("cpu")
    weights = tmm.weights_.detach().to("cpu")

    # 산점도 (하드 클러스터)
    hard = tmm.predict(X).to("cpu")
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
        logp = tmm.score_samples(grid).to("cpu").reshape(grid_size, grid_size)
    cs = ax.contour(xx.numpy(), yy.numpy(), logp.numpy(), levels=levels, linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=8)

    # 공분산 타원
    for k in range(tmm.n_components):
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