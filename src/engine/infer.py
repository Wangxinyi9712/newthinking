from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from losses.seg_losses import reliability_components


def _save_case_plot(
    image: torch.Tensor,
    pred: torch.Tensor,
    conf: torch.Tensor,
    reliability: torch.Tensor,
    ood_map: torch.Tensor,
    consistency_map: torch.Tensor,
    out_png: Path,
    error_map: torch.Tensor | None = None,
) -> None:
    if image.ndim == 5:
        d = image.shape[2] // 2
        img = image[0, 0, d].cpu().numpy()
        p = pred[0, 0, d].cpu().numpy()
        c = conf[0, 0, d].cpu().numpy()
        r = reliability[0, 0, d].cpu().numpy()
        o = ood_map[0, 0, d].cpu().numpy()
        s = consistency_map[0, 0, d].cpu().numpy()
        e = error_map[0, 0, d].cpu().numpy() if error_map is not None else None
    else:
        img = image[0, 0].cpu().numpy()
        p = pred[0, 0].cpu().numpy()
        c = conf[0, 0].cpu().numpy()
        r = reliability[0, 0].cpu().numpy()
        o = ood_map[0, 0].cpu().numpy()
        s = consistency_map[0, 0].cpu().numpy()
        e = error_map[0, 0].cpu().numpy() if error_map is not None else None

    cols = 7 if e is not None else 6
    fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("image")
    ax[1].imshow(p, cmap="viridis")
    ax[1].set_title("prediction")
    ax[2].imshow(c, cmap="magma", vmin=0, vmax=1)
    ax[2].set_title("confidence")
    ax[3].imshow(r, cmap="plasma", vmin=0, vmax=1)
    ax[3].set_title("reliability")
    ax[4].imshow(o, cmap="cividis", vmin=0, vmax=1)
    ax[4].set_title("ood_map")
    ax[5].imshow(s, cmap="inferno", vmin=0, vmax=1)
    ax[5].set_title("consistency_map")
    if e is not None:
        ax[6].imshow(e, cmap="Reds", vmin=0, vmax=1)
        ax[6].set_title("error_map")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def run_inference(model: torch.nn.Module, loader, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "visualizations").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    case_scores = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["image"].to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float().cpu()

            noisy_x = x + 0.03 * torch.randn_like(x)
            noisy_probs = torch.sigmoid(model(noisy_x))
            parts = reliability_components(noisy_probs, probs, x, enable_ood=True)
            reliability = (0.35 * parts["confidence_map"] + 0.25 * parts["entropy_map"] + 0.25 * parts["consistency_map"] + 0.15 * parts["ood_map"]).clamp(0, 1)

            save_obj = {
                "pred": pred,
                "confidence": probs.cpu(),
                "reliability": reliability.cpu(),
                "ood_map": parts["ood_map"].cpu(),
                "consistency_map": parts["consistency_map"].cpu(),
            }

            error_map = None
            if "label" in batch:
                y = batch["label"].float()
                error_map = (pred - y).abs().clamp(0, 1)
                save_obj["error_map"] = error_map
                case_scores.append((i, float(error_map.mean().item())))

            torch.save(save_obj, out_path / f"case_{i:04d}.pt")
            _save_case_plot(
                x.cpu(),
                pred,
                probs.cpu(),
                reliability.cpu(),
                parts["ood_map"].cpu(),
                parts["consistency_map"].cpu(),
                out_path / "visualizations" / f"case_{i:04d}.png",
                error_map=error_map,
            )

    if case_scores:
        case_scores.sort(key=lambda x: x[1], reverse=True)
        with open(out_path / "worst_cases.txt", "w", encoding="utf-8") as f:
            for idx, err in case_scores[:20]:
                f.write(f"case_{idx:04d}: mean_error={err:.6f}\n")