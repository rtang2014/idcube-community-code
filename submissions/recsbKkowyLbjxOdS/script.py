
# hsi_transformer_autoencoder.py
import os, math, argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

try:
    from scipy.io import loadmat, savemat
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------
# Data utilities
# -----------------------------
def load_cube(path):
    """
    Load an HxWxB hyperspectral cube.
    Supports .npy or .mat with key 'Images' (HxWxB).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        cube = np.load(path)
    elif ext == ".mat":
        if not HAS_SCIPY:
            raise RuntimeError("Install scipy to load .mat files.")
        md = loadmat(path)

        print("MAT keys:", [k for k in md.keys() if not k.startswith('__')])

        # Try common keys
        for k in ["Images", "Image", "cube", "CUBE"]:
            if k in md:
                cube = md[k]
                break
        else:
            raise KeyError("Cannot find 'Images' (or cube) in .mat file.")
    else:
        raise ValueError("Use a .npy or .mat file.")
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError("Cube must be HxWxB.")
    return cube  # H, W, B


def spectra_from_cube(cube):
    H, W, B = cube.shape
    X = cube.reshape(-1, B)  # N x B
    return X


def minmax_per_row(X, eps=1e-8):
    xmin = X.min(axis=1, keepdims=True)
    xmax = X.max(axis=1, keepdims=True)
    Xn = (X - xmin) / np.maximum(xmax - xmin, eps)
    return Xn, xmin, xmax


# -----------------------------
# Positional Encoding (optional but helpful)
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x):
        """
        x: [L, N, D]
        """
        L = x.size(0)
        return x + self.pe[:L].unsqueeze(1)  # broadcast over batch


# -----------------------------
# Model: Transformer Autoencoder
# -----------------------------
class HSITokenEmbed(nn.Module):
    """
    Project 1-channel sequence (per-band scalar) to d_model channels via 1x1 conv.
    Input for conv1d: [N, C=1, L=B]  →  [N, d_model, L]
    """
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(1, d_model, kernel_size=1)

    def forward(self, x):  # x: [N, 1, L]
        return self.conv(x)


class HSITransformerAutoencoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, d_ff=128, num_layers=2, seq_len=150, use_posenc=True):
        super().__init__()
        self.token = HSITokenEmbed(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=False, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.use_posenc = use_posenc
        self.posenc = SinusoidalPositionalEncoding(d_model, max_len=seq_len) if use_posenc else None
        # Reconstruction head: 1x1 conv back to 1 channel
        self.recon = nn.Conv1d(d_model, 1, kernel_size=1)
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [N, B] normalized spectra
        returns:
          y_hat: [N, B] reconstructed spectra
          z:     [N, D] embedding (mean-pooled encoder output)
        """
        N, B = x.shape
        x1 = x.unsqueeze(1)                                # [N, 1, B]
        h = self.token(x1)                                 # [N, D, B]
        # Transformer expects [L, N, D]
        h_tr = h.permute(2, 0, 1).contiguous()             # [B, N, D]
        if self.use_posenc:
            h_tr = self.posenc(h_tr)
        h_enc = self.encoder(h_tr)                         # [B, N, D]
        h_enc = self.ln_final(h_enc)                       # [B, N, D]
        # Mean pool over sequence length (bands)
        z = h_enc.mean(dim=0)                              # [N, D]
        # Back to [N, D, B] then 1x1 conv to 1 channel
        h_back = h_enc.permute(1, 2, 0).contiguous()       # [N, D, B]
        y_hat = self.recon(h_back).squeeze(1)              # [N, B]
        return y_hat, z


# -----------------------------
# Training loop
# -----------------------------
def train_autoencoder(model, Xn, epochs=30, batch_size=256, lr=1e-3, noise_sigma=0.01, device="cpu"):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    N, B = Xn.shape
    num_batches = math.ceil(N / batch_size)
    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        for bi in range(num_batches):
            idx = perm[bi * batch_size : (bi + 1) * batch_size]
            xb = torch.from_numpy(Xn[idx.numpy(), :]).to(device)
            with torch.no_grad():
                if noise_sigma > 0:
                    noise = noise_sigma * torch.randn_like(xb)
                    xb_in = (xb + noise).clamp(0.0, 1.0)
                else:
                    xb_in = xb
            y_hat, _ = model(xb_in)
            loss = loss_fn(y_hat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {ep:03d}/{epochs} | loss {epoch_loss/num_batches:.6f}")
    return model


# -----------------------------
# Inference utilities
# -----------------------------
@torch.no_grad()
def reconstruct_and_embed(model, Xn, batch_size=512, device="cpu"):
    model.eval().to(device)
    N, B = Xn.shape
    yhat = np.zeros_like(Xn, dtype=np.float32)
    # embedding size:
    test = torch.from_numpy(Xn[:2]).to(device)
    _, ztest = model(test)
    D = ztest.shape[1]
    Z = np.zeros((N, D), dtype=np.float32)

    num_batches = math.ceil(N / batch_size)
    for bi in range(num_batches):
        s = bi * batch_size
        e = min((bi + 1) * batch_size, N)
        xb = torch.from_numpy(Xn[s:e]).to(device)
        yh, z = model(xb)
        yhat[s:e] = yh.cpu().numpy()
        Z[s:e] = z.cpu().numpy()
    return yhat, Z


def unscale_rows(Xn, xmin, xmax, eps=1e-8):
    return Xn * np.maximum(xmax - xmin, eps) + xmin


# -----------------------------
# Main
# -----------------------------
def main():
    # Hardcoded path: cube.mat in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cube_path = os.path.join(script_dir, "cube.mat")

    if not os.path.isfile(cube_path):
        raise FileNotFoundError(f"cube.mat not found in {script_dir}")

    cube = load_cube(cube_path)
    H, W, B = cube.shape
    print(f"Loaded cube: {H}x{W}x{B}")

    # Prepare data
    X = spectra_from_cube(cube)                   # N x B
    Xn, xmin, xmax = minmax_per_row(X)           # normalize per-spectrum to [0,1]

    # Model
    model = HSITransformerAutoencoder(
        d_model=64,
        nhead=4,
        d_ff=128,
        num_layers=2,
        seq_len=B,
        use_posenc=True
    )

    # Train
    model = train_autoencoder(
        model, Xn, epochs=30, batch_size=256, lr=1e-3,
        noise_sigma=0.01, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Reconstruct + embeddings
    Xrec_n, Z = reconstruct_and_embed(model, Xn.astype(np.float32),
                                      batch_size=512,
                                      device="cuda" if torch.cuda.is_available() else "cpu")
    Xrec = unscale_rows(Xrec_n, xmin, xmax)  # unscaled reconstruction
    rec_cube = Xrec.reshape(H, W, B)         # reshape to cube

    # Cluster embeddings
    print("Clustering embeddings with k-means...")
    km = KMeans(n_clusters=5, n_init=10, random_state=42)
    labels = km.fit_predict(Z)
    seg_map = labels.reshape(H, W)

    # Simple metric
    mse_per_spec = ((Xn - Xrec_n) ** 2).mean(axis=1)
    print(f"Mean per-spectrum MSE (normalized): {mse_per_spec.mean():.6f}")

    # --- Always save outputs (timestamped) ---
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(script_dir, f"recCube_{ts}.npy"), rec_cube)
    np.save(os.path.join(script_dir, f"segMap_{ts}.npy"), seg_map)

    try:
        from scipy.io import savemat
        savemat(os.path.join(script_dir, f"out_{ts}.mat"), {
            "recCube": rec_cube,
            "segMap": seg_map,
            "embedFeatures": Z
        })
        print(f"Saved outputs to {script_dir}: recCube_{ts}.npy, segMap_{ts}.npy, out_{ts}.mat")
    except ImportError:
        print("scipy not installed → skipped .mat save (install with `pip install scipy`).")

    # Plot if available
    if HAS_PLT:
        mid = B // 2
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(seg_map)
        plt.title("Embedding Clusters")
        plt.axis("off"); plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(cube[..., mid])
        plt.title("Original (mid band)")
        plt.axis("off"); plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(rec_cube[..., mid])
        plt.title("Reconstructed (mid band)")
        plt.axis("off"); plt.colorbar()
        plt.tight_layout()
        plt.show()
    else:
        print("matplotlib not available; skipping plots.")

if __name__ == "__main__":
    main()