import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# ── Load data ────────────────────────────────────────────────────────────
df = pl.read_parquet("src/data/model_staging/tech_modeling_table.parquet")
print(f"Shape: {df.shape}")

# Separate features from targets
feature_cols = [c for c in df.columns if c not in ("symbol", "earnings_date", "target_return", "target_direction")]

X = df.select(feature_cols).to_numpy().astype(np.float32)
y = df["target_direction"].to_numpy().astype(np.float32)

# Handle infs/nulls
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


X.shape, y.shape

# ── Identify time-varying columns (pattern: name_t-10, name_t0, name_t+1) ─
import re
time_cols = [c for c in feature_cols if re.match(r".+_t[+-]?\d+$", c)]
static_cols = [c for c in feature_cols if not re.match(r".+_t[+-]?\d+$", c)]

# Parse step and base name
bases = sorted(set(re.sub(r"_t[+-]?\d+$", "", c) for c in time_cols))
steps = sorted({int(re.search(r"_t([+-]?\d+)$", c).group(1)) for c in time_cols})
n_steps, n_bases = len(steps), len(bases)
n_static = len(static_cols)

print(f"Time-varying: {n_bases} bases × {n_steps} steps = {n_bases * n_steps} cols")
print(f"Static: {n_static} cols")
print(f"Steps: {steps}")

# Build lookup: col name -> index
col_to_idx = {c: i for i, c in enumerate(feature_cols)}

# Build column grid: (step, base) -> col index
step_to_idx = {s: i for i, s in enumerate(steps)}
base_to_idx = {b: i for i, b in enumerate(bases)}
col_grid = np.full((n_steps, n_bases), -1, dtype=int)
for c in time_cols:
    s = int(re.search(r"_t([+-]?\d+)$", c).group(1))
    b = re.sub(r"_t[+-]?\d+$", "", c)
    col_grid[step_to_idx[s], base_to_idx[b]] = col_to_idx[c]

# Reshape X into (samples, steps, bases)
X_time = np.zeros((X.shape[0], n_steps, n_bases), dtype=np.float32)
for s in range(n_steps):
    for b in range(n_bases):
        idx = col_grid[s, b]
        if idx >= 0:
            X_time[:, s, b] = X[:, idx]

X_static = X[:, [col_to_idx[c] for c in static_cols]]

# Standardize
t_scaler = StandardScaler()
X_time = t_scaler.fit_transform(X_time.reshape(-1, n_bases)).reshape(X_time.shape)
if n_static > 0:
    s_scaler = StandardScaler()
    X_static = s_scaler.fit_transform(X_static)

# ── Simple LSTM model ────────────────────────────────────────────────────
class SimpleLSTM(nn.Module):
    def __init__(self, n_bases, n_static, hidden=64):
        super().__init__()
        self.n_static = n_static
        self.lstm = nn.LSTM(n_bases, hidden, batch_first=True)
        self.fc = nn.Linear(hidden + n_static, 1)

    def forward(self, x_seq, x_stat):
        out, (hn, _) = self.lstm(x_seq)
        last = hn[-1]
        if self.n_static > 0:
            last = torch.cat([last, x_stat], dim=1)
        return self.fc(last).squeeze(1)

class SimpleBiLSTM(nn.Module):
    def __init__(self, n_bases, n_static, hidden=64):
        super().__init__()
        self.n_static = n_static
        self.lstm = nn.LSTM(n_bases, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2 + n_static, 1)

    def forward(self, x_seq, x_stat):
        out, (hn, _) = self.lstm(x_seq)
        last = torch.cat((hn[-2], hn[-1]), dim=1)
        if self.n_static > 0:
            last = torch.cat([last, x_stat], dim=1)
        return self.fc(last).squeeze(1)

# ── Simple train loop ────────────────────────────────────────────────────
def train_model(model, loader, epochs=20, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        total_loss, n = 0, 0
        for x_seq, x_stat, y in loader:
            x_seq, x_stat, y = x_seq.to(device), x_stat.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x_seq, x_stat)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}: loss={total_loss/n:.4f}")

# ── Simple test loop ────────────────────────────────────────────────────
def eval_model(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x_seq, x_stat, y in loader:
            x_seq, x_stat = x_seq.to(device), x_stat.to(device)
            logits = model(x_seq, x_stat)
            preds.append(torch.sigmoid(logits).cpu().numpy())
            trues.append(y.numpy())
    preds = np.concatenate(preds) >= 0.5
    return accuracy_score(np.concatenate(trues), preds)

# ── Run LSTM and BiLSTM ─────────────────────────────────────────────────
BATCH_SIZE = 128
dataset = TensorDataset(
    torch.tensor(X_time), torch.tensor(X_static), torch.tensor(y)
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for name, Model in [("LSTM", SimpleLSTM), ("BiLSTM", SimpleBiLSTM)]:
    print(f"\n--- {name} ---")
    model = Model(n_bases, n_static).to(device)
    train_model(model, loader, epochs=20)
    acc = eval_model(model, loader)
    print(f"Train accuracy: {acc:.4f}")

# ── Baseline comparison ─────────────────────────────────────────────────
print(f"\n--- Baseline ---")
print(f"Random (DA):             {0.50:.4f}")
print(f"Always majority class:   {max(y.mean(), 1-y.mean()):.4f}")
