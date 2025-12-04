import requests
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
SYMBOL = "NKNUSDT"          # Binance symbol
N_TRADES = 10000            # how many most recent trades to use (we paginate)
N_VOL_BINS = 8              # number of volume bins
TAUS_SECONDS = [5, 10, 30, 60, 100]   # time lags τ to study (in seconds)
MIN_NOTIONAL = 0.0          # filter tiny trades (in quote currency), e.g. 5.0
BINANCE_URL = "https://api.binance.com"

# Autocorrelation config
MAX_SIGN_LAG = 50           # max lag (in number of trades) for sign autocorrelation


# =========================
# DATA FETCH
# =========================

def fetch_recent_agg_trades(symbol: str, n_trades: int) -> pd.DataFrame:
    """
    Fetch up to n_trades *most recent* aggregated trades from Binance.
    We paginate backwards using fromId.
    """
    trades = []
    remaining = n_trades
    last_id = None

    while remaining > 0:
        limit = min(remaining, 1000)
        params = {"symbol": symbol, "limit": limit}
        if last_id is not None:
            # fetch trades BEFORE last_id
            params["fromId"] = max(last_id - limit, 0)

        resp = requests.get(BINANCE_URL + "/api/v3/aggTrades",
                            params=params, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        trades = batch + trades
        remaining -= len(batch)
        last_id = batch[0]["a"]  # smallest id in this batch

        time.sleep(0.05)

        if len(batch) < limit:
            break

    if not trades:
        raise RuntimeError("No trades returned from Binance.")

    df = pd.DataFrame(trades)

    # Columns: a aggTradeId, p price, q qty, T timestamp ms, m isBuyerMaker (bool)
    df["price"] = df["p"].astype(float)
    df["qty"] = df["q"].astype(float)
    df["time_ms"] = df["T"].astype("int64")
    df["time"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True)

    # Trade notional in quote currency
    df["notional"] = df["price"] * df["qty"]

    # Trade sign ε_t: +1 for buyer-initiated, -1 for seller-initiated
    # Binance: m = True => buyer is maker => trade was SELL-initiated
    df["sign"] = np.where(df["m"], -1.0, 1.0)

    df = df.sort_values("time_ms").reset_index(drop=True)
    return df


# =========================
# RESPONSE FUNCTION R(V, τ)
# =========================

def compute_R_V_tau(df: pd.DataFrame,
                    taus_seconds,
                    n_bins: int,
                    min_notional: float = 0.0):
    """
    Compute Bouchaud-style response function:

        R(V_bin, τ) = E[ ε_t * (p(t+τ) - p(t)) / p(t) | trade volume in bin ]

    using trade price as a proxy for mid-price.

    Returns:
        log_vol_centers : np.array shape (n_bins,)
        taus            : np.array shape (n_tau,)
        R               : np.array shape (n_bins, n_tau)
    """
    # Filter by notional if requested
    if min_notional > 0:
        df = df[df["notional"] >= min_notional].reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Not enough trades after filtering to compute response.")

    times = df["time_ms"].to_numpy()
    prices = df["price"].to_numpy()
    signs = df["sign"].to_numpy()
    vols = df["notional"].to_numpy()  # use notional; switch to qty if you prefer

    # Volume bins (quantiles)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(vols, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) - 1 < n_bins:
        n_bins = len(bin_edges) - 1
        print(f"Reduced number of bins to {n_bins} due to many identical volumes.")
    taus = np.array(taus_seconds, dtype=float)
    n_tau = len(taus)

    R = np.full((n_bins, n_tau), np.nan)
    log_vol_centers = np.zeros(n_bins)

    for b in range(n_bins):
        v_min = bin_edges[b]
        v_max = bin_edges[b + 1]
        idx_bin = np.where((vols >= v_min) & (vols <= v_max))[0]
        if len(idx_bin) == 0:
            continue

        # center volume (for plotting) = mean notional in bin
        vol_center = vols[idx_bin].mean()
        log_vol_centers[b] = np.log(vol_center)

        for k, tau in enumerate(taus):
            tau_ms = tau * 1000
            contribs = []

            for i in idx_bin:
                t0 = times[i]
                target = t0 + tau_ms
                j = np.searchsorted(times, target, side="left")
                if j >= len(times):
                    continue

                p0 = prices[i]
                p_tau = prices[j]
                eps = signs[i]

                contrib = eps * (p_tau - p0) / p0
                contribs.append(contrib)

            if contribs:
                R[b, k] = np.mean(contribs)

    return log_vol_centers, taus, R


# =========================
# SIGN AUTOCORRELATION
# =========================

def compute_sign_autocorrelation(signs: np.ndarray, max_lag: int) -> (np.ndarray, np.ndarray):
    """
    Compute autocorrelation of trade signs:
        ρ(ℓ) = Corr(ε_t, ε_{t+ℓ})

    signs: array of +1/-1 (or any real values)
    max_lag: maximum lag in *number of trades*

    Returns:
        lags: np.array of lags [1..max_lag]
        rho: np.array of autocorrelations
    """
    signs = np.asarray(signs, dtype=float)
    n = len(signs)
    if n < max_lag + 1:
        max_lag = n - 1
        print(f"Reduced MAX_SIGN_LAG to {max_lag} due to short series.")

    lags = np.arange(1, max_lag + 1, dtype=int)
    rho = np.zeros_like(lags, dtype=float)

    mean = signs.mean()
    var = signs.var()
    if var == 0:
        return lags, np.zeros_like(lags, dtype=float)

    for idx, lag in enumerate(lags):
        x = signs[:-lag]
        y = signs[lag:]
        cov = np.mean((x - mean) * (y - mean))
        rho[idx] = cov / var

    return lags, rho


# =========================
# MAIN
# =========================

def main():
    print(f"Fetching last {N_TRADES} trades for {SYMBOL} from Binance...")
    df = fetch_recent_agg_trades(SYMBOL, N_TRADES)
    print(f"Fetched {len(df)} trades from {df['time'].min()} to {df['time'].max()}.")

    # ---------------------
    # RESPONSE FUNCTION R(V, τ)
    # ---------------------
    logV, taus, R = compute_R_V_tau(
        df,
        taus_seconds=TAUS_SECONDS,
        n_bins=N_VOL_BINS,
        min_notional=MIN_NOTIONAL,
    )

    # 1) Plot R(V, τ) curves
    plt.figure(figsize=(8, 5))
    for k, tau in enumerate(taus):
        if np.all(np.isnan(R[:, k])):
            continue
        plt.plot(
            logV,
            R[:, k],
            marker="o",
            label=f"τ = {int(tau)} s"
        )

    plt.xlabel("ln(volume in quote currency)")
    plt.ylabel("R(V, τ) = E[ ε · Δp / p | V ]")
    plt.title(f"Price Response Function R(V, τ) for {SYMBOL}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Bouchaud concavity fit: R(V, τ) ≈ A(τ) ln V + B(τ)
    slopes = []
    intercepts = []

    print("\nBouchaud concavity fit: R(V, τ) ≈ A(τ) ln V + B(τ)")
    print("τ (s)\tA(τ)\t\tB(τ)\t\tR^2")

    for k, tau in enumerate(taus):
        y = R[:, k]
        x = logV
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            slopes.append(np.nan)
            intercepts.append(np.nan)
            continue

        x_fit = x[mask]
        y_fit = y[mask]

        # linear regression using numpy.polyfit: y ≈ a*x + b
        a, b = np.polyfit(x_fit, y_fit, 1)

        # R^2
        y_pred = a * x_fit + b
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        slopes.append(a)
        intercepts.append(b)

        print(f"{int(tau):4d}\t{a: .3e}\t{b: .3e}\t{r2: .3f}")

    slopes = np.array(slopes)
    intercepts = np.array(intercepts)

    # Plot A(τ) vs τ
    plt.figure(figsize=(6, 4))
    plt.plot(taus, slopes, marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("τ (seconds)")
    plt.ylabel("A(τ)  (slope wrt ln V)")
    plt.title(f"Volume Concavity: A(τ) for {SYMBOL}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------------
    # 3) TRADE SIGN AUTOCORRELATION
    # ---------------------
    signs = df["sign"].to_numpy()
    lags, rho = compute_sign_autocorrelation(signs, MAX_SIGN_LAG)

    # Print a few values
    print("\nTrade sign autocorrelation ρ(ℓ) = Corr(ε_t, ε_{t+ℓ})")
    print("lag\tρ(lag)")
    for lag, r in zip(lags, rho):
        print(f"{lag:3d}\t{r: .4f}")

    # Plot autocorrelation
    plt.figure(figsize=(7, 4))
    plt.stem(lags, rho)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Lag (number of trades)")
    plt.ylabel("ρ(lag)")
    plt.title(f"Trade Sign Autocorrelation for {SYMBOL}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
