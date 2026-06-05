#!/usr/bin/env python3
"""L.U.M.U.S.-8 Alpha Engine TTL audit backtest.

The backtest intentionally uses today's obtainable universe. It is useful as an
operational TTL audit, not as a point-in-time constituent reconstruction.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TTL_LIST = [30, 60, 90, 120, 180]
START_DATE = "2018-01-01"
LOOKBACK_DAYS = 252
AUDIT_NOTE = "本検証は現在取得可能な銘柄ユニバースを用いた簡易バックテストであり、サバイバーシップバイアスを完全には除去していない。"
US_STATIC_FALLBACK = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "LLY", "JPM", "V", "WMT", "XOM", "CAT", "COST"]
JP_TICKERS = [
    "7203.T", "6758.T", "8306.T", "8035.T", "9984.T", "9432.T", "6861.T", "6098.T",
    "4063.T", "6954.T", "7974.T", "6301.T", "4568.T", "6501.T", "7741.T", "7267.T",
    "6273.T", "4543.T", "8058.T", "8001.T", "8031.T", "8053.T", "8002.T", "8316.T",
    "8411.T", "8766.T", "8801.T", "8802.T", "8591.T", "8725.T", "8750.T", "6857.T",
    "6146.T", "6723.T", "6920.T", "7735.T", "6981.T", "6503.T", "6702.T", "6752.T",
    "6506.T", "6965.T", "7729.T", "6869.T", "6971.T", "6315.T", "4062.T", "7701.T",
    "7011.T", "7012.T", "7013.T", "6367.T", "6113.T", "6481.T", "1801.T", "1802.T",
    "1803.T", "1812.T", "1925.T", "1928.T", "1808.T", "1721.T", "5803.T", "5802.T",
    "7201.T", "7269.T", "7270.T", "5401.T", "5713.T", "1605.T", "5020.T", "9101.T",
    "9104.T", "9107.T", "3407.T", "4188.T", "4452.T", "4911.T", "4183.T", "9983.T",
    "3382.T", "7453.T", "3092.T", "4661.T", "4385.T", "2413.T", "4689.T", "4755.T",
    "9735.T", "3659.T", "4307.T", "3088.T", "3064.T", "2802.T", "2502.T", "2503.T",
    "4502.T", "4519.T", "4503.T", "4523.T", "9020.T", "9021.T", "9022.T", "9201.T",
    "9202.T", "9501.T", "9502.T", "9503.T",
]


@dataclass
class BacktestResult:
    ttl: int
    use_exit: bool
    trading_cost: float
    equity: pd.Series
    drawdown: pd.Series
    metrics: dict[str, float | int]
    trades: pd.DataFrame
    exit_warnings: pd.DataFrame
    rebalances: pd.DataFrame


def get_tickers_lumus(timeout: int = 15) -> tuple[list[str], list[str], str]:
    """Return current S&P 500 constituents and the engine's curated JP list."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) >= 100:
            return tickers, JP_TICKERS.copy(), "Wikipedia"
    except Exception:
        pass
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        tickers = pd.read_csv(io.StringIO(response.text))["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) >= 100:
            return tickers, JP_TICKERS.copy(), "GitHub CSV"
    except Exception:
        pass
    return US_STATIC_FALLBACK.copy(), JP_TICKERS.copy(), "Static fallback"


def _close_frame(download: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if download.empty:
        return pd.DataFrame()
    if isinstance(download.columns, pd.MultiIndex):
        if "Close" not in download.columns.get_level_values(0):
            return pd.DataFrame()
        close = download["Close"]
    else:
        close = download[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def download_prices(tickers: Iterable[str], start: str, end: str | None = None, batch_size: int = 100) -> pd.DataFrame:
    """Download adjusted close prices in batches without backward-filling history."""
    tickers = list(dict.fromkeys(tickers))
    frames = []
    for offset in range(0, len(tickers), batch_size):
        batch = tickers[offset : offset + batch_size]
        raw = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False, threads=True)
        frames.append(_close_frame(raw, batch))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).loc[:, lambda frame: ~frame.columns.duplicated()].sort_index()


def safe_zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def calculate_scores(prices: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    """Calculate signals using only closes at or before signal_date."""
    history = prices.loc[:signal_date]
    records: dict[str, dict[str, float]] = {}
    for ticker in history.columns:
        series = history[ticker].dropna()
        if len(series) < LOOKBACK_DAYS:
            continue
        window = series.iloc[-LOOKBACK_DAYS:]
        daily_ret = window.pct_change(fill_method=None).dropna()
        vol = daily_ret.std() * math.sqrt(252)
        if not np.isfinite(vol) or vol <= 0:
            continue
        p_now = window.iloc[-1]
        composite_ret = ((p_now / window.iloc[-252] - 1) * 3 + (p_now / window.iloc[-126] - 1) * 2 + (p_now / window.iloc[-63] - 1)) / 6
        avg_pos = daily_ret[daily_ret > 0].mean()
        avg_neg = abs(daily_ret[daily_ret < 0].mean())
        quality = avg_pos / avg_neg if np.isfinite(avg_neg) and avg_neg > 0 else 1.0
        prox_high = p_now / window.max()
        records[ticker] = {
            "Efficiency": composite_ret / vol,
            "Quality": quality,
            "Valuation_Alt": (1 / prox_high) * (1 / vol),
            "Volatility": vol,
            "Composite_Ret": composite_ret,
        }
    scores = pd.DataFrame(records).T
    if scores.empty:
        return scores
    scores["Total_Score"] = safe_zscore(scores["Efficiency"]) * 0.4 + safe_zscore(scores["Quality"]) * 0.4 + safe_zscore(scores["Valuation_Alt"]) * 0.2
    return scores.sort_values("Total_Score", ascending=False)


def determine_exposure(index_prices: pd.DataFrame, signal_date: pd.Timestamp) -> tuple[float, str, str]:
    regimes = []
    for ticker in ["^GSPC", "^N225"]:
        series = index_prices.loc[:signal_date, ticker].dropna() if ticker in index_prices else pd.Series(dtype=float)
        regimes.append("BULL" if len(series) >= 200 and series.iloc[-1] > series.iloc[-200:].mean() else "BEAR")
    exposure = 1.0 if regimes == ["BULL", "BULL"] else 0.6 if "BULL" in regimes else 0.2
    return exposure, regimes[0], regimes[1]


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def _next_rebalance_date(calendar: pd.DatetimeIndex, current: pd.Timestamp, ttl: int) -> pd.Timestamp | None:
    candidates = calendar[calendar >= current + pd.Timedelta(days=ttl)]
    return candidates[0] if len(candidates) else None


def _portfolio_targets(scores_us: pd.DataFrame, scores_jp: pd.DataFrame, exposure: float) -> dict[str, float]:
    selected = pd.concat([scores_us.head(6), scores_jp.head(6)])
    if selected.empty:
        return {}
    inverse_vol = 1 / selected["Volatility"]
    return (inverse_vol / inverse_vol.sum() * exposure).to_dict()


def run_backtest(
    signal_prices: pd.DataFrame,
    return_prices: pd.DataFrame,
    index_prices: pd.DataFrame,
    us_tickers: list[str],
    jp_tickers: list[str],
    ttl: int,
    start_date: str = START_DATE,
    use_exit: bool = False,
    trading_cost: float = 0.0,
) -> BacktestResult:
    """Backtest one TTL. Signals use the prior available close; trades occur at today's close."""
    calendar = return_prices.loc[start_date:].index
    if len(calendar) < 2:
        raise ValueError("Not enough return-price observations in the requested backtest period")
    returns = return_prices.reindex(calendar).ffill().pct_change(fill_method=None).fillna(0.0)
    signal_prices = signal_prices.reindex(return_prices.index).ffill()
    index_prices = index_prices.reindex(return_prices.index).ffill()
    weights: dict[str, float] = {}
    entry_dates: dict[str, pd.Timestamp] = {}
    entry_prices: dict[str, float] = {}
    equity_value = 1.0
    next_rebalance = calendar[0]
    equity_rows, trade_rows, warning_rows, rebalance_rows = [], [], [], []
    turnover = 0.0

    def transact(ticker: str, side: str, weight: float, when: pd.Timestamp, reason: str) -> None:
        nonlocal equity_value, turnover
        if weight <= 1e-12:
            return
        turnover += weight
        equity_value *= max(0.0, 1 - weight * trading_cost)
        local_price = signal_prices.at[when, ticker] if ticker in signal_prices else np.nan
        trade_rows.append({"Date": when, "Ticker": ticker, "Side": side, "Weight": weight, "Reason": reason, "Local_Price": local_price})
        if side == "BUY":
            entry_dates[ticker] = when
            entry_prices[ticker] = local_price
        elif ticker in entry_dates:
            trade_rows[-1]["Holding_Days"] = (when - entry_dates.pop(ticker)).days
            buy_price = entry_prices.pop(ticker, np.nan)
            trade_rows[-1]["Trade_Return"] = local_price / buy_price - 1 if np.isfinite(buy_price) and buy_price else np.nan

    for position, current in enumerate(calendar):
        if position:
            daily_return = sum(weight * returns.at[current, ticker] for ticker, weight in weights.items() if ticker in returns)
            equity_value *= 1 + daily_return

        if use_exit and weights:
            # Exit signals also use the prior available close. This avoids deciding
            # from today's close and unrealistically selling at that same close.
            exit_signal_dates = signal_prices.index[signal_prices.index < current]
            exit_signal_date = exit_signal_dates[-1] if len(exit_signal_dates) else None
            for ticker in list(weights):
                close = signal_prices.loc[:exit_signal_date, ticker].dropna() if exit_signal_date is not None else pd.Series(dtype=float)
                if len(close) < 50:
                    continue
                current_price, sma20, sma50 = close.iloc[-1], close.iloc[-20:].mean(), close.iloc[-50:].mean()
                rsi14 = calculate_rsi(close).iloc[-1]
                if rsi14 >= 80:
                    warning_rows.append({"Signal_Date": exit_signal_date, "Trade_Date": current, "Ticker": ticker, "RSI14": rsi14, "Action": "WARNING_ONLY"})
                reason = "SMA50_EXIT" if current_price < sma50 else "SMA20_EXIT" if current_price < sma20 else None
                if reason:
                    old_weight = weights.pop(ticker)
                    transact(ticker, "SELL", old_weight, current, reason)

        if current >= next_rebalance:
            if position == 0:
                # There is no prior close on the requested calendar for the first day.
                # The broader downloaded history still supplies the signal close.
                signal_candidates = signal_prices.index[signal_prices.index < current]
            else:
                signal_candidates = signal_prices.index[signal_prices.index < current]
            if len(signal_candidates):
                signal_date = signal_candidates[-1]
                scores_us = calculate_scores(signal_prices[us_tickers], signal_date)
                scores_jp = calculate_scores(signal_prices[jp_tickers], signal_date)
                exposure, us_regime, jp_regime = determine_exposure(index_prices, signal_date)
                targets = _portfolio_targets(scores_us, scores_jp, exposure)
                for ticker, old_weight in list(weights.items()):
                    transact(ticker, "SELL", old_weight, current, "REBALANCE")
                weights = {}
                for ticker, target_weight in targets.items():
                    transact(ticker, "BUY", target_weight, current, "REBALANCE")
                    weights[ticker] = target_weight
                rebalance_rows.append({"Trade_Date": current, "Signal_Date": signal_date, "TTL": ttl, "Exposure": exposure, "US_Regime": us_regime, "JP_Regime": jp_regime, "Selected": len(weights)})
            following = _next_rebalance_date(calendar, current, ttl)
            next_rebalance = following if following is not None else calendar[-1] + pd.Timedelta(days=ttl)
        equity_rows.append((current, equity_value))

    for ticker, old_weight in list(weights.items()):
        transact(ticker, "SELL", old_weight, calendar[-1], "FINAL_LIQUIDATION")
    if equity_rows:
        equity_rows[-1] = (equity_rows[-1][0], equity_value)
    equity = pd.Series(dict(equity_rows), name=f"TTL {ttl}")
    drawdown = equity / equity.cummax() - 1
    trades = pd.DataFrame(trade_rows)
    metrics = calculate_metrics(equity, trades, turnover, len(rebalance_rows))
    return BacktestResult(ttl, use_exit, trading_cost, equity, drawdown, metrics, trades, pd.DataFrame(warning_rows), pd.DataFrame(rebalance_rows))


def calculate_metrics(equity: pd.Series, trades: pd.DataFrame, turnover: float, number_of_rebalances: int) -> dict[str, float | int]:
    daily = equity.pct_change(fill_method=None).dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = daily.std() * math.sqrt(252)
    downside = daily[daily < 0].std() * math.sqrt(252)
    mdd = (equity / equity.cummax() - 1).min()
    monthly = equity.resample("ME").last().pct_change(fill_method=None).dropna()
    sells = trades[trades["Side"] == "SELL"] if not trades.empty else pd.DataFrame()
    trade_returns = sells.get("Trade_Return", pd.Series(dtype=float)).dropna()
    wins, losses = trade_returns[trade_returns > 0], trade_returns[trade_returns < 0]
    return {
        "CAGR": cagr, "Annualized_Volatility": vol, "Sharpe_Ratio": daily.mean() / daily.std() * math.sqrt(252) if daily.std() else np.nan,
        "Sortino_Ratio": daily.mean() / downside * 252 if downside else np.nan, "Max_Drawdown": mdd, "Calmar_Ratio": cagr / abs(mdd) if mdd else np.nan,
        "Total_Return": equity.iloc[-1] - 1, "Monthly_Win_Rate": (monthly > 0).mean() if len(monthly) else np.nan,
        "Number_of_Rebalances": number_of_rebalances, "Number_of_Trades": len(trades), "Average_Holding_Days": sells.get("Holding_Days", pd.Series(dtype=float)).mean(),
        "Turnover": turnover, "Best_Month": monthly.max() if len(monthly) else np.nan, "Worst_Month": monthly.min() if len(monthly) else np.nan,
        "Average_Trade_Return": trade_returns.mean(), "Win_Rate_per_Trade": (trade_returns > 0).mean() if len(trade_returns) else np.nan,
        "Average_Win": wins.mean(), "Average_Loss": losses.mean(), "Profit_Factor": wins.sum() / abs(losses.sum()) if len(losses) else np.nan,
        "Largest_Losing_Trade": losses.min() if len(losses) else np.nan,
    }


def comparison_table(results: list[BacktestResult]) -> pd.DataFrame:
    frame = pd.DataFrame({result.ttl: result.metrics for result in results}).T
    frame.index.name = "TTL"
    return frame


def recommendation(table: pd.DataFrame) -> tuple[int, str, str]:
    ranks = pd.DataFrame(index=table.index)
    ranks["Calmar"] = table["Calmar_Ratio"].rank(ascending=False)
    ranks["Sortino"] = table["Sortino_Ratio"].rank(ascending=False)
    ranks["MDD"] = table["Max_Drawdown"].rank(ascending=False)
    ranks["CAGR"] = table["CAGR"].rank(ascending=False)
    ranks["Turnover"] = table["Turnover"].rank(ascending=True)
    winner = int(ranks.mean(axis=1).idxmin())
    verdict = "A：60日TTLは維持" if winner == 60 else "B：90日TTLへ変更検討" if winner == 90 else "C：30日TTLへ短縮検討" if winner == 30 else "D：120日以上へ延長検討"
    reason = f"主要5指標の平均順位が最良（{ranks.mean(axis=1).loc[winner]:.2f}位）。{verdict}。"
    return winner, reason, AUDIT_NOTE


def save_charts(results: list[BacktestResult], output_dir: Path, case_slug: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.concat([result.equity for result in results], axis=1)
    drawdown = pd.concat([result.drawdown.rename(f"TTL {result.ttl}") for result in results], axis=1)
    table = comparison_table(results)
    for frame, title, ylabel, filename in [(equity, "TTL Equity Curves", "Growth of 1.0", "equity_curves"), (drawdown, "TTL Drawdown Curves", "Drawdown", "drawdown_curves")]:
        ax = frame.plot(figsize=(11, 6), title=title)
        ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(output_dir / f"{case_slug}_{filename}.png", dpi=150); plt.close(ax.figure)
    ax = table[["CAGR", "Max_Drawdown", "Calmar_Ratio"]].plot(kind="bar", figsize=(11, 6), title="TTL CAGR / MDD / Calmar")
    ax.grid(axis="y", alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(output_dir / f"{case_slug}_risk_metrics.png", dpi=150); plt.close(ax.figure)
    ax = table["Number_of_Trades"].plot(kind="bar", figsize=(9, 5), title="TTL Number of Trades")
    ax.grid(axis="y", alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(output_dir / f"{case_slug}_trades.png", dpi=150); plt.close(ax.figure)


def synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Deterministic end-to-end demo data; never presented as an empirical result."""
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2016-12-01", "2025-12-31")
    us, jp = [f"US{i:02d}" for i in range(10)], [f"JP{i:02d}.T" for i in range(10)]
    tickers = us + jp
    factor = rng.normal(.00025, .009, (len(idx), 1))
    noise = rng.normal(0, .007, (len(idx), len(tickers)))
    drifts = np.linspace(.00005, .00045, len(tickers))
    local = pd.DataFrame(100 * np.exp(np.cumsum(factor + noise + drifts, axis=0)), index=idx, columns=tickers)
    usdjpy = pd.Series(110 * np.exp(np.cumsum(rng.normal(.00003, .002, len(idx)))), index=idx)
    valuation = local.copy(); valuation[us] = valuation[us].mul(usdjpy, axis=0)
    indices = pd.DataFrame({"^GSPC": local[us].mean(axis=1), "^N225": local[jp].mean(axis=1)}, index=idx)
    return local, valuation, indices, us, jp


def load_market_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], str]:
    if args.demo:
        local, valuation, indices, us, jp = synthetic_data()
        return local, valuation, indices, us, jp, "Synthetic demo data"
    us, jp, source = get_tickers_lumus()
    warmup = (pd.Timestamp(args.start_date) - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    all_prices = download_prices(us + jp + ["^GSPC", "^N225", "JPY=X"], warmup, args.end_date)
    missing = [ticker for ticker in ["^GSPC", "^N225", "JPY=X"] if ticker not in all_prices]
    if missing:
        raise RuntimeError(f"Required market series missing: {missing}")
    local = all_prices.reindex(columns=us + jp).dropna(axis=1, how="all")
    us, jp = [t for t in us if t in local], [t for t in jp if t in local]
    valuation = local.copy()
    valuation[us] = valuation[us].mul(all_prices["JPY=X"], axis=0)
    return local, valuation, all_prices[["^GSPC", "^N225"]], us, jp, source


def format_percent_table(table: pd.DataFrame) -> str:
    concise = table[["CAGR", "Annualized_Volatility", "Sharpe_Ratio", "Sortino_Ratio", "Max_Drawdown", "Calmar_Ratio", "Number_of_Trades", "Turnover"]].copy()
    concise.columns = ["CAGR", "Vol", "Sharpe", "Sortino", "MDD", "Calmar", "Trades", "Turnover"]
    for column in ["CAGR", "Vol", "MDD"]:
        concise[column] = concise[column].map(lambda value: f"{value:.2%}")
    for column in ["Sharpe", "Sortino", "Calmar", "Turnover"]:
        concise[column] = concise[column].map(lambda value: f"{value:.2f}")
    concise["Trades"] = concise["Trades"].astype(int)
    return concise.to_markdown()


def write_report(output_dir: Path, tables: dict[str, pd.DataFrame], source: str, demo: bool) -> None:
    winner, reason, note = recommendation(tables["ttl_only_cost_0"])
    label = "合成データによる動作確認（実証結果ではない）" if demo else "取得済み市場データによる簡易バックテスト"
    report = ["# L.U.M.U.S.-8 Alpha Engine TTL Audit", "", f"- データ種別: {label}", f"- ユニバース取得元: {source}", f"- 生成日: {date.today().isoformat()}", "", f"> {AUDIT_NOTE}", ""]
    for slug, title in [("ttl_only_cost_0", "TTL Only（取引コスト 0%）"), ("ttl_exit_cost_0", "TTL + Exit（取引コスト 0%）"), ("ttl_only_cost_001", "TTL Only（片道取引コスト 0.1%）"), ("ttl_exit_cost_001", "TTL + Exit（片道取引コスト 0.1%）")]:
        report += [f"## {title}", "", format_percent_table(tables[slug]), ""]
    report += ["## 結論表", "", "| 推奨TTL | 理由 | 注意点 |", "| ---: | --- | --- |", f"| {winner}日 | {reason} | {note} |", "", "## 監査コメント", "", "- シグナルは必ず取引日の前営業日までの終値で算出し、同日終値を見て同日終値で購入する未来リークを避けています。", "- Exit Protocolでは50日線割れまたは20日線割れを全売却として扱い、売却資金は次回リバランスまで利回り0%のキャッシュで保持します。", "- RSI14が80以上の場合は裁量的な一部利確を再現せず、警告ログのみ記録します。", "- 日本株と米国株を合算する評価損益はJPY換算し、シグナル計算は現行戦略に合わせて各市場のローカル通貨価格を使います。", "- 判定はTTL OnlyのCalmar、Sortino、MDD、CAGR、Turnoverの平均順位に基づきます。Exitありだけを見てTTLを評価しません。", ""]
    if demo:
        report += ["> **注意:** このレポートの数値は合成データによるCLI動作確認結果です。投資判断に利用できません。実データ監査には `python lumus_ttl_backtest.py --output-dir artifacts/live` を実行してください。", ""]
    (output_dir / "audit_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=None, help="Exclusive yfinance end date; default is latest obtainable close")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/live"))
    parser.add_argument("--demo", action="store_true", help="Use deterministic synthetic data for an offline end-to-end check")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal, valuation, indices, us, jp, source = load_market_data(args)
    tables: dict[str, pd.DataFrame] = {}
    for use_exit in [False, True]:
        for cost in [0.0, 0.001]:
            slug = f"ttl_{'exit' if use_exit else 'only'}_cost_{'001' if cost else '0'}"
            results = [run_backtest(signal, valuation, indices, us, jp, ttl, args.start_date, use_exit, cost) for ttl in TTL_LIST]
            table = comparison_table(results)
            table.to_csv(args.output_dir / f"{slug}.csv")
            pd.concat([result.trades.assign(TTL=result.ttl) for result in results]).to_csv(args.output_dir / f"{slug}_trades.csv", index=False)
            pd.concat([result.exit_warnings.assign(TTL=result.ttl) for result in results]).to_csv(args.output_dir / f"{slug}_rsi_warnings.csv", index=False)
            save_charts(results, args.output_dir, slug)
            tables[slug] = table
    metadata = {"generated_at": date.today().isoformat(), "source": source, "demo": args.demo, "start_date": args.start_date, "end_date": args.end_date or "latest obtainable close", "ttl_list": TTL_LIST, "audit_note": AUDIT_NOTE}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir, tables, source, args.demo)
    print(f"Audit outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
