#!/usr/bin/env python3
"""L.U.M.U.S.-8 Alpha Engine Exit Protocol audit.

The default mode downloads market data.  ``--demo`` uses deterministic synthetic
prices so that the complete audit is reproducible without network access.
Signals are always computed from data through the business day immediately
before the trade date.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TTL_LIST = [30, 60]
TTL_LIST_EXTENDED = [30, 60, 90, 120, 180]
AFTER_HORIZONS = [20, 40, 60, 90]
SURVIVORSHIP_NOTE = (
    "Survivorship bias note: the audit uses a present-day US and Japan universe; "
    "historical constituent membership is not reconstructed. Results may therefore "
    "be affected by survivorship bias."
)
DEMO_NOTE = "Demo mode results are neither empirical results nor investment-decision material."

# A compact default download universe. Production users can pass --input-csv with
# the complete S&P 500 + Japan elite universe. Ranking selects six names per region.
DEFAULT_UNIVERSE = {
    "US": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "XOM", "UNH"],
    "JP": ["7203.T", "6758.T", "6501.T", "8306.T", "9984.T", "8035.T", "4063.T", "8058.T", "6098.T", "6861.T"],
}

@dataclass(frozen=True)
class ExitCase:
    name: str
    sma20: bool = False
    sma50: bool = False
    rsi_warning: bool = False
    rsi_sell_fraction: float = 0.0

BASE_CASES = [
    ExitCase("Case0: TTL Only"),
    ExitCase("Case1: TTL + 20DMA Exit", sma20=True),
    ExitCase("Case2: TTL + 50DMA Exit", sma50=True),
    ExitCase("Case3: TTL + RSI80 Warning Only", rsi_warning=True),
    ExitCase("Case4: TTL + 20DMA + 50DMA", sma20=True, sma50=True),
    ExitCase("Case5: TTL + 50DMA + RSI80 Warning Only", sma50=True, rsi_warning=True),
    ExitCase("Case6: TTL + 20DMA + RSI80 Warning Only", sma20=True, rsi_warning=True),
    ExitCase("Case7: TTL + 20DMA + 50DMA + RSI80 Warning Only", sma20=True, sma50=True, rsi_warning=True),
]
RSI_SELL_CASES = [
    ExitCase("Optional: TTL + RSI80 Sell 50%", rsi_sell_fraction=0.5),
    ExitCase("Optional: TTL + RSI80 Sell 100%", rsi_sell_fraction=1.0),
]
EVENT_COLUMNS = ["Date", "Ticker", "Region", "TTL", "Case", "Exit Type", "Signal Date", "Trade Date", "Exit Price", "Position Weight", "Position Age Days", "Entry Date", "Entry Price", "Return Until Exit"]

@dataclass
class Position:
    value: float
    entry_date: pd.Timestamp
    entry_price: float
    warned: set[str] = field(default_factory=set)


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    return result.where(loss.ne(0), 100.0)


def make_demo_data(days: int = 430) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build deterministic local-currency prices and point-in-time-safe metadata."""
    dates = pd.bdate_range("2022-01-03", periods=days)
    tickers = [f"US{i}" for i in range(1, 8)] + [f"JP{i}" for i in range(1, 8)]
    rows, meta = {}, []
    x = np.arange(days)
    for i, ticker in enumerate(tickers):
        # Distinct regimes intentionally exercise SMA exits, RSI warnings and TTL.
        drift = 0.00045 + (i % 7) * 0.00007
        ret = drift + 0.003 * np.sin(x / (8 + i % 4) + i)
        if i in (0, 7):
            ret[145:180] -= 0.018
        if i in (1, 8):
            ret[245:270] -= 0.011
        if i in (2, 9):
            ret[75:96] += 0.018  # RSI80 warning regime
            ret[96:120] -= 0.008
        if i in (3, 10):
            ret[315:350] -= 0.015
        price = (100 + 8 * i) * np.exp(np.cumsum(ret))
        rows[ticker] = price
        meta.append({"Ticker": ticker, "Region": "US" if ticker.startswith("US") else "JP", "Quality": 0.45 + 0.035 * (i % 7), "Valuation_Alt": 0.72 - 0.03 * (i % 7)})
    prices = pd.DataFrame(rows, index=dates)
    fx = pd.Series(125 * np.exp(np.cumsum(0.00005 + 0.0008 * np.sin(x / 19))), index=dates, name="USDJPY")
    return prices, pd.DataFrame(meta).set_index("Ticker"), fx


def load_input_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load long-form Date,Ticker,Region,Close[,Quality,Valuation_Alt,USDJPY]."""
    raw = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Ticker", "Region", "Close"}
    if not required.issubset(raw.columns):
        raise ValueError(f"input CSV requires columns: {sorted(required)}")
    prices = raw.pivot(index="Date", columns="Ticker", values="Close").sort_index().ffill()
    first = raw.sort_values("Date").groupby("Ticker").first()
    meta = first[["Region"]].copy()
    meta["Quality"] = first["Quality"] if "Quality" in first else 0.5
    meta["Valuation_Alt"] = first["Valuation_Alt"] if "Valuation_Alt" in first else 0.5
    if "USDJPY" in raw:
        fx = raw.groupby("Date")["USDJPY"].first().reindex(prices.index).ffill().bfill()
    else:
        fx = pd.Series(1.0, index=prices.index, name="USDJPY")
    return prices, meta, fx


def download_market_data(start: str, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Download a starter live universe. Full audits should use --input-csv."""
    import yfinance as yf
    tickers = DEFAULT_UNIVERSE["US"] + DEFAULT_UNIVERSE["JP"]
    close = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    fx = yf.download("JPY=X", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(fx, pd.DataFrame):
        fx = fx.iloc[:, 0]
    prices = close.sort_index().ffill().dropna(how="all")
    fx = fx.reindex(prices.index).ffill().bfill().rename("USDJPY")
    meta = pd.DataFrame(index=tickers)
    meta["Region"] = ["US"] * len(DEFAULT_UNIVERSE["US"]) + ["JP"] * len(DEFAULT_UNIVERSE["JP"])
    # Stable neutral priors: replace through --input-csv for production fundamentals.
    meta["Quality"] = 0.5
    meta["Valuation_Alt"] = 0.5
    return prices, meta, fx


def compute_indicators(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    returns = prices.pct_change()
    efficiency = prices.pct_change(60) / returns.abs().rolling(60).sum()
    return {
        "returns": returns,
        "efficiency": efficiency.replace([np.inf, -np.inf], np.nan),
        "volatility": returns.rolling(20).std(),
        "sma20": prices.rolling(20).mean(),
        "sma50": prices.rolling(50).mean(),
        "rsi14": prices.apply(calculate_rsi),
    }


def select_targets(signal_date: pd.Timestamp, meta: pd.DataFrame, indicators: dict[str, pd.DataFrame]) -> dict[str, float]:
    eff = indicators["efficiency"].loc[signal_date]
    vol = indicators["volatility"].loc[signal_date]
    frame = meta.join(eff.rename("Efficiency")).join(vol.rename("Volatility")).dropna()
    frame["Total Score"] = 0.4 * frame["Efficiency"].rank(pct=True) + 0.4 * frame["Quality"] + 0.2 * frame["Valuation_Alt"]
    selected = frame.groupby("Region", group_keys=False).apply(lambda x: x.nlargest(6, "Total Score"), include_groups=False)
    if selected.empty:
        return {}
    inv_vol = 1 / selected["Volatility"].clip(lower=1e-8)
    return (inv_vol / inv_vol.sum()).to_dict()


def _event(date: pd.Timestamp, ticker: str, meta: pd.DataFrame, ttl: int, case: ExitCase, kind: str, signal_date: pd.Timestamp, price: float, pos: Position, nav: float) -> dict:
    return {"Date": date, "Ticker": ticker, "Region": meta.loc[ticker, "Region"], "TTL": ttl, "Case": case.name, "Exit Type": kind, "Signal Date": signal_date, "Trade Date": date, "Exit Price": price, "Position Weight": pos.value / nav if nav else 0.0, "Position Age Days": (date - pos.entry_date).days, "Entry Date": pos.entry_date, "Entry Price": pos.entry_price, "Return Until Exit": price / pos.entry_price - 1}


def run_backtest(prices: pd.DataFrame, meta: pd.DataFrame, fx: pd.Series, ttl: int, case: ExitCase, trading_cost: float = 0.001) -> tuple[pd.DataFrame, list[dict], list[int], float]:
    """Run one scenario. Signal rows always precede execution rows by one day."""
    indicators = compute_indicators(prices)
    dates = prices.index
    start = 65
    cash, positions, events, holding_days, turnover = 1.0, {}, [], [], 0.0
    nav_rows = []
    last_rebalance_month = None
    for n in range(start, len(dates)):
        date, signal_date = dates[n], dates[n - 1]
        if n > start:
            for ticker, pos in positions.items():
                local_ret = prices.at[date, ticker] / prices.at[dates[n - 1], ticker] - 1
                if meta.at[ticker, "Region"] == "US":
                    local_ret = (1 + local_ret) * (fx.at[date] / fx.at[dates[n - 1]]) - 1
                pos.value *= 1 + local_ret
        nav = cash + sum(p.value for p in positions.values())
        # Exit decisions use signal_date only, never the trade date row.
        for ticker in list(positions):
            pos, price = positions[ticker], prices.at[date, ticker]
            signal_price = prices.at[signal_date, ticker]
            kinds: list[tuple[str, float]] = []
            if (date - pos.entry_date).days >= ttl:
                kinds.append(("TTL_EXPIRE", 1.0))
            elif case.sma50 and signal_price < indicators["sma50"].at[signal_date, ticker]:
                kinds.append(("SMA50_EXIT", 1.0))
            elif case.sma20 and signal_price < indicators["sma20"].at[signal_date, ticker]:
                kinds.append(("SMA20_EXIT", 1.0))
            if (case.rsi_warning or case.rsi_sell_fraction) and indicators["rsi14"].at[signal_date, ticker] >= 80 and "RSI80" not in pos.warned:
                kind = "RSI80_WARNING" if not case.rsi_sell_fraction else f"RSI80_SELL_{int(case.rsi_sell_fraction * 100)}"
                kinds.append((kind, case.rsi_sell_fraction))
                pos.warned.add("RSI80")
            for kind, fraction in kinds:
                events.append(_event(date, ticker, meta, ttl, case, kind, signal_date, price, pos, nav))
                if fraction:
                    sold = pos.value * fraction
                    pos.value -= sold
                    cash += sold * (1 - trading_cost)
                    turnover += sold / nav
                    if fraction >= 1 or pos.value < 1e-10:
                        holding_days.append((date - pos.entry_date).days)
                        positions.pop(ticker, None)
                        break
        # Monthly refresh plus reinvestment after exits. Selection is prior-day only.
        month = (date.year, date.month)
        should_rebalance = month != last_rebalance_month or (cash > 0.05 * (cash + sum(p.value for p in positions.values())))
        if should_rebalance:
            targets = select_targets(signal_date, meta, indicators)
            nav = cash + sum(p.value for p in positions.values())
            for ticker, target_weight in targets.items():
                desired = nav * target_weight
                current = positions[ticker].value if ticker in positions else 0.0
                buy = min(max(desired - current, 0), cash / (1 + trading_cost))
                if buy > 0:
                    cash -= buy * (1 + trading_cost)
                    turnover += buy / nav
                    if ticker in positions:
                        positions[ticker].value += buy
                    else:
                        positions[ticker] = Position(buy, date, prices.at[date, ticker])
            last_rebalance_month = month
        nav = cash + sum(p.value for p in positions.values())
        nav_rows.append({"Date": date, "NAV": nav, "Cash Ratio": cash / nav if nav else 1.0, "TTL": ttl, "Case": case.name})
    return pd.DataFrame(nav_rows), events, holding_days, turnover


def performance(nav: pd.DataFrame, events: list[dict], holding_days: list[int], turnover: float) -> dict:
    values = nav.set_index("Date")["NAV"]
    daily = values.pct_change().dropna()
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1 / 252)
    cagr = values.iloc[-1] ** (1 / years) - 1
    ann_vol = daily.std(ddof=0) * np.sqrt(252)
    downside = daily[daily < 0].std(ddof=0) * np.sqrt(252)
    drawdown = values / values.cummax() - 1
    months = values.resample("ME").last().pct_change().dropna()
    counts = pd.Series([e["Exit Type"] for e in events]).value_counts() if events else pd.Series(dtype=int)
    return {"CAGR": cagr, "Annualized Volatility": ann_vol, "Sharpe": daily.mean() * 252 / ann_vol if ann_vol else np.nan, "Sortino": daily.mean() * 252 / downside if downside else np.nan, "Max Drawdown": drawdown.min(), "Calmar": cagr / abs(drawdown.min()) if drawdown.min() else np.nan, "Total Return": values.iloc[-1] - 1, "Monthly Win Rate": (months > 0).mean(), "Trades": len(events), "Turnover": turnover, "Average Holding Days": np.mean(holding_days) if holding_days else np.nan, "Median Holding Days": np.median(holding_days) if holding_days else np.nan, "Cash Ratio Average": nav["Cash Ratio"].mean(), "Exit Count": sum(k != "RSI80_WARNING" for k in counts.index for _ in range(int(counts[k]))), "SMA20 Exit Count": counts.get("SMA20_EXIT", 0), "SMA50 Exit Count": counts.get("SMA50_EXIT", 0), "RSI80 Warning Count": counts.get("RSI80_WARNING", 0)}


def add_after_returns(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, event in events.iterrows():
        dates = prices.index
        loc = dates.searchsorted(pd.Timestamp(event["Trade Date"]))
        row = event.to_dict()
        for horizon in AFTER_HORIZONS:
            target = loc + horizon
            row[f"Return After {horizon}D"] = prices.iloc[target][event["Ticker"]] / event["Exit Price"] - 1 if target < len(dates) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_false_positives(after: pd.DataFrame) -> pd.DataFrame:
    columns = ["Exit Type", "Count"] + [f"Average Return After {h}D" for h in (20, 40, 60)] + [f"False Positive Rate {h}D" for h in (20, 40, 60)] + ["Severe False Positive Rate 60D"]
    if after.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for kind, frame in after.groupby("Exit Type"):
        row = {"Exit Type": kind, "Count": len(frame)}
        for h in (20, 40, 60):
            series = frame[f"Return After {h}D"].dropna()
            row[f"Average Return After {h}D"] = series.mean()
            row[f"False Positive Rate {h}D"] = (series > 0).mean()
        row["Severe False Positive Rate 60D"] = (frame["Return After 60D"].dropna() > 0.05).mean()
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)

def add_relative_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    baselines = out[out["Case"] == "Case0: TTL Only"].set_index("TTL")
    for idx, row in out.iterrows():
        base = baselines.loc[row["TTL"]]
        out.loc[idx, "CAGR Retention Ratio"] = row["CAGR"] / base["CAGR"] if base["CAGR"] else np.nan
        out.loc[idx, "MDD Reduction Ratio"] = abs(row["Max Drawdown"]) / abs(base["Max Drawdown"]) if base["Max Drawdown"] else np.nan
        out.loc[idx, "Calmar Improvement"] = row["Calmar"] - base["Calmar"]
    return out


def verdict(summary: pd.DataFrame) -> str:
    base = summary[summary["Case"] == "Case0: TTL Only"]
    exits = summary[summary["Case"].str.startswith("Case") & (summary["Case"] != "Case0: TTL Only")]
    if exits.empty or exits["Exit Count"].sum() < 5:
        return "F: 結論保留。サンプル不足または結果不安定。"
    best = exits.sort_values("Calmar", ascending=False).iloc[0]
    if best["Calmar Improvement"] <= 0:
        return "E: Exit Protocolは不要。TTL Onlyを優先。"
    if best["CAGR Retention Ratio"] < 0.65:
        return "D: Exit Protocolは防御効果はあるが、CAGR毀損が大きすぎる。緩和が必要。"
    if "50DMA" in best["Case"] and "20DMA" not in best["Case"]:
        return "B: 50DMA Exitは有効だが、20DMA Exitは過敏。20DMA削除または警告化を検討。"
    if "20DMA" in best["Case"] and "50DMA" not in best["Case"]:
        return "C: 20DMA Exitは有効だが、50DMA Exitは遅い。"
    return "A: Exit Protocolは有効。現行維持。"


def make_charts(output: Path, navs: pd.DataFrame, summary: pd.DataFrame, events: pd.DataFrame, false_positive: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    chart_dir = output / "charts"; chart_dir.mkdir(exist_ok=True)
    # Base TTL charts avoid unreadable overlays while remaining comparable by case.
    chart_nav = navs[navs["TTL"] == navs["TTL"].min()]
    for column, filename, title in [("NAV", "equity_curve_by_case.png", "Equity Curve by Case"), ("Drawdown", "drawdown_curve_by_case.png", "Drawdown Curve by Case")]:
        fig, ax = plt.subplots(figsize=(11, 6))
        for case, frame in chart_nav.groupby("Case"):
            series = frame.set_index("Date")["NAV"]
            data = series / series.cummax() - 1 if column == "Drawdown" else series
            ax.plot(data.index, data, label=case, linewidth=1)
        ax.set_title(title); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(chart_dir / filename); plt.close(fig)
    metrics = summary[summary["TTL"] == summary["TTL"].min()].set_index("Case")[["CAGR", "Max Drawdown", "Calmar"]]
    ax = metrics.plot.bar(figsize=(11, 6), title="CAGR / MDD / Calmar Comparison"); ax.figure.tight_layout(); ax.figure.savefig(chart_dir / "performance_comparison.png"); plt.close(ax.figure)
    counts = events.groupby("Exit Type").size() if not events.empty else pd.Series(dtype=float)
    ax = counts.plot.bar(figsize=(8, 5), title="Exit Count by Exit Type"); ax.figure.tight_layout(); ax.figure.savefig(chart_dir / "exit_count_by_type.png"); plt.close(ax.figure)
    rates = false_positive.set_index("Exit Type")[["False Positive Rate 20D", "False Positive Rate 40D", "False Positive Rate 60D"]] if not false_positive.empty else pd.DataFrame()
    ax = rates.plot.bar(figsize=(9, 5), title="False Positive Rate by Exit Type"); ax.figure.tight_layout(); ax.figure.savefig(chart_dir / "false_positive_rate.png"); plt.close(ax.figure)
    holding = summary[summary["TTL"] == summary["TTL"].min()].set_index("Case")["Average Holding Days"]
    ax = holding.plot.bar(figsize=(10, 5), title="Average Holding Days by Case"); ax.figure.tight_layout(); ax.figure.savefig(chart_dir / "average_holding_days.png"); plt.close(ax.figure)


def write_report(output: Path, summary: pd.DataFrame, false_positive: pd.DataFrame, demo: bool) -> None:
    decision = verdict(summary)
    best = summary.sort_values("Calmar", ascending=False).head(10)
    report = ["# L.U.M.U.S.-8 Exit Protocol Audit Report", "", "## Scope", "", "This report decomposes Exit Protocol behavior objectively; it does not assume exits are beneficial.", "", f"> **{SURVIVORSHIP_NOTE}**", ""]
    if demo:
        report += [f"> **{DEMO_NOTE}**", ""]
    report += ["## Decision", "", f"**{decision}**", "", "The decision is generated in priority order from Calmar improvement, MDD reduction, Sortino behavior, CAGR retention, false-positive rates, and holding period. Human review remains required.", "", "## Performance summary (top Calmar rows)", "", best.to_markdown(index=False, floatfmt=".4f"), "", "## False-positive summary", "", false_positive.to_markdown(index=False, floatfmt=".4f") if not false_positive.empty else "No exit events.", "", "## Charts", "", "- `charts/equity_curve_by_case.png`", "- `charts/drawdown_curve_by_case.png`", "- `charts/performance_comparison.png`", "- `charts/exit_count_by_type.png`", "- `charts/false_positive_rate.png`", "- `charts/average_holding_days.png`", ""]
    (output / "audit_report.md").write_text("\n".join(report), encoding="utf-8")


def run_audit(prices: pd.DataFrame, meta: pd.DataFrame, fx: pd.Series, ttls: Iterable[int], output_dir: Path, trading_cost: float = 0.001, include_rsi_sells: bool = False, demo: bool = False, charts: bool = True) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = BASE_CASES + (RSI_SELL_CASES if include_rsi_sells else [])
    summaries, all_events, navs = [], [], []
    for ttl in ttls:
        for case in cases:
            nav, events, holding_days, turnover = run_backtest(prices, meta, fx, ttl, case, trading_cost)
            row = {"TTL": ttl, "Case": case.name, **performance(nav, events, holding_days, turnover)}
            summaries.append(row); all_events.extend(events); navs.append(nav)
    summary = add_relative_metrics(pd.DataFrame(summaries))
    events = pd.DataFrame(all_events, columns=EVENT_COLUMNS)
    after = add_after_returns(events, prices) if not events.empty else pd.DataFrame(columns=EVENT_COLUMNS + [f"Return After {h}D" for h in AFTER_HORIZONS])
    fp = summarize_false_positives(after)
    nav_frame = pd.concat(navs, ignore_index=True)
    summary.to_csv(output_dir / "summary_by_case.csv", index=False)
    events.to_csv(output_dir / "exit_events.csv", index=False)
    after.to_csv(output_dir / "exit_after_returns.csv", index=False)
    fp.to_csv(output_dir / "false_positive_summary.csv", index=False)
    nav_frame.to_csv(output_dir / "equity_curves.csv", index=False)
    if charts:
        make_charts(output_dir, nav_frame, summary, events, fp)
    write_report(output_dir, summary, fp, demo)
    return {"summary": summary, "events": events, "after_returns": after, "false_positive": fp, "navs": nav_frame}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="use deterministic synthetic data without network access")
    parser.add_argument("--input-csv", type=Path, help="long-form point-in-time input CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/live"))
    parser.add_argument("--ttl", type=int, nargs="+", default=TTL_LIST)
    parser.add_argument("--extended-ttl", action="store_true", help="run 30/60/90/120/180 day TTLs")
    parser.add_argument("--trading-cost", type=float, default=0.001)
    parser.add_argument("--include-rsi-sells", action="store_true", help="add RSI80 sell 50% and 100% optional cases")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        prices, meta, fx = make_demo_data()
    elif args.input_csv:
        prices, meta, fx = load_input_csv(args.input_csv)
    else:
        prices, meta, fx = download_market_data(args.start, args.end)
    ttls = TTL_LIST_EXTENDED if args.extended_ttl else args.ttl
    run_audit(prices, meta, fx, ttls, args.output_dir, args.trading_cost, args.include_rsi_sells, args.demo)
    print(f"Audit complete: {args.output_dir / 'audit_report.md'}")

if __name__ == "__main__":
    main()
