from pathlib import Path
import pandas as pd
from lumus_exit_audit import BASE_CASES, DEMO_NOTE, SURVIVORSHIP_NOTE, make_demo_data, run_audit, run_backtest


def audit(tmp_path, cost=0.001):
    prices, meta, fx = make_demo_data()
    return run_audit(prices, meta, fx, [30, 60], tmp_path, trading_cost=cost, demo=True, charts=False)


def test_signal_date_precedes_trade_date(tmp_path):
    result = audit(tmp_path)
    events = result["events"]
    assert not events.empty
    assert (pd.to_datetime(events["Signal Date"]) < pd.to_datetime(events["Trade Date"])).all()


def test_future_changes_do_not_change_past_signals():
    prices, meta, fx = make_demo_data()
    cutoff = prices.index[260]
    nav1, events1, _, _ = run_backtest(prices.loc[:cutoff], meta, fx.loc[:cutoff], 60, BASE_CASES[7])
    changed = prices.copy(); changed.loc[changed.index > cutoff] *= 9
    nav2, events2, _, _ = run_backtest(changed.loc[:cutoff], meta, fx.loc[:cutoff], 60, BASE_CASES[7])
    pd.testing.assert_frame_equal(nav1, nav2)
    assert events1 == events2


def test_all_cases_and_ttls_run(tmp_path):
    summary = audit(tmp_path)["summary"]
    assert set(summary["Case"]) == {case.name for case in BASE_CASES}
    assert set(summary["TTL"]) == {30, 60}
    assert len(summary) == len(BASE_CASES) * 2


def test_required_exit_types_are_logged(tmp_path):
    kinds = set(audit(tmp_path)["events"]["Exit Type"])
    assert {"SMA20_EXIT", "SMA50_EXIT", "RSI80_WARNING"} <= kinds


def test_cost_does_not_improve_performance(tmp_path):
    zero = audit(tmp_path / "zero", cost=0)["summary"].set_index(["TTL", "Case"])
    costly = audit(tmp_path / "costly", cost=0.01)["summary"].set_index(["TTL", "Case"])
    assert (costly["Total Return"] <= zero["Total Return"] + 1e-12).all()


def test_notes_exist_in_readme_and_report(tmp_path):
    audit(tmp_path)
    readme = Path("README.md").read_text()
    report = (tmp_path / "audit_report.md").read_text()
    assert SURVIVORSHIP_NOTE in readme and SURVIVORSHIP_NOTE in report
    assert DEMO_NOTE in readme and DEMO_NOTE in report


def test_demo_outputs_without_external_data(tmp_path):
    result = audit(tmp_path)
    for name in ["summary_by_case.csv", "exit_events.csv", "exit_after_returns.csv", "false_positive_summary.csv", "audit_report.md"]:
        assert (tmp_path / name).exists()
    assert not result["false_positive"].empty
