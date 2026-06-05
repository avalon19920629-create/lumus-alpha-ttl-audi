import pandas as pd
from lumus_ttl_backtest import AUDIT_NOTE, TTL_LIST, calculate_scores, comparison_table, run_backtest, synthetic_data


def test_score_uses_only_data_available_at_signal_date():
    signal, _, _, us, _ = synthetic_data()
    signal_date = signal.index[300]
    baseline = calculate_scores(signal[us], signal_date)
    changed_future = signal.copy()
    changed_future.loc[changed_future.index > signal_date, us] *= 100
    pd.testing.assert_frame_equal(baseline, calculate_scores(changed_future[us], signal_date))


def test_all_ttls_run_in_both_cases_without_future_leak():
    signal, valuation, indices, us, jp = synthetic_data()
    results = []
    for ttl in TTL_LIST:
        result = run_backtest(signal, valuation, indices, us, jp, ttl, "2018-01-01", use_exit=True, trading_cost=.001)
        results.append(result)
        assert not result.equity.empty
        assert (result.rebalances["Signal_Date"] < result.rebalances["Trade_Date"]).all()
        assert set(result.trades["Side"]) == {"BUY", "SELL"}
    assert list(comparison_table(results).index) == TTL_LIST


def test_trading_cost_never_improves_equity_for_same_trade_path_without_exit():
    signal, valuation, indices, us, jp = synthetic_data()
    free = run_backtest(signal, valuation, indices, us, jp, 60, "2018-01-01", trading_cost=0)
    costly = run_backtest(signal, valuation, indices, us, jp, 60, "2018-01-01", trading_cost=.001)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_required_survivorship_bias_note_is_present():
    assert "サバイバーシップバイアスを完全には除去していない" in AUDIT_NOTE
