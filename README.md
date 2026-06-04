# L.U.M.U.S.-8 Exit Protocol Audit

TTL監査とは分離した、L.U.M.U.S.-8 Alpha Engine向けExit Protocol分解監査です。Exitを肯定するためではなく、利益保護・利益毀損、条件別の有効性、過敏性を客観的に比較します。

> **Survivorship bias note: the audit uses a present-day US and Japan universe; historical constituent membership is not reconstructed. Results may therefore be affected by survivorship bias.**
>
> **Demo mode results are neither empirical results nor investment-decision material.** Demo modeの結果は実証結果でも投資判断材料でもありません。

## 固定条件

- スコア: `Efficiency : Quality : Valuation_Alt = 40 : 40 : 20`
- 地域別上位6銘柄（米国6 + 日本6 = 最大12銘柄）
- 逆ボラティリティRisk Parity
- 評価額はJPY換算。テクニカル・シグナルは各市場のローカル通貨終値で計算
- シグナル計算・Exit判定はTrade Dateの前営業日まで。常に `Signal Date < Trade Date`
- 基本TTL: 30 / 60日。`--extended-ttl` で30 / 60 / 90 / 120 / 180日

## Exitケース

`Case0`〜`Case7` はTTL Only、20DMA、50DMA、RSI80 Warning Onlyとその組み合わせです。`--include-rsi-sells` を指定すると、RSI80 Sell 50% / 100%も追加比較します。主分析はWarning Onlyです。

## 実行

```bash
python lumus_exit_audit.py --output-dir artifacts/live
python lumus_exit_audit.py --demo --output-dir artifacts/demo
python lumus_exit_audit.py --ttl 30 60 --trading-cost 0.001 --output-dir artifacts/live
python lumus_exit_audit.py --demo --extended-ttl --include-rsi-sells --output-dir artifacts/demo-extended
```

通常実行は`yfinance`からスターター・ユニバースを取得します。本番監査ではポイントインタイムに整備した完全なS&P 500 + 日本株精鋭リストを `--input-csv` で投入してください。CSVはlong形式の `Date,Ticker,Region,Close` が必須で、任意列は `Quality,Valuation_Alt,USDJPY` です。

## 成果物

出力先に `summary_by_case.csv`, `exit_events.csv`, `exit_after_returns.csv`, `false_positive_summary.csv`, `equity_curves.csv`, `audit_report.md` と6種類のグラフを生成します。Exit後20 / 40 / 60 / 90営業日リターン、偽陽性率、厳格偽陽性率（60D > +5%）を含みます。

## テスト

```bash
pytest -q
```
