# L.U.M.U.S.-8 Alpha Engine TTL Audit

L.U.M.U.S.-8の保有期限（TTL）を、`30 / 60 / 90 / 120 / 180`日の同一条件で比較する監査用バックテストです。TTL Only、TTL + Exit Protocol、および各ケースの片道取引コスト0.1%版を生成します。

> 本検証は現在取得可能な銘柄ユニバースを用いた簡易バックテストであり、サバイバーシップバイアスを完全には除去していない。

## セットアップ

```bash
python -m pip install -r requirements.txt
```

## 実データ監査

```bash
python lumus_ttl_backtest.py --output-dir artifacts/live
```

終了日を固定して再現性を高める場合は、`yfinance`の排他的な終了日を指定します。

```bash
python lumus_ttl_backtest.py --end-date 2026-06-03 --output-dir artifacts/live
```

米国株ユニバースは Wikipedia、GitHub CSV、static fallback の順で取得します。日本株は現行の精鋭リストを使います。シグナルはローカル通貨価格で計算し、日米を合算する評価損益はJPY換算します。

## オフライン動作確認

```bash
python lumus_ttl_backtest.py --demo --output-dir artifacts/demo
```

`--demo` は決定論的な合成データを使います。生成される数値は実証結果でも投資判断材料でもありません。

## 生成物

各ケースについて比較CSV、取引ログ、RSI80警告ログ、エクイティカーブ、ドローダウンカーブ、CAGR / MDD / Calmar比較、売買回数比較を生成します。全ケースを要約した `audit_report.md` と実行条件を保存する `metadata.json` も出力します。

## 実装上の監査ポイント

- Signal Date は Trade Date より前の営業日です。
- 稼働率はUS / JP指数の200日線判定により、100% / 60% / 20%に切り替えます。
- Exitありでは、終値が50日線または20日線を下回った銘柄を全売却し、次回リバランスまでキャッシュ保持します。
- RSI14が80以上の場合は一部利確を行わず、警告ログのみ記録します。
- 推奨TTLはTTL OnlyのCalmar、Sortino、MDD、CAGR、Turnoverの平均順位で決め、CAGRだけでは判断しません。
