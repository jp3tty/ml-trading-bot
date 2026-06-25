"""
Paper Trade Validator

Runs the ML trader in paper mode and records all signals + outcomes to a
persistent CSV log. Run this once per trading day (at or after market open).

Two-pass scan (mirrors ml_trader.py):
  Pass 1 — SELL: checks every held Alpaca position for an exit signal.
  Pass 2 — BUY:  checks FinViz watchlist (excluding held) for entry signals.

Usage:
    python paper_trade_validator.py               # scan + log signals
    python paper_trade_validator.py --report      # show log summary only (no scan)
    python paper_trade_validator.py --dry-run     # scan but don't place orders
    python paper_trade_validator.py --symbols AAPL TSLA NVDA
"""

import argparse
import csv
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from alpaca_trading import AlpacaConnection
from ml.binary_predictor import BinaryBuyPredictor
from ml.binary_sell_predictor import BinarySellPredictor
from ml_trader import MLTrader, _get_indicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

SIGNAL_LOG    = "paper_trade_log/signals.csv"
ORDER_LOG     = "paper_trade_log/orders.csv"
MAX_POSITIONS = 20
MIN_SELL_PNL_PCT = 0.005  # ML SELL won't fire below +0.5% gain — mirrors ml_trader.py
STOP_LOSS_PCT    = 0.020  # -2.0%: wider than training labels to survive intraday noise on 4h bars
TAKE_PROFIT_PCT  = 0.025  # +2.5%: maintains 1.25× R/R ratio with wider stop

# 'side' = BUY or SELL; 'signal_fired' = whether the detector triggered
SIGNAL_FIELDS = [
    'timestamp', 'symbol', 'side', 'probability', 'threshold',
    'signal_fired', 'action_taken', 'price'
]
ORDER_FIELDS = [
    'timestamp', 'symbol', 'side', 'qty', 'entry_price',
    'take_profit', 'stop_loss', 'order_id', 'confidence', 'rsi', 'momentum',
]


def ensure_log_files():
    Path("paper_trade_log").mkdir(exist_ok=True)

    if not os.path.exists(SIGNAL_LOG):
        with open(SIGNAL_LOG, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=SIGNAL_FIELDS).writeheader()

    if not os.path.exists(ORDER_LOG):
        with open(ORDER_LOG, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=ORDER_FIELDS).writeheader()


def log_signal(symbol, side, prediction, action_taken, price):
    signal_key = 'is_buy' if side == 'BUY' else 'is_sell'
    row = {
        'timestamp':    datetime.now().isoformat(),
        'symbol':       symbol,
        'side':         side,
        'probability':  round(prediction['probability'], 4),
        'threshold':    round(prediction['threshold'], 4),
        'signal_fired': prediction.get(signal_key, False),
        'action_taken': action_taken,
        'price':        round(price, 4),
    }
    with open(SIGNAL_LOG, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=SIGNAL_FIELDS).writerow(row)


def log_order(symbol, side, qty, entry_price, take_profit, stop_loss, order, confidence,
              rsi=None, momentum=None):
    row = {
        'timestamp':   datetime.now().isoformat(),
        'symbol':      symbol,
        'side':        side,
        'qty':         qty,
        'entry_price': entry_price,
        'take_profit': take_profit if take_profit is not None else '',
        'stop_loss':   stop_loss if stop_loss is not None else '',
        'order_id':    getattr(order, 'id', 'N/A') if order is not None else 'N/A',
        'confidence':  round(confidence, 4) if confidence is not None else '',
        'rsi':         rsi if rsi is not None else '',
        'momentum':    momentum if momentum is not None else '',
    }
    with open(ORDER_LOG, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=ORDER_FIELDS).writerow(row)


def sync_bracket_exits(conn, alpaca_held_symbols):
    """Detect and log positions closed by bracket SL/TP since the last run.

    Compares open positions in orders.csv against Alpaca's live positions.
    Any symbol open in the log but absent from Alpaca was closed by a bracket
    order — we fetch the actual fill price and write a BRACKET_EXIT sell row.

    Strategy 1 (primary): query the parent bracket order by ID and inspect its
    legs.  Alpaca only exposes bracket child orders through the parent; they do
    not reliably appear as separate entries in list_orders.

    Strategy 2 (fallback): paginate list_orders for any symbol not resolved by
    the legs lookup.

    Any symbol still unresolved after both strategies gets a RECONCILE row with
    an approximate live price so the log stays in sync.
    """
    if not os.path.exists(ORDER_LOG) or os.path.getsize(ORDER_LOG) == 0:
        return

    with open(ORDER_LOG, newline='') as f:
        rows = list(csv.DictReader(f))

    # Exclude synthetic rows so they are never treated as real Alpaca fills
    existing_order_ids = {
        r['order_id'] for r in rows
        if r.get('order_id') and r['order_id'] not in ('N/A', '', 'RECONCILE')
    }

    # Net open qty per symbol; track the last BUY timestamp and order ID
    net_qty = defaultdict(int)
    last_buy_ts = {}
    last_buy_order_id = {}
    for r in rows:
        try:
            qty = int(r.get('qty', 0) or 0)
        except ValueError:
            continue
        if r['side'] == 'BUY':
            net_qty[r['symbol']] += qty
            last_buy_ts[r['symbol']] = r['timestamp'][:19]
            oid = r.get('order_id', '')
            if oid and oid not in ('N/A', '', 'RECONCILE'):
                last_buy_order_id[r['symbol']] = oid
        elif r['side'] == 'SELL':
            net_qty[r['symbol']] -= qty

    log_open = {sym for sym, qty in net_qty.items() if qty > 0}
    bracket_closed = log_open - alpaca_held_symbols

    if not bracket_closed:
        logging.info("sync_bracket_exits: log is in sync with Alpaca — no unrecorded exits")
        return

    logging.info(
        f"sync_bracket_exits: {len(bracket_closed)} bracket-closed position(s) detected: "
        f"{sorted(bracket_closed)}"
    )

    sell_fills = defaultdict(list)

    # ------------------------------------------------------------------
    # Strategy 1: inspect parent bracket order legs directly.
    # Bracket child (SL/TP) orders are only reliably accessible via the
    # parent; list_orders does not return them as top-level entries.
    # ------------------------------------------------------------------
    for symbol in sorted(bracket_closed):
        parent_id = last_buy_order_id.get(symbol)
        if not parent_id:
            continue
        try:
            parent = conn.api.get_order(parent_id)
            legs = getattr(parent, 'legs', None) or []
            for leg in legs:
                if (
                    getattr(leg, 'side', '') == 'sell'
                    and getattr(leg, 'status', '') == 'filled'
                    and getattr(leg, 'filled_avg_price', None)
                    and float(getattr(leg, 'filled_avg_price', 0) or 0) > 0
                    and leg.id not in existing_order_ids
                ):
                    sell_fills[symbol].append(leg)
                    logging.info(
                        f"{symbol}: found bracket exit via parent legs — "
                        f"leg_id={leg.id}  price={leg.filled_avg_price}"
                    )
                    break
        except Exception as e:
            logging.warning(
                f"sync_bracket_exits: could not fetch parent order {parent_id} "
                f"for {symbol}: {e}"
            )

    # ------------------------------------------------------------------
    # Strategy 2: paginate list_orders for symbols not resolved above.
    # ------------------------------------------------------------------
    still_unfound = bracket_closed - set(sell_fills)
    if still_unfound:
        page_after = min(
            last_buy_ts.get(sym, rows[0]['timestamp'][:19]) for sym in still_unfound
        )
        MAX_PAGES = 20
        for _ in range(MAX_PAGES):
            try:
                batch = conn.api.list_orders(
                    status='closed',
                    limit=500,
                    direction='asc',
                    after=page_after,
                )
            except Exception as e:
                logging.error(f"sync_bracket_exits: could not fetch order history: {e}")
                break

            if not batch:
                break

            for o in batch:
                if (
                    o.side == 'sell'
                    and o.filled_avg_price
                    and float(getattr(o, 'filled_qty', 0) or 0) > 0
                    and o.id not in existing_order_ids
                    and o.symbol in still_unfound
                ):
                    sell_fills[o.symbol].append(o)

            if len(batch) < 500:
                break

            last_submitted = getattr(batch[-1], 'submitted_at', None)
            if last_submitted is None:
                break
            next_after = str(last_submitted)[:19]
            if next_after == page_after:
                break
            page_after = next_after

    logged = 0
    reconciled = 0
    for symbol in sorted(bracket_closed):
        remaining = net_qty[symbol]

        for fill in sell_fills.get(symbol, []):
            exit_price = round(float(fill.filled_avg_price), 4)
            exit_qty   = int(float(getattr(fill, 'filled_qty', remaining) or remaining))
            remaining -= exit_qty

            row = {
                'timestamp':   datetime.now().isoformat(),
                'symbol':      symbol,
                'side':        'SELL',
                'qty':         exit_qty,
                'entry_price': exit_price,
                'take_profit': '',
                'stop_loss':   '',
                'order_id':    fill.id,
                'confidence':  '',
                'rsi':         '',
                'momentum':    '',
            }
            with open(ORDER_LOG, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=ORDER_FIELDS).writerow(row)

            existing_order_ids.add(fill.id)
            logging.info(
                f"{symbol}: logged BRACKET_EXIT — qty={exit_qty} @ ${exit_price:.2f}  "
                f"order_id={fill.id}"
            )
            logged += 1

        # Both strategies exhausted — write RECONCILE with approximate live price.
        if remaining > 0:
            approx_price = None
            try:
                approx_price = conn.get_live_price(symbol)
            except Exception:
                pass

            row = {
                'timestamp':   datetime.now().isoformat(),
                'symbol':      symbol,
                'side':        'SELL',
                'qty':         remaining,
                'entry_price': round(approx_price, 4) if approx_price else '',
                'take_profit': '',
                'stop_loss':   '',
                'order_id':    'RECONCILE',
                'confidence':  '',
                'rsi':         '',
                'momentum':    '',
            }
            with open(ORDER_LOG, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=ORDER_FIELDS).writerow(row)
            price_note = f"approx exit price ${approx_price:.4f}" if approx_price else "fill price unavailable"
            logging.info(
                f"{symbol}: reconciled {remaining} phantom share(s) — "
                f"Alpaca confirms position closed, {price_note}"
            )
            reconciled += 1

    if logged:
        logging.info(f"sync_bracket_exits: logged {logged} bracket exit(s)")
    if reconciled:
        logging.info(f"sync_bracket_exits: reconciled {reconciled} stale position(s)")


def print_model_info(buy_predictor, sell_predictor=None):
    def _fmt(predictor, label):
        info = predictor.get_model_info()
        p = info['params']
        print(f"\n  {label}")
        print(f"  {'─' * 50}")
        print(f"  Classifier:  {p.get('classifier', 'N/A')}")
        print(f"  Window:      {p.get('window_size')}   Horizon: {p.get('horizon')}")
        print(f"  Threshold:   {info['threshold']:.4f}")
        print(f"  Precision:   {info['precision']}   Recall: {info['recall']}   F1: {info['f1']}")

    print("\n" + "=" * 55)
    print("  Loaded Models")
    print("=" * 55)
    _fmt(buy_predictor, "BUY Detector")
    if sell_predictor:
        _fmt(sell_predictor, "SELL Detector")
    else:
        print("\n  SELL Detector")
        print("  " + "─" * 50)
        print("  Not loaded — run ml/binary_sell_search.py to train one.")
    print("=" * 55)


def print_open_positions(conn: AlpacaConnection):
    try:
        positions = conn.api.list_positions()
    except Exception as e:
        print(f"  Could not fetch positions: {e}")
        return []

    print("\n" + "=" * 55)
    print("  Open Positions (Paper Account)")
    print("=" * 55)
    if not positions:
        print("  None")
    else:
        for p in positions:
            pl     = float(p.unrealized_pl)
            pl_pct = float(p.unrealized_plpc) * 100
            print(
                f"  {p.symbol:<6}  qty={p.qty:<4}  "
                f"entry=${float(p.avg_entry_price):.2f}  "
                f"current=${float(p.current_price):.2f}  "
                f"P&L=${pl:+.2f} ({pl_pct:+.1f}%)"
            )
    print("=" * 55)
    return positions


def print_report():
    print("\n" + "=" * 55)
    print("  Paper Trade Signal Log Summary")
    print("=" * 55)

    if not os.path.exists(SIGNAL_LOG) or os.path.getsize(SIGNAL_LOG) == 0:
        print("  No signals logged yet.")
        return

    df = pd.read_csv(SIGNAL_LOG)

    # Handle legacy files that may have old column names
    if 'is_buy' in df.columns and 'signal_fired' not in df.columns:
        df = df.rename(columns={'is_buy': 'signal_fired'})
        df['side'] = 'BUY'

    total = len(df)
    print(f"  Total signals logged:    {total}")
    print(f"  Date range:              {df['timestamp'].min()[:10]}  →  {df['timestamp'].max()[:10]}")

    for side in ['BUY', 'SELL']:
        side_df = df[df['side'] == side] if 'side' in df.columns else df
        fired   = side_df['signal_fired'].map(lambda v: str(v).strip().lower() == 'true').sum() if len(side_df) else 0
        acted   = (side_df['action_taken'].isin([f'{side}_ORDER', f'DRY_RUN_{side}'])).sum()
        if len(side_df) == 0:
            continue
        print(f"\n  {side} signals:")
        print(f"    Scanned:        {len(side_df)}")
        print(f"    Fired:          {fired}  ({fired/len(side_df)*100:.1f}%)")
        print(f"    Orders placed:  {acted}")
        if fired > 0:
            fired_rows = side_df[side_df['signal_fired'].map(lambda v: str(v).strip().lower() == 'true')]
            print(f"    Avg prob:       {fired_rows['probability'].mean():.4f}")
            top = fired_rows['symbol'].value_counts().head(5)
            print(f"    Top symbols:    {', '.join(f'{s}({c})' for s, c in top.items())}")

    print("=" * 55)

    if os.path.exists(ORDER_LOG) and os.path.getsize(ORDER_LOG) > 0:
        orders = pd.read_csv(ORDER_LOG)
        print(f"\n  Orders placed: {len(orders)}")
        for _, row in orders.iterrows():
            side = row.get('side', 'BUY')
            if side == 'BUY':
                print(
                    f"    {row['timestamp'][:16]}  BUY  {row['symbol']}  "
                    f"x{row['qty']} @ ${row['entry_price']}  "
                    f"TP=${row['take_profit']}  SL=${row['stop_loss']}"
                )
            else:
                print(
                    f"    {row['timestamp'][:16]}  SELL {row['symbol']}  "
                    f"x{row['qty']} @ ${row['entry_price']}"
                )
        print("=" * 55)


def run_scan(symbols=None, min_confidence=0.6, min_sell_confidence=0.3, dry_run=False):
    ensure_log_files()

    conn   = AlpacaConnection(paper=True)
    trader = MLTrader(paper=True)

    clock = conn.api.get_clock()
    if not clock.is_open:
        logging.warning("Market is closed — running in signal-preview mode anyway.")

    # Load detectors
    buy_predictor  = trader.buy_detector
    sell_predictor = trader.sell_detector

    print_model_info(buy_predictor, sell_predictor)

    # Fetch held positions once — drives the SELL pass
    held = trader.get_held_positions()

    # Sync any bracket SL/TP exits that fired since the last run
    sync_bracket_exits(conn, set(held.keys()))

    print_open_positions(conn)

    # Guard against same-day duplicate buys: if this symbol was already
    # purchased earlier today it may not yet appear in list_positions() due
    # to Alpaca paper-trading settlement lag, so exclude it from the BUY pass.
    today = datetime.now().strftime('%Y-%m-%d')
    today_buys = set()
    if os.path.exists(ORDER_LOG):
        with open(ORDER_LOG, newline='') as f:
            for row in csv.DictReader(f):
                if row.get('side') == 'BUY' and row['timestamp'].startswith(today):
                    today_buys.add(row['symbol'])
    unsettled = today_buys - set(held.keys())
    if unsettled:
        logging.info(
            f"Same-day buy guard: {sorted(unsettled)} already purchased today "
            f"but not yet reflected in Alpaca positions — will be excluded from BUY pass"
        )

    session_signals = []
    session_orders  = []

    # ------------------------------------------------------------------
    # Pass 1: SELL — every held position, regardless of today's watchlist
    # ------------------------------------------------------------------
    logging.info(f"--- SELL pass: {len(held)} held positions ---")

    for symbol, position in held.items():
        try:
            df = trader.fetch_recent_data(symbol)
            if df is None or len(df) < 50:
                logging.warning(f"{symbol}: insufficient data for SELL check")
                continue

            current_price = float(df['close'].iloc[-1])
            avg_entry     = float(position.avg_entry_price)
            pnl_pct       = (current_price - avg_entry) / avg_entry

            if sell_predictor is None:
                logging.info(f"{symbol}: SELL detector not loaded — skipping")
                continue

            prediction = sell_predictor.predict(df)
            if prediction is None:
                logging.warning(f"{symbol}: SELL features could not be built")
                continue

            action = 'NO_ACTION'

            if prediction['is_sell'] and prediction['probability'] >= min_sell_confidence:
                if pnl_pct < MIN_SELL_PNL_PCT:
                    action = 'PNL_GATE'
                    logging.info(
                        f"{symbol}: SELL gated — P&L {pnl_pct*100:+.2f}% "
                        f"below minimum +{MIN_SELL_PNL_PCT*100:.1f}%  "
                        f"(prob={prediction['probability']:.4f}  entry=${avg_entry:.2f})"
                    )
                elif dry_run:
                    action = 'DRY_RUN_SELL'
                    logging.info(
                        f"{symbol}: [DRY RUN] SELL  prob={prediction['probability']:.4f}  "
                        f"@ ${current_price:.2f}  "
                        f"[entry=${avg_entry:.2f}  P&L={pnl_pct*100:+.2f}%]"
                    )
                else:
                    try:
                        open_orders = conn.api.list_orders(status='open', symbols=[symbol])
                        for o in open_orders:
                            conn.api.cancel_order(o.id)
                        conn.api.close_position(symbol)
                        action = 'SELL_ORDER'
                        ind = _get_indicators(df)
                        log_order(symbol, 'SELL', position.qty, current_price,
                                  None, None, None, prediction['probability'],
                                  rsi=ind['rsi'], momentum=ind['momentum'])
                        session_orders.append(('SELL', symbol))
                        logging.info(
                            f"{symbol}: SELL order placed @ ${current_price:.2f}  "
                            f"[entry=${avg_entry:.2f}  P&L={pnl_pct*100:+.2f}%]"
                        )
                    except Exception as e:
                        logging.error(f"{symbol}: error closing position: {e}")
            else:
                logging.info(
                    f"{symbol}: hold  prob={prediction['probability']:.4f}  "
                    f"(model_threshold={prediction['threshold']:.4f}  "
                    f"sell_confidence_floor={min_sell_confidence:.4f})  "
                    f"entry=${avg_entry:.2f}  P&L={pnl_pct*100:+.2f}%"
                )

            log_signal(symbol, 'SELL', prediction, action, current_price)
            session_signals.append({
                'symbol':      symbol,
                'side':        'SELL',
                'probability': prediction['probability'],
                'fired':       prediction['is_sell'],
                'action':      action,
                'price':       current_price,
            })

        except Exception as e:
            logging.error(f"{symbol} (SELL pass): {e}")
            continue

    # ------------------------------------------------------------------
    # Pass 2: BUY — watchlist symbols not already held
    # ------------------------------------------------------------------
    if symbols is None:
        symbols = trader.get_watchlist()

    buy_candidates  = [s for s in symbols if s not in held and s not in today_buys]
    skipped         = len(symbols) - len(buy_candidates)
    open_slots      = max(0, MAX_POSITIONS - len(held))
    logging.info(
        f"--- BUY pass: {len(buy_candidates)} candidates "
        f"({skipped} skipped — {len(held)} held, {len(unsettled)} same-day guard) | "
        f"position slots available: {open_slots}/{MAX_POSITIONS} ---"
    )

    if open_slots == 0:
        logging.info("  At position limit — BUY pass skipped.")
        buy_candidates = []

    # Step 1: Score all candidates
    scored_candidates = []
    for symbol in buy_candidates:
        try:
            df = trader.fetch_recent_data(symbol)
            if df is None or len(df) < 100:
                logging.warning(f"{symbol}: insufficient data, skipping")
                continue

            current_price = float(df['close'].iloc[-1])
            prediction    = buy_predictor.predict(df)

            if prediction is None:
                logging.warning(f"{symbol}: BUY features could not be built")
                continue

            scored_candidates.append((symbol, prediction, current_price, df))
            logging.info(
                f"{symbol}: scored  prob={prediction['probability']:.4f}  "
                f"(threshold={prediction['threshold']:.4f})"
            )

        except Exception as e:
            logging.error(f"{symbol} (BUY pass): {e}")
            continue

    # Step 2: Rank by probability, keep top N (capped by open slots and 5-per-run limit)
    scored_candidates.sort(key=lambda x: x[1]['probability'], reverse=True)
    max_buys     = min(5, open_slots)
    top5_symbols = {s for s, _, _, _ in scored_candidates[:max_buys]}
    logging.info(
        f"  Top {max_buys} of {len(scored_candidates)} scored: "
        f"{[s for s, _, _, _ in scored_candidates[:max_buys]]}"
    )

    # Step 3: Act on top 5 only; log all
    for symbol, prediction, current_price, df in scored_candidates:
        action = 'NO_ACTION'
        in_top5 = symbol in top5_symbols

        if in_top5 and prediction['is_buy'] and prediction['probability'] >= min_confidence:
            if dry_run:
                action = 'DRY_RUN_BUY'
                logging.info(
                    f"{symbol}: [DRY RUN] BUY  prob={prediction['probability']:.4f}  "
                    f"@ ${current_price:.2f}"
                )
            else:
                live_price = conn.get_live_price(symbol) or current_price
                qty        = trader.calculate_position_size(symbol, live_price)

                entry_price = round(live_price, 2)
                stop_loss   = round(live_price * (1 - STOP_LOSS_PCT), 2)
                take_profit = round(live_price * (1 + TAKE_PROFIT_PCT), 2)

                order = conn.place_bracket_order(
                    symbol=symbol,
                    qty=qty,
                    entry_price=entry_price,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                )
                action = 'BUY_ORDER'
                ind = _get_indicators(df)
                log_order(symbol, 'BUY', qty, entry_price,
                          take_profit, stop_loss, order,
                          prediction['probability'],
                          rsi=ind['rsi'], momentum=ind['momentum'])
                session_orders.append(('BUY', symbol))
                logging.info(
                    f"{symbol}: BUY ORDER  qty={qty}  "
                    f"prob={prediction['probability']:.4f}  @ ${entry_price}  "
                    f"TP=${take_profit}  SL=${stop_loss}"
                )
        elif not in_top5:
            logging.info(
                f"{symbol}: skipped (not top 5)  prob={prediction['probability']:.4f}"
            )
        else:
            logging.info(
                f"{symbol}: no signal  prob={prediction['probability']:.4f}  "
                f"(threshold={prediction['threshold']:.4f})"
            )

        log_signal(symbol, 'BUY', prediction, action, current_price)
        session_signals.append({
            'symbol':      symbol,
            'side':        'BUY',
            'probability': prediction['probability'],
            'fired':       prediction['is_buy'],
            'action':      action,
            'price':       current_price,
        })

    # Session summary
    buy_signals  = [s for s in session_signals if s['side'] == 'BUY'  and s['fired']]
    sell_signals = [s for s in session_signals if s['side'] == 'SELL' and s['fired']]
    buy_orders   = [o for o in session_orders if o[0] == 'BUY']
    sell_orders  = [o for o in session_orders if o[0] == 'SELL']

    print("\n" + "=" * 55)
    print(f"  Session complete — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)
    print(f"  Symbols scanned:    {len(session_signals)}")
    print(f"  SELL signals fired: {len(sell_signals)}  →  orders: {len(sell_orders)}")
    print(f"  BUY signals fired:  {len(buy_signals)}   →  orders: {len(buy_orders)}")

    if sell_signals:
        print("\n  SELL signals this session:")
        for s in sorted(sell_signals, key=lambda x: x['probability'], reverse=True):
            print(f"    {s['symbol']:<6}  prob={s['probability']:.4f}  @ ${s['price']:.2f}  →  {s['action']}")

    if buy_signals:
        print("\n  BUY signals this session:")
        for s in sorted(buy_signals, key=lambda x: x['probability'], reverse=True):
            print(f"    {s['symbol']:<6}  prob={s['probability']:.4f}  @ ${s['price']:.2f}  →  {s['action']}")

    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Paper trade validator")
    parser.add_argument('--report',     action='store_true', help='Show log summary only (no scan)')
    parser.add_argument('--dry-run',    action='store_true', help='Scan but do not place orders')
    parser.add_argument('--symbols',    nargs='+',           help='Specific symbols to scan for BUY')
    parser.add_argument('--confidence',      type=float, default=0.6,  help='Min BUY confidence threshold')
    parser.add_argument('--sell-confidence', type=float, default=0.3,  help='Min SELL confidence floor (overrides model threshold)')
    args = parser.parse_args()

    if args.report:
        print_report()
        return

    run_scan(
        symbols=args.symbols,
        min_confidence=args.confidence,
        min_sell_confidence=args.sell_confidence,
        dry_run=args.dry_run,
    )

    print_report()


if __name__ == "__main__":
    main()
