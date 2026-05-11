import os

import pandas as pd
import plotly.express as px
import streamlit as st
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

st.set_page_config(page_title="ML Trading Dashboard", layout="wide", page_icon="📈")
st.title("ML Trading Dashboard")
st.caption("Paper Trading · Data refreshes every 60s (account/positions) or 5 min (orders/signals)")


# ── Alpaca connection ──────────────────────────────────────────────────────────

def get_client():
    key    = st.secrets.get("ALPACA_API_KEY")    or os.getenv("ALPACA_API_KEY")
    secret = st.secrets.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        st.error("Alpaca credentials not found. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to Streamlit Secrets.")
        st.stop()
    return TradingClient(api_key=key, secret_key=secret, paper=True)


# ── Data helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_account():
    a = get_client().get_account()
    return {
        'portfolio_value': float(a.portfolio_value),
        'buying_power':    float(a.buying_power),
        'equity':          float(a.equity),
        'last_equity':     float(a.last_equity),
    }

@st.cache_data(ttl=60)
def fetch_positions():
    rows = []
    for p in get_client().get_all_positions():
        rows.append({
            'symbol':        p.symbol,
            'qty':           float(p.qty),
            'entry_price':   float(p.avg_entry_price),
            'current_price': float(p.current_price),
            'unrealized_pl': float(p.unrealized_pl),
            'pl_pct':        float(p.unrealized_plpc) * 100,
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=300)
def fetch_filled_orders():
    request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500)
    orders = get_client().get_orders(filter=request)
    rows = []
    for o in orders:
        if o.filled_avg_price is None:
            continue
        rows.append({
            'order_id':   str(o.id),
            'symbol':     o.symbol,
            'side':       o.side.value,
            'qty':        float(o.filled_qty or 0),
            'fill_price': float(o.filled_avg_price),
            'filled_at':  pd.to_datetime(o.filled_at, utc=True),
            'order_type': str(o.order_class or o.type),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

@st.cache_data(ttl=300)
def load_orders_log():
    path = "paper_trade_log/orders.csv"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['timestamp'])

@st.cache_data(ttl=300)
def load_signals_log():
    path = "paper_trade_log/signals.csv"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['timestamp'])


# ── Account Summary ────────────────────────────────────────────────────────────

try:
    acct    = fetch_account()
    day_pl  = acct['equity'] - acct['last_equity']
    day_pct = day_pl / acct['last_equity'] * 100 if acct['last_equity'] else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", f"${acct['portfolio_value']:,.2f}")
    c2.metric("Buying Power",    f"${acct['buying_power']:,.2f}")
    c3.metric("Day P&L",         f"${day_pl:+,.2f}", delta=f"{day_pct:+.2f}%")
    c4.metric("Open Positions",  len(fetch_positions()))
except Exception as e:
    st.error(f"Could not load account data: {e}")

st.divider()


# ── Active Positions ───────────────────────────────────────────────────────────

st.subheader("Active Positions")
positions = fetch_positions()
orders_log = load_orders_log()

if not positions.empty:
    display = positions.copy()

    if not orders_log.empty and 'rsi' in orders_log.columns:
        buys = orders_log[orders_log['side'] == 'BUY']
        latest_buys = buys.sort_values('timestamp').groupby('symbol').last().reset_index()
        indicator_map = latest_buys.set_index('symbol')[['rsi', 'momentum', 'confidence']]
        display = display.join(indicator_map, on='symbol')

    display['entry_price']   = display['entry_price'].map('${:.2f}'.format)
    display['current_price'] = display['current_price'].map('${:.2f}'.format)
    display['unrealized_pl'] = display['unrealized_pl'].map('${:+,.2f}'.format)
    display['pl_pct']        = display['pl_pct'].map('{:+.1f}%'.format)
    display.columns          = [c.replace('_', ' ').title() for c in display.columns]

    st.dataframe(display, use_container_width=True, hide_index=True)
else:
    st.info("No open positions.")

st.divider()


# ── Trade History ──────────────────────────────────────────────────────────────

st.subheader("Trade History")
filled = fetch_filled_orders()

if not filled.empty:
    buys  = filled[filled['side'] == 'buy'].sort_values('filled_at').reset_index(drop=True)
    sells = filled[filled['side'] == 'sell'].sort_values('filled_at').reset_index(drop=True)

    rows = []
    for _, buy in buys.iterrows():
        entry_price = buy['fill_price']

        ind = {}
        if not orders_log.empty:
            match = orders_log[
                (orders_log['symbol'] == buy['symbol']) &
                (orders_log['side'] == 'BUY')
            ]
            if not match.empty:
                last = match.sort_values('timestamp').iloc[-1]
                ind['confidence']  = round(float(last.get('confidence') or 0), 3)
                ind['rsi']         = round(float(last['rsi']), 1) if 'rsi' in last.index and pd.notna(last['rsi']) else '—'
                ind['momentum %']  = round(float(last['momentum']), 4) if 'momentum' in last.index and pd.notna(last['momentum']) else '—'
                ind['take_profit'] = float(last.get('take_profit') or 0)
                ind['stop_loss']   = float(last.get('stop_loss') or 0)

        after = sells[
            (sells['symbol'] == buy['symbol']) &
            (sells['filled_at'] > buy['filled_at'])
        ]

        if not after.empty:
            sell       = after.iloc[0]
            exit_price = sell['fill_price']
            pl_pct     = (exit_price - entry_price) / entry_price * 100
            pl_abs     = (exit_price - entry_price) * buy['qty']

            tp = ind.get('take_profit', 0)
            sl = ind.get('stop_loss', 0)
            if tp and abs(exit_price - tp) / tp < 0.01:
                exit_via = '✅ Take Profit'
            elif sl and abs(exit_price - sl) / sl < 0.01:
                exit_via = '🛑 Stop Loss'
            else:
                exit_via = '🤖 SELL Signal'

            rows.append({
                'Symbol':         buy['symbol'],
                'Entry Date':     buy['filled_at'].strftime('%Y-%m-%d %H:%M'),
                'Exit Date':      sell['filled_at'].strftime('%Y-%m-%d %H:%M'),
                'Entry Price':    f"${entry_price:.2f}",
                'Exit Price':     f"${exit_price:.2f}",
                'P&L':            f"${pl_abs:+.2f}",
                'P&L %':          f"{pl_pct:+.1f}%",
                'Exit Via':       exit_via,
                'Confidence':     ind.get('confidence', '—'),
                'RSI at Entry':   ind.get('rsi', '—'),
                'Mom % at Entry': ind.get('momentum %', '—'),
            })
        else:
            rows.append({
                'Symbol':         buy['symbol'],
                'Entry Date':     buy['filled_at'].strftime('%Y-%m-%d %H:%M'),
                'Exit Date':      '—',
                'Entry Price':    f"${entry_price:.2f}",
                'Exit Price':     '—',
                'P&L':            '—',
                'P&L %':          '—',
                'Exit Via':       '⏳ Open',
                'Confidence':     ind.get('confidence', '—'),
                'RSI at Entry':   ind.get('rsi', '—'),
                'Mom % at Entry': ind.get('momentum %', '—'),
            })

    if rows:
        history_df = pd.DataFrame(rows)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        closed = history_df[~history_df['P&L %'].isin(['—', '⏳ Open'])].copy()
        if len(closed) >= 2:
            closed['pl_val'] = (
                closed['P&L %']
                .str.replace('%', '', regex=False)
                .str.replace('+', '', regex=False)
                .astype(float)
            )
            closed = closed.sort_values('Exit Date')
            closed['Cumulative P&L %'] = closed['pl_val'].cumsum()
            fig = px.line(
                closed, x='Exit Date', y='Cumulative P&L %',
                markers=True, title='Cumulative P&L % — Closed Trades',
            )
            fig.add_hline(y=0, line_dash='dash', line_color='gray')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No filled orders yet.")
else:
    st.info("No trade history yet. First live run is Monday 9:30 AM ET.")

st.divider()


# ── Signal Log ─────────────────────────────────────────────────────────────────

st.subheader("Recent Signals")
signals = load_signals_log()
if not signals.empty:
    recent = signals.sort_values('timestamp', ascending=False).head(200)
    st.dataframe(recent, use_container_width=True, hide_index=True)
else:
    st.info("No signals logged yet.")
