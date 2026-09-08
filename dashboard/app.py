import os
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

st.set_page_config(page_title="ML Trading Dashboard", layout="wide", page_icon="📈")
st.title("ML Trading Dashboard")

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"],
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] * {
    color: inherit !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="false"],
.stTabs [data-baseweb="tab-list"] button[aria-selected="false"] * {
    color: #888888 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #888888 !important;
}
</style>
""", unsafe_allow_html=True)


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


# ── Load data ──────────────────────────────────────────────────────────────────

positions  = fetch_positions()
orders_log = load_orders_log()
filled     = fetch_filled_orders()
signals    = load_signals_log()

all_symbols = sorted(set(positions['symbol']) if not positions.empty else set()
                      | (set(filled['symbol']) if not filled.empty else set())
                      | (set(signals['symbol']) if not signals.empty else set()))

all_dates = pd.concat([
    filled['filled_at'].dt.tz_localize(None) if not filled.empty else pd.Series(dtype='datetime64[ns]'),
    signals['timestamp'] if not signals.empty else pd.Series(dtype='datetime64[ns]'),
])
if not all_dates.empty:
    min_date = all_dates.min().date()
    max_date = all_dates.max().date()
else:
    max_date = pd.Timestamp.utcnow().date()
    min_date = max_date - timedelta(days=30)


# ── Sidebar: filters & controls ─────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    symbol_filter = st.multiselect("Symbols", options=all_symbols, default=[],
                                    help="Leave empty to show all symbols.")

    date_range = st.date_input("Date range (history & signals)",
                                value=(min_date, max_date),
                                min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        range_start, range_end = date_range
    else:
        range_start, range_end = min_date, max_date

    st.divider()
    st.caption("Paper Trading · Data refreshes every 60s (account/positions) or 5 min (orders/signals)")


def in_symbols(df, col='symbol'):
    return df if not symbol_filter else df[df[col].isin(symbol_filter)]

def in_date_range(df, col):
    ts = pd.to_datetime(df[col])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    return df[(ts.dt.date >= range_start) & (ts.dt.date <= range_end)]


# ── Account Summary ────────────────────────────────────────────────────────────

try:
    acct    = fetch_account()
    day_pl  = acct['equity'] - acct['last_equity']
    day_pct = day_pl / acct['last_equity'] * 100 if acct['last_equity'] else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", f"${acct['portfolio_value']:,.2f}")
    c2.metric("Buying Power",    f"${acct['buying_power']:,.2f}")
    c3.metric("Day P&L",         f"${day_pl:+,.2f}", delta=f"{day_pct:+.2f}%")
    c4.metric("Open Positions",  len(positions))
except Exception as e:
    st.error(f"Could not load account data: {e}")

st.divider()


tab_positions, tab_history, tab_signals = st.tabs(
    ["📊 Active Positions", "📜 Trade History", "🔔 Recent Signals"]
)


# ── Active Positions ───────────────────────────────────────────────────────────

with tab_positions:
    view = in_symbols(positions)

    if not view.empty:
        display = view.copy()

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


# ── Trade History ──────────────────────────────────────────────────────────────

with tab_history:
    filled_view = in_symbols(filled) if not filled.empty else filled
    filled_view = in_date_range(filled_view, 'filled_at') if not filled_view.empty else filled_view

    if not filled_view.empty:
        buys  = filled_view[filled_view['side'] == 'buy'].sort_values('filled_at').reset_index(drop=True)
        sells = filled_view[filled_view['side'] == 'sell'].sort_values('filled_at').reset_index(drop=True)

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

                sl = ind.get('stop_loss', 0)
                if sell.get('order_type') in ('bracket', 'oto'):
                    # place_bracket_order() sets order_class to 'bracket' (TP+SL) or
                    # 'oto' (SL only) — never 'oco'. A fill with this class is always
                    # one of the two bracket legs, so distinguish by price proximity
                    # to the logged stop-loss and default to Take Profit otherwise.
                    if sl and abs(exit_price - sl) / sl < 0.01:
                        exit_via = '🛑 Stop Loss'
                    else:
                        exit_via = '✅ Take Profit'
                else:
                    exit_via = '🤖 SELL Signal'

                rows.append({
                    'Symbol':         buy['symbol'],
                    'Entry Date':     buy['filled_at'].strftime('%Y-%m-%d %H:%M'),
                    'Exit Date':      sell['filled_at'].strftime('%Y-%m-%d %H:%M'),
                    'Entry Price':    f"${entry_price:.2f}",
                    'Exit Price':     f"${exit_price:.2f}",
                    'Price Diff':     f"${exit_price - entry_price:+.2f}",
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
                    'Price Diff':     '—',
                    'P&L':            '—',
                    'P&L %':          '—',
                    'Exit Via':       '⏳ Open',
                    'Confidence':     ind.get('confidence', '—'),
                    'RSI at Entry':   ind.get('rsi', '—'),
                    'Mom % at Entry': ind.get('momentum %', '—'),
                })

        if rows:
            history_df = pd.DataFrame(rows).sort_values('Entry Date', ascending=False)

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
                    height=380,
                )
                fig.add_hline(y=0, line_dash='dash', line_color='gray')
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("No filled orders yet.")
    else:
        st.info("No trade history in the selected filters. First live run is Monday 9:30 AM ET.")


# ── Signal Log ─────────────────────────────────────────────────────────────────

with tab_signals:
    signals_view = in_symbols(signals) if not signals.empty else signals
    signals_view = in_date_range(signals_view, 'timestamp') if not signals_view.empty else signals_view

    if not signals_view.empty:
        recent = signals_view.sort_values('timestamp', ascending=False).head(200)
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No signals logged for the selected filters.")
