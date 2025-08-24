"""
Streamlit Dashboard

Real-time dashboard for monitoring the arbitrage trading system.
Provides portfolio overview, P&L tracking, signal monitoring, and risk management.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import requests
import websocket
import threading
from queue import Queue
import altair as alt

from ..config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Crypto Arbitrage Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
    }
    
    .profit-positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .profit-negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-active {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-inactive {
        color: #dc3545;
        font-weight: bold;
    }
    
    .alert-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    
    .alert-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)


class DashboardState:
    """Manage dashboard state and data"""
    
    def __init__(self):
        self.settings = get_settings()
        self.api_base_url = f"http://{self.settings.api.host}:{self.settings.api.port}"
        self.websocket_url = f"ws://{self.settings.api.host}:{self.settings.api.port}/ws"
        
        # Authentication token (in production, handle securely)
        self.auth_token = "dummy_token_for_demo"
        self.headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        # Data storage
        self.system_status = {}
        self.balances = []
        self.positions = []
        self.orders = []
        self.portfolio_history = []
        self.signals = []
        self.risk_status = {}
        
        # Real-time data
        self.realtime_data = Queue()
        self.websocket_client = None
        self.websocket_thread = None
        
        # Initialize session state
        if 'last_update' not in st.session_state:
            st.session_state.last_update = 0
        
        if 'websocket_connected' not in st.session_state:
            st.session_state.websocket_connected = False


# Initialize dashboard state
@st.cache_resource
def get_dashboard_state():
    """Get singleton dashboard state"""
    return DashboardState()


dashboard_state = get_dashboard_state()


def make_api_request(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Make API request to the trading system"""
    try:
        url = f"{dashboard_state.api_base_url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, headers=dashboard_state.headers, timeout=5)
        elif method == "POST":
            response = requests.post(url, headers=dashboard_state.headers, json=data, timeout=5)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed: {response.status_code} - {response.text}")
            return {}
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to trading system: {e}")
        return {}


def websocket_client():
    """WebSocket client for real-time updates"""
    def on_message(ws, message):
        try:
            data = json.loads(message)
            dashboard_state.realtime_data.put(data)
        except json.JSONDecodeError:
            pass
    
    def on_error(ws, error):
        logger.error(f"WebSocket error: {error}")
        st.session_state.websocket_connected = False
    
    def on_close(ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")
        st.session_state.websocket_connected = False
    
    def on_open(ws):
        logger.info("WebSocket connection opened")
        st.session_state.websocket_connected = True
    
    ws = websocket.WebSocketApp(
        dashboard_state.websocket_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    ws.run_forever()


def start_websocket():
    """Start WebSocket connection in background thread"""
    if not st.session_state.websocket_connected:
        dashboard_state.websocket_thread = threading.Thread(target=websocket_client, daemon=True)
        dashboard_state.websocket_thread.start()


def update_data():
    """Update dashboard data from API"""
    current_time = time.time()
    
    # Update every 5 seconds
    if current_time - st.session_state.last_update < 5:
        return
    
    # Get system status
    dashboard_state.system_status = make_api_request("/status")
    
    # Get balances
    dashboard_state.balances = make_api_request("/balances")
    
    # Get positions
    dashboard_state.positions = make_api_request("/positions")
    
    # Get recent orders
    dashboard_state.orders = make_api_request("/orders?limit=50")
    
    # Get portfolio history
    portfolio_response = make_api_request("/portfolio/history?hours=24")
    dashboard_state.portfolio_history = portfolio_response.get("history", [])
    
    # Get risk status
    dashboard_state.risk_status = make_api_request("/risk/status")
    
    st.session_state.last_update = current_time


def render_header():
    """Render dashboard header"""
    st.title("🏦 Crypto Arbitrage Trading Dashboard")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if dashboard_state.system_status:
            status = dashboard_state.system_status.get("status", "unknown")
            if status == "active":
                st.markdown(f'<span class="status-active">🟢 System Status: {status.upper()}</span>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="status-inactive">🔴 System Status: {status.upper()}</span>', 
                           unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-inactive">🔴 System Status: DISCONNECTED</span>', 
                       unsafe_allow_html=True)
    
    with col2:
        if st.session_state.websocket_connected:
            st.markdown('<span class="status-active">🟢 WebSocket: Connected</span>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-inactive">🔴 WebSocket: Disconnected</span>', 
                       unsafe_allow_html=True)
    
    with col3:
        last_update = datetime.fromtimestamp(st.session_state.last_update)
        st.markdown(f"📅 Last Update: {last_update.strftime('%H:%M:%S')}")


def render_portfolio_overview():
    """Render portfolio overview section"""
    st.header("💰 Portfolio Overview")
    
    if not dashboard_state.balances:
        st.warning("No balance data available")
        return
    
    # Calculate total portfolio value
    total_value = sum(balance.get("total_value_usd", 0) for balance in dashboard_state.balances)
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Portfolio Value",
            value=f"${total_value:,.2f}",
            delta=None
        )
    
    with col2:
        total_positions = len(dashboard_state.positions)
        st.metric(
            label="Active Positions",
            value=total_positions
        )
    
    with col3:
        if dashboard_state.system_status:
            today_orders = dashboard_state.system_status.get("total_orders_today", 0)
            st.metric(
                label="Orders Today",
                value=today_orders
            )
    
    with col4:
        if dashboard_state.system_status:
            today_signals = dashboard_state.system_status.get("total_signals_today", 0)
            st.metric(
                label="Signals Today",
                value=today_signals
            )
    
    # Balance breakdown
    if dashboard_state.balances:
        st.subheader("Balance by Exchange")
        
        balance_data = []
        for balance in dashboard_state.balances:
            for currency, amount in balance.get("balances", {}).items():
                balance_data.append({
                    "Exchange": balance["exchange"],
                    "Currency": currency,
                    "Amount": float(amount),
                    "Value USD": float(balance.get("total_value_usd", 0))
                })
        
        if balance_data:
            balance_df = pd.DataFrame(balance_data)
            
            # Exchange distribution pie chart
            exchange_values = balance_df.groupby("Exchange")["Value USD"].sum().reset_index()
            
            if len(exchange_values) > 0:
                fig = px.pie(
                    exchange_values,
                    values="Value USD",
                    names="Exchange",
                    title="Portfolio Distribution by Exchange"
                )
                st.plotly_chart(fig, use_container_width=True)


def render_pnl_chart():
    """Render P&L chart"""
    st.header("📊 Portfolio Performance")
    
    if not dashboard_state.portfolio_history:
        st.warning("No portfolio history available")
        return
    
    # Process portfolio history
    history_data = []
    for snapshot in dashboard_state.portfolio_history:
        metadata = json.loads(snapshot.get("metadata", "{}"))
        history_data.append({
            "timestamp": datetime.fromtimestamp(snapshot["timestamp"]),
            "total_value": float(snapshot["total_value_usd"]),
            "realized_pnl": float(snapshot["total_realized_pnl"]),
            "unrealized_pnl": float(snapshot["total_unrealized_pnl"]),
            "daily_pnl": float(snapshot["daily_pnl"])
        })
    
    if not history_data:
        st.warning("No valid portfolio history data")
        return
    
    history_df = pd.DataFrame(history_data)
    history_df = history_df.sort_values("timestamp")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Portfolio Value Over Time", "Daily P&L"),
        vertical_spacing=0.1
    )
    
    # Portfolio value chart
    fig.add_trace(
        go.Scatter(
            x=history_df["timestamp"],
            y=history_df["total_value"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#1f77b4", width=2)
        ),
        row=1, col=1
    )
    
    # Daily P&L bar chart
    colors = ["green" if pnl >= 0 else "red" for pnl in history_df["daily_pnl"]]
    fig.add_trace(
        go.Bar(
            x=history_df["timestamp"],
            y=history_df["daily_pnl"],
            name="Daily P&L",
            marker_color=colors
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    if len(history_df) > 1:
        col1, col2, col3, col4 = st.columns(4)
        
        total_return = ((history_df["total_value"].iloc[-1] - history_df["total_value"].iloc[0]) / 
                       history_df["total_value"].iloc[0] * 100)
        
        avg_daily_pnl = history_df["daily_pnl"].mean()
        total_realized_pnl = history_df["realized_pnl"].iloc[-1]
        total_unrealized_pnl = history_df["unrealized_pnl"].iloc[-1]
        
        with col1:
            st.metric(
                label="24h Return",
                value=f"{total_return:.2f}%",
                delta=None
            )
        
        with col2:
            color = "profit-positive" if avg_daily_pnl >= 0 else "profit-negative"
            st.markdown(f'<div class="metric-card">Avg Daily P&L<br/>'
                       f'<span class="{color}">${avg_daily_pnl:.2f}</span></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            color = "profit-positive" if total_realized_pnl >= 0 else "profit-negative"
            st.markdown(f'<div class="metric-card">Realized P&L<br/>'
                       f'<span class="{color}">${total_realized_pnl:.2f}</span></div>', 
                       unsafe_allow_html=True)
        
        with col4:
            color = "profit-positive" if total_unrealized_pnl >= 0 else "profit-negative"
            st.markdown(f'<div class="metric-card">Unrealized P&L<br/>'
                       f'<span class="{color}">${total_unrealized_pnl:.2f}</span></div>', 
                       unsafe_allow_html=True)


def render_positions():
    """Render current positions"""
    st.header("📈 Current Positions")
    
    if not dashboard_state.positions:
        st.info("No active positions")
        return
    
    # Create positions DataFrame
    positions_data = []
    for pos in dashboard_state.positions:
        positions_data.append({
            "Exchange": pos["exchange"],
            "Symbol": pos["symbol"],
            "Side": pos["side"],
            "Size": float(pos["size"]),
            "Entry Price": float(pos["entry_price"]),
            "Current Price": float(pos["current_price"]),
            "Unrealized P&L": float(pos["unrealized_pnl"]),
            "Realized P&L": float(pos["realized_pnl"])
        })
    
    positions_df = pd.DataFrame(positions_data)
    
    # Style the DataFrame
    def style_pnl(val):
        color = 'color: green' if val >= 0 else 'color: red'
        return color
    
    styled_df = positions_df.style.applymap(style_pnl, subset=['Unrealized P&L', 'Realized P&L'])
    
    st.dataframe(styled_df, use_container_width=True)


def render_recent_orders():
    """Render recent orders"""
    st.header("📋 Recent Orders")
    
    if not dashboard_state.orders:
        st.info("No recent orders")
        return
    
    # Create orders DataFrame
    orders_data = []
    for order in dashboard_state.orders:
        orders_data.append({
            "Time": datetime.fromtimestamp(order["created_time"]).strftime("%H:%M:%S"),
            "Exchange": order["exchange"],
            "Symbol": order["symbol"],
            "Side": order["side"],
            "Type": order["order_type"],
            "Quantity": float(order["quantity"]),
            "Price": f"${float(order['price']):.4f}" if order["price"] else "Market",
            "Status": order["status"],
            "Filled": f"{float(order['filled_quantity']):.6f}"
        })
    
    orders_df = pd.DataFrame(orders_data)
    
    # Color code status
    def style_status(val):
        if val == "FILLED":
            return 'color: green'
        elif val == "CANCELLED":
            return 'color: red'
        elif val == "PENDING":
            return 'color: orange'
        return ''
    
    styled_df = orders_df.style.applymap(style_status, subset=['Status'])
    
    st.dataframe(styled_df, use_container_width=True, height=300)


def render_risk_management():
    """Render risk management section"""
    st.header("⚠️ Risk Management")
    
    if not dashboard_state.risk_status:
        st.warning("Risk status not available")
        return
    
    # Risk metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        circuit_breaker = dashboard_state.risk_status.get("circuit_breaker_active", False)
        if circuit_breaker:
            st.markdown('<div class="alert-high">🚨 Circuit Breaker: ACTIVE</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">🟢 Circuit Breaker: Inactive</div>', 
                       unsafe_allow_html=True)
    
    with col2:
        position_limits = dashboard_state.risk_status.get("position_limits_breached", False)
        if position_limits:
            st.markdown('<div class="alert-high">⚠️ Position Limits: BREACHED</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">✅ Position Limits: OK</div>', 
                       unsafe_allow_html=True)
    
    with col3:
        daily_loss_limit = dashboard_state.risk_status.get("daily_loss_limit_breached", False)
        if daily_loss_limit:
            st.markdown('<div class="alert-high">💰 Daily Loss Limit: BREACHED</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">💚 Daily Loss Limit: OK</div>', 
                       unsafe_allow_html=True)
    
    # Daily P&L vs Limit
    current_pnl = float(dashboard_state.risk_status.get("current_daily_pnl", 0))
    max_loss = float(dashboard_state.risk_status.get("max_daily_loss", 1000))
    
    col1, col2 = st.columns(2)
    
    with col1:
        pnl_color = "profit-positive" if current_pnl >= 0 else "profit-negative"
        st.markdown(f'<div class="metric-card">Current Daily P&L<br/>'
                   f'<span class="{pnl_color}">${current_pnl:.2f}</span></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-card">Daily Loss Limit<br/>'
                   f'<span>${max_loss:.2f}</span></div>', 
                   unsafe_allow_html=True)
    
    # Kill switch
    st.subheader("Emergency Controls")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🛑 EMERGENCY KILL SWITCH", type="primary"):
            st.session_state.show_kill_switch_confirm = True
    
    with col2:
        if st.session_state.get("show_kill_switch_confirm", False):
            reason = st.text_input("Reason for kill switch activation:")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Confirm Kill Switch"):
                    response = make_api_request(
                        "/risk/kill-switch", 
                        "POST", 
                        {"reason": reason, "emergency": True}
                    )
                    if response:
                        st.success("Kill switch activated!")
                    st.session_state.show_kill_switch_confirm = False
            
            with col_b:
                if st.button("Cancel"):
                    st.session_state.show_kill_switch_confirm = False


def render_real_time_feed():
    """Render real-time data feed in sidebar"""
    st.sidebar.header("📡 Real-Time Feed")
    
    # Process real-time data
    recent_messages = []
    while not dashboard_state.realtime_data.empty():
        try:
            message = dashboard_state.realtime_data.get_nowait()
            recent_messages.append(message)
        except:
            break
    
    # Display recent messages
    if recent_messages:
        for msg in recent_messages[-5:]:  # Show last 5 messages
            msg_type = msg.get("type", "unknown")
            timestamp = datetime.fromtimestamp(msg.get("timestamp", time.time()))
            
            if msg_type == "signal":
                data = msg["data"]
                profit = data.get("expected_profit_pct", 0)
                st.sidebar.markdown(
                    f"**🎯 New Signal** - {timestamp.strftime('%H:%M:%S')}\n\n"
                    f"Symbol: {data.get('symbol', 'N/A')}\n\n"
                    f"Profit: {profit:.2f}%"
                )
            
            elif msg_type == "order_update":
                data = msg["data"]
                st.sidebar.markdown(
                    f"**📋 Order Update** - {timestamp.strftime('%H:%M:%S')}\n\n"
                    f"Order: {data.get('order_id', 'N/A')[:8]}\n\n"
                    f"Status: {data.get('status', 'N/A')}"
                )
    
    # Connection status
    if st.session_state.websocket_connected:
        st.sidebar.success("🟢 Real-time feed active")
    else:
        st.sidebar.error("🔴 Real-time feed disconnected")
        if st.sidebar.button("Reconnect WebSocket"):
            start_websocket()


def render_backtest_section():
    """Render backtesting section"""
    st.header("🔄 Backtesting")
    
    with st.expander("Run Backtest", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
            end_date = st.date_input("End Date", value=datetime.now())
            initial_balance = st.number_input("Initial Balance (USDT)", value=10000.0)
        
        with col2:
            exchanges = st.multiselect(
                "Exchanges",
                ["binance", "kraken", "coinbase", "bybit"],
                default=["binance", "kraken"]
            )
            symbols = st.multiselect(
                "Symbols",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"],
                default=["BTC/USDT"]
            )
        
        enable_slippage = st.checkbox("Enable Slippage Simulation", value=True)
        enable_latency = st.checkbox("Enable Latency Simulation", value=True)
        
        if st.button("Run Backtest"):
            backtest_data = {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_balance": {"USDT": initial_balance},
                "exchanges": exchanges,
                "symbols": symbols,
                "enable_slippage": enable_slippage,
                "enable_latency": enable_latency
            }
            
            response = make_api_request("/backtest", "POST", backtest_data)
            if response:
                st.success(f"Backtest started: {response.get('backtest_id', 'unknown')}")
            else:
                st.error("Failed to start backtest")


def main():
    """Main dashboard function"""
    # Initialize
    update_data()
    start_websocket()
    
    # Render sections
    render_header()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio", "📈 Positions", "📋 Orders", "⚠️ Risk", "🔄 Backtest"
    ])
    
    with tab1:
        render_portfolio_overview()
        render_pnl_chart()
    
    with tab2:
        render_positions()
    
    with tab3:
        render_recent_orders()
    
    with tab4:
        render_risk_management()
    
    with tab5:
        render_backtest_section()
    
    # Sidebar
    render_real_time_feed()
    
    # Auto-refresh
    time.sleep(1)
    st.rerun()


if __name__ == "__main__":
    main()
