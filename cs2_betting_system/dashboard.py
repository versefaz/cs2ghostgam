import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import redis
import streamlit as st

st.set_page_config(page_title="CS2 Paper Trading Dashboard", layout="wide")

r = redis.Redis(host='localhost', port=6379)

st.title("🎮 CS2 Betting Paper Trading Dashboard")

state = r.get('paper_trading_state')
positions = []
if state:
    state_data = json.loads(state)
    positions = state_data.get('positions', [])
else:
    st.info("No state found in Redis. Start the paper bot to populate data.")
    state_data = {'balance': 0, 'positions': []}

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Balance", f"${state_data.get('balance', 0):.2f}")
with col2:
    total_bets = len(positions)
    st.metric("Total Bets", total_bets)
with col3:
    completed = [b for b in positions if b.get('status') in ['won', 'lost']]
    won = [b for b in completed if b.get('status') == 'won']
    st.metric("🎯 Win Rate", f"{(len(won)/len(completed)*100):.1f}%" if completed else "0.0%")
with col4:
    total_profit = sum((b.get('returns', 0) - b.get('amount', 0)) for b in completed)
    total_staked = sum(b.get('amount', 0) for b in completed)
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    st.metric("💰 Total P&L", f"${total_profit:,.2f}", delta=f"{roi:+.1f}% ROI")

st.subheader("📊 Active Positions")
active_positions = [p for p in positions if p.get('status') == 'pending']
if active_positions:
    df = pd.DataFrame(active_positions)
    st.dataframe(df[['match_id', 'team', 'odds', 'amount', 'confidence', 'timestamp']], use_container_width=True)
else:
    st.info("No active positions at the moment")

st.subheader("📡 Recent Bets")
if completed:
    dfc = pd.DataFrame(completed)
    dfc['profit'] = dfc.get('returns', 0) - dfc.get('amount', 0)
    st.dataframe(dfc[['match_id', 'team', 'odds', 'amount', 'status', 'returns', 'profit', 'completed_at']], use_container_width=True, height=350)

st.subheader("📈 Performance Over Time")
if completed:
    chart_data = []
    running_balance = 10000
    for bet in sorted(completed, key=lambda x: x.get('timestamp', '')):
        profit = (bet.get('returns', 0) - bet.get('amount', 0))
        running_balance += profit
        chart_data.append({
            'Date': bet.get('timestamp', datetime.now()),
            'Balance': running_balance,
            'Result': 'Win' if bet.get('status') == 'won' else 'Loss'
        })
    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        fig = px.line(chart_df, x='Date', y='Balance', title='Bankroll Performance', markers=True)
        colors = ['green' if r == 'Win' else 'red' for r in chart_df['Result']]
        fig.update_traces(marker=dict(color=colors, size=8))
        st.plotly_chart(fig, use_container_width=True)
