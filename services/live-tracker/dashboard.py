import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import redis
import streamlit as st
import psycopg2

st.set_page_config(page_title="CS2 Prediction Tracker", page_icon="🎮", layout="wide")

POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'cs2_predictions')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'your_password')
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', '5432'))

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_DB', '0'))


@st.cache_resource
def get_pg_conn():
    return psycopg2.connect(host=POSTGRES_HOST, database=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD, port=POSTGRES_PORT)


def get_recent_predictions(hours=24):
    query = """
        SELECT * FROM predictions
        WHERE timestamp > NOW() - INTERVAL %s
        ORDER BY timestamp DESC
    """
    with get_pg_conn() as conn:
        df = pd.read_sql(query, conn, params=(f'{hours} hours',))
    return df


def get_performance_metrics():
    query = """
        SELECT 
            COUNT(*) as total_predictions,
            SUM(CASE WHEN is_correct = true THEN 1 ELSE 0 END) as correct,
            AVG(confidence_score) as avg_confidence,
            SUM(profit_loss) as total_profit,
            AVG(CASE WHEN is_correct IS NOT NULL THEN 
                CASE WHEN is_correct = true THEN 1 ELSE 0 END 
            END) as accuracy
        FROM predictions
        WHERE timestamp > NOW() - INTERVAL '7 days'
    """
    with get_pg_conn().cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchone()
    return {
        'total_predictions': result[0] or 0,
        'correct_predictions': result[1] or 0,
        'avg_confidence': float(result[2] or 0),
        'total_profit': float(result[3] or 0),
        'accuracy': float(result[4] or 0),
    }


def get_live_predictions():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    keys = r.keys("prediction:*")
    rows = []
    for k in keys[:50]:
        try:
            rows.append(json.loads(r.get(k)))
        except Exception:
            pass
    return pd.DataFrame(rows)


def main():
    st.title("🎮 CS2 Live Prediction Tracker")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Settings")
        time_range = st.selectbox("Time Range", ["Last 24 Hours", "Last 7 Days", "Last 30 Days"])
        hours = 24 if time_range == "Last 24 Hours" else 24 * 7 if time_range == "Last 7 Days" else 24 * 30

    metrics = get_performance_metrics()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Predictions", f"{metrics['total_predictions']:,}")
    c2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c3.metric("Avg Confidence", f"{metrics['avg_confidence']:.1%}")
    c4.metric("Total Profit", f"${metrics['total_profit']:,.2f}")
    roi = (metrics['total_profit'] / (max(metrics['total_predictions'], 1) * 100)) * 100
    c5.metric("ROI", f"{roi:.1f}%")

    st.markdown("---")
    left, right = st.columns([2, 1])
    with left:
        st.subheader("🔴 Live Predictions")
        live_df = get_live_predictions()
        if not live_df.empty:
            for _, pred in live_df.iterrows():
                confidence_color = "🟢" if pred.get('confidence_score', 0) > 0.7 else ("🟡" if pred.get('confidence_score', 0) > 0.6 else "🔴")
                cols = st.columns([3, 2, 2, 2])
                cols[0].write(f"**{pred.get('team1','?')} vs {pred.get('team2','?')}**")
                cols[1].write(f"Winner: **{pred.get('predicted_winner','?')}**")
                cols[2].write(f"{confidence_color} {pred.get('confidence_score',0):.1%}")
                cols[3].write(f"EV: {pred.get('expected_value',0):.3f}")
        else:
            st.info("No live predictions at the moment")
    with right:
        st.subheader("⚡ High Confidence Alerts")
        if not live_df.empty:
            filt = live_df[live_df['confidence_score'] > 0.75]
            if not filt.empty:
                for _, a in filt.iterrows():
                    st.success(f"{a['predicted_winner']} | Conf: {a['confidence_score']:.1%} | EV: {a['expected_value']:.3f}")
            else:
                st.info("No high confidence predictions")
        else:
            st.info("No data")

    st.markdown("---")
    st.subheader("📜 Recent Predictions History")
    recent_df = get_recent_predictions(hours)
    if not recent_df.empty:
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        display_cols = ['timestamp', 'team1', 'team2', 'predicted_winner', 'confidence_score', 'actual_winner', 'is_correct', 'profit_loss']
        st.dataframe(recent_df[display_cols], use_container_width=True, height=450)
    else:
        st.info("No recent predictions")


if __name__ == "__main__":
    main()
