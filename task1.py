# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# 1. Load Data
# -------------------------------
@st.cache_data
def load_data():
    balls = pd.read_csv("IPL_BallByBall2008_2024(Updated).csv")
    teams = pd.read_csv("ipl_teams_2024_info.csv")

    # Ensure lowercase column names for safety
    balls.columns = [c.lower() for c in balls.columns]
    teams.columns = [c.lower() for c in teams.columns]

    return balls, teams

balls, teams = load_data()

# -------------------------------
# 2. Preprocessing
# -------------------------------
# Ensure Season Column is Clean
if "season" in balls.columns:
    # Try to clean it
    balls["season"] = pd.to_numeric(balls["season"], errors="coerce")
    balls = balls.dropna(subset=["season"])
    balls["season"] = balls["season"].astype(int)

elif "date" in balls.columns:
    # Extract from date column
    balls["season"] = pd.to_datetime(balls["date"], errors="coerce").dt.year
    balls = balls.dropna(subset=["season"])
    balls["season"] = balls["season"].astype(int)

else:
    st.error("No 'season' or 'date' column found in dataset. Please check the file.")
    st.stop()


# Placeholder match dataset for wins (if available in your CSVs)
# Some IPL datasets keep match info separate - you may need a 'matches.csv'
# For now, we'll simulate with team1/team2/winner if present
matches_cols = {"match_id", "season", "team1", "team2", "winner"}
matches = None
if matches_cols.issubset(balls.columns):
    matches = balls.drop_duplicates("match_id")[list(matches_cols)]

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.set_page_config(page_title="IPL Season Insights", layout="wide")
st.title("🏏 IPL Season Insights Dashboard")

season = st.sidebar.selectbox("Select a Season", season_list)

st.markdown(f"### Insights for Season **{season}**")

# -------------------------------
# 4. Season Data
# -------------------------------
season_data = balls[balls["season"] == season]

# Runs
total_runs = season_data["batsman_runs"].sum()

# Wickets
total_wickets = season_data["player_dismissed"].notna().sum()

# Top Batsmen
top_batsmen = (
    season_data.groupby("batsman")["batsman_runs"]
    .sum()
    .reset_index()
    .sort_values("batsman_runs", ascending=False)
    .head(5)
)

# Top Bowlers
top_bowlers = (
    season_data[season_data["player_dismissed"].notna()]
    .groupby("bowler")["player_dismissed"]
    .count()
    .reset_index()
    .sort_values("player_dismissed", ascending=False)
    .head(5)
)

# Win Ratios (if match data exists)
win_ratios = None
if matches is not None:
    season_matches = matches[matches["season"] == season]
    team_played = (
        pd.concat([season_matches["team1"], season_matches["team2"]])
        .value_counts()
        .rename("matches_played")
    )
    team_won = season_matches["winner"].value_counts().rename("matches_won")
    win_ratios = (
        pd.concat([team_played, team_won], axis=1).fillna(0).reset_index()
    )
    win_ratios["win_ratio"] = (
        win_ratios["matches_won"] / win_ratios["matches_played"]
    ).round(2)
    win_ratios.rename(columns={"index": "team"}, inplace=True)

# -------------------------------
# 5. Display Metrics
# -------------------------------
col1, col2 = st.columns(2)
col1.metric("Total Runs", f"{total_runs:,}")
col2.metric("Total Wickets", f"{total_wickets:,}")

# -------------------------------
# 6. Charts
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Batsmen")
    fig_bat = px.bar(
        top_batsmen,
        x="batsman",
        y="batsman_runs",
        text="batsman_runs",
        color="batsman_runs",
        title="Top 5 Run Scorers",
    )
    st.plotly_chart(fig_bat, use_container_width=True)

with col2:
    st.subheader("Top Bowlers")
    fig_bowl = px.bar(
        top_bowlers,
        x="bowler",
        y="player_dismissed",
        text="player_dismissed",
        color="player_dismissed",
        title="Top 5 Wicket Takers",
    )
    st.plotly_chart(fig_bowl, use_container_width=True)

if win_ratios is not None:
    st.subheader("Team Win Ratios")
    fig_win = px.bar(
        win_ratios,
        x="team",
        y="win_ratio",
        text="win_ratio",
        color="win_ratio",
        title="Win Ratios per Team",
    )
    st.plotly_chart(fig_win, use_container_width=True)

# -------------------------------
# 7. Season Trends
# -------------------------------
st.subheader("Season Trends (All Seasons)")
trend_runs = balls.groupby("season")["batsman_runs"].sum().reset_index()
trend_wickets = (
    balls[balls["player_dismissed"].notna()]
    .groupby("season")["player_dismissed"]
    .count()
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:
    fig_trend_runs = px.line(
        trend_runs, x="season", y="batsman_runs", markers=True, title="Runs Trend"
    )
    st.plotly_chart(fig_trend_runs, use_container_width=True)

with col2:
    fig_trend_wickets = px.line(
        trend_wickets, x="season", y="player_dismissed", markers=True, title="Wickets Trend"
    )
    st.plotly_chart(fig_trend_wickets, use_container_width=True)
