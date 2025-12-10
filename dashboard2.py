
"""
Streamlit IPL Analytics Dashboard (single-file)

Files expected (place in same folder as this file):
- IPL_BallByBall2008_2024(Updated).csv
- team_performance_dataset_2008to2024.csv
- Players_Info_2024.csv
- ipl_teams_2024_info.csv

Dependencies:
- pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit, plotly, joblib

Run:
    streamlit run IPL_Dashboard_Streamlit_App.py

This app implements the 10 tasks listed in the provided image (Season insights, Venue impact, Head-to-Head,
Orange/Purple cap tracker + predictor, Player career tracker, Match outcome & score predictor, Best XI and Fantasy recommender).

Note: Training models may take a few moments depending on your machine. Models are cached and saved with joblib to speed up repeated runs.
"""

import streamlit as st
st.set_page_config(page_title='IPL Analytics Dashboard', layout='wide', initial_sidebar_state='expanded')

# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import os
from datetime import datetime

# --- Constants / File paths ---
BALL_BY_BALL = 'IPL_BallByBall2008_2024(Updated).csv'
TEAM_PERF = 'team_performance_dataset_2008to2024.csv'
PLAYERS = 'Players_Info_2024.csv'
TEAMS = 'ipl_teams_2024_info.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Color Theme (modern navy + accent) ---
PRIMARY_BG = '#0f1724'   # dark navy
CARD_BG = '#0b1220'
ACCENT = '#ff6b6b'       # warm coral accent
ACCENT2 = '#7c3aed'      # purple accent
TEXT = '#e6eef8'

st.markdown(f"""
<style>
body {{background-color: {PRIMARY_BG}; color: {TEXT};}}
section.main {{background-color: {PRIMARY_BG};}}
.css-18e3th9 {{background-color: {PRIMARY_BG};}}
.card {{background-color: {CARD_BG}; border-radius: 12px; padding: 12px;}}
</style>
""", unsafe_allow_html=True)

# --- Utility functions ---
@st.cache_data
def load_data():
    bb = pd.read_csv(BALL_BY_BALL, low_memory=False)
    tp = pd.read_csv(TEAM_PERF, low_memory=False)
    players = pd.read_csv(PLAYERS, low_memory=False)
    teams = pd.read_csv(TEAMS, low_memory=False)

    # Standardize column names
    bb.columns = [c.strip() for c in bb.columns]
    tp.columns = [c.strip() for c in tp.columns]
    players.columns = [c.strip() for c in players.columns]
    teams.columns = [c.strip() for c in teams.columns]

    # Convert Date columns
    for df in (bb, tp):
        for col in df.columns:
            if 'Date' in col:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass
    return bb, tp, players, teams

bb, tp, players_df, teams_df = load_data()

# --- Lightweight preprocessing and aggregates ---
@st.cache_data
def compute_aggregates(bb, tp, players):
    # Ensure Season column exists in tp
    if 'Season' not in tp.columns and 'Date' in tp.columns:
        tp = tp.copy()
        tp['Season'] = pd.to_datetime(tp['Date'], errors='coerce').dt.year

    # Ensure Season exists in bb too
    if 'Season' not in bb.columns and 'Date' in bb.columns:
        bb = bb.copy()
        bb['Season'] = pd.to_datetime(bb['Date'], errors='coerce').dt.year

    # Season aggregates
    season_runs = tp.groupby('Season').agg({
        'First_Innings_Score': 'sum',
        'Second_Innings_Score': 'sum',
        'Match_ID': 'count'
    }).rename(columns={
        'First_Innings_Score': 'Runs_1st',
        'Second_Innings_Score': 'Runs_2nd',
        'Match_ID': 'Matches'
    }).reset_index()
    season_runs['Total_Runs'] = season_runs['Runs_1st'] + season_runs['Runs_2nd']

    # Wickets per season (from ball-by-ball)
    wickets_per_season = bb.groupby('Season')['Player Out'].count().reset_index(name='Wickets')

    # Wickets per season (from ball-by-ball)
    wickets_per_season = bb.groupby('Season')['Player Out'].count().reset_index(name='Wickets')

    # 🔑 Clean and force dtype alignment
    def clean_season(val):
        if pd.isna(val):
            return None
        if isinstance(val, str) and "/" in val:
            return int(val.split("/")[0])  # keep first year from "2007/08"
        return int(val)

    season_runs['Season'] = season_runs['Season'].apply(clean_season).astype('Int64')
    wickets_per_season['Season'] = wickets_per_season['Season'].apply(clean_season).astype('Int64')

    # Merge safely
    season_runs = season_runs.merge(wickets_per_season, on='Season', how='left')

    # Top performers per season
    batsman_runs = bb.groupby(['Season','Striker'])['runs_scored'].sum().reset_index()
    top_batsman = batsman_runs.sort_values(['Season','runs_scored'], ascending=[True,False]).groupby('Season').head(5)
    top_batsman = top_batsman.rename(columns={'Striker': 'Batsman', 'runs_scored': 'Runs'})

    wicket_rows = bb[bb['Player Out'].notna()]
    bowler_wk = wicket_rows.groupby(['Season','Bowler'])['Player Out'].count().reset_index().rename(columns={'Player Out':'Wickets'})
    top_bowlers = bowler_wk.sort_values(['Season','Wickets'], ascending=[True,False]).groupby('Season').head(5)

    # Venue impact
    venue_avg = tp.groupby('Venue').agg({
        'First_Innings_Score': 'mean',
        'Match_ID':'count'
    }).rename(columns={'First_Innings_Score':'Avg_First_Innings','Match_ID':'Matches'}).reset_index()

    return season_runs, top_batsman, top_bowlers, venue_avg

season_runs, top_batsman, top_bowlers, venue_avg = compute_aggregates(bb, tp, players_df)


st.sidebar.title('IPL Analytics')
page = st.sidebar.radio('Select Dashboard / Tool', [
    'Home',
    'Season Insights',
    'Venue Impact',
    'Head-to-Head',
    'Orange & Purple Cap Tracker',
    'Player Career Tracker',
    'Match Outcome Predictor',
    'Score Predictor',
    'Best Playing XI Recommender',
    'Fantasy Team Recommender'
])

# --- Home Page ---
if page == 'Home':
    st.title('🏏 IPL Analytics Dashboard')
    st.markdown('A modern interactive dashboard covering season insights, stadium impacts, head-to-heads, player career tracking, modeling, and team recommenders.')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Seasons covered', int(bb['Season'].nunique()))
    with col2:
        st.metric('Matches (ball-by-ball rows)', f"{bb['Match id'].nunique()} matches")
    with col3:
        st.metric('Players tracked', players_df['Player Name'].nunique())

    st.markdown('---')
    st.markdown('Use the sidebar to explore different tools and insights.')
elif page == 'Season Insights':
    st.header('Season Insights')
    st.markdown('Runs, wickets, win ratios and top performers per season')

    st.plotly_chart(
        px.bar(
            season_runs,
            x='Season',
            y='Total_Runs',
            title='Total Runs per Season',
            template='plotly_dark',
            color='Total_Runs',
            color_continuous_scale='sunsetdark'
        )
    )
    st.plotly_chart(
        px.bar(season_runs, x='Season', y='Wickets',
               title='Total Wickets per Season',
               template='plotly_dark',
               color='Wickets',
               color_continuous_scale='plasma')
    )

    season_choice = st.selectbox('Choose season', sorted(bb['Season'].dropna().unique()))
    st.subheader(f'Top performers in {season_choice}')
    # Top batsmen chart
    tb = top_batsman[top_batsman['Season'] == season_choice]
    if not tb.empty:
        st.plotly_chart(
            px.bar(
                tb,
                x='Batsman',
                y='Runs',
                title=f'Top Batsmen in {season_choice}',
                template='plotly_dark',
                text='Runs',
                color='Runs',
                color_continuous_scale='sunsetdark'
            )
        )

    # Top bowlers chart
    tbo = top_bowlers[top_bowlers['Season'] == season_choice]
    if not tbo.empty:
        st.plotly_chart(
            px.bar(
                tbo,
                x='Bowler',
                y='Wickets',
                title=f'Top Bowlers in {season_choice}',
                template='plotly_dark',
                text='Wickets',
                color='Wickets',
                color_continuous_scale='plasma'
            )
        )
# --- Venue Impact ---
if page == 'Venue Impact':
    st.header('Venue Impact & Toss Analysis')
    if venue_avg.empty:
        st.info('Venue-level aggregates not available in team performance file.')
    else:
        st.plotly_chart(px.bar(venue_avg.head(25), x='Venue', y='Avg_First_Innings', title='Top venues by avg first innings score', template='plotly_dark'))

    st.markdown('### Toss impact')
    if 'Toss_Winner' in tp.columns and 'Match_Winner' in tp.columns:
        toss = tp.copy()
        toss['Toss_Won_and_Winner'] = toss['Toss_Winner'] == toss['Match_Winner']

        toss_rate = toss.groupby('Toss_Winner')['Toss_Won_and_Winner'].mean().reset_index()
        toss_rate.rename(columns={'Toss_Won_and_Winner': 'WinRate'}, inplace=True)

        # Create bar chart
        colors = px.colors.qualitative.Plotly  # use same color palette as other charts
        fig_toss = px.bar(
            toss_rate.sort_values('WinRate', ascending=False),
            x='Toss_Winner',
            y='WinRate',
            color='Toss_Winner',
            color_discrete_sequence=colors,
            title='Toss Impact on Winning',
            template='plotly_dark',
            text=toss_rate['WinRate'].apply(lambda x: f"{x:.2%}")  # show win % on bars
        )

        fig_toss.update_layout(yaxis_tickformat='%')  # format y-axis as %

        st.plotly_chart(fig_toss)

    else:
        st.info('Toss and result columns not present in team dataset.')

# --- Head-to-Head ---
if page == 'Head-to-Head':
    st.header('Head-to-Head Analysis')

    # --- Step 1: Detect the correct team column in teams_df ---
    team_col = None
    for col in teams_df.columns:
        if 'team' in col.lower():  # case-insensitive match
            team_col = col
            break

    if team_col is None:
        st.error("No team column found in ipl_teams_2024_info.csv")
    else:
        # --- Team selection ---
        teams = sorted(teams_df[team_col].dropna().unique())
        t1 = st.selectbox('Team 1', teams, index=0)
        t2 = st.selectbox('Team 2', teams, index=1)

        # --- Step 2: Filter matches where both selected teams played ---
        def match_between_teams(row, team_a, team_b):
            teams_col = row.get('Teams')
            if pd.isna(teams_col):
                return False
            teams_list = [x.strip() for x in str(teams_col).split(' vs ')]
            return team_a in teams_list and team_b in teams_list

        h2h = tp[tp.apply(lambda r: match_between_teams(r, t1, t2), axis=1)]

        st.markdown(f"### {t1} vs {t2} — Across seasons")

        if h2h.empty:
            st.info("No matches found between these teams in team performance dataset.")
        else:
            # --- Wins summary ---
            if 'Match_Winner' in h2h.columns:
                wins = h2h.groupby('Match_Winner').size().reset_index(name='Wins')
                st.subheader('Head-to-Head Wins')
                st.table(wins)
                st.plotly_chart(
                    px.pie(
                        wins,
                        names='Match_Winner',
                        values='Wins',
                        title=f'{t1} vs {t2} Wins',
                        template='plotly_dark'
                    )
                )

            # --- Total runs summary ---
            if 'First_Innings_Score' in h2h.columns and 'Second_Innings_Score' in h2h.columns:
                h2h['Total_Runs'] = h2h['First_Innings_Score'] + h2h['Second_Innings_Score']
                runs_summary = h2h.groupby('Teams')['Total_Runs'].sum().reset_index()
                st.subheader('Total Runs Scored Across Matches')
                st.dataframe(runs_summary)

            # --- Top wicket-takers in these matches (optional) ---
            if 'Player Out' in bb.columns and 'Match id' in bb.columns and 'Match_ID' in h2h.columns:
                match_ids = h2h['Match_ID'].unique()
                wickets_h2h = bb[bb['Match id'].isin(match_ids) & bb['Player Out'].notna()]
                wickets_summary = (
                    wickets_h2h.groupby('Bowler')['Player Out']
                    .count()
                    .reset_index(name='Wickets')
                    .sort_values('Wickets', ascending=False)
                )
                st.subheader('Top Wicket-Takers in These Matches')
                st.dataframe(wickets_summary.head(10))

# --- Orange & Purple Cap Tracker ---
if page == 'Orange & Purple Cap Tracker':
    st.header('Orange & Purple Cap Tracker')
    st.markdown('Top run-scorers and wicket-takers overall and by season')

    # --- Top overall ---
    batsman_runs_all = (
        bb.groupby('Striker')['runs_scored'].sum()
        .reset_index()
        .rename(columns={'Striker':'Batsman','runs_scored':'Runs'})
        .sort_values('Runs', ascending=False)
    )
    bowler_wk_all = (
        bb[bb['Player Out'].notna()]
        .groupby('Bowler')['Player Out']
        .count()
        .reset_index()
        .rename(columns={'Player Out':'Wickets'})
        .sort_values('Wickets', ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Top Run Scorers (All-Time)')
        st.plotly_chart(
            px.bar(
                batsman_runs_all.head(10),
                x='Batsman',
                y='Runs',
                text='Runs',
                title='Top 10 Run Scorers (All-Time)',
                template='plotly_dark',
                color='Runs',
                color_continuous_scale='sunsetdark'
            )
        )
    with col2:
        st.subheader('Top Wicket Takers (All-Time)')
        st.plotly_chart(
            px.bar(
                bowler_wk_all.head(10),
                x='Bowler',
                y='Wickets',
                text='Wickets',
                title='Top 10 Wicket Takers (All-Time)',
                template='plotly_dark',
                color='Wickets',
                color_continuous_scale='plasma'
            )
        )

    # --- Season-wise top performers ---
    season_choice = st.selectbox('Season for Top Players', sorted(bb['Season'].dropna().unique()), key='opc')

    # Batsmen
    br = (
        bb[bb['Season']==season_choice]
        .groupby('Striker')['runs_scored'].sum()
        .reset_index()
        .rename(columns={'Striker':'Batsman','runs_scored':'Runs'})
        .sort_values('Runs', ascending=False)
    )

    # Bowlers
    bw = bb[bb['Season']==season_choice]
    if 'Bowler' in bw.columns:
        bw = (
            bw[bw['Player Out'].notna()]
            .groupby('Bowler')['Player Out']
            .count()
            .reset_index()
            .rename(columns={'Player Out':'Wickets'})
            .sort_values('Wickets', ascending=False)
        )

    st.subheader(f'Top Batsmen in Season {season_choice}')
    st.plotly_chart(
        px.bar(
            br.head(10),
            x='Batsman',
            y='Runs',
            text='Runs',
            title=f'Top 10 Batsmen in {season_choice}',
            template='plotly_dark',
            color='Runs',
            color_continuous_scale='sunsetdark'
        )
    )

    st.subheader(f'Top Bowlers in Season {season_choice}')
    st.plotly_chart(
        px.bar(
            bw.head(10),
            x='Bowler',
            y='Wickets',
            text='Wickets',
            title=f'Top 10 Bowlers in {season_choice}',
            template='plotly_dark',
            color='Wickets',
            color_continuous_scale='plasma'
        )
    )

# --- Player Career Tracker ---
if page == 'Player Career Tracker':
    st.header('Player Career Tracker')

    # Select player
    player = st.selectbox('Choose player', sorted(players_df['Player Name'].dropna().unique()))

    # Get player role info
    role_info = players_df[players_df['Player Name']==player][['Batting Style','Bowling Style']].iloc[0]
    batting_style = role_info['Batting Style'] if pd.notna(role_info['Batting Style']) else "N/A"
    bowling_style = role_info['Bowling Style'] if pd.notna(role_info['Bowling Style']) else "N/A"

    st.markdown(f"**Player:** {player}")
    st.markdown(f"**Batting Style:** {batting_style}")
    st.markdown(f"**Bowling Style:** {bowling_style}")

    # Filter ball-by-ball data
    pbb = bb[(bb['Striker']==player) | (bb.get('Bowler',None)==player) | (bb.get('Player Out',None)==player)]

    if pbb.empty:
        st.info('No ball-by-ball data for this player.')
    else:
        # --- Batting stats ---
        if 'Striker' in pbb.columns:
            batting = pbb[pbb['Striker']==player]
            runs_by_season = batting.groupby('Season')['runs_scored'].sum().reset_index()
            st.subheader('Season-wise Runs')
            st.plotly_chart(
                px.bar(
                    runs_by_season,
                    x='Season',
                    y='runs_scored',
                    text='runs_scored',
                    title=f'{player} - Runs by Season',
                    template='plotly_dark',
                    color='runs_scored',
                    color_continuous_scale='sunsetdark'
                )
            )

        # --- Bowling stats ---
        if 'Bowler' in pbb.columns and 'Player Out' in pbb.columns:
            bowling = pbb[(pbb['Bowler']==player) & (pbb['Player Out'].notna())]
            wickets_by_season = bowling.groupby('Season')['Player Out'].count().reset_index().rename(columns={'Player Out':'Wickets'})
            st.subheader('Season-wise Wickets')
            st.plotly_chart(
                px.bar(
                    wickets_by_season,
                    x='Season',
                    y='Wickets',
                    text='Wickets',
                    title=f'{player} - Wickets by Season',
                    template='plotly_dark',
                    color='Wickets',
                    color_continuous_scale='plasma'
                )
            )

# --- Match Outcome Predictor ---
if page == 'Match Outcome Predictor':
    st.header('Match Outcome Predictor')
    st.markdown('Predict match winner using toss, venue, teams, and recent stats')

    # Prepare features from team performance dataset
    required_cols = ['Match_ID','Date','Teams','Venue','Toss_Winner','Match_Winner']
    if all(c in tp.columns for c in required_cols):
        dfm = tp[required_cols].dropna()
        # Extract team names
        dfm[['Team1','Team2']] = dfm['Teams'].str.split(' vs ', expand=True)

        # Encode categorical
        le = LabelEncoder()
        dfm['Venue_enc'] = le.fit_transform(dfm['Venue'])

        # Fit LabelEncoder on all possible labels including Draw/No Result
        all_teams = pd.concat([dfm['Team1'], dfm['Team2'], dfm['Toss_Winner'], dfm['Match_Winner']]).unique()
        team_le = LabelEncoder()
        team_le.fit(all_teams)

        dfm['Team1_enc'] = team_le.transform(dfm['Team1'])
        dfm['Team2_enc'] = team_le.transform(dfm['Team2'])
        dfm['Toss_enc'] = team_le.transform(dfm['Toss_Winner'])
        dfm['Winner_enc'] = team_le.transform(dfm['Match_Winner'])

        # Features: venue, team1, team2, toss
        X = dfm[['Venue_enc','Team1_enc','Team2_enc','Toss_enc']]
        y = dfm['Winner_enc']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        clf_path = os.path.join(MODEL_DIR, 'match_outcome_rf.pkl')
        if os.path.exists(clf_path):
            clf = joblib.load(clf_path)
        else:
            clf = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(X_train, y_train)
            joblib.dump(clf, clf_path)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        # st.metric('Model accuracy (test)', f"{acc:.2f}")

        # Prediction UI
        st.subheader('Predict a single match')
        venue = st.selectbox('Venue', sorted(dfm['Venue'].unique()))
        t1 = st.selectbox('Team 1', sorted([t for t in team_le.classes_ if t != 'Draw/No Result']))
        t2 = st.selectbox('Team 2', sorted([t for t in team_le.classes_ if t not in [t1,'Draw/No Result']]))
        toss = st.selectbox('Toss winner', [t1,t2])
        if st.button('Predict winner'):
            feat = pd.DataFrame([{
                'Venue_enc': le.transform([venue])[0],
                'Team1_enc': team_le.transform([t1])[0],
                'Team2_enc': team_le.transform([t2])[0],
                'Toss_enc': team_le.transform([toss])[0]
            }])
            pred_enc = clf.predict(feat)[0]
            pred_team = team_le.inverse_transform([pred_enc])[0]
            st.success(f'Predicted winner: {pred_team}')
    else:
        st.warning('Required columns for outcome prediction not available in team performance dataset.')

# --- Score Predictor (first innings) ---
if page == 'Score Predictor':
    st.header('Score Predictor (First Innings)')
    st.markdown('Forecast 1st innings score range for a match using venue and teams')
    if 'First_Innings_Score' in tp.columns and 'Venue' in tp.columns and 'Teams' in tp.columns:
        dfsc = tp[['First_Innings_Score','Venue','Teams']].dropna()
        # Split Teams
        dfsc[['Team1','Team2']] = dfsc['Teams'].str.split(' vs ', expand=True)
        # Encode
        le_v = LabelEncoder(); dfsc['Venue_enc']=le_v.fit_transform(dfsc['Venue'])
        le_t = LabelEncoder(); dfsc['Team1_enc']=le_t.fit_transform(dfsc['Team1']); dfsc['Team2_enc']=le_t.transform(dfsc['Team2'])
        X = dfsc[['Venue_enc','Team1_enc','Team2_enc']]
        y = dfsc['First_Innings_Score']
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=42)
        reg_path = os.path.join(MODEL_DIR, 'score_rf.pkl')
        if os.path.exists(reg_path):
            reg = joblib.load(reg_path)
        else:
            reg = RandomForestRegressor(n_estimators=200, random_state=42)
            reg.fit(X_train, y_train)
            joblib.dump(reg, reg_path)
        ypred = reg.predict(X_test)
        mae = mean_absolute_error(y_test, ypred)
        # st.metric('MAE on test set', f"{mae:.1f} runs")

        # UI for prediction
        v = st.selectbox('Venue', sorted(dfsc['Venue'].unique()))
        t1 = st.selectbox('Batting Team', sorted(le_t.classes_))
        t2 = st.selectbox('Bowling Team', [t for t in sorted(le_t.classes_) if t!=t1])
        if st.button('Predict 1st innings score'):
            feat = pd.DataFrame([{
                'Venue_enc': le_v.transform([v])[0],
                'Team1_enc': le_t.transform([t1])[0],
                'Team2_enc': le_t.transform([t2])[0]
            }])
            pred = reg.predict(feat)[0]
            st.success(f'Predicted 1st innings score: {pred:.0f} runs (±{mae:.0f})')
    else:
        st.warning('First innings score or venue columns not present in team dataset.')

# --- Best Playing XI Recommender ---
if page == 'Best Playing XI Recommender':
    st.header('Best Playing XI Recommender')
    st.markdown('Automatically suggest optimal team combination using past data. Roles are respected and all-rounders are prioritized.')

    # Detect correct team column in players_df
    team_column = 'Team Name' if 'Team Name' in players_df.columns else players_df.columns[0]

    # Select team
    team_choice = st.selectbox('Select team', sorted(players_df[team_column].dropna().unique()))

    # Calculate player stats from ball-by-ball data
    bats = bb.groupby('Striker')['runs_scored'].sum().reset_index().rename(columns={'Striker': 'Player', 'runs_scored': 'Runs'})
    wk = bb[bb['Player Out'].notna()].groupby('Bowler')['Player Out'].count().reset_index().rename(columns={'Bowler': 'Player', 'Player Out': 'Wickets'})

    # Merge stats with player info
    players = players_df.copy()
    players = pd.merge(players, bats, left_on='Player Name', right_on='Player', how='left')
    players = pd.merge(players, wk, left_on='Player Name', right_on='Player', how='left')
    players = players.drop(columns=['Player_x', 'Player_y'], errors='ignore')
    players['Runs'] = players['Runs'].fillna(0)
    players['Wickets'] = players['Wickets'].fillna(0)
    players['AllRoundScore'] = players['Runs'] / 1000 + players['Wickets'] / 20

    # Filter players for selected team
    team_players = players[players[team_column] == team_choice].copy()

    # Initialize Recommended XI
    recommended_xi = []

    # 1️⃣ Top 5 batsmen (exclude keepers)
    batsmen = team_players[
        team_players['Batting Style'].notna() &
        ~team_players['Batting Style'].str.contains('Wicketkeeper|Keeper', na=False)
    ].sort_values('Runs', ascending=False)
    recommended_xi += batsmen.head(5)['Player Name'].tolist()

    # 2️⃣ 1 Wicketkeeper
    keeper = team_players[team_players['Batting Style'].str.contains('Wicketkeeper|Keeper', na=False)]
    if not keeper.empty:
        recommended_xi.append(keeper.iloc[0]['Player Name'])

    # 3️⃣ Top 3 bowlers
    bowlers = team_players[team_players['Bowling Style'].notna()].sort_values('Wickets', ascending=False)
    recommended_xi += [p for p in bowlers.head(3)['Player Name'].tolist() if p not in recommended_xi]

    # 4️⃣ Top 2 all-rounders
    allrounders = team_players.sort_values('AllRoundScore', ascending=False)
    recommended_xi += [p for p in allrounders.head(2)['Player Name'].tolist() if p not in recommended_xi]

    # Ensure unique and exactly 11 players
    recommended_xi = list(dict.fromkeys(recommended_xi))
    if len(recommended_xi) < 11:
        for p in team_players['Player Name'].tolist():
            if p not in recommended_xi:
                recommended_xi.append(p)
            if len(recommended_xi) == 11:
                break
    else:
        recommended_xi = recommended_xi[:11]

    # Display Recommended XI in Fantasy-style UI
    st.subheader('🎯 Recommended Playing XI')
    num_cols = 3
    for i in range(0, len(recommended_xi), num_cols):
        cols = st.columns(num_cols)
        for j, player in enumerate(recommended_xi[i:i + num_cols]):
            cols[j].markdown(f"**🏏 {player}**")

    # Display full team table with numbering starting from 1
    columns_to_show = ['Player Name', 'Batting Style', 'Bowling Style', 'Player Salary', 'Runs', 'Wickets', 'AllRoundScore']
    display_df = team_players[columns_to_show].sort_values('AllRoundScore', ascending=False).reset_index(drop=True)
    display_df.insert(0,'No.', range(1, len(display_df) + 1))

    # Highlight selected XI
    def highlight_selected(row):
        return ['background-color: lightgreen; color: black' if row['Player Name'] in recommended_xi else '' for _ in row]

    st.subheader(f'All players of {team_choice}')
    st.dataframe(display_df.style.apply(highlight_selected, axis=1))

# --- Fantasy Team Recommender ---
if page == 'Fantasy Team Recommender':
    st.header('Fantasy Team Recommender')
    st.markdown('Select two teams and a budget to get an optimal fantasy XI. Highlighted players are recommended.')

    # Inputs
    budget = st.number_input('Budget (e.g., 100)', min_value=50, max_value=1000, value=100)
    team_options = sorted(players_df['Team Name'].dropna().unique())
    team1 = st.selectbox('Select Team 1', options=['Any'] + team_options, key='team1')
    team2 = st.selectbox('Select Team 2', options=['Any'] + team_options, key='team2')

    # Copy players data
    dfp = players_df.copy()

    # Parse salary
    def parse_salary(x):
        try:
            s = str(x).replace(',','').replace('$','').replace('INR','').strip()
            return float(''.join([c for c in s if c.isdigit() or c=='.']))
        except:
            return np.nan
    dfp['SalaryNum'] = dfp['Player Salary'].apply(parse_salary).fillna(5) if 'Player Salary' in dfp.columns else 5

    # Compute historical performance
    bats = bb.groupby('Striker')['runs_scored'].sum().reset_index().rename(columns={'Striker':'Player','runs_scored':'Runs'})
    wk = bb[bb['Player Out'].notna()].groupby('Bowler')['Player Out'].count().reset_index().rename(columns={'Bowler':'Player','Player Out':'Wickets'})
    dfp = dfp.merge(bats, left_on='Player Name', right_on='Player', how='left')
    dfp = dfp.merge(wk, left_on='Player Name', right_on='Player', how='left')
    dfp['Runs'] = dfp['Runs'].fillna(0)
    dfp['Wickets'] = dfp['Wickets'].fillna(0)
    dfp['ValueScore'] = (dfp['Runs']/500 + dfp['Wickets']/15) / (dfp['SalaryNum']/10)

    teams_selected = []
    if team1 != 'Any':
        teams_selected.append(team1)
    if team2 != 'Any':
        teams_selected.append(team2)

    if teams_selected:
        dfp = dfp[dfp['Team Name'].isin(teams_selected)]

    # Sort by value
    dfp = dfp.sort_values('ValueScore', ascending=False)

    # Prepare candidate pools
    keeper_candidates = dfp[dfp['Batting Style'].str.contains('Wicketkeeper|Keeper', na=False)]
    batsmen_candidates = dfp[dfp['Batting Style'].notna()]
    bowlers_candidates = dfp[dfp['Bowling Style'].notna()]

    # Selection under budget
    selected = []
    total_cost = 0

    # Pick keeper
    if not keeper_candidates.empty:
        k = keeper_candidates.iloc[0]
        selected.append(k['Player Name'])
        total_cost += k['SalaryNum']

    # Pick top batsmen (total 5 including keeper)
    for _, r in batsmen_candidates.iterrows():
        if len([p for p in selected if p]) >= 5:
            break
        if r['Player Name'] in selected: continue
        if total_cost + r['SalaryNum'] > budget: continue
        selected.append(r['Player Name'])
        total_cost += r['SalaryNum']

    # Pick top bowlers (total 3)
    for _, r in bowlers_candidates.iterrows():
        if len([p for p in selected if p]) >= 8:
            break
        if r['Player Name'] in selected: continue
        if total_cost + r['SalaryNum'] > budget: continue
        selected.append(r['Player Name'])
        total_cost += r['SalaryNum']

    # Fill rest with best value (allrounders or remaining)
    for _, r in dfp.iterrows():
        if len(selected) >= 11: break
        if r['Player Name'] in selected: continue
        if total_cost + r['SalaryNum'] > budget: continue
        selected.append(r['Player Name'])
        total_cost += r['SalaryNum']

    # Display results
    st.subheader('Recommended Fantasy XI')

    # Display selected XI in a visually appealing way
    num_cols = 3  # number of columns per row
    for i in range(0, len(selected), num_cols):
        cols = st.columns(num_cols)
        for j, player in enumerate(selected[i:i + num_cols]):
            cols[j].markdown(f"**🎯 {player}**")


    st.metric('Estimated total cost', f"{total_cost:.1f} (budget {budget})")

# --- Footer / End ---
st.markdown('---')
st.markdown('Built with ❤️ using Streamlit, pandas, scikit-learn, plotly and matplotlib. Modify the file paths at the top if your csv filenames differ.')