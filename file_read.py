import pandas as pd

df_ball_by_ball = pd.read_csv("IPL_BallByBall2008_2024(Updated).csv")


df_teams_info = pd.read_csv("ipl_teams_2024_info.csv")


df_players_info = pd.read_csv("Players_Info_2024.csv")


df_team_performance = pd.read_csv("team_performance_dataset_2008to2024.csv")


print("Preview of df_ball_by_ball:")
print(df_ball_by_ball)
