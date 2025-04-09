import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@st.cache_data
def load_data():
    return pd.read_csv("final_cleaned.csv")  # Ensure this CSV exists

df = load_data()


features = [
    'minutes', 'goals_scored', 'assists', 'clean_sheets', 'saves', 
    'expected_goals', 'expected_assists', 'bonus', 'form', 
    'opponent_difficulty', 'home_difficulty', 'form_variability', 
    'position_encoded', 'team_encoded'
]
target = 'fantasy_points'


X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)


xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)
st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")


position_mapping = {0: "DEF", 1: "FWD", 2: "GK", 3: "MID"}
df["position"] = df["position_encoded"].map(position_mapping)

def select_best_fpl_team(df, selected_fixtures, budget_limit=1000):
    filtered_df = df[df["team_encoded"].isin(selected_fixtures)].copy()
    if filtered_df.empty:
        st.error("No players found for the selected teams. Try different teams!")
        return pd.DataFrame(), 0
    
    filtered_df["efficiency"] = filtered_df["fantasy_points"] / filtered_df["now_cost"]
    filtered_df = filtered_df.sort_values(by=["efficiency", "fantasy_points"], ascending=[False, False])
    
    squad_requirements = {2: 2, 0: 5, 3: 5, 1: 3}
    selected_team = []
    budget_used = 0
    
    for position, count in squad_requirements.items():
        position_players = filtered_df[filtered_df["position_encoded"] == position]
        selected_players = []
        for _, player in position_players.iterrows():
            if len(selected_players) < count and budget_used + player["now_cost"] <= budget_limit:
                selected_players.append(player)
                budget_used += player["now_cost"]
            if len(selected_players) == count:
                break
        selected_team.extend(selected_players)
    
    remaining_budget = budget_limit - budget_used
    remaining_slots = 15 - len(selected_team)
    available_players = filtered_df[~filtered_df["full_name"].isin([p["full_name"] for p in selected_team])]
    available_players = available_players.sort_values(by="efficiency", ascending=False)
    
    for _, player in available_players.iterrows():
        if len(selected_team) < 15 and budget_used + player["now_cost"] <= budget_limit:
            selected_team.append(player)
            budget_used += player["now_cost"]
    
    return pd.DataFrame(selected_team), budget_used

def plot_formation(team_df):
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("green")
    ax.set_title("FPL Team Formation", fontsize=14, fontweight='bold', color="white")
    
    # Draw goalposts and pitch lines
    ax.plot([1, 9, 9, 1, 1], [1, 1, 13, 13, 1], color="white", linewidth=2)
    ax.plot([4, 6, 6, 4, 4], [1, 1, 3, 3, 1], color="white", linewidth=2)
    ax.plot([4, 6, 6, 4, 4], [11, 11, 13, 13, 11], color="white", linewidth=2)
    
    positions = {"GK": [(5, 1.5), (5, 2.5)], "DEF": [(2, 4), (4, 4), (6, 4), (8, 4), (5, 5)],
                 "MID": [(1, 7), (3, 7), (5, 7), (7, 7), (9, 7)], "FWD": [(3, 10), (5, 10), (7, 10)]}
    
    pos_categories = {2: "GK", 0: "DEF", 3: "MID", 1: "FWD"}
    player_positions = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    
    for _, player in team_df.iterrows():
        pos = pos_categories.get(player["position_encoded"], "MID")
        if len(player_positions[pos]) < len(positions[pos]):
            player_positions[pos].append(player)
    
    for pos, coords in positions.items():
        for i, coord in enumerate(coords):
            if i < len(player_positions[pos]):
                player = player_positions[pos][i]
                rect = patches.Rectangle((coord[0] - 0.5, coord[1] - 0.3), 1, 0.6, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                ax.text(coord[0], coord[1], player["full_name"].split()[0], fontsize=10, ha="center", va="center", color="black", fontweight='bold')
    
    st.pyplot(fig)

st.title("⚽ FPL Team Selection Optimizer")
num_matches = st.number_input("Enter the number of matches:", min_value=1, max_value=38, value=10, step=1)
selected_fixtures = []
for i in range(1, num_matches + 1):
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.number_input(f"Match {i} Home Team (encoded):", min_value=0, max_value=19, step=1)
    with col2:
        away_team = st.number_input(f"Match {i} Away Team (encoded):", min_value=0, max_value=19, step=1)
    selected_fixtures.append(home_team)
    selected_fixtures.append(away_team)

if st.button("Generate Best Team"):
    best_team, budget_used = select_best_fpl_team(df, selected_fixtures)
    if not best_team.empty:
        st.success(f"✅ Total Players Selected: {len(best_team)} / 15")
        st.success(f"💰 Total Budget Used: {budget_used:.1f} / 1000 credits")
        st.dataframe(best_team[["full_name", "position_encoded", "team_encoded", "now_cost", "fantasy_points"]])
        plot_formation(best_team)
