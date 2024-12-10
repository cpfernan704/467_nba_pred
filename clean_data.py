import pandas as pd
from datetime import datetime

def combine_and_clean_data():
    # compile files into single df
    dfs = []
    file_paths = [
        './NBA-Data-2010-2024-main/regular_season_box_scores_2010_2024_part_1.csv',
        './NBA-Data-2010-2024-main/regular_season_box_scores_2010_2024_part_2.csv',
        './NBA-Data-2010-2024-main/regular_season_box_scores_2010_2024_part_3.csv'
    ]

    for file_path in file_paths:
        df = pd.read_csv(file_path, low_memory=False)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print("Total rows before cleaning:", len(combined_df))

    # filter players
    combined_df['game_date'] = pd.to_datetime(combined_df['game_date'])
    player_first_games = combined_df.groupby('personId')['game_date'].min()
    non_rookie_players = player_first_games[player_first_games < '2022-10-01'].index
    recent_players = combined_df[(combined_df['game_date'] >= '2022-01-01') & (combined_df['personId'].isin(non_rookie_players))]
    
    player_set = set(recent_players['personId'].unique())
    cleaned_df = combined_df[combined_df['personId'].isin(player_set)].copy()
    cleaned_df = cleaned_df.sort_values(['personId', 'game_date']).reset_index(drop=True)

    # feature engineering
    cleaned_df['minutes_played'] = cleaned_df['minutes'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 if pd.notna(x) and ':' in str(x) else 0)
    cleaned_df['is_home'] = cleaned_df['matchup'].str.contains(' vs. ').astype(int)
    cleaned_df['opponent'] = cleaned_df['matchup'].apply(lambda x: x.split(' vs. ')[1] if ' vs. ' in x else x.split(' @ ')[1])

    cleaned_df['days_rest'] = cleaned_df.groupby('personId')['game_date'].diff().dt.days.fillna(3)
    cleaned_df['is_back_to_back'] = (cleaned_df['days_rest'] == 1).astype(int)
    
    cleaned_df['season_year'] = cleaned_df['game_date'].dt.year
    cleaned_df['season_month'] = cleaned_df['game_date'].dt.month
    cleaned_df['season_progress'] = ((cleaned_df['season_month'] - 10) % 12) / 11
    
    for stat in ['points', 'assists', 'reboundsTotal', 'fieldGoalsAttempted', 'fieldGoalsMade', 'threePointersMade', 'freeThrowsAttempted', 'turnovers', 'minutes_played']:
        cleaned_df[f'{stat}_avg_5g'] = (cleaned_df.groupby('personId')[stat].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
        cleaned_df[f'{stat}_avg_10g'] = (cleaned_df.groupby('personId')[stat].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()))
    
    cleaned_df['rebounds_plus_assists'] = cleaned_df['reboundsTotal'] + cleaned_df['assists']
    cleaned_df['points_rebounds_assists'] = cleaned_df['points'] + cleaned_df['reboundsTotal'] + cleaned_df['assists']
    cleaned_df['points_assists'] = cleaned_df['points'] + cleaned_df['assists']
    cleaned_df['points_rebounds'] = cleaned_df['points'] + cleaned_df['reboundsTotal']
    cleaned_df['blocks_steals'] = cleaned_df['blocks'] + cleaned_df['steals']
    
    cleaned_df = cleaned_df.fillna(0)
    print("Total rows after cleaning:", len(cleaned_df))
    
    target_vars = [
        'points', 'reboundsTotal', 'assists', 'threePointersMade', 
        'turnovers', 'blocks', 'steals', 'rebounds_plus_assists',
        'points_rebounds_assists', 'points_assists', 'points_rebounds',
        'blocks_steals'
    ]
    
    val_start_date = '2022-10-18'
    test_start_date = '2023-10-24'
    
    train_df = cleaned_df[cleaned_df['game_date'] < val_start_date].copy()
    val_df = cleaned_df[(cleaned_df['game_date'] >= val_start_date) & (cleaned_df['game_date'] < test_start_date)].copy()
    test_df = cleaned_df[cleaned_df['game_date'] >= test_start_date].copy()
    
    drop_cols = ['personId', 'teamId', 'gameId', 'game_date', 'matchup', 'position', 'opponent', 'minutes', 'personName', 'teamName', 'opposing_teamId']
    
    for stat in target_vars:
        drop_cols.append(stat)
        drop_cols.append(f'{stat}_pred')
    
    current_game_stats = [
        'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPercentage',
        'threePointersAttempted', 'threePointersPercentage',
        'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPercentage',
        'reboundsOffensive', 'reboundsDefensive', 'foulsPersonal',
        'plusMinusPoints'
    ]

    drop_cols.extend(current_game_stats)
    
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors='ignore')
    val_df = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns], errors='ignore')
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors='ignore')
    
    train_df.to_csv('./cleaned_data/train_data.csv', index=False)
    val_df.to_csv('./cleaned_data/val_data.csv', index=False)
    test_df.to_csv('./cleaned_data/test_data.csv', index=False)
    
    train_df.sample(n=50).to_csv('./cleaned_data/samples/sample_train_data.csv', index=False)
    val_df.sample(n=25).to_csv('./cleaned_data/samples/sample_val_data.csv', index=False)
    test_df.sample(n=25).to_csv('./cleaned_data/samples/sample_test_data.csv', index=False)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = combine_and_clean_data()
