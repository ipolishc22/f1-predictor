import fastf1
import pandas as pd 

def _quali_features(year:int, round:int) -> pd.DataFrame:
    q = fastf1.get_session(year, round, 'Q')
    q.load()

    res = q.results.copy()
    res['q_best_s'] = res[['Q1', 'Q2', 'Q3']].min(axis=1).dt.total_seconds()
    pole = res['q_best_s'].min()
    res['gap_to_pole_s'] = res['q_best_s'] = pole
    res['made_q3'] = res['Q3'].notna()
    res_sorted = res.sort_values('q_best_s')
    res['gap_to_ahead_s'] = res_sorted['q_best_s'].diff().fillna(0.0)

    r = fastf1.get_session(year, round, 'R')
    r.load()

    grid = r.results[['GridPosition']]

    out = res[['Abbreviation', 'q_best_s', 'gap_to_pole_s', 'gap_to_ahead_s', 'made_q3']].join(grid)
    return out