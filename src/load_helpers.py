import fastf1
import pandas as pd 

def _quali_features(year:int, round:int) -> pd.DataFrame:
    q = fastf1.get_session(year, round, 'Q')
    q.load(laps=False, telemetry=False, weather=False, messages=False)

    res = q.results.copy()
    res['q_best_s'] = res[['Q1', 'Q2', 'Q3']].min(axis=1).dt.total_seconds()
    pole = res['q_best_s'].min()
    res['gap_to_pole_s'] = res['q_best_s'] - pole
    res['made_q3'] = res['Q3'].notna()
    res_sorted = res.sort_values('q_best_s')
    res['gap_to_ahead_s'] = res_sorted['q_best_s'].diff().fillna(0.0)

    r = fastf1.get_session(year, round, 'R')
    r.load(laps=False, telemetry=False, weather=False, messages=False)

    grid = r.results[['GridPosition']]

    out = res[['Abbreviation', 'q_best_s', 'gap_to_pole_s', 'gap_to_ahead_s', 'made_q3']].join(grid)
    return out


def _target_top_3(year:int, round:int) -> pd.DataFrame:
    r = fastf1.get_session(year, round, 'R')
    r.load(laps=False, telemetry=False, weather=False, messages=False)

    res_r = r.results.copy()
    res_r['finished_top_3'] = res_r['Position'] <= 3
    return res_r[['finished_top_3']]


def load_prerace_features(year:int, round:int, *, for_training: bool=True) -> pd.DataFrame:
    feats = _quali_features(year, round)
    if for_training:
        feats = feats.join(_target_top_3(year, round))
    else:
        feats['finished_top_3'] = pd.NA

    feats['year'] = year
    feats['round'] = round

    return feats.reset_index()


def build_dataset(races, *, for_training=True):
    frames = [load_prerace_features(yr, rd, for_training=for_training)
              for yr, rd in races]
    return pd.concat(frames, ignore_index=True)
    