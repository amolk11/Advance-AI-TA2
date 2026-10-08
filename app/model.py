"""
Bayesian Air Quality Risk & Mitigation System - Core Modeling & Inference Engine
Dataset: Pune Shivajinagar - IITM Station 11613
Model: Temporal Dynamic Bayesian Network with Dirichlet BDeu Parameter Smoothing
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

STATES = ['Low', 'Moderate', 'High']
SEASONS = ['Monsoon', 'Post-Monsoon', 'Winter', 'Summer']
POLLUTANTS = ['co', 'no2', 'o3']

STATE_COLORS = {
    'Low': '#10b981',       # Emerald Green
    'Moderate': '#f59e0b',  # Amber
    'High': '#ef4444'       # Red
}

def get_season(dt):
    """Categorize astronomical calendar season from datetime object."""
    m = dt.month
    if m in [6, 7, 8, 9]:
        return 'Monsoon'
    elif m in [10, 11]:
        return 'Post-Monsoon'
    elif m in [12, 1, 2]:
        return 'Winter'
    else:
        return 'Summer'

def get_aqi_state(val):
    """Discretize continuous AQI value into epidemiological risk states."""
    if pd.isna(val):
        return None
    if val <= 100:
        return 'Low'
    elif val <= 200:
        return 'Moderate'
    else:
        return 'High'

def find_data_file():
    """Locate the station CSV dataset across typical execution directories."""
    candidate_paths = [
        Path('data/station_1_Revenue_Colony-Shivajinagar_Pu.csv'),
        Path('../data/station_1_Revenue_Colony-Shivajinagar_Pu.csv'),
        Path(__file__).parent.parent / 'data' / 'station_1_Revenue_Colony-Shivajinagar_Pu.csv',
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find station_1_Revenue_Colony-Shivajinagar_Pu.csv")

def load_and_preprocess_data():
    """Load raw dataset and build leak-free consecutive 1-day Markov transition pairs."""
    data_path = find_data_file()
    raw_df = pd.read_csv(data_path)
    
    # Preprocess dates
    df = raw_df.dropna(subset=['aqi']).copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 2025+ clean continuous series
    df_2025 = df[df['date'] >= '2025-01-01'].sort_values('date').reset_index(drop=True)
    
    df_2025['season'] = df_2025['date'].apply(get_season)
    df_2025['aqi_state'] = df_2025['aqi'].apply(get_aqi_state)
    
    # Lag features for t-1 -> t
    df_2025['date_diff'] = df_2025['date'].diff().dt.days
    df_2025['previous_aqi'] = df_2025['aqi'].shift(1)
    df_2025['previous_aqi_state'] = df_2025['aqi_state'].shift(1)
    df_2025['previous_dominant_pollutant'] = df_2025['dominant_pollutant'].shift(1)
    df_2025['target_aqi_state'] = df_2025['aqi_state']
    
    # Strictly consecutive 1-day pairs
    modeling_df = df_2025[df_2025['date_diff'] == 1].dropna(
        subset=['previous_aqi_state', 'previous_dominant_pollutant']
    ).copy().reset_index(drop=True)
    
    modeling_df['previous_aqi_state'] = pd.Categorical(modeling_df['previous_aqi_state'], categories=STATES)
    modeling_df['target_aqi_state'] = pd.Categorical(modeling_df['target_aqi_state'], categories=STATES)
    modeling_df['season'] = pd.Categorical(modeling_df['season'], categories=SEASONS)
    modeling_df['previous_dominant_pollutant'] = pd.Categorical(modeling_df['previous_dominant_pollutant'], categories=POLLUTANTS)
    
    return raw_df, df_2025, modeling_df

def build_and_train_bayesian_network(modeling_df, train_split=0.8, equivalent_sample_size=10):
    """
    Train 1st-order Markov Temporal Bayesian Network with Dirichlet BDeu smoothing.
    """
    split_idx = int(len(modeling_df) * train_split)
    train_df = modeling_df.iloc[:split_idx].copy()
    test_df = modeling_df.iloc[split_idx:].copy()
    
    edges = [
        ('previous_aqi_state', 'target_aqi_state'),
        ('season', 'target_aqi_state'),
        ('previous_dominant_pollutant', 'target_aqi_state')
    ]
    model = DiscreteBayesianNetwork(edges)
    
    features = ['previous_aqi_state', 'season', 'previous_dominant_pollutant', 'target_aqi_state']
    state_dict = {
        'previous_aqi_state': STATES,
        'target_aqi_state': STATES,
        'season': SEASONS,
        'previous_dominant_pollutant': POLLUTANTS
    }
    
    estimator = BayesianEstimator(model, train_df[features], state_names=state_dict)
    cpds = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=equivalent_sample_size)
    model.add_cpds(*cpds)
    
    infer = VariableElimination(model)
    return model, infer, train_df, test_df

def query_posterior_risk(infer_engine, previous_state, season, dominant_pollutant='co'):
    """
    Compute exact posterior distribution: P(AQI_t | AQI_{t-1}, Pollutant_{t-1}, Season_t).
    """
    evidence = {
        'previous_aqi_state': previous_state,
        'season': season,
        'previous_dominant_pollutant': dominant_pollutant.lower()
    }
    q = infer_engine.query(variables=['target_aqi_state'], evidence=evidence, show_progress=False)
    p_dict = dict(zip(q.state_names['target_aqi_state'], q.values))
    return {s: float(p_dict.get(s, 0.0)) for s in STATES}

def simulate_markov_trajectory(infer_engine, initial_state, season, dominant_pollutant='co', steps=7):
    """
    Simulate forward probability evolution over multi-day horizon without teacher forcing.
    """
    current_dist = np.array([1.0 if s == initial_state else 0.0 for s in STATES])
    trajectory = [current_dist]
    
    for _ in range(steps):
        next_dist = np.zeros(3)
        for i, prev_s in enumerate(STATES):
            p_dict = query_posterior_risk(infer_engine, prev_s, season, dominant_pollutant=dominant_pollutant)
            p_vec = np.array([p_dict[s] for s in STATES])
            next_dist += current_dist[i] * p_vec
        trajectory.append(next_dist)
        current_dist = next_dist
        
    traj_df = pd.DataFrame(trajectory, columns=STATES)
    traj_df['Day'] = [f"t+{i}" if i > 0 else "t-1 (Observed)" for i in range(steps + 1)]
    return traj_df

def get_decision_advisory(prob_dict):
    """
    Determine municipal risk tier and actionable health/curb guidelines based on Bayesian posterior.
    """
    p_low = prob_dict['Low']
    p_mod = prob_dict['Moderate']
    p_high = prob_dict['High']
    
    if p_high >= 0.50:
        return {
            "tier": "Severe Alert",
            "code": "TIER-4",
            "color": "#ef4444",
            "badge_class": "advisory-severe",
            "summary": "High probability of hazardous pollutant accumulation exceeding ambient safety limits.",
            "health_advice": "Vulnerable groups must avoid outdoor exertion. General population should minimize exposure and use N95 respirators.",
            "municipal_action": "Issue formal public health alert. Trigger mandatory industrial emission caps and emergency traffic rationing."
        }
    elif p_high >= 0.20 or p_mod >= 0.70:
        return {
            "tier": "High Vigilance",
            "code": "TIER-3",
            "color": "#ea580c",
            "badge_class": "advisory-high",
            "summary": "Strong atmospheric persistence indicating elevated probability of prolonged moderate-to-high AQI states.",
            "health_advice": "Sensitive demographics (respiratory/cardiac conditions, elderly, children) should reduce prolonged outdoor exertion.",
            "municipal_action": "Deploy mechanized sweeping on high-density corridors; activate traffic diversion protocols to reduce congestion."
        }
    elif p_mod >= 0.30:
        return {
            "tier": "Elevated Advisory",
            "code": "TIER-2",
            "color": "#d97706",
            "badge_class": "advisory-elevated",
            "summary": "Moderate risk of air quality deterioration. Conditions warrant operational readiness.",
            "health_advice": "Sensitive individuals should monitor symptoms during morning and evening rush hours.",
            "municipal_action": "Increase monitoring cadence across ward stations; prepare water misting trucks for major transit junctions."
        }
    else:
        return {
            "tier": "Routine / Low Risk",
            "code": "TIER-1",
            "color": "#059669",
            "badge_class": "advisory-routine",
            "summary": "High statistical confidence in favorable dispersion and low pollutant concentration.",
            "health_advice": "Air quality is within acceptable limits. Standard outdoor activities are unrestricted.",
            "municipal_action": "Maintain baseline environmental telemetry. No regulatory interventions required."
        }
