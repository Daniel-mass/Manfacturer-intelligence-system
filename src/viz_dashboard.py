#src/viz_dashboard.py
import solara
import joblib
import polars as pl
import numpy as np
from mesa.visualization import SolaraViz, make_plot_component
from mesa.visualization.components import make_altair_space 
from src.simulation import FactoryFleet, NASA_MODEL_PATH, NASA_FEATURES_PATH, EXCLUDE_COLS

# 1. Pre-load the heavy data once
lgbm_model = joblib.load(NASA_MODEL_PATH)
df = pl.read_parquet(NASA_FEATURES_PATH).to_pandas()
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
eligible = df[df["rul"].between(60, 125)]

# 2. Define how to draw the Engines (The Portrayal)
def engine_portrayal(agent):
    # Managed = Green Circle, Unmanaged = Red Square
    color = "tab:green" if agent.is_managed else "tab:red"
    marker = "circle" if agent.is_managed else "square"
    alpha = 1.0 if agent.is_active else 0.2
    
    return {
        "color": color,
        "size": 50,
        "marker": marker,
        "alpha": alpha
    }

# 3. Setup Model Parameters
# Note: sample_rows and start_ruls MUST be here to satisfy Mesa 3.0 signature checks
model_params = {
    "n_agents": solara.reactive(50), 
    "lgbm_model": lgbm_model,
    "feature_cols": feature_cols,
    "sample_rows": eligible.sample(100).reset_index(drop=True),
    "start_ruls": np.random.choice(eligible["rul"].values, size=100)
}

# 4. Define UI Components
# make_altair_space requires these None arguments in some Mesa 3.0 versions
SpaceComponent = make_altair_space(
    agent_portrayal=engine_portrayal,
    propertylayer_portrayal=None,
    post_process=None
)

# Line chart for survival percentage
# Line chart for survival percentage (Final project version)
SurvivalPlot = make_plot_component([
    "survival_pct_managed", 
    "survival_pct_unmanaged"
])

# 5. Helper function to create the initial model instance
def create_initial_model(params):
    n = int(params["n_agents"].value)
    # Re-sample for the initial run
    s_rows = eligible.sample(n * 2).reset_index(drop=True)
    s_ruls = np.random.choice(eligible["rul"].values, size=n * 2)
    
    return FactoryFleet(
        n_agents=n,
        lgbm_model=lgbm_model,
        feature_cols=feature_cols,
        sample_rows=s_rows,
        start_ruls=s_ruls
    )

# Create the first instance
initial_model = create_initial_model(model_params)

# 6. Launch the Viz
# Passing the instance (model) and the blueprint (model_params)
Page = SolaraViz(
    model=initial_model,
    model_params=model_params,
    components=[SpaceComponent, SurvivalPlot],
    name="NASA Engine Digital Twin - Fleet Management"
)