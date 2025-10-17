
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from prob_agent import ProbabilityAgent, Fixture, load_ratings_from_json, load_league_params_from_json

app = FastAPI(title="ModelProbGen", version="1.1.0",
              description="Generate football model probabilities + form stability from fixture + date.")

RATINGS_PATH = "ratings.json"
LEAGUE_PARAMS_PATH = "league_params.json"
ratings = load_ratings_from_json(RATINGS_PATH)
league_params = load_league_params_from_json(LEAGUE_PARAMS_PATH)
agent = ProbabilityAgent(ratings, league_params)

class PredictResponse(BaseModel):
    ModelHomeProb: float
    ModelDrawProb: float
    ModelAwayProb: float
    ModelOverProb: float
    ModelUnderProb: float
    ModelYesProb: float
    ModelNoProb: float
    ModelHomeDNBProb: float
    ModelAwayDNBProb: float
    Model1XProb: float
    Model12Prob: float
    ModelX2Prob: float
    LambdaHome: float
    LambdaAway: float
    HomeFormStability: float
    AwayFormStability: float
    FormStability: float

@app.get("/predict", response_model=PredictResponse)
def predict(home: str, away: str, league: str, date: str,
            ou_line: float = Query(2.5, description="Over/Under line, e.g. 2.5"),
            max_goals: int = Query(10, ge=6, le=15)):
    try:
        fx = Fixture(home_team=home, away_team=away, league=league, date=date)
        probs = agent.predict(fx, ou_line=ou_line, max_goals=max_goals)
        return probs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
