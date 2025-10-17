
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
import json

def _poisson_pmf(lmbda: float, k: int) -> float:
    if lmbda <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)

def _score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 10, dc_rho: float = 0.0):
    mat = {}
    for i in range(max_goals + 1):
        p_i = _poisson_pmf(lambda_home, i)
        for j in range(max_goals + 1):
            p_j = _poisson_pmf(lambda_away, j)
            mat[(i, j)] = p_i * p_j
    if dc_rho != 0.0:
        keys = [(0,0), (1,0), (0,1), (1,1)]
        corr = {(0,0): 1 + dc_rho, (1,0): 1 - dc_rho, (0,1): 1 - dc_rho, (1,1): 1 + dc_rho}
        for k in keys:
            if k in mat:
                mat[k] *= corr[k]
        s = sum(mat.values())
        for k in mat:
            mat[k] /= s
    return mat

def _sum_outcomes(mat):
    p_home = sum(p for (i, j), p in mat.items() if i > j)
    p_draw = sum(p for (i, j), p in mat.items() if i == j)
    p_away = sum(p for (i, j), p in mat.items() if i < j)
    return p_home, p_draw, p_away

def _over_under(mat, line: float):
    p_over = sum(p for (i, j), p in mat.items() if (i + j) > line)
    p_under = 1.0 - p_over
    return p_over, p_under

def _btts(mat):
    p_gh0 = sum(p for (i, j), p in mat.items() if i == 0)
    p_ga0 = sum(p for (i, j), p in mat.items() if j == 0)
    p_00  = mat.get((0, 0), 0.0)
    p_yes = 1.0 - p_gh0 - p_ga0 + p_00
    p_no  = 1.0 - p_yes
    return p_yes, p_no

def _safe(p: float) -> float:
    return max(0.0, min(1.0, p))

@dataclass
class TeamRating:
    attack: float
    defense: float
    last_updated: Optional[str] = None

@dataclass
class LeagueParams:
    mu: float
    home_advantage: float
    dc_rho: float = 0.0

@dataclass
class Fixture:
    home_team: str
    away_team: str
    league: str
    date: str

class ProbabilityAgent:
    def __init__(self, ratings: Dict[str, TeamRating], league_params: Dict[str, LeagueParams]):
        self.ratings = ratings
        self.league_params = league_params

    def _lambdas(self, fx: Fixture):
        if fx.league not in self.league_params:
            raise ValueError(f"League '{fx.league}' not found.")
        lp = self.league_params[fx.league]
        if fx.home_team not in self.ratings:
            raise ValueError(f"Missing rating for home team '{fx.home_team}'.")
        if fx.away_team not in self.ratings:
            raise ValueError(f"Missing rating for away team '{fx.away_team}'.")
        h = self.ratings[fx.home_team]
        a = self.ratings[fx.away_team]
        lambda_home = lp.mu * math.exp(lp.home_advantage + h.attack - a.defense)
        lambda_away = lp.mu * math.exp(a.attack - h.defense)
        return lambda_home, lambda_away, lp

    def predict(self, fx: Fixture, ou_line: float = 2.5, max_goals: int = 10):
        lambda_home, lambda_away, lp = self._lambdas(fx)
        mat = _score_matrix(lambda_home, lambda_away, max_goals=max_goals, dc_rho=lp.dc_rho)
        pH, pD, pA = _sum_outcomes(mat)
        pOver, pUnder = _over_under(mat, ou_line)
        pYes, pNo = _btts(mat)
        if (1.0 - pD) > 0:
            pHomeDNB = pH / (1.0 - pD)
            pAwayDNB = pA / (1.0 - pD)
        else:
            pHomeDNB, pAwayDNB = 0.5, 0.5
        p1X = pH + pD
        p12 = pH + pA
        pX2 = pD + pA
        return {
            "ModelHomeProb": _safe(pH),
            "ModelDrawProb": _safe(pD),
            "ModelAwayProb": _safe(pA),
            "ModelOverProb": _safe(pOver),
            "ModelUnderProb": _safe(pUnder),
            "ModelYesProb": _safe(pYes),
            "ModelNoProb": _safe(pNo),
            "ModelHomeDNBProb": _safe(pHomeDNB),
            "ModelAwayDNBProb": _safe(pAwayDNB),
            "Model1XProb": _safe(p1X),
            "Model12Prob": _safe(p12),
            "ModelX2Prob": _safe(pX2),
            "LambdaHome": lambda_home,
            "LambdaAway": lambda_away,
        }

def load_ratings_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: TeamRating(**v) for k, v in raw.items()}

def load_league_params_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: LeagueParams(**v) for k, v in raw.items()}
