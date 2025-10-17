
# ModelProbGen API

A tiny FastAPI microservice that returns **model probabilities** for 1X2, O/U, BTTS, DNB and Double Chance
from just **fixture + date** (and optional O/U line).

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Open: http://127.0.0.1:8000/docs

### Example
```
GET /predict?home=Arsenal&away=Chelsea&league=EPL&date=2025-10-19&ou_line=2.5
```

## Docker
```bash
docker build -t modelprobgen:1.0 .
docker run -p 8000:8000 modelprobgen:1.0
```

## Customize data
Replace `ratings.json` and `league_params.json` with your calibrated values,
or wire them to a DB. The engine uses a Poisson/Dixon–Coles core for coherence across markets.
