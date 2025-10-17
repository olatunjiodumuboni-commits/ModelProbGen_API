
# ModelProbGen API v1.1.0

Returns model probabilities (1X2, O/U, BTTS, DNB, Double Chance) and form stability metrics:
- HomeFormStability, AwayFormStability (0–1 each)
- FormStability (harmonic mean)

## Run
pip install -r requirements.txt
uvicorn main:app --reload

## Docker
docker build -t modelprobgen:1.1 .
docker run -p 8000:8000 modelprobgen:1.1
