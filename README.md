\# Payment Acceptance API (Calibrated + CI)



Endpoints:

\- `GET /health`

\- `POST /predict` → JSON: {MidName, Bin, Application2, Amount}

\- `POST /predict\_batch` → CSV/Parquet con columnas: MidName,Bin,Application2,Amount



\### Local

```bash

docker build -t payment-api:latest .

docker run -p 8080:8080 payment-api:latest

curl http://localhost:8080/health



