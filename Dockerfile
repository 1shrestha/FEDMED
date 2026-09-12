# Deploys the central FL aggregation server + live metrics WebSocket.
# Build & run from the REPO ROOT (needs access to model/, data/, privacy/, federated/):
#   docker build -f backend/Dockerfile -t fedmed-backend .
#   docker run -p 8080:8080 -p 8765:8765 fedmed-backend
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ model/
COPY data/ data/
COPY privacy/ privacy/
COPY federated/ federated/
COPY backend/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

# 8080 = Flower gRPC (hospital nodes connect here)
# 8765 = WebSocket metrics stream (dashboard connects here)
EXPOSE 8080 8765

CMD ["./entrypoint.sh"]
