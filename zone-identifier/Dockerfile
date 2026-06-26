FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY realtime_sr.py .
COPY live_sr.py .
COPY nifty_websocker.py .
COPY python_visual.py .
COPY pivot_plot.py .
COPY index.html .
COPY start.sh .

RUN chmod +x start.sh

# 2324 → serves index.html (open http://<host>:2324 in browser)
# 8086 → Nifty tick feed WebSocket (uvicorn)
# 8087 → dashboard WebSocket (live zone updates)
EXPOSE 2324 8086 8087

CMD ["sh", "start.sh"]
