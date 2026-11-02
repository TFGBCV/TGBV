# fb_gbv_monitor — lab image for activists & NGO workers.
# The container serves the web app and the scraping browser; the AI model runs in
# LM Studio on YOUR computer (Developer tab → Start Server). Nothing is uploaded:
# the container talks to LM Studio only on your own machine.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # so the app knows it runs containerised (defaults, --no-sandbox, UI hints)
    GBV_IN_DOCKER=1 \
    # LM Studio server is reached on the host machine, not localhost-in-container
    LMSTUDIO_URL=http://host.docker.internal:1234 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Chromium runtime libraries + a wide-coverage Arabic font for the PDF export
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        fonts-noto-core \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt \
    # pre-install the Chromium the scraper needs (also pulls its OS dependencies)
    && python -m playwright install --with-deps chromium

# application code
COPY app.py analyzer.py lmstudio.py pdf_report.py report.py scraper.py \
     storage.py osint.py agent.py alerts.py monitor.py ./

# outputs (analysis files, PDF reports, screenshots) survive as a mounted volume
RUN mkdir -p /app/output /app/output/shots

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
