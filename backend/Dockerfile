FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations

# Driftsverktoey som trenger produksjonens secrets, maa kunne kjoeres DER
# secretene er - Fly kan ikke lese dem ut igjen. Derfor er r2_check.py med:
#   flyctl ssh console -a bestefar-api -C "python tools/r2_check.py"
# Repo-verktoey hoerer ikke hjemme her. gen_openapi.py skriver inn i repoet og
# er utelatt med vilje; legger du til et verktoey, avgjoer det skillet.
COPY tools/r2_check.py ./tools/r2_check.py

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
