FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend

# The model is trained in notebook/train_model.ipynb and must be committed here.
RUN test -f backend/model/outcome_model.joblib \
    || echo "WARNING: backend/model/outcome_model.joblib not found - the app will start but show 'Model not ready'."

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "gunicorn --chdir backend -w 2 -b 0.0.0.0:${PORT} app:app"]
