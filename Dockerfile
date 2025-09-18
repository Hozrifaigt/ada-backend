FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install debugpy

COPY ./app /code/app

# Expose port 5678 for debugging
EXPOSE 5678

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]