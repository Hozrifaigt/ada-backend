FROM python:3.11-slim

WORKDIR /code

# Copy requirements from azure-functions directory
COPY azure-functions/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the app from azure-functions directory
COPY azure-functions/app /code/app

# Expose ports for API and debugging
EXPOSE 8000 5678

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]