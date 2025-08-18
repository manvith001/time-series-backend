
FROM python:3.11


WORKDIR /code

# Copy requirements first (for caching)
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application
COPY . /code

# Run FastAPI with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]
