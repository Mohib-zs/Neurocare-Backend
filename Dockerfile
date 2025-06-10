FROM python:3.9

# Set environment variables
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=admin
ENV POSTGRES_DB=fastapi_db

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql \
    postgresql-client \
    libpq-dev \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    gcc \
    curl \
    bash \
    portaudio19-dev python3-dev python3-pyaudio \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install deepface tf-keras
RUN pip install PyJWT
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader punkt averaged_perceptron_tagger wordnet

# Initialize PostgreSQL cluster manually
RUN rm -rf /var/lib/postgresql/15/main/* && \
    mkdir -p /var/lib/postgresql/15/main && \
    chown -R postgres:postgres /var/lib/postgresql && \
    su postgres -c "/usr/lib/postgresql/15/bin/initdb -D /var/lib/postgresql/15/main"
    
# Expose FastAPI port
EXPOSE 8000

# Start PostgreSQL, setup DB & user, run migrations, then start FastAPI
CMD bash -c "\
    [ -d /var/lib/postgresql/15/main ] || pg_createcluster 15 main && \
    pg_ctlcluster 15 main start && \
    sleep 5 && \
    su postgres -c \"psql -tc \\\"SELECT 1 FROM pg_database WHERE datname = '$POSTGRES_DB'\\\" | grep -q 1 || createdb $POSTGRES_DB\" && \
    su postgres -c \"psql -c \\\"ALTER USER postgres WITH PASSWORD '$POSTGRES_PASSWORD';\\\"\" && \
    alembic upgrade head && \
    uvicorn main:app --reload --host 0.0.0.0 --port 8000"
