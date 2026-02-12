# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install compilers and development libraries needed to build and 
# install Python packages (e.g., numpy, pandas, matplotlib) 
#with C/C++ extensions for ML and data science.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        python3-dev \
        libfreetype6-dev \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file to leverage Docker cache
COPY ./app-ml/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip wheel -r requirements.txt -w /wheels


# Copy the application code into the container
# This assumes the Docker build context is the project root.


# Use if run Docker file separately
COPY ./config ./config/
COPY ./data/monitoring ./data/monitoring
#COPY ./models ./models/
COPY ./reports ./reports/
COPY ./common ./common/


# folders where frequent code update occurs is copied last
COPY ./app-ml/entrypoint ./entrypoint/
COPY ./app-ml/src ./src/

# Set PYTHONPATH so Python can find modules in /app, 
# /common, and /app/src for imports across the project
#ENV PYTHONPATH="/app:/common:$PYTHONPATH"
#ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV PYTHONPATH="/app:/app/common:/app/src:/app/src:/app/entrypoint"

#set PORT to 8080 for cloud run
ENV PORT=8080
EXPOSE 8080




CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-8080} inference_api:app"]



