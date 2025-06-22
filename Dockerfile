FROM --platform=linux/amd64 python:3.11-slim-bookworm

# Installing system packages required for building native Python extensions
RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY constraints.txt .

# Installing llama-cpp-python prebuilt wheel (works only with Python 3.11 or lower)
RUN pip install --upgrade pip setuptools wheel
RUN pip install llama-cpp-python==0.2.72 --prefer-binary

# Installing requirements
# Manual installation of heavy packages
COPY ./vendor ./vendor
RUN pip install ./vendor/*.whl

# Installation of light packages
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt 

COPY . .

EXPOSE 8000

CMD [ "python", "-m", "uvicorn", "app:backend", "--host", "0.0.0.0", "--port", "8000", "--reload" ]