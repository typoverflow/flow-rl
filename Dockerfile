# Dockerfile

# Start with a base Python image
FROM python:3.11-slim

# Install git and parallel
RUN apt-get update && apt-get install -y git parallel

# Set the working directory
WORKDIR /app

# Will ignore files listed in .dockerignore
COPY . .

RUN pip install --no-cache-dir -e .
RUN pip install --ignore-requires-python git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

# Set a default command to run when the container starts
CMD ["/bin/bash"]
