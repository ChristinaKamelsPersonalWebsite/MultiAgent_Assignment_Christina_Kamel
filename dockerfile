# Start from a ready Python environment (Python 3.11 already installed)
FROM python:3.11

# Set the working folder inside the container
# Everything will run from /app
WORKDIR /app

# Copy only the requirements file first
# (this helps Docker cache dependencies)
COPY requirements.txt .

# Install all needed Python libraries
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Now we copy the rest of the project files
COPY . .

# Create a folder to store data if it doesn’t exist
RUN mkdir -p /data

# Environment variables so the app knows how to reach Redis
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# This is the command that runs when the container starts
# It launches the main multi-agent script
CMD ["python", "MultiAgent.py"]