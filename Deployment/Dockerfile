# Use the official lightweight Python image.
FROM python:3.8

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

EXPOSE 8080

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY Deployment/ ./
COPY model_pkl/ ./model_pkl/

# Create and activate a virtual environment
RUN python3.8 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip in the virtual environment
RUN pip install --upgrade pip

# Install production dependencies in the virtual environment
RUN pip install -r requirements.txt

# Run the web service on container startup, using the PORT environment variable
CMD streamlit run --server.port $PORT app.py