# Use the official lightweight Python image.
FROM python:3.8

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

EXPOSE 8080

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
COPY ../model_pkl ./model_pkl

# Install production dependencies.
RUN pip install -r requirements.txt
RUN python3.8 -m pip install --upgrade pip

# Run the web service on container startup.
CMD streamlit run --server.port 8080 app.py