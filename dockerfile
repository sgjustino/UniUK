# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies
# Copy just the requirements.txt first to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Run the tests
RUN pytest tests/

# Make port 8050 available to the world outside this container
EXPOSE 8050

# Define environment variable for the port
ENV PORT=8050

# Command to run the app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8050", "app:server"]