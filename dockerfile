# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy all project files
COPY . .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "chatpdf.py", "--server.port=8501", "--server.address=0.0.0.0"]