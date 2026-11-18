FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies before copying code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining project files
COPY . .

# Expose port 8000
EXPOSE 8000

# Start uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
