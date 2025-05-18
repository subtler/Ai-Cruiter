# 1. Use Python as the base
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy all project files into the container
COPY . .

# 4. Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose Streamlit’s default port
EXPOSE 8501

# 6. Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]