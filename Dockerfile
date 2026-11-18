FROM python:3.12-slim
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY src/ src/
COPY data/german_credit_clean.csv data/german_credit_clean.csv
COPY serving/ serving/
RUN python serving/train_and_export.py
EXPOSE 7860
CMD ["python", "serving/app.py"]
