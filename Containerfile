FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -e .[dev]
CMD ["pipeline"]
