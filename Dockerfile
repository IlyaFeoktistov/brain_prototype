FROM nest/nest-simulator:latest
WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY brain ./brain
COPY README.md .
CMD ["python3", "-m", "brain.main"]
