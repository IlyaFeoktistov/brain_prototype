FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY brain ./brain
COPY README.md .
CMD ["python3", "-m", "brain.server"]
