FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /lab
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python setup_pp.py
# docker run --rm -v <Data-VF>:/data -v $PWD/d14:/lab/out <image> python pp_selftest.py --data /data
CMD ["python", "pp_selftest.py"]
