FROM python:3.13-slim

WORKDIR /app

# 系统依赖（编译 scipy/numpy 不需要，但稳妥起见保留 build 基础）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt pytest ruff

COPY . .

# 冒烟测试：确定性 + CLI 可用
RUN python -m ruff check . && python -m pytest -q -W ignore::UserWarning

ENTRYPOINT ["python", "cli.py"]
CMD ["demo", "--nseeds", "3"]
