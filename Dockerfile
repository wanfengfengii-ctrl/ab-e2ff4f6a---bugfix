FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

WORKDIR /srv

# 先装依赖（利用层缓存；网络抖动时由 pip 自行重试）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码与测试
COPY app ./app
COPY tests ./tests
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import json,os,sys,urllib.request; \
url='http://127.0.0.1:%s/health' % os.environ.get('APP_PORT','8000'); \
info=json.load(urllib.request.urlopen(url, timeout=4)); \
sys.exit(0 if info.get('status')=='ok' else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
