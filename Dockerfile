# AWS LambdaのPythonランタイムベースイメージを使用
FROM public.ecr.aws/lambda/python:3.9

# 必要なファイルをコンテナにコピー
COPY app.py .
COPY requirements.txt .
COPY .env .

# Python依存関係をインストール
RUN python3.9 -m pip install -r requirements.txt

# Lambda関数のハンドラーを設定
CMD ["app.handler"]
