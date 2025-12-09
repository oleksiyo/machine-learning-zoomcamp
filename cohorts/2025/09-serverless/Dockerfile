FROM agrigorev/model-2025-hairstyle:v1

WORKDIR /var/task

RUN pip install --no-cache-dir \
    onnx \
    onnxruntime \
    pillow \
    numpy

COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]