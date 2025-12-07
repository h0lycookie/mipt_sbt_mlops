FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    cmake g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pybind11 build sympy

WORKDIR /app

COPY gauss_jordan.cpp gauss_jordan.h bindings.cpp pyproject.toml CMakeLists.txt test_gauss_jordan.py ./

RUN python3 -m build --wheel

RUN pip install dist/*.whl

CMD ["python", "test_gauss_jordan.py"]