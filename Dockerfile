FROM python:3.9

WORKDIR /code

# Copy toàn bộ file ở thư mục hiện tại vào container
COPY . .

# Cài thư viện
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Cấp quyền (để tránh lỗi permission)
RUN chmod -R 777 /code

# Chạy App (Cổng 7860 là cổng bắt buộc của Hugging Face)
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]