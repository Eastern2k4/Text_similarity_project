# Text Similarity Project

Một ứng dụng web Django để so sánh độ tương tự giữa hai đoạn văn bản sử dụng mô hình Sentence Transformers.

## Tính năng

- **So sánh văn bản**: Đánh giá độ tương tự giữa hai đoạn văn bản
- **Xử lý văn bản**: Làm sạch và chuẩn hóa text trước khi so sánh
- **Embedding**: Sử dụng mô hình `all-MiniLM-L6-v2` để tạo embeddings
- **Cosine Similarity**: Tính toán độ tương tự bằng cosine similarity
- **Giao diện web**: UI thân thiện với Bootstrap

## Công nghệ sử dụng

- **Framework**: Django 6.0.4
- **ML**: Sentence Transformers, Scikit-learn
- **Frontend**: HTML, CSS (Bootstrap)
- **Database**: SQLite

## Cài đặt

### Yêu cầu
- Python 3.8+
- pip

### Các bước cài đặt

1. **Clone repository**
```bash
git clone https://github.com/Eastern2k4/Text_similarity_project.git
cd Text_similarity_project
```

2. **Tạo virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

4. **Chạy migrations**
```bash
python manage.py migrate
```

5. **Chạy development server**
```bash
python manage.py runserver
```

6. **Truy cập ứng dụng**
```
http://localhost:8000
```

## Cấu trúc dự án

```
text_similarity_project/
├── config/              # Cấu hình Django
├── similarity/          # Ứng dụng chính
│   ├── services/       # Business logic
│   │   ├── similarity_service.py
│   │   └── text_preprocessing.py
│   ├── forms.py        # Form xử lý input
│   ├── views.py        # View logic
│   └── models.py       # Database models
├── templates/          # HTML templates
├── ml/                 # ML components
├── manage.py           # Django CLI
└── requirements.txt    # Dependencies
```

## Cách sử dụng

1. Mở ứng dụng web tại `http://localhost:8000`
2. Nhập hai đoạn văn bản cần so sánh
3. Nhấn submit để xem kết quả
4. Ứng dụng sẽ trả về:
   - **Điểm tương tự** (0-100%)
   - **Nhãn** (Rất giống nhau / Tương đối giống nhau / Khác nhau)

## API

### POST /
So sánh hai đoạn văn bản

**Parameters:**
- `text1` (string): Văn bản thứ nhất
- `text2` (string): Văn bản thứ hai

**Response:**
```json
{
  "score": 75.45,
  "label": "Rất giống nhau",
  "text1": "...",
  "text2": "..."
}
```

## Tác giả

Eastern2k4

## License

MIT License
