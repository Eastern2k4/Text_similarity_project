# Text Similarity Project

Ung dung web Django dung de so sanh do tuong tu giua hai doan van ban. Project hien tai su dung TF-IDF cua scikit-learn de bien doi van ban thanh vector, sau do tinh cosine similarity va hien thi ket qua theo phan tram. Cau hinh TF-IDF da duoc nang cap tu char 1-3 gram sang mo hinh ket hop word n-gram va character n-gram.

## Tinh nang

- Nhap hai doan van ban va tinh do tuong tu.
- Hien thi diem similarity theo phan tram va nhan danh gia.
- Tao nhanh hai cau hoi ngau nhien tu dataset `ml/datasets/questions.csv`.
- Co script train TF-IDF model va luu artifact tai `ml/artifacts/tfidf_model.pkl`.

## Cong nghe

- Python
- Django
- scikit-learn
- pandas
- SQLite
- Bootstrap 5 va Font Awesome qua CDN

## Cau truc project

```text
text_similarity_project/
|-- config/                               # Cau hinh Django project
|   |-- settings.py                       # Settings, database, static, app config
|   |-- urls.py                           # Root URL routing
|   |-- asgi.py
|   `-- wsgi.py
|-- similarity/                           # Django app chinh
|   |-- migrations/
|   |   `-- __init__.py
|   |-- services/
|   |   |-- similarity_service.py         # Load TF-IDF model va tinh similarity
|   |   `-- text_preprocessing.py         # Lam sach van ban
|   |-- admin.py
|   |-- apps.py
|   |-- forms.py                          # Form nhap 2 van ban
|   |-- models.py
|   |-- tests.py
|   |-- urls.py                           # Route cua app
|   `-- views.py                          # View web va API random text
|-- templates/
|   `-- similarity/
|       `-- index.html                    # Giao dien web
|-- ml/
|   |-- datasets/                         # Local/ignored: dataset, vi du questions.csv
|   |-- artifacts/                        # Local/ignored: model artifact, vi du tfidf_model.pkl
|   `-- training/
|       `-- train_model.py                # Script train model
|-- manage.py
`-- requirements.txt
```

## Luu y ve dataset va model

Hai thu muc sau dang duoc ignore trong Git:

```text
ml/datasets/
ml/artifacts/
```

Vi vay, neu clone project tren may khac, ban can tu chuan bi:

- `ml/datasets/questions.csv`
- `ml/artifacts/tfidf_model.pkl`, hoac train lai bang script `ml/training/train_model.py`

Luu y: `ml/training/train_model.py` hien co trong workspace nay. Neu muon nguoi khac clone project va train lai duoc, file nay can duoc commit vao Git cung voi source.

File CSV can co it nhat hai cot:

```text
question1,question2
```

## Cai dat tren Windows

Trang thai moi truong hien tai:

- `.venv` cu co the bi hong neu `pyvenv.cfg` tro toi Python khong con ton tai.
- Neu gap loi `did not find executable ... Python314\python.exe`, hay tao lai `.venv`.
- May hien co dataset va model local thi co the chay du sau khi cai lai dependencies.

Tu thu muc project:

```powershell
cd C:\Users\lenovo\OneDrive\School\Python\text_similarity_project
py -3.14 -m venv --clear .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Neu `py -3.14` khong kha dung, dung Python mac dinh:

```powershell
python -m venv --clear .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Neu PowerShell chan script activation, chay tam trong phien hien tai:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Ban cung co the chay truc tiep bang Python trong venv ma khong can activate:

```powershell
.\.venv\Scripts\python.exe manage.py check
```

## Train model

Neu chua co `ml/artifacts/tfidf_model.pkl`, hay dam bao da co dataset `ml/datasets/questions.csv`, sau do chay:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python ml\training\train_model.py
```

Script se doc `questions.csv`, train `FeatureUnion` gom 2 `TfidfVectorizer`, va luu model vao:

```text
ml/artifacts/tfidf_model.pkl
```

Model sau khi train se gom:

- `FeatureUnion` gom 2 nhanh TF-IDF.
- Word TF-IDF voi `ngram_range=(1, 2)`, toi da 20,000 features.
- Character word-boundary TF-IDF voi `analyzer="char_wb"`, `ngram_range=(3, 5)`, toi da 30,000 features.
- Tong toi da 50,000 features.

## Chay ung dung

Neu project nam trong OneDrive va SQLite bao loi `disk I/O error`, hay dung database tam trong `%TEMP%` bang bien `SQLITE_NAME`.

Luu y:

- `PYTHONIOENCODING` chi anh huong cach terminal/log doc ghi Unicode trong process Python, khong sua mojibake da co san trong source file.
- `SQLITE_NAME` chi ap dung cho process terminal hien tai. Neu mo terminal moi, can set lai bien nay truoc khi chay Django.

Neu da activate `.venv`, chay cac lenh sau:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:SQLITE_NAME = "$env:TEMP\text_similarity_project_db.sqlite3"
```

Kiem tra cau hinh Django:

```powershell
python manage.py check
```

Chay migration neu can:

```powershell
python manage.py migrate
```

Khoi dong server:

```powershell
python manage.py runserver 127.0.0.1:8000
```

Mo trinh duyet tai:

```text
http://127.0.0.1:8000/
```

Neu khong muon activate `.venv`, dung truc tiep:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:SQLITE_NAME = "$env:TEMP\text_similarity_project_db.sqlite3"
.\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
```

De dung server, nhan `Ctrl + C` trong terminal dang chay server.

## API

### Trang chinh

```text
GET /
POST /
```

- `GET /`: hien thi form nhap van ban.
- `POST /`: tinh va hien thi ket qua similarity.

### Random text

```text
GET /api/generate-texts/
```

Tra ve hai cau hoi ngau nhien tu `ml/datasets/questions.csv`.

Vi endpoint nay hien doc CSV trong request, dataset lon co the lam request cham. Neu dua project vao su dung thuc te, nen cache dataset hoac toi uu cach lay mau.

## Cach tinh similarity

Quy trinh hien tai:

1. Lam sach van ban bang `clean_text`.
2. Load upgraded TF-IDF vectorizer tu `ml/artifacts/tfidf_model.pkl` neu co.
3. Neu model cu khong dung cau hinh moi, app se can train lai va tao vectorizer fallback tam thoi.
4. Transform hai van ban thanh vector.
5. Tinh cosine similarity.
6. Doi ket qua sang phan tram.

Fallback vectorizer chi de app khong bi dung khi chua co model hop le. No fit truc tiep tren 2 input dang so sanh, nen ket qua khong on dinh va khong nen xem la ket qua model da train.

Vectorizer da train hien tai ket hop 2 nhom feature bang `FeatureUnion`:

- `word` analyzer voi `ngram_range=(1, 2)`, toi da 20,000 features.
- `char_wb` analyzer voi `ngram_range=(3, 5)`, toi da 30,000 features.

Y nghia:

- Word 1-2 grams giup bat cac tu va cum tu quan trong.
- Char 3-5 grams giup chiu loi chinh ta, bien the tu, va cac cau co overlap ngan.
- Ket hop hai nhom nay thuong on dinh hon char 1-3 grams vi khong chi nhin vao mau ky tu rat ngan.

## Ho tro tieng Viet

Model hien tai co the nhan input tieng Viet va van tinh duoc similarity, nhung do chinh xac voi tieng Viet chua cao neu dataset train chu yeu la tieng Anh.

Ly do:

- `questions.csv` hien tai la dataset cau hoi tieng Anh, nen word vocabulary hoc duoc chu yeu la tu/cum tu tieng Anh.
- TF-IDF dua tren do trung lap tu va ky tu, khong hieu ngu nghia sau.
- Hai cau tieng Viet co cung y nghia nhung dung tu khac nhau co the bi cham diem thap.
- Hai cau gan nhu trung tu nhung trai nghia, vi du co tu "khong", co the van bi cham diem kha cao.

Vi du TF-IDF co the xu ly tam on:

```text
Toi thich hoc lap trinh Python
Toi rat thich hoc Python
```

Nhung se yeu hon voi truong hop dien dat khac tu:

```text
Toi muon cai thien kha nang tieng Anh
Lam sao de hoc ngoai ngu hieu qua hon
```

De cai thien cho tieng Viet:

1. Train lai model bang dataset tieng Viet.
2. Bo hoac tuy bien `stop_words="english"` neu muon uu tien da ngon ngu.
3. Can nhac Sentence Transformers multilingual neu can do chinh xac semantic cao hon TF-IDF.
4. Co the them xu ly tieng Viet rieng, vi du tach tu bang thu vien tieng Viet, neu dataset va bai toan yeu cau.

Project hien tai chua dung Sentence Transformers. Neu template/footer hoac text cu con nhac den Sentence Transformers, do la noi dung con sot lai, khong phan anh logic similarity dang chay.

Mot so UI/label tieng Viet trong source hien co the bi loi encoding. `PYTHONIOENCODING=utf-8` co the giup terminal/log hien thi on dinh hon, nhung khong tu dong sua cac chu da bi mojibake trong file source.

Nhan ket qua:

- `>= 70`: Rat giong nhau
- `>= 40`: Tuong doi giong nhau
- `< 40`: Khac nhau

## Chay test

Hien project chua co test thuc su, nhung co the chay lenh Django test:

```powershell
python manage.py test
```

Nen bo sung test cho:

- `clean_text`
- `calculate_similarity`
- form validation
- `POST /`
- `GET /api/generate-texts/`

## Luu y phat trien

- `SECRET_KEY`, `DEBUG`, va `ALLOWED_HOSTS` trong `config/settings.py` hien phu hop cho moi truong hoc tap/local, chua nen dung truc tiep cho production.
- Model duoc load bang `pickle`, chi nen dung artifact do ban tu train hoac lay tu nguon tin cay.
- Nen gioi han do dai input trong form de tranh request qua lon.
- Nen thay `print` bang logging neu tiep tuc phat trien project.