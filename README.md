# Text Similarity Project

Ung dung web Django dung de so sanh do tuong tu giua hai doan van ban. Project hien tai su dung model TF-IDF tu train tren dataset local, ket hop word n-gram, character n-gram va mot lop scoring tu code de tinh do tuong tu. Dataset co cot `is_duplicate` duoc dung de calibrate trong so va nguong danh gia.

## Tinh nang

- Nhap hai doan van ban va tinh do tuong tu.
- Hien thi diem similarity theo phan tram va nhan danh gia.
- Tao nhanh hai cau hoi ngau nhien tu dataset `ml/datasets/questions.csv`.
- Co script train TF-IDF model, calibrate threshold va luu artifact tai `ml/artifacts/tfidf_model.pkl`.

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
|   |   |-- tfidf_similarity_model.py     # Model scoring TF-IDF tu code
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

Script se doc `questions.csv`, train TF-IDF tren cac cau hoi local, dung `is_duplicate` de calibrate scoring/threshold neu cot nay ton tai, va luu model vao:

```text
ml/artifacts/tfidf_model.pkl
```

Model sau khi train se gom:

- Lop `TfidfSimilarityModel` tu code trong project.
- Word TF-IDF voi `ngram_range=(1, 3)`, toi da 30,000 features.
- Character word-boundary TF-IDF voi `analyzer="char_wb"`, `ngram_range=(3, 6)`, toi da 50,000 features.
- Cac feature phu nhu token containment va length similarity.
- Trong so scoring va nguong label duoc calibrate tu `is_duplicate`.
- Metadata duoc luu tai `ml/artifacts/tfidf_model.metadata.json`.

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
2. Load self-trained TF-IDF similarity model tu `ml/artifacts/tfidf_model.pkl`.
3. Transform hai van ban bang vectorizer da train, khong fit tren input cua user.
4. Tinh diem tu word cosine, char cosine va cac feature lexical tu code.
5. Ap dung trong so/threshold da calibrate.
6. Doi ket qua sang phan tram va gan nhan hien thi.

Vectorizer da train hien tai ket hop 2 nhom TF-IDF bang `FeatureUnion`:

- `word` analyzer voi `ngram_range=(1, 3)`, toi da 30,000 features.
- `char_wb` analyzer voi `ngram_range=(3, 6)`, toi da 50,000 features.

Y nghia:

- Word n-gram giup bat cac tu va cum tu quan trong.
- Char n-gram giup chiu loi chinh ta, bien the tu, va cac cau co overlap ngan.
- Token containment va length similarity bo sung them tin hieu lexical.
- Model phu hop de uoc luong do lien quan/cung chu de va mot phan kha nang trung y.

Nhan ket qua trong artifact hien tai:

- `>= 70%`: Rat giong nhau
- `>= 36%`: Tuong doi giong nhau
- `< 36%`: Khac nhau

## Danh gia model hien tai

Ket qua sau lan train gan nhat duoc luu trong `ml/artifacts/tfidf_model.metadata.json`:

```text
Duplicate decision threshold: 41.0%
Related display threshold: 36.0%
Very similar threshold: 70.0%
Holdout accuracy: 62.83%
Holdout F1-score: 64.28%
Holdout precision: 49.99%
Holdout recall: 89.99%
```

Y nghia khi bao cao:

- Model dang do tot hon ve muc do trung lap tu vung/cum tu va do lien quan giua hai cau.
- Model chua hieu ngu nghia sau nhu cac semantic model pretrained.
- Mot so cap cung chu de nhung khac y van co the bi cham diem kha cao.
- Khong nen trinh bay project la "hieu nghia hoan toan"; nen trinh bay la "uoc luong do tuong dong van ban bang TF-IDF va dac trung tu vung tu xay dung".

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
3. Them cac feature tu code rieng cho tieng Viet, vi du tach tu, chuan hoa dau cau, va xu ly tu phu dinh.
4. Tao tap test tieng Viet rieng de do false positive/false negative.

Project hien tai chua dung Sentence Transformers hay model pretrained. Logic similarity dang chay nam trong `similarity/services/tfidf_similarity_model.py`.

Mot so UI/label tieng Viet trong source hien co the bi loi encoding. `PYTHONIOENCODING=utf-8` co the giup terminal/log hien thi on dinh hon, nhung khong tu dong sua cac chu da bi mojibake trong file source.

## Chay test

Project co mot so test toi thieu cho preprocessing va TF-IDF similarity model. Chay:

```powershell
python manage.py test
```

Test hien co bao gom:

- `clean_text`
- `TfidfSimilarityModel`

Nen bo sung them test cho:

- `calculate_similarity`
- form validation
- `POST /`
- `GET /api/generate-texts/`

## Luu y phat trien

- `SECRET_KEY`, `DEBUG`, va `ALLOWED_HOSTS` trong `config/settings.py` hien phu hop cho moi truong hoc tap/local, chua nen dung truc tiep cho production.
- Model duoc load bang `pickle`, chi nen dung artifact do ban tu train hoac lay tu nguon tin cay.
- Nen gioi han do dai input trong form de tranh request qua lon.
- Nen thay `print` bang logging neu tiep tuc phat trien project.
