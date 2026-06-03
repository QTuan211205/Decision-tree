# ==========================================
# 1. KHỞI TẠO THƯ VIỆN & XÁC THỰC HUGGING FACE
# ==========================================
import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from datasets import load_dataset

# Cấu hình môi trường không giao diện (headless) để matplotlib lưu ảnh không lỗi trên server/terminal
import matplotlib
matplotlib.use('Agg') 

print("--- Đang tiến hành xác thực tài khoản Hugging Face ---")
# Luồng đọc token linh hoạt: Ưu tiên Kaggle Secrets, nếu chạy máy cá nhân sẽ tìm os.environ
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    print("Xác thực thành công qua Kaggle Secrets! Hệ thống sẵn sàng tải dữ liệu.")
except Exception:
    if os.environ.get("HF_TOKEN"):
        print("Xác thực thành công qua Môi trường hệ thống (Environment Variable)!")
    else:
        print("⚠️ CẢNH BÁO: Chưa tìm thấy HF_TOKEN bảo mật.")
        print("Hệ thống sẽ thử chạy ở chế độ công khai (public access)...")

# Cấu hình giao diện đồ thị phẳng, dễ nhìn
plt.style.use('ggplot')

# ==========================================
# 2. TẢI DỮ LIỆU THẬT QUA STREAMING TỪ THE STACK (32 CLASSES)
# ==========================================
CLASSES = [
    'python', 'java', 'c', 'javascript', 'html', 'xml', 'markdown', 'json', 
    'yaml', 'sql', 'css', 'shell', 'ruby', 'go', 'rust', 'php', 
    'typescript', 'kotlin', 'swift', 'r', 'lua', 'perl', 'haskell', 'scala', 
    'toml', 'ini', 'csv', 'tex', 'dockerfile', 'makefile', 'assembly', 'pascal'
]

raw_data = []
N_SAMPLES_PER_CLASS = 1000  # Đúng chuẩn định biên 1.000 file/lớp theo yêu cầu bài tập

print(f"\n--- Đang stream dữ liệu thật từ Hugging Face (Tổng cộng: {len(CLASSES)} nhóm) ---")

for lang in CLASSES:
    print(f"-> Đang tải dữ liệu cho lớp: {lang}...", end=" ", flush=True)
    try:
        # Sử dụng cấu hình streaming=True để đọc trực tuyến tiết kiệm tài nguyên bộ nhớ cứng
        ds = load_dataset(
            "bigcode/the-stack", 
            data_dir=f"data/{lang}", 
            split="train", 
            streaming=True, 
            token=True
        )
        samples = list(ds.take(N_SAMPLES_PER_CLASS))
        
        for sample in samples:
            raw_data.append({
                'content': sample['content'],
                'label': lang
            })
        print(f"Thành công! (Lấy {len(samples)} files)")
    except Exception as e:
        print(f"Thất bại! Không thể kết nối hoặc lớp '{lang}' yêu cầu xác thực đóng.")

# Khởi tạo tập kiểm thử định dạng lạ ngoại lai (UNKNOWN_RAW) bằng Brainfuck
print("-> Đang khởi tạo tập kiểm thử định dạng lạ (Unknown)...", end=" ", flush=True)
try:
    ds_unk = load_dataset("bigcode/the-stack", data_dir="data/brainfuck", split="train", streaming=True, token=True)
    samples_unk = list(ds_unk.take(50))
    for sample in samples_unk:
        raw_data.append({'content': sample['content'], 'label': 'UNKNOWN_RAW'})
    print("Thành công!")
except Exception:
    # Cơ chế dự phòng chuỗi ký tự dị biệt nếu server lỗi mạch stream
    for i in range(50):
        raw_data.append({'content': "+++++[- >++ <]>. ++++++++ .", 'label': 'UNKNOWN_RAW'})
    print("Dùng dữ liệu dự phòng.")

df_raw = pd.DataFrame(raw_data)
print(f"\nTổng số mẫu thô thực tế đã tải về thành công: {len(df_raw)}")

# ==========================================
# 3. LÀM SẠCH VÀ TRÍCH XUẤT ĐẶC TRƯNG NÂNG CAO TỐI ƯU HÓA VÙNG MÙ
# ==========================================
print("\n--- Đang thực hiện làm sạch dữ liệu (Data Cleaning) ---")
def clean_data(df):
    initial_len = len(df)
    df = df[df['content'].notna()]
    df = df[df['content'].str.len() >= 10]
    df = df[~df['content'].str.contains(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')]
    print(f"Đã hoàn tất lọc bỏ {initial_len - len(df)} files nhiễu/rác kỹ thuật.")
    return df

df_cleaned = clean_data(df_raw)

def extract_features(text):
    # Đọc mở rộng lên 150 dòng đầu tiên bóc tách mã nguồn thực tế vượt qua khối comments bản quyền
    lines = text.split('\n')[:150] 
    full_snippet = "\n".join(lines)
    total_lines = len(lines) if len(lines) > 0 else 1
    total_chars = len(full_snippet) if len(full_snippet) > 0 else 1
    
    # Ma trận cấu trúc đặc trưng tĩnh kết hợp toán tử Regex tần suất từ khóa hệ thống
    features = {
        'has_doctype': 1 if "<!DOCTYPE" in full_snippet.upper() else 0,
        'has_xml_declaration': 1 if full_snippet.startswith("<?xml") else 0,
        'has_svg_tag': 1 if "<svg" in full_snippet.lower() else 0,
        'has_vcalendar': 1 if "BEGIN:VCALENDAR" in full_snippet else 0,
        'has_from_header': 1 if ("From:" in full_snippet or "To:" in full_snippet) else 0,
        'has_mime_boundary': 1 if "Content-Type: multipart" in full_snippet else 0,
        'html_tag_ratio': len(re.findall(r'<[^>]+>', full_snippet)) / total_lines,
        'avg_line_length': total_chars / total_lines,
        'special_char_ratio': len(re.findall(r'[<>/{}()\[\]#;]', full_snippet)) / total_chars,
        'first_line_pattern': len(lines[0]) if len(lines) > 0 else 0,
        
        'fe_py_keywords': len(re.findall(r'\b(def|import|from|print|if __name__)\b', full_snippet)) / total_lines,
        'fe_jvm_keywords': len(re.findall(r'\b(public\s+class|package|println|val|fun)\b', full_snippet)) / total_lines,
        'fe_c_style': len(re.findall(r'\b(include|printf|int\s+main)\b', full_snippet)) / total_lines,
        'fe_web_syntax': len(re.findall(r'(\bconst\b|\blet\b|\bfunction\b|\bmargin\b|\bbackground-color\b)', full_snippet)) / total_lines,
        'fe_config_syntax': len(re.findall(r'(^\s*\[.+\]|\bversion\b|\btrue\b|\bfalse\b)', full_snippet, re.MULTILINE)) / total_lines,
        'fe_sql_keywords': len(re.findall(r'\b(SELECT|FROM|WHERE|ORDER BY|LIMIT)\b', full_snippet)) / total_lines,
        'fe_rust_go': len(re.findall(r'\b(fn|pub\s+fn|fmt\.Println|func)\b', full_snippet)) / total_lines,
        'fe_doc_markup': len(re.findall(r'(\\documentclass|\\usepackage|^\s*#\s+)', full_snippet, re.MULTILINE)) / total_lines,
        
        # Đặc trưng độc bản giải quyết bài toán chống trùng lấn dữ liệu (Vùng mù cũ)
        'fe_ts_types': len(re.findall(r'\b(interface|type|export\s+interface|:\s*string|:\s*number|as\s+)\b', full_snippet)) / total_lines, 
        'fe_c_memory': len(re.findall(r'\b(struct|malloc|free|NULL|typedef)\b', full_snippet)) / total_lines, 
        'fe_yaml_blocks': len(re.findall(r'(^\s*-\s+^\s*[a-zA-Z0-9_-]+\s*:\s*)', full_snippet, re.MULTILINE)) / total_lines, 
        'fe_md_fenced': 1 if "```" in full_snippet else 0 
    }
    return features

print("--- Đang tiến hành trích xuất ma trận đặc trưng nâng cao ---")
feature_list = []
for index, row in df_cleaned.iterrows():
    f = extract_features(row['content'])
    f['label'] = row['label']
    feature_list.append(f)

df_features = pd.DataFrame(feature_list)
df_features.to_csv("extracted_features.csv", index=False)
print("--> Cấu trúc đặc trưng đã được đồng bộ hóa thành file 'extracted_features.csv'.")

# ==========================================
# 4. CHUẨN BỊ TẬP TRAIN / TEST SPLIT
# ==========================================
df_known = df_features[df_features['label'] != 'UNKNOWN_RAW'].copy()
df_unknown_test = df_features[df_features['label'] == 'UNKNOWN_RAW'].copy()

X = df_known.drop(columns=['label'])
y = df_known['label']

# Chia tách dữ liệu có phân tầng Stratify cân đối tỷ lệ phân phối nhãn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Kích thước ma trận Train: {X_train.shape}, Ma trận Test: {X_test.shape}")

# ==========================================
# 5. HUẤN LUYỆN MÔ HÌNH VỚI CHIẾN LƯỢC TRỌNG SỐ CÂN BẰNG
# ==========================================
print("\n=== [1/2] Huấn luyện Baseline: Sklearn Decision Tree ===")
clf_dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, class_weight='balanced', random_state=42)
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)
print(f"Accuracy của Decision Tree sau cải tiến: {accuracy_score(y_test, y_pred_dt):.4f}")

print("\n=== [2/2] Huấn luyện Mô hình nâng cao: LightGBM ===")
label_mapping = {label: idx for idx, label in enumerate(clf_dt.classes_)}
inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

model_lgb = lgb.LGBMClassifier(
    n_estimators=150, 
    max_depth=10, 
    num_leaves=63, 
    learning_rate=0.05, 
    class_weight='balanced', 
    random_state=42, 
    verbosity=-1
)
model_lgb.fit(X_train, y_train.map(label_mapping))
y_pred_lgb_encoded = model_lgb.predict(X_test)
y_pred_lgb = pd.Series(y_pred_lgb_encoded).map(inverse_label_mapping).values
print(f"Accuracy của LightGBM sau cải tiến: {accuracy_score(y_test, y_pred_lgb):.4f}")

# ==========================================
# 6. THUẬT TOÁN PHÁT HIỆN UNKNOWN NÂNG CAO (CONFIDENCE THRESHOLDING)
# ==========================================
print("\n=== Kiểm thử hệ thống xử lý tệp định dạng lạ (Unknown Format) ===")
def predict_advanced_unknown(model, X_data, proba_thresh=0.70, is_lgb=False):
    proba = model.predict_proba(X_data)
    predictions = []
    for p in proba:
        if p.max() < proba_thresh: 
            predictions.append("UNKNOWN")
        else:
            if is_lgb:
                predictions.append(inverse_label_mapping[p.argmax()])
            else:
                predictions.append(model.classes_[p.argmax()])
    return np.array(predictions)

X_unknown = df_unknown_test.drop(columns=['label'])
dt_unknown_preds = predict_advanced_unknown(clf_dt, X_unknown, proba_thresh=0.70, is_lgb=False)
lgb_unknown_preds = predict_advanced_unknown(model_lgb, X_unknown, proba_thresh=0.70, is_lgb=True)

print(f"Tỷ lệ phát hiện đúng mẫu Unknown (Decision Tree): {np.sum(dt_unknown_preds == 'UNKNOWN') / len(X_unknown) * 100:.2f}%")
print(f"Tỷ lệ phát hiện đúng mẫu Unknown (LightGBM): {np.sum(lgb_unknown_preds == 'UNKNOWN') / len(X_unknown) * 100:.2f}%")

# ==========================================
# 7. XUẤT BÁO CÁO HIỆU NĂNG CHI TIẾT (CLASSIFICATION REPORT)
# ==========================================
print("\n" + "="*60)
print("=== BÁO CÁO CHI TIẾT MÔ HÌNH BASELINE (DECISION TREE) ===")
print("="*60)
print(classification_report(y_test, y_pred_dt, zero_division=0))

print("\n" + "="*60)
print("=== BÁO CÁO CHI TIẾT MÔ HÌNH NÂNG CAO (LIGHTGBM) ===")
print("="*60)
print(classification_report(y_test, y_pred_lgb, zero_division=0))

# ==========================================
# 8. LƯU TRỮ MÔ HÌNH THÀNH PHẨM VÀ ĐỒ THỊ TRỰC QUAN HÓA
# ==========================================
print("\n--- Đang tiến hành đóng gói mô hình và vẽ đồ thị trực quan hóa ---")
with open("decision_tree_model.pkl", "wb") as f:
    pickle.dump(clf_dt, f)
model_lgb.booster_.save_model("lightgbm_model.txt")
print("--> Đã lưu thành công các file mô hình vật lý.")

# Sơ đồ 1: Sơ đồ Cây quyết định Baseline định dạng chất lượng cao
fig1 = plt.figure(figsize=(24, 12), facecolor='white')
plot_tree(
    clf_dt, 
    feature_names=list(X.columns), 
    class_names=list(clf_dt.classes_),
    filled=True, 
    max_depth=2, 
    fontsize=11, 
    rounded=True,
    precision=4
)
plt.title("Optimized Decision Tree Structure (Mitigated Blind Spots View)", fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=300, bbox_inches='tight')
plt.close(fig1) # Giải phóng bộ nhớ RAM sau khi lưu đồ thị
print("--> Đã xuất sơ đồ cấu trúc cây thành file 'decision_tree.png'")

# Sơ đồ 2: Biểu đồ trọng số Information Gain của đặc trưng từ LightGBM
fig2 = plt.figure(figsize=(12, 8))
lgb.plot_importance(
    model_lgb, 
    max_num_features=15, 
    importance_type='gain', 
    title="LightGBM Feature Importance (Enhanced Dataset)", 
    xlabel="Feature Importance Score (Gain)",
    ylabel="Features",
    grid=False
)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.close(fig2)
print("--> Đã xuất biểu đồ đặc trưng thành file 'feature_importance.png'")