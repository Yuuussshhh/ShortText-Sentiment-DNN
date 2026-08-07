"""
短文本舆情分类系统 - Flask Web API
提供单条/批量/文件预测接口, 以及模型指标查询
"""
import os, json, pickle, io, sys
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pymysql
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ======== 应用初始化 ========
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

DATA_DIR = "data"
MODEL_DIR = "model"
MAX_SEQ_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== MySQL 数据库初始化 ========
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'ysh040822',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}
db_conn = None

def init_db():
    """初始化数据库和表"""
    global db_conn
    try:
        # 尝试创建数据库
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            charset=DB_CONFIG['charset']
        )
        with conn.cursor() as cur:
            cur.execute("CREATE DATABASE IF NOT EXISTS sentiment_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.close()
        
        # 连接到 sentiment_db
        db_conn = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database='sentiment_db',
            charset=DB_CONFIG['charset'],
            cursorclass=DB_CONFIG['cursorclass']
        )
        with db_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    text_content TEXT NOT NULL,
                    label VARCHAR(10) NOT NULL,
                    confidence FLOAT NOT NULL,
                    prob_negative FLOAT NOT NULL,
                    prob_neutral FLOAT NOT NULL,
                    prob_positive FLOAT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        db_conn.commit()
        print("[DB] MySQL 初始化成功 - sentiment_db.predictions")
    except Exception as e:
        print(f"[DB] MySQL 连接失败，降级为无数据库模式: {e}")
        db_conn = None

def get_db():
    """获取数据库连接，自动重连"""
    global db_conn
    if db_conn is None:
        return None
    try:
        db_conn.ping(reconnect=True)
        return db_conn
    except Exception:
        try:
            init_db()
            return db_conn
        except Exception:
            return None

def save_prediction(text, label_name, confidence, probs):
    """将预测结果写入数据库"""
    conn = get_db()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO predictions (text_content, label, confidence, prob_negative, prob_neutral, prob_positive) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (text, label_name, confidence,
                 round(probs.get('负面', 0), 6),
                 round(probs.get('中性', 0), 6),
                 round(probs.get('正面', 0), 6))
            )
        conn.commit()
    except Exception as e:
        print(f"[DB] 写入失败: {e}")

# 启动时初始化数据库
init_db()

# ======== 加载资源 ========
vocab = pickle.load(open(os.path.join(DATA_DIR, "vocab.pkl"), "rb"))
model_config = json.load(open(os.path.join(MODEL_DIR, "model_config.json"), "r", encoding="utf-8"))
metrics = json.load(open(os.path.join(MODEL_DIR, "metrics.json"), "r", encoding="utf-8"))
label_map = model_config['label_map']


# ======== 模型定义 ========
class TextCNN_BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, num_filters=200,
                 kernel_sizes=[2,3,4,5], lstm_hidden=256, lstm_layers=2,
                 num_classes=3, dropout=0.35, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.edrop = nn.Dropout(dropout * 0.6)
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(nn.Conv1d(embed_dim, num_filters, k, padding=k//2))
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in kernel_sizes])
        self.cdrop = nn.Dropout(dropout * 0.6)
        total_filters = num_filters * len(kernel_sizes)
        self.lstm = nn.LSTM(total_filters, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0)
        self.ldrop = nn.Dropout(dropout)
        lstm_out = lstm_hidden * 2
        self.attn = nn.MultiheadAttention(lstm_out, num_heads=4,
                                               dropout=dropout*0.5, batch_first=True)
        self.anorm = nn.LayerNorm(lstm_out)
        self.fc1 = nn.Linear(lstm_out, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fdrop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        B, L = x.shape
        emb = self.emb(x)
        emb = self.edrop(emb)
        emb = emb.permute(0, 2, 1)
        conv_outs = []
        for conv, bn in zip(self.convs, self.bns):
            out = conv(emb)
            out = F.adaptive_max_pool1d(out, L)
            out = bn(out)
            out = F.relu(out)
            conv_outs.append(out)
        conv_cat = torch.cat(conv_outs, dim=1)
        conv_cat = self.cdrop(conv_cat)
        conv_cat = conv_cat.permute(0, 2, 1)
        lstm_out, _ = self.lstm(conv_cat)
        lstm_out = self.ldrop(lstm_out)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        lstm_out = self.anorm(lstm_out + attn_out)
        pooled = F.adaptive_max_pool1d(lstm_out.permute(0, 2, 1), 1).squeeze(-1)
        out = self.fc1(pooled)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fdrop(out)
        return self.fc2(out)


def load_model():
    checkpoint = torch.load(
        os.path.join(MODEL_DIR, "model_export.pt"),
        map_location=DEVICE,
        weights_only=False
    )
    mc = model_config
    model = TextCNN_BiLSTM_Attention(
        vocab_size=mc['vocab_size'], embed_dim=mc['embed_dim'],
        num_filters=mc['num_filters'], kernel_sizes=mc['kernel_sizes'],
        lstm_hidden=mc['lstm_hidden'], lstm_layers=mc['lstm_layers'],
        num_classes=mc['num_classes'], dropout=mc['dropout']
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

model = load_model()

# ======== 预处理 ========
def clean_text(text):
    text = re.sub(r'#\S+#', '', text)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()


def text_to_ids(text):
    """字符级编码 - 与预处理保持一致"""
    text = clean_text(text)
    if not text:
        return None
    chars = list(text)
    ids = [vocab.get(ch, 1) for ch in chars[:MAX_SEQ_LEN]]  # 1 = <UNK>
    pad_len = MAX_SEQ_LEN - len(ids)
    ids = ids + [0] * pad_len
    return np.array([ids], dtype=np.int32)

def predict_single(text):
    """预测单条文本"""
    ids = text_to_ids(text)
    if ids is None:
        return None
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    pred = int(logits.argmax(1).cpu().numpy()[0])
    return {
        "label": pred,
        "label_name": label_map[str(pred)],
        "probabilities": {
            label_map[str(i)]: round(float(probs[i]), 6)
            for i in range(len(probs))
        },
        "confidence": round(float(probs[pred]), 4)
    }

# ======== 路由 ========

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    单条预测
    POST JSON: {"text": "..."}
    """
    data = request.get_json(force=True, silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "缺少 text 字段"}), 400
    text = data['text'].strip()
    if not text:
        return jsonify({"error": "文本为空"}), 400
    result = predict_single(text)
    if result is None:
        return jsonify({"error": "文本解析失败"}), 400
    # 写入数据库
    save_prediction(text, result['label_name'], result['confidence'], result['probabilities'])
    return jsonify({"code": 0, "data": result})

@app.route('/api/predict_batch', methods=['POST'])
def api_predict_batch():
    """
    批量预测
    POST JSON: {"texts": ["...", "..."]}
    """
    data = request.get_json(force=True, silent=True)
    if not data or 'texts' not in data:
        return jsonify({"error": "缺少 texts 字段"}), 400
    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "texts 需为非空列表"}), 400

    ids_list = []
    valid_indices = []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            continue
        ids = text_to_ids(t)
        if ids is not None:
            ids_list.append(ids[0])
            valid_indices.append(i)

    if not ids_list:
        return jsonify({"error": "无有效文本"}), 400

    x = torch.tensor(np.array(ids_list), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

    results = []
    for vi, i in enumerate(valid_indices):
        results.append({
            "index": vi,
            "text": texts[i][:80],
            "label": int(preds[vi]),
            "label_name": label_map[str(int(preds[vi]))],
            "confidence": round(float(probs[vi][preds[vi]]), 4),
            "probabilities": {
                label_map[str(j)]: round(float(probs[vi][j]), 6)
                for j in range(len(probs[vi]))
            }
        })

    # 批量结果写入数据库
    for r, i in zip(results, valid_indices):
        save_prediction(texts[i], r['label_name'], r['confidence'], r['probabilities'])
    return jsonify({"code": 0, "data": {
        "total": len(texts),
        "valid": len(results),
        "results": results
    }})

@app.route('/api/predict_file', methods=['POST'])
def api_predict_file():
    """文件上传预测, 支持 txt/csv"""
    if 'file' not in request.files:
        return jsonify({"error": "缺少 file"}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    filename = secure_filename(f.filename or "upload.txt")
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'txt'
    content = f.read()
    try:
        content = content.decode('utf-8')
    except UnicodeDecodeError:
        content = content.decode('gbk', errors='replace')

    lines = []
    if ext == 'csv':
        import csv, io as stdio
        reader = csv.reader(stdio.StringIO(content))
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                lines.append(row[1].strip())
            elif len(row) >= 1:
                lines.append(row[0].strip())
    else:
        lines = [l.strip() for l in content.split('\n') if l.strip()]

    if not lines:
        return jsonify({"error": "文件无有效内容"}), 400

    results = []
    stats = {label_map[str(i)]: 0 for i in range(3)}
    for text in lines:
        r = predict_single(text)
        if r:
            results.append({"text": text[:80], **r})
            stats[r['label_name']] += 1
            # 写入数据库
            save_prediction(text, r['label_name'], r['confidence'], r['probabilities'])

    return jsonify({"code": 0, "data": {
        "total": len(results),
        "stats": stats,
        "results": results
    }})

@app.route('/api/history', methods=['GET'])
def api_history():
    """获取历史记录列表"""
    conn = get_db()
    if conn is None:
        return jsonify({"code": 0, "data": [], "message": "数据库不可用"})
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, text_content, label, confidence, prob_negative, prob_neutral, prob_positive, created_at "
                "FROM predictions ORDER BY id DESC LIMIT 50"
            )
            rows = cur.fetchall()
        history = []
        for row in rows:
            history.append({
                "id": row['id'],
                "text": row['text_content'],
                "label_name": row['label'],
                "confidence": round(float(row['confidence']), 4),
                "probabilities": {
                    "负面": round(float(row['prob_negative']), 6),
                    "中性": round(float(row['prob_neutral']), 6),
                    "正面": round(float(row['prob_positive']), 6)
                },
                "time": row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if row['created_at'] else ''
            })
        return jsonify({"code": 0, "data": history})
    except Exception as e:
        return jsonify({"code": -1, "error": str(e)}), 500

@app.route('/api/history', methods=['DELETE'])
def api_history_clear():
    """清空历史记录"""
    conn = get_db()
    if conn is None:
        return jsonify({"code": -1, "error": "数据库不可用"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE predictions")
        conn.commit()
        return jsonify({"code": 0, "message": "历史记录已清空"})
    except Exception as e:
        return jsonify({"code": -1, "error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """分类统计查询"""
    conn = get_db()
    if conn is None:
        return jsonify({"code": 0, "data": {"负面": 0, "中性": 0, "正面": 0}})
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT label, COUNT(*) AS cnt FROM predictions GROUP BY label")
            rows = cur.fetchall()
        stats = {"负面": 0, "中性": 0, "正面": 0}
        for row in rows:
            if row['label'] in stats:
                stats[row['label']] = row['cnt']
        stats['总计'] = sum(stats.values())
        return jsonify({"code": 0, "data": stats})
    except Exception as e:
        return jsonify({"code": -1, "error": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """获取模型指标"""
    return jsonify({"code": 0, "data": {
        "test_accuracy": metrics['test_accuracy'],
        "best_val_accuracy": metrics['best_val_accuracy'],
        "num_params": metrics['num_params'],
        "classification_report": metrics['classification_report'],
        "confusion_matrix": metrics['confusion_matrix'],
        "label_map": label_map,
        "model_type": metrics['model_type'],
    }})

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# ======== 启动 ========
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
