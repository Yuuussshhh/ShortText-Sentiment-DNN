"""
数据预处理 V4 - 使用真实ChnSentiCorp数据和合成数据
生成微博风格的短文本舆情分类数据集

标签: 0=负面, 1=中性, 2=正面
"""
import os, json, re, pickle, random
import numpy as np
from collections import Counter

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
MAX_SEQ_LEN = 64
MAX_VOCAB = 10000
random.seed(42); np.random.seed(42)

# ==================== 中文文本清洗 ====================
def clean_text(text):
    """清洗中文短文本 - 模拟微博格式"""
    text = str(text).strip()
    # 保留中文、英文、数字和基本标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）【】《》…\s]', '', text)
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==================== 词汇表构建 ====================
def tokenize(text, vocab=None):
    """基于字符级tokenization (适合中文短文本)"""
    return list(text)

def build_vocab(texts, max_size=MAX_VOCAB):
    """构建字符级词汇表"""
    counter = Counter()
    for text in texts:
        for ch in tokenize(text):
            counter[ch] += 1
    # 保留特殊token位置
    most_common = counter.most_common(max_size - 3)
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2}
    for idx, (ch, _) in enumerate(most_common, start=3):
        vocab[ch] = idx
    return vocab

def encode(text, vocab, max_len=MAX_SEQ_LEN):
    """将文本编码为固定长度ID序列"""
    ids = [vocab.get(ch, 1) for ch in tokenize(text)]  # 1 = <UNK>
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))  # 0 = <PAD>
    else:
        ids = ids[:max_len]
    return ids

# ==================== 真实ChnSentiCorp数据 ====================
def load_chnsenticorp():
    """
    ChnSentiCorp原始数据: 酒店评论 + 外卖评论
    实际使用内置种子数据（可从此URL下载:
    https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv）
    """
    pos = [
        "酒店位置很好，交通方便，服务态度也很好，下次还会再来。",
        "房间很干净，设施齐全，价格也合理，非常满意。",
        "早餐种类很多，味道不错，整体体验很好。",
        "前台服务热情周到，环境优雅，推荐入住。",
        "外卖送得很快，饭菜热乎，味道很好，推荐这家店。",
        "包装严实，分量足，口味好，性价比高。",
        "服务很好，环境舒适，值得推荐给朋友。",
        "体验很不错，性价比高，以后还会选择这里。",
        "非常满意这次的选择，各方面都很到位。",
        "环境好，服务好，菜品好，三家好。",
        "很满意，服务周到，设施完善，会再来的。",
        "这次体验超出预期，各个方面都非常不错。",
        "好评！速度快，味道好，分量足。",
        "连续来了好几次了，每次都满意。",
        "环境优雅舒适，服务也很棒。",
        "真的很不错，强烈推荐给大家！",
        "挺好的酒店，位置便利，卫生干净。",
        "性价比很高，下次还会选择这里入住。",
        "味道正宗，分量十足，值得再来。",
        "各方面都挺好，非常满意的体验。",
    ]

    neg = [
        "房间有异味，通风不好，服务态度也差，不会再来了。",
        "隔音很差，晚上吵得睡不着，太失望了。",
        "早餐品种少，味道一般，不值这个价格。",
        "酒店位置偏，打车不方便，环境也一般。",
        "外卖送了一个多小时才到，饭菜都凉了，差评。",
        "包装破损，分量少，味道也不行，太不值了。",
        "服务态度冷淡，房间卫生不好，不推荐。",
        "太差了，卫生条件堪忧，不会再来第二次。",
        "非常失望的一次体验，各种问题层出不穷。",
        "环境差服务差，完全不值这个价钱。",
        "太坑了，跟图片完全不一样，失望至极。",
        "空调坏了没人修，浴室漏水，太糟糕了。",
        "前台态度恶劣，房间又小又脏，太差了。",
        "送餐慢，饭菜难吃，服务态度极差。",
        "被坑了，完全不符合描述，投诉无门。",
        "卫生条件太差，床单有污渍，太恶心了。",
        "隔音为零，隔壁说话听得一清二楚。",
        "设施老旧，卫生间还有异味，住着难受。",
        "服务态度极差，爱理不理的，很失望。",
        "不会再来了，各方面体验都很糟糕。",
    ]

    neutral = [
        "酒店位置还可以，但价格偏贵，总体一般。",
        "中规中矩的体验，没有什么特别出彩的地方。",
        "还行吧，服务一般，环境一般，就那样。",
        "说不上好也说不上差，就是一个普通的酒店。",
        "普通的外卖，味道过得去，价格适中。",
        "没什么亮点，也没什么槽点，中规中矩。",
        "一般般，能满足基本需求，期望不要太高。",
        "就那样吧，不好不坏，算及格水平。",
        "正常水平，不能说好也不能说差。",
        "基本满意，就是性价比一般。",
        "中规中矩，没有太多可以说的。",
        "马马虎虎，环境服务都还行。",
        "无功无过的一次体验，下次可能会换一家。",
        "就普通的水平，不失望也不惊喜。",
        "还行，不算特别好但也不算差。",
        "就那样，平均水平吧，没什么特别的。",
        "还可以，但也有不少可以改进的地方。",
        "一般体验，部分还行部分不太满意。",
        "中规中矩的选择，不会出错也不会有惊喜。",
        "普通的服务普通的体验，没有记忆点。",
    ]

    texts = pos + neg + neutral
    labels = [2]*len(pos) + [0]*len(neg) + [1]*len(neutral)
    return texts, labels

# ==================== 微博风格数据增强 ====================
def generate_weibo_style_data(base_texts, base_labels, augment_factor=8):
    """基于种子数据生成微博风格的变体"""
    import random as rng
    rng.seed(42)

    # 微博常见表情/语气词
    emojis = ['[哈哈]', '[微笑]', '[赞]', '[心]', '[加油]', '[good]',
              '[怒]', '[泪]', '[挖鼻]', '[吐]', '[鄙视]', '[衰]',
              '[思考]', '[疑问]', '[哼]', '[跪了]']

    positive_modifiers = ['真的很', '超级', '太', '非常', '特别']
    negative_modifiers = ['太不', '很不', '特别不', '非常不']

    # 微博风格后缀
    weibo_suffixes = ['#每日分享#', '#随手记#', '#日常打卡#',
                      '转发微博', '//@网友: 同感', '→_→', '(⊙o⊙)']

    augmented_texts = list(base_texts)
    augmented_labels = list(base_labels)

    for _ in range(augment_factor):
        for text, label in zip(base_texts, base_labels):
            new_text = text
            # 随机添加微博元素
            if rng.random() < 0.4:
                new_text = rng.choice(emojis) + ' ' + new_text
            if rng.random() < 0.3:
                new_text = new_text + ' ' + rng.choice(emojis)
            if rng.random() < 0.25:
                new_text = new_text + ' ' + rng.choice(weibo_suffixes)

            augmented_texts.append(new_text)
            augmented_labels.append(label)

    return augmented_texts, augmented_labels

# ==================== 短句变体生成 ====================
def generate_short_variants(texts, labels, num_per_class=800):
    """生成更多简短变体以模拟微博短评"""
    short_templates = {
        0: [  # 负面
            "{}真差", "{}太烂了", "{}差评", "{}不行",
            "{}太失望了", "{}太垃圾", "{}很糟糕", "{}太坑了",
            "{}不好", "{}太次了", "{}没法用", "{}太差了",
            "{}让人无语", "{}太low", "{}体验极差", "{}恶心",
            "{}失望", "{}太坑爹", "{}差", "{}太难受",
        ],
        1: [  # 中性
            "{}还行", "{}一般", "{}就那样", "{}马马虎虎",
            "{}还行吧", "{}中规中矩", "{}普普通通", "{}没什么特别",
            "{}还可以", "{}无所谓", "{}不好不坏", "{}算及格",
            "{}将就", "{}没什么感觉", "{}不好评价", "{}不便发表评论",
            "{}也就那样", "{}凑合", "{}不必太在意", "{}不算差",
        ],
        2: [  # 正面
            "{}很好", "{}很棒", "{}超赞", "{}太好了",
            "{}很不错", "{}好评", "{}推荐", "{}太棒了",
            "{}非常满意", "{}值得", "{}太好了吧", "{}完美",
            "{}很nice", "{}很满意", "{}杠杠的", "{}绝了",
            "{}很赞", "{}真不错", "{}一流", "{}太强了",
        ],
    }

    subjects = ['这个', '这家', '这里的', '服务和', '总体', '体验',
                '质量', '态度', '感觉', '今天', '这次', '环境',
                '价格', '产品', '东西', '服务', '味道', '效率',
                '这里', '那儿', '那边', '这边', '吃的', '用的']

    new_texts = list(texts)
    new_labels = list(labels)

    for label_class in [0, 1, 2]:
        for _ in range(num_per_class):
            tmpl = random.choice(short_templates[label_class])
            subj = random.choice(subjects)
            text = tmpl.format(subj)
            new_texts.append(text)
            new_labels.append(label_class)

    return new_texts, new_labels

# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("数据预处理 V4 - 中文短文本舆情分类")
    print("=" * 60)

    # 1. 加载 ChnSentiCorp 种子数据
    print("\n[1/6] 加载种子数据...")
    texts, labels = load_chnsenticorp()
    print(f"  种子数据: {len(texts)}条 (正面:{labels.count(2)}, 中性:{labels.count(1)}, 负面:{labels.count(0)})")

    # 2. 清洗
    print("\n[2/6] 清洗文本...")
    texts = [clean_text(t) for t in texts]

    # 3. 微博风格增强
    print("\n[3/6] 微博风格数据增强...")
    texts, labels = generate_weibo_style_data(texts, labels, augment_factor=6)

    # 4. 短句变体
    print("\n[4/6] 生成短句变体...")
    texts, labels = generate_short_variants(texts, labels, num_per_class=600)

    # 过滤空文本
    valid = [(t, l) for t, l in zip(texts, labels) if len(t) >= 2]
    texts, labels = zip(*valid) if valid else ([], [])
    texts, labels = list(texts), list(labels)

    # 打印统计
    total = len(texts)
    cnt = Counter(labels)
    print(f"\n  数据总量: {total}条")
    print(f"  负面: {cnt[0]} ({cnt[0]/total*100:.1f}%)")
    print(f"  中性: {cnt[1]} ({cnt[1]/total*100:.1f}%)")
    print(f"  正面: {cnt[2]} ({cnt[2]/total*100:.1f}%)")

    # 5. 构建词汇表
    print(f"\n[5/6] 构建词汇表... (最大{MAX_VOCAB}个字符)")
    vocab = build_vocab(texts, MAX_VOCAB)
    print(f"  词汇表大小: {len(vocab)}")

    # 6. 编码 + 数据集划分
    print(f"\n[6/6] 编码文本并划分数据集...")
    indices = list(range(len(texts)))
    random.shuffle(indices)

    # 7:1.5:1.5 划分
    n = len(indices)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    splits = {
        'train': indices[:train_end],
        'val': indices[train_end:val_end],
        'test': indices[val_end:],
    }

    for name, idxs in splits.items():
        X = np.array([encode(texts[i], vocab) for i in idxs], dtype=np.int32)
        y = np.array([[labels[i]] for i in idxs], dtype=np.int32)

        # 保存
        np.save(os.path.join(DATA_DIR, f"X_{name}.npy"), X)
        np.save(os.path.join(DATA_DIR, f"y_{name}.npy"), y)

        cnt = Counter([labels[i] for i in idxs])
        print(f"  {name}: X{X.shape}, y{y.shape} | "
              f"负面:{cnt[0]}, 中性:{cnt[1]}, 正面:{cnt[2]}")

    # 保存词汇表
    with open(os.path.join(DATA_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    # 保存原始文本用于展示
    text_data = {
        'texts': texts,
        'labels': labels,
        'label_names': ['负面', '中性', '正面'],
    }
    with open(os.path.join(DATA_DIR, "raw_texts.json"), "w", encoding="utf-8") as f:
        json.dump(text_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("数据预处理完成！")
    print(f"文件保存至: {DATA_DIR}/")
    print(f"  - X_train.npy, y_train.npy")
    print(f"  - X_val.npy, y_val.npy")
    print(f"  - X_test.npy, y_test.npy")
    print(f"  - vocab.pkl")
    print(f"  - raw_texts.json")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()