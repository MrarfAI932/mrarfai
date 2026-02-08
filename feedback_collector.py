"""MRARFAI 用户反馈收集器 v1.0
Week 5-6 交付物：收集禾苗内部试用反馈
"""
import json
import os
from datetime import datetime

FEEDBACK_DIR = os.environ.get('FEEDBACK_DIR', './feedback')

def _ensure_dir():
    os.makedirs(FEEDBACK_DIR, exist_ok=True)

def save_feedback(rating: int, useful_tabs: list, pain_points: str, suggestions: str, user_name: str = "匿名"):
    """保存一条用户反馈"""
    _ensure_dir()
    entry = {
        'timestamp': datetime.now().isoformat(),
        'user': user_name,
        'rating': rating,
        'useful_tabs': useful_tabs,
        'pain_points': pain_points,
        'suggestions': suggestions,
    }
    path = os.path.join(FEEDBACK_DIR, f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    return path

def load_all_feedback() -> list:
    """加载所有反馈（管理员用）"""
    _ensure_dir()
    entries = []
    for fname in sorted(os.listdir(FEEDBACK_DIR)):
        if fname.endswith('.json'):
            with open(os.path.join(FEEDBACK_DIR, fname), 'r', encoding='utf-8') as f:
                entries.append(json.load(f))
    return entries

def get_summary() -> dict:
    """反馈摘要统计"""
    entries = load_all_feedback()
    if not entries:
        return {'count': 0, 'avg_rating': 0, 'tab_votes': {}, 'latest': None}
    
    ratings = [e['rating'] for e in entries]
    tab_votes = {}
    for e in entries:
        for t in e.get('useful_tabs', []):
            tab_votes[t] = tab_votes.get(t, 0) + 1
    
    return {
        'count': len(entries),
        'avg_rating': sum(ratings) / len(ratings),
        'tab_votes': dict(sorted(tab_votes.items(), key=lambda x: x[1], reverse=True)),
        'latest': entries[-1],
    }

def log_usage(action: str, detail: str = ""):
    """记录使用行为（匿名）"""
    _ensure_dir()
    log_path = os.path.join(FEEDBACK_DIR, 'usage_log.jsonl')
    entry = {'ts': datetime.now().isoformat(), 'action': action, 'detail': detail}
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
