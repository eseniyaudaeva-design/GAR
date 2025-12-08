import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import re
from collections import Counter, defaultdict
import math
import concurrent.futures
from urllib.parse import urlparse
import inspect
import time
import json

# ==========================================
# 0. ПАТЧ СОВМЕСТИМОСТИ (Для NLP)
# ==========================================
if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

# ==========================================
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO", page_icon="📊")

# ==========================================
# 2. АВТОРИЗАЦИЯ
# ==========================================
def check_password():
    if st.session_state.get("authenticated"):
        return True
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <style>
            .auth-container {
                display: flex; flex-direction: column; align-items: center;
                justify-content: center; padding: 2rem; background-color: white;
                border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-top: 5rem;
            }
            </style>
            <div class="auth-container">
                <h3>📊 GAR PRO</h3>
                <h3>Вход в систему</h3>
            </div>
        """, unsafe_allow_html=True)
        
        password = st.text_input("Пароль", type="password", key="password_input", label_visibility="collapsed")
        
        if st.button("ВОЙТИ", type="primary", use_container_width=True):
            if password == "jfV6Xel-Q7vp-_s2UYPO":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Неверный пароль")
    return False

if not check_password():
    st.stop()

# ==========================================
# 3. НАСТРОЙКИ API И РЕГИОНОВ
# ==========================================
# Убедитесь, что этот токен актуален!
ARSENKIN_TOKEN = "43acbbb60cb7989c05914ff21be45379"

# Словарь регионов (Название -> {yandex_id, google_id})
REGION_MAP = {
    "Москва": {"ya": 213, "go": 1011969},
    "Санкт-Петербург": {"ya": 2, "go": 1011966},
    "Екатеринбург": {"ya": 54, "go": 1011868},
    "Новосибирск": {"ya": 65, "go": 1011928},
    "Казань": {"ya": 43, "go": 1011904},
    "Нижний Новгород": {"ya": 47, "go": 1011918},
    "Самара": {"ya": 51, "go": 1011956},
    "Челябинск": {"ya": 56, "go": 1011882},
    "Омск": {"ya": 66, "go": 1011931},
    "Краснодар": {"ya": 35, "go": 1011894},
    "Киев (UA)": {"ya": 143, "go": 1012852},
    "Минск (BY)": {"ya": 157, "go": 1001493},
    "Алматы (KZ)": {"ya": 162, "go": 1014601}
}

DEFAULT_EXCLUDE_DOMAINS = [
    "yandex.ru", "avito.ru", "beru.ru", "tiu.ru", "aliexpress.com", "ebay.com",
    "auto.ru", "2gis.ru", "sravni.ru", "toshop.ru", "price.ru", "pandao.ru",
    "instagram.com", "wikipedia.org", "rambler.ru", "hh.ru", "banki.ru", 
    "regmarkets.ru", "zoon.ru", "pulscen.ru", "prodoctorov.ru", "blizko.ru", 
    "domclick.ru", "satom.ru", "quto.ru", "edadeal.ru", "cataloxy.ru", 
    "irr.ru", "onliner.by", "shop.by", "deal.by", "yell.ru", "profi.ru", 
    "irecommend.ru", "otzovik.com", "ozon.ru", "ozon.by", "market.yandex.ru", 
    "youtube.com", "gosuslugi.ru", "dzen.ru", "2gis.by", "wildberries.ru", 
    "rutube.ru", "vk.com", "facebook.com"
]
DEFAULT_EXCLUDE = "\n".join(DEFAULT_EXCLUDE_DOMAINS)
DEFAULT_STOPS = "рублей\nруб\nкупить\nцена\nшт\nсм\nмм\nкг\nкв\nм2\nстр\nул"

# Цвета
PRIMARY_COLOR = "#277EFF"
PRIMARY_DARK = "#1E63C4"
TEXT_COLOR = "#3D4858"
LIGHT_BG_MAIN = "#F1F5F9"
BORDER_COLOR = "#E2E8F0"
HEADER_BG = "#F0F7FF"
ROW_BORDER_COLOR = "#DBEAFE" 

st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .stApp {{ background-color: #FFFFFF !important; color: {TEXT_COLOR} !important; }}
        html, body, p, li, h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; color: {TEXT_COLOR} !important; }}
        .stButton button {{ background-color: {PRIMARY_COLOR} !important; color: white !important; border: none; border-radius: 6px; }}
        .stButton button:hover {{ background-color: {PRIMARY_DARK} !important; }}
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {LIGHT_BG_MAIN} !important; color: {TEXT_COLOR} !important; border: 1px solid {BORDER_COLOR} !important;
        }}
        div[data-testid="stDataFrame"] {{ border: 2px solid {PRIMARY_COLOR} !important; border-radius: 8px !important; }}
        div[data-testid="stDataFrame"] div[role="columnheader"] {{
            background-color: {HEADER_BG} !important; color: {PRIMARY_COLOR} !important; font-weight: 700 !important; border-bottom: 2px solid {PRIMARY_COLOR} !important;
        }}
        div[data-testid="stDataFrame"] div[role="gridcell"] {{
            background-color: #FFFFFF !important; color: {TEXT_COLOR} !important; border-bottom: 1px solid {ROW_BORDER_COLOR} !important;
        }}
        .legend-box {{ padding: 10px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 5px; font-size: 14px; margin-bottom: 10px; }}
        .text-red {{ color: #D32F2F; font-weight: bold; }}
        .text-bold {{ font-weight: 600; }}
        .sort-container {{ background-color: {LIGHT_BG_MAIN}; padding: 10px; border-radius: 8px; margin-bottom: 10px; border: 1px solid {BORDER_COLOR}; }}
        section[data-testid="stSidebar"] {{ background-color: #FFFFFF !important; border-left: 1px solid {BORDER_COLOR} !important; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. ЛОГИКА (БЭКЕНД)
# ==========================================

# Инициализация NLP
try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    USE_NLP = True
except Exception as e:
    morph = None
    USE_NLP = False
    st.sidebar.error(f"Ошибка загрузки NLP: {e}")

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# --- ФУНКЦИЯ РАБОТЫ С API ARSENKIN (ФИНАЛЬНОЕ, ИСПРАВЛЕННОЕ РЕШЕНИЕ) ---
def get_arsenkin_urls(query, engine_type, region_name, depth_val=10):
    url_set = "https://arsenkin.ru/api/tools/set"
    url_check = "https://arsenkin.ru/api/tools/check"  # Для проверки статуса
    url_get = "https://arsenkin.ru/api/tools/get"    # Для получения результата
    
    headers = {
        "Authorization": f"Bearer {ARSENKIN_TOKEN}",
        "Content-type": "application/json"
    }
    
    reg_ids = REGION_MAP.get(region_name, {"ya": 213, "go": 1011969})
    se_params = []
    
    if "Яндекс" in engine_type:
        se_params.append({"type": 2, "region": reg_ids['ya']})
    if "Google" in engine_type:
        se_params.append({"type": 11, "region": reg_ids['go']})
        
    payload = {
        "tools_name": "check-top",
        "data": {
            "queries": [query],
            "is_snippet": False,
            "noreask": True,
            "se": se_params,
            "depth": depth_val
        }
    }
    
    # 1. Постановка задачи
    try:
        r = requests.post(url_set, headers=headers, json=payload, timeout=15)
        resp_json = r.json()
        
        if "error" in resp_json or "task_id" not in resp_json:
            st.error(f"❌ Ошибка API (старт): {resp_json}")
            return []
            
        task_id = resp_json["task_id"]
        st.toast(f"Задача ID {task_id} запущена")
        
    except Exception as e:
        st.error(f"❌ Ошибка сети при постановке задачи: {e}")
        return []
    
    # 2. Ожидание и проверка статуса (через /check)
    status = "process"
    attempts = 0
    max_attempts = 40 # Макс 200 секунд
    
    progress_info = st.empty()
    bar = st.progress(0)
    
    while status == "process" and attempts < max_attempts:
        time.sleep(5) # Ждем 5 сек для обхода 429 ошибки
        attempts += 1
        bar.progress(attempts / max_attempts)
        progress_info.text(f"Ожидание ответа API... ({attempts*5} сек)")
        
        try:
            # Запрашиваем статус
            r_check = requests.post(url_check, headers=headers, json={"task_id": task_id})
            res_check_data = r_check.json()
            
            # Если получили статус 'finish', значит можно запрашивать данные
            if res_check_data.get("status") == "finish":
                status = "done"
                break
            
            # Проверка на лимиты
            if str(res_check_data.get("code")) == "429":
                continue 
                    
        except Exception:
            pass # Игнорируем ошибки при проверке статуса
            
    bar.empty()
    progress_info.empty()
        
    if status != "done":
        st.error(f"⏳ Время вышло. Статус: {res_check_data.get('status', 'Unknown')}")
        st.write("JSON-ответ сервера (если есть):")
        st.json(res_check_data)
        return []
        
    # 3. Получение результата (через /get)
    res_data = {}
    try:
        st.info("Статус 'finish' получен. Запрашиваем финальный результат...")
        r_final = requests.post(url_get, headers=headers, json={"task_id": task_id}, timeout=30)
        res_data = r_final.json()
        
        # Проверяем, что пришел именно результат, а не статус (code должен быть TASK_RESULT)
        if res_data.get("code") != "TASK_RESULT":
            st.error(f"❌ Ошибка: API не вернул финальный результат (TASK_RESULT).")
            st.write("JSON-ответ сервера:")
            st.json(res_data)
            return []
            
    except Exception as e:
        st.error(f"❌ Ошибка сети при получении результата: {e}")
        st.write("JSON-ответ сервера:")
        st.json(res_data)
        return []

    # 4. Парсинг
    urls = []
    try:
        collect = res_data['result']['result']['collect']
        
        def extract_urls(obj):
            found = []
            if isinstance(obj, list):
                for item in obj:
                    found.extend(extract_urls(item))
            elif isinstance(obj, str):
                if obj.startswith('http'):
                    found.append(obj)
            return found
            
        urls = extract_urls(collect)
        
    except Exception:
        st.error("❌ Ошибка чтения и парсинга финального JSON-ответа")
        st.write("JSON, который не удалось разобрать:")
        st.json(res_data) 
        return []
        
    return list(set(urls))


def process_text_detailed(text, settings, n_gram=1):
    if settings['numbers']:
        pattern = r'[а-яА-ЯёЁ0-9a-zA-Z]+' 
    else:
        pattern = r'[а-яА-ЯёЁa-zA-Z]+'
        
    words = re.findall(pattern, text.lower())
    stops = set(w.lower() for w in settings['custom_stops'])
    
    lemmas = []
    forms_map = defaultdict(set)
    
    for w in words:
        if len(w) < 2: continue
        if w in stops: continue
        
        lemma = w
        if USE_NLP and n_gram == 1: 
            p = morph.parse(w)[0]
            if 'PREP' in p.tag or 'CONJ' in p.tag or 'PRCL' in p.tag or 'NPRO' in p.tag: continue
            lemma = p.normal_form
        
        lemmas.append(lemma)
        forms_map[lemma].add(w)
    
    if n_gram > 1:
        ngrams = []
        for i in range(len(lemmas) - n_gram + 1):
            phrase = " ".join(lemmas[i:i+n_gram])
            ngrams.append(phrase)
        return ngrams, {}
        
    return lemmas, forms_map

def parse_page(url, settings):
    headers = {'User-Agent': settings['ua']}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200: return None
        soup = BeautifulSoup(r.text, 'html.parser')
        
        tags_to_remove = ['script', 'style', 'head']
        if settings['noindex']:
            tags_to_remove.extend(['noindex', 'nav', 'footer', 'header', 'aside'])
        
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments: c.extract()
        for t in soup.find_all(tags_to_remove): t.decompose()
            
        anchors_list = [a.get_text(strip=True) for a in soup.find_all('a') if a.get_text(strip=True)]
        anchor_text = " ".join(anchors_list)
        
        extra_text = []
        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): extra_text.append(img['alt'])
            for t in soup.find_all(title=True): extra_text.append(t['title'])
        body_text = soup.get_text(separator=' ') + " " + " ".join(extra_text)
        return {'url': url, 'domain': urlparse(url).netloc, 'body_text': body_text, 'anchor_text': anchor_text}
    except: return None

def calculate_metrics(comp_data, my_data, settings):
    all_forms_map = defaultdict(set)

    # 1. Ваш сайт
    if not my_data or not my_data['body_text']:
        my_lemmas, my_forms, my_anchors, my_len = [], {}, [], 0
    else:
        my_lemmas, my_forms = process_text_detailed(my_data['body_text'], settings)
        my_anchors, _ = process_text_detailed(my_data['anchor_text'], settings)
        my_len = len(my_lemmas)
        for k, v in my_forms.items():
            all_forms_map[k].update(v)
    
    # 2. Конкуренты
    comp_docs = []
    for p in comp_data:
        body, c_forms = process_text_detailed(p['body_text'], settings)
        anchor, _ = process_text_detailed(p['anchor_text'], settings)
        comp_docs.append({'body': body, 'anchor': anchor})
        for k, v in c_forms.items():
            all_forms_map[k].update(v)
    
    if not comp_docs:
        return {"depth": pd.DataFrame(), "hybrid": pd.DataFrame(), "ngrams": pd.DataFrame(), "relevance_top": pd.DataFrame(), "my_score": {"width": 0, "depth": 0}}

    avg_len = np.mean([len(d['body']) for d in comp_docs])
    norm_k = (my_len / avg_len) if (settings['norm'] and my_len > 0 and avg_len > 0) else 1.0
    
    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    vocab = sorted(list(vocab))
    N = len(comp_docs)
    doc_freqs = Counter()
    for d in comp_docs:
        for w in set(d['body']): doc_freqs[w] += 1
        
    table_depth, table_hybrid = [], []
    for word in vocab:
        df = doc_freqs[word]
        if df < 2 and word not in my_lemmas: continue 
        
        my_tf_total = my_lemmas.count(word)        
        my_tf_anchor = my_anchors.count(word)      
        my_tf_text = max(0, my_tf_total - my_tf_anchor) 
        
        forms_set = all_forms_map.get(word, set())
        forms_str = ", ".join(sorted(list(forms_set))) if forms_set else word
        
        c_total_tfs = [d['body'].count(word) for d in comp_docs]
        c_anchor_tfs = [d['anchor'].count(word) for d in comp_docs]
        
        sum_in_top = sum(c_total_tfs)
        mean_total = np.mean(c_total_tfs)
        med_total = np.median(c_total_tfs)
        max_total = np.max(c_total_tfs)
        med_anchor = np.median(c_anchor_tfs)
        
        rec_min = int(round(min(mean_total, med_total) * norm_k))
        rec_max = int(round(max_total * norm_k))
        rec_anchor = int(round(med_anchor * norm_k)) 
        
        diff_total = 0
        if my_tf_total < rec_min: diff_total = rec_min - my_tf_total 
        elif my_tf_total > rec_max: diff_total = rec_max - my_tf_total 
        
        diff_anchor = rec_anchor - my_tf_anchor
        rec_text_min = max(0, rec_min - rec_anchor)
        rec_text_max = max(0, rec_max - rec_anchor)
        diff_text = 0
        if my_tf_text < rec_text_min: diff_text = rec_text_min - my_tf_text
        elif my_tf_text > rec_text_max: diff_text = rec_text_max - my_tf_text

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        idf = max(0.1, idf) 
        spam_percent = 0
        if my_tf_total > rec_max and rec_max > 0:
            spam_percent = round(((my_tf_total - rec_max) / rec_max) * 100, 1)
        elif my_tf_total > 0 and rec_max == 0:
            spam_percent = 100 
        spam_idf = round(spam_percent * idf, 1)
        abs_diff = abs(diff_total)

        if med_total > 0.5 or my_tf_total > 0:
            table_depth.append({
                "Слово": word, "Словоформы": forms_str, "Повторы у вас": my_tf_total,
                "Повторов в ТОПе": sum_in_top, "Минимум (рек)": rec_min, "Максимум (рек)": rec_max,
                "Добавить/Убрать": diff_total, "Тег A у вас": my_tf_anchor, "Тег A (рек)": rec_anchor,
                "Тег A +/-": diff_anchor, "Текст у вас": my_tf_text, "Текст (рек)": rec_text_min,
                "Текст +/-": diff_text, "Переспам %": spam_percent, "Переспам*IDF": spam_idf,
                "diff_abs": abs_diff, "is_missing": (my_tf_total == 0)
            })
            table_hybrid.append({
                "Слово": word, "TF-IDF ТОП": round(med_total * idf, 2), "TF-IDF у вас": round(my_tf_total * idf, 2),
                "Сайтов": df, "Переспам": max_total
            })

    table_ngrams = []
    if comp_docs and my_data:
        try:
            my_bi, _ = process_text_detailed(my_data['body_text'], settings, 2)
            comp_bi = [process_text_detailed(p['body_text'], settings, 2)[0] for p in comp_data]
            all_bi = set(my_bi)
            for c in comp_bi: all_bi.update(c)
            bi_freqs = Counter()
            for c in comp_bi: 
                for b_ in set(c): bi_freqs[b_] += 1
            for bg in all_bi:
                df = bi_freqs[bg]
                if df < 2 and bg not in my_bi: continue
                
                my_c = my_bi.count(bg)
                comp_c = [c.count(bg) for c in comp_bi]
                med_c = np.median(comp_c) if comp_c else 0
                
                rec_ngram = int(round(med_c * norm_k))
                diff_ngram = 0
                if my_c < rec_ngram: diff_ngram = rec_ngram - my_c
                elif my_c > rec_ngram: diff_ngram = rec_ngram - my_c
                
                if med_c > 0 or my_c > 0:
                    table_ngrams.append({
                        "N-грамма": bg, "Сайтов": df, "У вас": my_c,
                        "Медиана (рек)": rec_ngram, "Добавить/Убрать": diff_ngram,
                        "TF-IDF": round(my_c * math.log(N/df if df>0 else 1), 3),
                        "diff_abs": abs(diff_ngram), "is_missing": (my_c == 0)
                    })
        except: pass

    # --- ТОП РЕЛЕВАНТНОСТИ ---
    table_rel = []
    competitor_stats = []
    
    # 1. Сбор конкурентов
    for i, p in enumerate(comp_data):
        p_lemmas, _ = process_text_detailed(p['body_text'], settings)
        relevant_lemmas = [w for w in p_lemmas if w in vocab]
        
        raw_width = len(set(relevant_lemmas))
        raw_depth = len(relevant_lemmas)
        
        competitor_stats.append({
            "domain": p['domain'], "pos": i + 1,
            "raw_w": raw_width, "raw_d": raw_depth
        })
        
    # 2. Максимумы (Эталон)
    max_width_top = max([c['raw_w'] for c in competitor_stats]) if competitor_stats else 1
    max_depth_top = max([c['raw_d'] for c in competitor_stats]) if competitor_stats else 1
    
    # 3. Баллы конкурентов
    for c in competitor_stats:
        score_w = int(round((c['raw_w'] / max_width_top) * 100))
        score_d = int(round((c['raw_d'] / max_depth_top) * 100))
        
        table_rel.append({
            "Домен": c['domain'], "Позиция": c['pos'],
            "Ширина (балл)": score_w, "Глубина (балл)": score_d
        })
        
    # 4. Баллы для ВАШЕГО сайта
    my_relevant = [w for w in my_lemmas if w in vocab]
    my_raw_w = len(set(my_relevant))
    my_raw_d = len(my_relevant)
    
    my_score_w = int(round((my_raw_w / max_width_top) * 100))
    my_score_d = int(round((my_raw_d / max_depth_top) * 100))
    
    # Добавляем ВАШ сайт в таблицу
    if my_data and my_data.get('domain'):
        my_label = f"{my_data['domain']} (Вы)"
    else:
        my_label = "Ваш сайт"
        
    table_rel.append({
        "Домен": my_label, "Позиция": 0, # Позиция 0 чтобы был сверху при сортировке
        "Ширина (балл)": my_score_w, "Глубина (балл)": my_score_d
    })
        
    return {
        "depth": pd.DataFrame(table_depth), "hybrid": pd.DataFrame(table_hybrid),
        "ngrams": pd.DataFrame(table_ngrams), "relevance_top": pd.DataFrame(table_rel),
        "my_score": {"width": my_score_w, "depth": my_score_d}
    }

# ==========================================
# 5. ФУНКЦИЯ ОТОБРАЖЕНИЯ (FINAL)
# ==========================================

def render_paginated_table(df, title_text, key_prefix, default_sort_col=None, use_abs_sort_default=False):
    if df.empty:
        st.info(f"{title_text}: Нет данных.")
        return

    st.markdown(f"### {title_text}")
    
    # БЛОК СОРТИРОВКИ
    if f'{key_prefix}_sort_col' not in st.session_state:
        st.session_state[f'{key_prefix}_sort_col'] = default_sort_col if default_sort_col in df.columns else df.columns[0]
    if f'{key_prefix}_sort_order' not in st.session_state:
        st.session_state[f'{key_prefix}_sort_order'] = "Убывание" 

    with st.container():
        st.markdown("<div class='sort-container'>", unsafe_allow_html=True)
        col_s1, col_s2, col_sp = st.columns([2, 2, 4])
        with col_s1:
            sort_col = st.selectbox(
                "🗂 Сортировать весь список по:", 
                df.columns, 
                key=f"{key_prefix}_sort_box",
                index=list(df.columns).index(st.session_state[f'{key_prefix}_sort_col']) if st.session_state[f'{key_prefix}_sort_col'] in df.columns else 0
            )
            st.session_state[f'{key_prefix}_sort_col'] = sort_col
        with col_s2:
            sort_order = st.radio(
                "Порядок:", 
                ["Убывание", "Возрастание"], 
                horizontal=True,
                key=f"{key_prefix}_order_box",
                index=0 if st.session_state[f'{key_prefix}_sort_order'] == "Убывание" else 1
            )
            st.session_state[f'{key_prefix}_sort_order'] = sort_order
        st.markdown("</div>", unsafe_allow_html=True)

    # СОРТИРОВКА
    ascending = (sort_order == "Возрастание")
    if "Добавить" in sort_col or "+/-" in sort_col:
        df['_temp_sort'] = df[sort_col].abs()
        df = df.sort_values(by='_temp_sort', ascending=ascending).drop(columns=['_temp_sort'])
    else:
        df = df.sort_values(by=sort_col, ascending=ascending)

    # ПАГИНАЦИЯ (20 строк)
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    
    ROWS_PER_PAGE = 20
    if f'{key_prefix}_page' not in st.session_state:
        st.session_state[f'{key_prefix}_page'] = 1
        
    total_rows = len(df)
    total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
    current_page = st.session_state[f'{key_prefix}_page']
    
    if current_page > total_pages: current_page = total_pages
    if current_page < 1: current_page = 1
    
    start_idx = (current_page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE
    
    df_view = df.iloc[start_idx:end_idx]

    # ПОКРАСКА ЯЧЕЕК
    def highlight_rows(row):
        base_style = 'background-color: #FFFFFF; color: #3D4858; border-bottom: 1px solid #DBEAFE;'
        styles = []
        for _ in row:
            if 'is_missing' in row and row['is_missing']:
                styles.append(base_style + 'color: #D32F2F; font-weight: bold;')
            else:
                styles.append(base_style + 'font-weight: 600;')
        return styles
    
    cols_to_hide = ["diff_abs", "is_missing"]
    
    styled_df = df_view.style.apply(highlight_rows, axis=1)
    
    # ВЫВОД ТАБЛИЦЫ
    dynamic_height = (len(df_view) * 35) + 40 
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=dynamic_height, 
        column_config={c: None for c in cols_to_hide}
    )
    
    # КНОПКИ ПЕРЕКЛЮЧЕНИЯ
    c_spacer, c_btn_prev, c_info, c_btn_next = st.columns([6, 1, 1, 1])
    with c_btn_prev:
        if st.button("⬅️", key=f"{key_prefix}_prev", disabled=(current_page <= 1), use_container_width=True):
            st.session_state[f'{key_prefix}_page'] -= 1
            st.rerun()
    with c_info:
        st.markdown(f"<div style='text-align: center; margin-top: 10px; color:{TEXT_COLOR}'><b>{current_page}</b> / {total_pages}</div>", unsafe_allow_html=True)
    with c_btn_next:
        if st.button("➡️", key=f"{key_prefix}_next", disabled=(current_page >= total_pages), use_container_width=True):
            st.session_state[f'{key_prefix}_page'] += 1
            st.rerun()
    st.markdown("---")

# ==========================================
# 6. ИНТЕРФЕЙС
# ==========================================

col_main, col_sidebar = st.columns([65, 35]) 

with col_main:
    st.title("SEO Анализатор Релевантности")

    st.markdown("### URL или код страницы Вашего сайта")
    my_input_type = st.radio("Тип страницы", ["Релевантная страница на вашем сайте", "Исходный код страницы или текст", "Без страницы"], horizontal=True, label_visibility="collapsed", key="my_page_source_radio")

    my_url = ""
    my_page_content = ""
    if my_input_type == "Релевантная страница на вашем сайте":
        my_url = st.text_input("URL страницы", placeholder="https://site.ru/catalog/tovar", label_visibility="collapsed", key="my_url_input")
    elif my_input_type == "Исходный код страницы или текст":
        my_page_content = st.text_area("Исходный код или текст", height=200, label_visibility="collapsed", placeholder="Вставьте HTML", key="my_content_input")

    st.markdown("### Поисковой запрос")
    query = st.text_input("Основной запрос", placeholder="Например: купить пластиковые окна", label_visibility="collapsed", key="query_input")

    st.markdown("### Поиск или URL страниц конкурентов")
    source_type_new = st.radio("Источник конкурентов", ["Поиск через API Arsenkin (TOP-10)", "Список url-адресов ваших конкурентов"], horizontal=True, label_visibility="collapsed", key="competitor_source_radio")
    source_type = "API" if "API" in source_type_new else "Ручной список" 

    if source_type == "Ручной список":
        st.markdown("### Введите список URL")
        st.text_area("Вставьте ссылки здесь (каждая с новой строки)", height=200, key="manual_urls_ui")

    st.markdown("### Редактируемые списки")
    excludes = st.text_area("Не учитывать домены", DEFAULT_EXCLUDE, height=200, key="settings_excludes")
    c_stops = st.text_area("Стоп-слова", DEFAULT_STOPS, height=200, key="settings_stops")

    st.markdown("---")
    
    if st.button("ЗАПУСТИТЬ АНАЛИЗ", type="primary", use_container_width=True, key="start_analysis_btn"):
        for key in list(st.session_state.keys()):
            if key.endswith('_page'): st.session_state[key] = 1
        st.session_state.start_analysis_flag = True

with col_sidebar:
    st.markdown("#####⚙️ Настройки")
    ua = st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0"], key="settings_ua")
    search_engine = st.selectbox("Поисковая система", ["Яндекс", "Google", "Яндекс + Google"], key="settings_search_engine")
    region = st.selectbox("Регион поиска", list(REGION_MAP.keys()), key="settings_region")
    device = st.selectbox("Устройство", ["Desktop", "Mobile"], key="settings_device")
    top_n = st.selectbox("Глубина сбора (ТОП)", [10, 20, 30, 50], index=0, key="settings_top_n")
    
    st.markdown("---")
    st.selectbox("Учитывать тип страниц по url", ["Все страницы", "Главные страницы", "Внутренние страницы"], key="settings_url_type")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.checkbox("Исключать noindex/script", True, key="settings_noindex")
        st.checkbox("Учитывать Alt/Title", False, key="settings_alt")
        st.checkbox("Учитывать числа", False, key="settings_numbers")
    with col_c2:
        st.checkbox("Нормировать по длине", True, key="settings_norm")
        st.checkbox("Исключать агрегаторы", True, key="settings_agg")

# ==========================================
# 7. ВЫПОЛНЕНИЕ
# ==========================================
if st.session_state.get('start_analysis_flag'):
    st.session_state.start_analysis_flag = False

    if my_input_type == "Релевантная страница на вашем сайте" and not st.session_state.get('my_url_input'):
        st.error("Введите URL!")
        st.stop()
    if my_input_type == "Исходный код страницы или текст" and not st.session_state.get('my_content_input', '').strip():
        st.error("Введите исходный код!")
        st.stop()
    if source_type == "API" and not st.session_state.get('query_input'):
        st.error("Введите поисковой запрос!")
        st.stop()
    if source_type == "Ручной список" and not st.session_state.get("manual_urls_ui", "").strip():
        st.error("Введите список URL конкурентов!")
        st.stop()


    settings = {
        'noindex': st.session_state.settings_noindex, 
        'alt_title': st.session_state.settings_alt, 
        'numbers': st.session_state.settings_numbers,
        'norm': st.session_state.settings_norm, 
        'ua': st.session_state.settings_ua, 
        'custom_stops': st.session_state.settings_stops.split()
    }
    
    target_urls = []
    
    if source_type == "API":
        # Логика работы через Arsenkin API
        
        with st.spinner(f"Сбор ТОПа через Arsenkin API..."):
            found_urls = get_arsenkin_urls(
                query=st.session_state.query_input, 
                engine_type=st.session_state.settings_search_engine,
                region_name=st.session_state.settings_region,
                depth_val=st.session_state.settings_top_n
            )
            
        if not found_urls:
            # Здесь ошибка была выведена функцией get_arsenkin_urls
            st.error("API не вернул ссылки. Проверьте **JSON-ответ сервера** (если он выведен выше).")
            st.stop()
            
        # Фильтрация
        excl = [d.strip() for d in st.session_state.settings_excludes.split('\n') if d.strip()]
        if st.session_state.settings_agg: 
            excl.extend(["avito", "ozon", "wildberries", "market", "tiu", "youtube", "vk.com", "yandex.ru"])
            
        for u in found_urls:
            # Исключаем свой сайт
            if my_input_type == "Релевантная страница на вашем сайте" and st.session_state.my_url_input and urlparse(st.session_state.my_url_input).netloc in urlparse(u).netloc: continue
            # Исключаем домены из списка исключений
            if any(x in urlparse(u).netloc for x in excl): continue
            
            target_urls.append(u)
            
        st.info(f"Получено уникальных URL от API: {len(found_urls)}. После фильтрации: {len(target_urls)}")

    else:
        # Ручной режим
        raw_urls = st.session_state.get("manual_urls_ui", "")
        if raw_urls:
            target_urls = [u.strip() for u in raw_urls.split('\n') if u.strip()]
        else:
            target_urls = []

    if not target_urls:
        st.error("Нет конкурентов для анализа после фильтрации. Проверьте список исключаемых доменов.")
        st.stop()
        
    my_data = None
    if my_input_type == "Релевантная страница на вашем сайте":
        with st.spinner("Скачивание вашей страницы..."):
            my_data = parse_page(st.session_state.my_url_input, settings)
            if not my_data:
                st.error("Не удалось скачать вашу страницу. Проверьте URL или настройки User-Agent.")
                st.stop()
    elif my_input_type == "Исходный код страницы или текст":
        my_data = {'url': 'Local', 'domain': 'local', 'body_text': st.session_state.my_content_input, 'anchor_text': ''}

    comp_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(parse_page, u, settings): u for u in target_urls}
        done = 0
        total = len(target_urls)
        prog = st.progress(0)
        stat = st.empty()
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: comp_data.append(res)
            done += 1
            prog.progress(done / total)
            stat.text(f"Скачивание страниц конкурентов: {done}/{total}")
    prog.empty()
    stat.empty()

    if not comp_data:
        st.error("Не удалось скачать контент со страниц конкурентов (возможно, блокировка ботов).")
        st.stop()

    with st.spinner("Анализ данных..."):
        st.session_state.analysis_results = calculate_metrics(comp_data, my_data, settings)
        st.session_state.analysis_done = True
        st.rerun()

if st.session_state.analysis_done and st.session_state.analysis_results:
    results = st.session_state.analysis_results
    st.success("Анализ готов!")
    
    st.markdown(f"""
        <div style='background-color: {LIGHT_BG_MAIN}; padding: 15px; border-radius: 8px; border: 1px solid {BORDER_COLOR}; margin-bottom: 20px;'>
            <h4 style='margin:0; color: {PRIMARY_COLOR};'>Результат вашего сайта (в баллах от 0 до 100)</h4>
            <p style='margin:5px 0 0 0;'>Ширина (охват семантики): <b>{results['my_score']['width']}</b> | Глубина (оптимизация): <b>{results['my_score']['depth']}</b></p>
        </div>
        <div class="legend-box">
            <span class="text-red">Красный</span>: слова, которых нет у вас. <span class="text-bold">Жирный</span>: слова, участвующие в анализе.<br>
            Минимум: min(среднее, медиана). Переспам: % превышения макс. диапазона. <br>
            ℹ️ Для сортировки всего списка используйте меню над таблицей.
        </div>
    """, unsafe_allow_html=True)

    render_paginated_table(results['depth'], "1. Рекомендации по глубине", "tbl_depth_1", default_sort_col="Добавить/Убрать", use_abs_sort_default=True)
    render_paginated_table(results['hybrid'], "3. Гибридный ТОП (TF-IDF)", "tbl_hybrid", default_sort_col="TF-IDF ТОП", use_abs_sort_default=False)
    render_paginated_table(results['ngrams'], "4. N-граммы (Фразы)", "tbl_ngrams", default_sort_col="Добавить/Убрать", use_abs_sort_default=True)
    render_paginated_table(results['relevance_top'], "5. ТОП релевантности (Баллы 0-100)", "tbl_rel", default_sort_col="Ширина (балл)", use_abs_sort_default=False)
