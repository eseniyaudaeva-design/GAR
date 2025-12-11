import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import re
from collections import Counter, defaultdict
import math
import concurrent.futures
from urllib.parse import urlparse, urljoin
import inspect
import time
import json
import io 
import os

# Попытка импорта openai
try:
    import openai
except ImportError:
    openai = None

# ==========================================
# 0. ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ (SESSION STATE)
# ==========================================
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

if 'ai_generated_df' not in st.session_state:
    st.session_state.ai_generated_df = None
if 'ai_excel_bytes' not in st.session_state:
    st.session_state.ai_excel_bytes = None

if 'tags_html_result' not in st.session_state:
    st.session_state.tags_html_result = None

if 'table_html_result' not in st.session_state:
    st.session_state.table_html_result = None

if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

# ==========================================
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ И СПИСКИ
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO", page_icon="📊")

GARBAGE_LATIN_STOPLIST = {
    'whatsapp', 'viber', 'telegram', 'skype', 'vk', 'instagram', 'facebook', 'youtube', 'twitter',
    'cookie', 'cookies', 'policy', 'privacy', 'agreement', 'terms',
    'click', 'submit', 'send', 'zakaz', 'basket', 'cart', 'order', 'call', 'back', 'callback',
    'login', 'logout', 'sign', 'register', 'auth', 'account', 'profile',
    'search', 'menu', 'nav', 'navigation', 'footer', 'header', 'sidebar',
    'img', 'jpg', 'png', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'svg',
    'ok', 'error', 'undefined', 'null', 'true', 'false', 'var', 'let', 'const', 'function', 'return',
    'ru', 'en', 'com', 'net', 'org', 'biz', 'shop', 'store',
    'phone', 'email', 'tel', 'fax', 'mob', 'address', 'copyright', 'all', 'rights', 'reserved',
    'div', 'span', 'class', 'id', 'style', 'script', 'body', 'html', 'head', 'meta', 'link'
}

# ==========================================
# 2. АВТОРИЗАЦИЯ
# ==========================================
def check_password():
    if st.session_state.get("authenticated"):
        return True
    
    st.markdown("""
        <style>
        .main {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .auth-logo-box {
            text-align: center;
            margin-bottom: 1rem;
            padding-top: 0; 
        }
        .login-box h3 {
            margin-top: 0;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-logo-box">', unsafe_allow_html=True)
        try:
            st.image("logo.png", width=250) 
        except Exception:
            st.markdown("<h3 style='color: #D32F2F; font-size: 14px; margin-top: 0;'>LOGO (Не найден)</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("<h3>Вход в систему</h3>", unsafe_allow_html=True)
        
        password = st.text_input("Пароль", type="password", key="password_input", label_visibility="collapsed")
        
        if st.button("ВОЙТИ", type="primary", use_container_width=True):
            if password == "jfV6Xel-Q7vp-_s2UYPO":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Неверный пароль")
        st.markdown('</div>', unsafe_allow_html=True)
        
    return False

if not check_password():
    st.stop()

# ==========================================
# 3. НАСТРОЙКИ API И РЕГИОНОВ
# ==========================================
ARSENKIN_TOKEN = "43acbbb60cb7989c05914ff21be45379"

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
    "irr.ru", "onliner.by", "shop.by", "deal.by", "yell.ru", 
    "profi.ru", "irecommend.ru", "otzovik.com", "ozon.ru", "ozon.by", 
    "market.yandex.ru", "youtube.com", "gosuslugi.ru", "dzen.ru", 
    "2gis.by", "wildberries.ru", "rutube.ru", "vk.com", "facebook.com"
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
        .text-green {{ color: #2E7D32; font-weight: bold; }}
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

# --- ФУНКЦИЯ РАБОТЫ С API ARSENKIN ---
def get_arsenkin_urls(query, engine_type, region_name, depth_val=10):
    url_set = "https://arsenkin.ru/api/tools/set"
    url_check = "https://arsenkin.ru/api/tools/check" 
    url_get = "https://arsenkin.ru/api/tools/get"
    
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
    
    status = "process"
    attempts = 0
    max_attempts = 40
    progress_info = st.empty()
    bar = st.progress(0)
    res_check_data = {}
    
    while status == "process" and attempts < max_attempts:
        time.sleep(5)
        attempts += 1
        bar.progress(attempts / max_attempts)
        progress_info.text(f"Ожидание ответа API... ({attempts*5} сек)")
        
        try:
            r_check = requests.post(url_check, headers=headers, json={"task_id": task_id})
            res_check_data = r_check.json()
            if res_check_data.get("status") == "finish":
                status = "done"
                break
            if str(res_check_data.get("code")) == "429":
                continue 
        except Exception:
            pass
            
    bar.empty()
    progress_info.empty()
        
    if status != "done":
        st.error(f"⏳ Время вышло. Статус: {res_check_data.get('status', 'Unknown')}")
        st.write("JSON-ответ сервера:")
        st.json(res_check_data)
        return []
        
    res_data = {}
    try:
        st.info("Статус 'finish' получен. Запрашиваем результат...")
        r_final = requests.post(url_get, headers=headers, json={"task_id": task_id}, timeout=30)
        res_data = r_final.json()
        if res_data.get("code") != "TASK_RESULT":
            st.error(f"❌ Ошибка: API не вернул финальный результат.")
            st.json(res_data)
            return []
    except Exception as e:
        st.error(f"❌ Ошибка сети при получении результата: {e}")
        st.json(res_data)
        return []

    results_list = []
    try:
        if 'result' in res_data and 'result' in res_data['result'] and 'collect' in res_data['result']['result']:
            collect = res_data['result']['result']['collect']
        else:
            st.error("❌ Ошибка парсинга: Нет поля 'collect'.")
            st.json(res_data)
            return []

        final_url_list = []
        if collect and isinstance(collect, list) and len(collect) > 0 and \
           collect[0] and isinstance(collect[0], list) and len(collect[0]) > 0 and \
           collect[0][0] and isinstance(collect[0][0], list):
            final_url_list = collect[0][0]
        else:
             unique_urls = set()
             for engine_data in collect:
                if isinstance(engine_data, dict):
                    for engine_id, serps in engine_data.items():
                        if isinstance(serps, list):
                            for item in serps:
                                url = item.get('url')
                                pos = item.get('pos')
                                if url and pos:
                                    if url not in unique_urls:
                                        results_list.append({'url': url, 'pos': pos})
                                        unique_urls.add(url)
                                    else:
                                        for res in results_list:
                                            if res['url'] == url and pos < res['pos']:
                                                res['pos'] = pos
             return results_list 

        if final_url_list:
            for index, url in enumerate(final_url_list):
                pos = index + 1
                results_list.append({'url': url, 'pos': pos})
    except Exception as e:
        st.error(f"❌ Ошибка парсинга JSON: {e}")
        st.json(res_data) 
        return []
    return results_list

def process_text_detailed(text, settings, n_gram=1):
    # Приводим к нижнему регистру и меняем 'ё' на 'е' ПЕРЕД всем остальным
    text = text.lower().replace('ё', 'е')
    
    pattern = r'[а-яА-ЯёЁ0-9a-zA-Z]+' 
    words = re.findall(pattern, text)
    
    # Стоп-слова тоже нормализуем (ё -> е)
    stops = set(w.lower().replace('ё', 'е') for w in settings['custom_stops'])
    
    lemmas = []
    forms_map = defaultdict(set)
    
    for w in words:
        if len(w) < 2: continue
        
        if not settings['numbers'] and w.isdigit():
            continue
            
        if w in stops: continue
        
        lemma = w
        if USE_NLP and n_gram == 1: 
            p = morph.parse(w)[0]
            if 'PREP' in p.tag or 'CONJ' in p.tag or 'PRCL' in p.tag or 'NPRO' in p.tag: continue
            
            # !FIX: ПРИНУДИТЕЛЬНАЯ ЗАМЕНА Ё НА Е В ЛЕММЕ
            lemma = p.normal_form.replace('ё', 'е')
        
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
        
        tags_to_remove = []
        
        if settings['noindex']:
            tags_to_remove.append('noindex') 
        
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments: c.extract()
        
        if tags_to_remove:
            for t in soup.find_all(tags_to_remove): t.decompose()
            
        anchors_list = [a.get_text(strip=True) for a in soup.find_all('a') if a.get_text(strip=True)]
        anchor_text = " ".join(anchors_list)
        
        extra_text = []
        
        # Всегда собираем Meta Description и Keywords
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            extra_text.append(meta_desc['content'])
            
        meta_kw = soup.find('meta', attrs={'name': 'keywords'})
        if meta_kw and meta_kw.get('content'):
            extra_text.append(meta_kw['content'])
            
        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): extra_text.append(img['alt'])
            for t in soup.find_all(title=True): extra_text.append(t['title'])
            
        body_text_raw = soup.get_text(separator=' ') + " " + " ".join(extra_text)
        body_text = re.sub(r'\s+', ' ', body_text_raw).strip()
        
        if not body_text:
            return None 

        return {'url': url, 'domain': urlparse(url).netloc, 'body_text': body_text, 'anchor_text': anchor_text}
    except: 
        return None

def calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, original_results):
    all_forms_map = defaultdict(set)
    
    # --- 1. Обработка данных вашего сайта ---
    if not my_data or not my_data.get('body_text'):
        my_lemmas, my_forms, my_anchors, my_len = [], {}, [], 0
    else:
        my_lemmas, my_forms = process_text_detailed(my_data['body_text'], settings)
        my_anchors, _ = process_text_detailed(my_data['anchor_text'], settings)
        my_len = len(my_lemmas)
        for k, v in my_forms.items():
            all_forms_map[k].update(v)

    # --- 2. Обработка конкурентов ---
    comp_data_parsed = [d for d in comp_data_full if d.get('body_text')]
    
    comp_docs = []
    for p in comp_data_parsed:
        body, c_forms = process_text_detailed(p['body_text'], settings)
        anchor, _ = process_text_detailed(p['anchor_text'], settings)
        comp_docs.append({'body': body, 'anchor': anchor})
        for k, v in c_forms.items():
            all_forms_map[k].update(v)
    
    # Заглушка, если нет конкурентов
    if not comp_docs:
        table_rel_fallback = []
        for item in original_results:
            domain = urlparse(item['url']).netloc
            table_rel_fallback.append({
                "Домен": domain, 
                "Позиция": item['pos'],
                "Ширина (балл)": 0, "Глубина (балл)": 0
            })
        
        if my_data and my_data.get('domain'):
            my_label = f"{my_data['domain']} (Вы)"
        else:
            my_label = "Ваш сайт"
        
        table_rel_fallback.append({
            "Домен": my_label, 
            "Позиция": my_serp_pos if my_serp_pos > 0 else len(original_results) + 1,
            "Ширина (балл)": 0, "Глубина (балл)": 0
        })
        
        table_rel_df = pd.DataFrame(table_rel_fallback).sort_values(by='Позиция', ascending=True).reset_index(drop=True)
        return {
            "depth": pd.DataFrame(), 
            "hybrid": pd.DataFrame(), 
            "relevance_top": table_rel_df, 
            "my_score": {"width": 0, "depth": 0}, 
            "missing_semantics_high": [], 
            "missing_semantics_low": []
        }

    # Длины текстов
    c_lens = [len(d['body']) for d in comp_docs]
    median_len = np.median(c_lens)
    
    # AvgL (Средняя длина текста конкурентов) для BM25
    avg_dl = np.mean(c_lens) if c_lens else 0
    
    if median_len > 0 and my_len > 0 and settings['norm']:
        norm_k = my_len / median_len
    else:
        norm_k = 1.0
    
    # Полный словарь
    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    vocab = sorted(list(vocab))
    N = len(comp_docs) 
    
    doc_freqs = Counter()
    for d in comp_docs:
        for w in set(d['body']): doc_freqs[w] += 1
        
    word_counts_per_doc = []
    for d in comp_docs:
        word_counts_per_doc.append(Counter(d['body']))

    # --- ЭТАП 1: TF-IDF (вес слов) ---
    word_idf_map = {}
    for lemma in vocab:
        df = doc_freqs[lemma]
        if df == 0: continue
        # Стандартный IDF
        idf = math.log((N + 1) / (df + 1)) + 1
        word_idf_map[lemma] = idf

    # --- ЭТАП 2: ЯДРО (S_WIDTH_CORE) ---
    S_WIDTH_CORE = set()
    missing_semantics_high = []
    missing_semantics_low = []
    
    my_full_lemmas_set = set(my_lemmas) | set(my_anchors)
    lsi_candidates_weighted = [] # Для таблиц

    # Расчет статистик слов
    for lemma in vocab:
        if lemma in GARBAGE_LATIN_STOPLIST: continue
        
        c_counts = [word_counts_per_doc[i][lemma] for i in range(N)]
        med_val = np.median(c_counts)
        percent = int((doc_freqs[lemma] / N) * 100)
        
        # Получаем вес слова для сортировки
        # Для сортировки используем упрощенный вес: IDF * Median_TF
        # (в таблице Hybrid используется сложный TF-IDF, здесь упростим для скорости сбора ядра)
        weight_simple = word_idf_map.get(lemma, 0) * med_val
        
        if med_val > 0:
            lsi_candidates_weighted.append((lemma, weight_simple))

        # === УСЛОВИЕ ЯДРА (Strict) ===
        is_width_word = False
        if med_val >= 1: 
            S_WIDTH_CORE.add(lemma)
            is_width_word = True
        
        # Списки упущенного
        if lemma not in my_full_lemmas_set:
            if len(lemma) < 2: continue
            if lemma.isdigit(): continue
            
            item = {'word': lemma, 'percent': percent, 'weight': weight_simple}
            
            if is_width_word:
                missing_semantics_high.append(item)
            elif percent >= 30:
                 missing_semantics_low.append(item)

    missing_semantics_high.sort(key=lambda x: x['weight'], reverse=True)
    missing_semantics_low.sort(key=lambda x: x['percent'], reverse=True)
    
    # Для таблицы Depth сортируем по весу
    lsi_candidates_weighted.sort(key=lambda x: x[1], reverse=True)
    S_DEPTH_TOP70 = set([x[0] for x in lsi_candidates_weighted[:70]])

    # --- ЭТАП 3: РАСЧЕТ BM25 (НОВАЯ ЛОГИКА ГЛУБИНЫ) ---
    
    def calculate_bm25_for_doc(doc_tokens, doc_len):
        """
        Считает 'сырой' BM25 документа, суммируя веса только по словам из S_WIDTH_CORE.
        Формула: Sum( IDF * (TF * 2.2) / (TF + 1.2 * (0.25 + 0.75 * L/AvgL)) )
        """
        if avg_dl == 0 or doc_len == 0: return 0
        
        score = 0
        counts = Counter(doc_tokens)
        
        # Знаменатель K, зависящий от длины документа
        # 1.2 * (0.25 + 0.75 * L/AvgL)
        K = 1.2 * (0.25 + 0.75 * (doc_len / avg_dl))
        
        # Суммируем ТОЛЬКО по значимым словам (S_WIDTH_CORE)
        # Если суммировать все, мусор забьет сигнал.
        # Если S_WIDTH_CORE пустой, берем S_DEPTH_TOP70
        target_words = S_WIDTH_CORE if S_WIDTH_CORE else S_DEPTH_TOP70
        
        for word in target_words:
            if word not in counts: continue
            
            tf = counts[word]
            idf = word_idf_map.get(word, 0)
            
            # Формула BM25 (компонента TF)
            # (TF * 2.2) / (TF + K)
            term_weight = (tf * 2.2) / (tf + K)
            
            score += idf * term_weight
            
        return score

    # 3.1. BM25 для конкурентов
    comp_bm25_scores = []
    for i in range(N):
        s = calculate_bm25_for_doc(comp_docs[i]['body'], c_lens[i])
        comp_bm25_scores.append(s)
        
    # 3.2. Медиана BM25 ТОПа
    if comp_bm25_scores:
        median_bm25_top = np.median(comp_bm25_scores)
    else:
        median_bm25_top = 0
        
    # 3.3. Лимит спама (100 баллов)
    # Если медиана 0, ставим заглушку 1, чтобы не делить на 0
    spam_limit = median_bm25_top * 1.25
    if spam_limit == 0: spam_limit = 1 

    # 3.4. BM25 для ВАС
    my_bm25_raw = calculate_bm25_for_doc(my_lemmas, my_len)
    
    # 3.5. Итоговый балл глубины для ВАС
    my_depth_score_final = int(round((my_bm25_raw / spam_limit) * 100))

    # --- ЭТАП 4: ТАБЛИЦЫ ДЕТАЛИЗАЦИИ ---
    table_depth, table_hybrid = [], []
    words_in_range_depth = 0
    total_important_words_depth = 0
    
    for lemma in vocab:
        if lemma in GARBAGE_LATIN_STOPLIST: continue
        
        df = doc_freqs[lemma]
        if df < 2 and lemma not in my_lemmas: continue 
        
        my_tf_count = my_lemmas.count(lemma)        
        forms_set = all_forms_map.get(lemma, set())
        forms_str = ", ".join(sorted(list(forms_set))) if forms_set else lemma
        
        c_counts = [word_counts_per_doc[i][lemma] for i in range(N)]
        
        med_total = np.median(c_counts)
        max_total = np.max(c_counts)
        mean_total = np.mean(c_counts)
        
        base_min = min(mean_total, med_total)
        
        # Округление рекомендаций ВВЕРХ (ceil)
        rec_min = int(math.ceil(base_min * norm_k))
        rec_max = int(round(max_total * norm_k)) # Max можно обычно округлять
        if rec_max < rec_min: rec_max = rec_min # Защита
        
        rec_median = med_total * norm_k 
        
        status = "Норма"
        action_diff = 0
        action_text = "✅"
        
        if my_tf_count < rec_min:
            status = "Недоспам"
            action_diff = int(round(rec_min - my_tf_count))
            if action_diff == 0: action_diff = 1
            action_text = f"+{action_diff}"
        elif my_tf_count > rec_max:
            status = "Переспам"
            action_diff = int(round(my_tf_count - rec_max))
            if action_diff == 0: action_diff = 1
            action_text = f"-{action_diff}"
        
        # Расчет "процентной глубины" для конкретного слова (визуализация в таблице)
        depth_percent = 0
        if rec_median > 0.1:
            depth_percent = int(round((my_tf_count / rec_median) * 100))
        else:
            depth_percent = 0 if my_tf_count == 0 else 100
        depth_percent = min(100, depth_percent)

        # Для таблицы гибрид нужен вес
        weight_hybrid = word_idf_map.get(lemma, 0) * (my_tf_count / my_len if my_len > 0 else 0)

        table_depth.append({
            "Слово": lemma, 
            "Словоформы": forms_str, 
            "Вхождений у вас": my_tf_count,
            "Медиана": round(med_total, 1), 
            "Минимум (рек)": rec_min, 
            "Максимум (рек)": rec_max, 
            "Глубина %": depth_percent,
            "Статус": status,
            "Рекомендация": action_text,
            "is_missing": (status == "Недоспам" and my_tf_count == 0),
            "sort_val": abs(action_diff) if status != "Норма" else 0
        })
        
        table_hybrid.append({
            "Слово": lemma, 
            "TF-IDF ТОП": round(word_idf_map.get(lemma, 0) * (med_total / avg_dl if avg_dl > 0 else 0), 4), 
            "TF-IDF у вас": round(weight_hybrid, 4),
            "Сайтов": df, 
            "Переспам": max_total
        })

    # --- ЭТАП 5: РАСЧЕТ ИТОГОВ (ШИРИНА) ---
    
    total_width_core_count = len(S_WIDTH_CORE)
    
    def calculate_width_score_rule_90(lemmas_set):
        if total_width_core_count == 0: return 0
        intersection_count = len(lemmas_set.intersection(S_WIDTH_CORE))
        ratio = intersection_count / total_width_core_count
        if ratio >= 0.9: return 100
        else: return int(round((ratio / 0.9) * 100))

    table_rel = []
    
    # Конкуренты в таблице
    for i, item in enumerate(original_results):
        url = item['url']
        pos = item['pos']
        domain = urlparse(url).netloc
        
        # Находим данные конкурента
        # Используем индекс i, так как comp_bm25_scores мы строили по порядку (0..N)
        # Но original_results может быть больше N (если часть не скачалась), нужно сопоставлять аккуратно
        
        # Ищем распарсенные данные
        parsed_entry = next((d for d in comp_data_full if d.get('url') == url), None)
        
        width_score_val = 0
        depth_score_val_bm25 = 0 
        
        if parsed_entry and parsed_entry.get('body_text'):
            p_lemmas, _ = process_text_detailed(parsed_entry['body_text'], settings)
            p_set = set(p_lemmas)
            
            # Ширина
            width_score_val = calculate_width_score_rule_90(p_set)
            
            # Глубина (BM25)
            # Нам нужно найти этот документ в comp_docs, чтобы взять его посчитанный BM25
            # Или пересчитать (проще пересчитать, т.к. индексы могут не совпасть при фильтрации)
            c_score_raw = calculate_bm25_for_doc(p_lemmas, len(p_lemmas))
            depth_score_val_bm25 = int(round((c_score_raw / spam_limit) * 100))
                
            width_score_val = min(100, width_score_val)
            # depth_score_val_bm25 НЕ ограничиваем 100, чтобы видеть переспам (140 и т.д.)
            
        table_rel.append({
            "Домен": domain, "Позиция": pos,
            "Ширина (балл)": width_score_val,
            "Глубина (балл)": depth_score_val_bm25
        })
        
    # Вы в таблице
    my_score_w = calculate_width_score_rule_90(my_full_lemmas_set)
    my_score_w = min(100, my_score_w)
    
    if my_data and my_data.get('domain'):
        my_label = f"{my_data['domain']} (Вы)"
    else:
        my_label = "Ваш сайт"
        
    table_rel.append({
        "Домен": my_label, 
        "Позиция": my_serp_pos if my_serp_pos > 0 else len(original_results) + 1,
        "Ширина (балл)": my_score_w, "Глубина (балл)": my_depth_score_final
    })
    
    table_rel_df = pd.DataFrame(table_rel).sort_values(by='Позиция', ascending=True).reset_index(drop=True)
        
    return {
        "depth": pd.DataFrame(table_depth), "hybrid": pd.DataFrame(table_hybrid),
        "relevance_top": table_rel_df,
        "my_score": {"width": my_score_w, "depth": my_depth_score_final},
        "missing_semantics_high": missing_semantics_high,
        "missing_semantics_low": missing_semantics_low
    }
# ==========================================
# 5. ФУНКЦИЯ ОТОБРАЖЕНИЯ (PAGINATION + EXCEL)
# ==========================================

def render_paginated_table(df, title_text, key_prefix, default_sort_col=None, use_abs_sort_default=False):
    if df.empty:
        st.info(f"{title_text}: Нет данных.")
        return

    # Заголовок и кнопка скачивания в одной строке
    col_t1, col_t2 = st.columns([7, 3])
    with col_t1:
        st.markdown(f"### {title_text}")
    
    # 1. Сортировка (до фильтрации)
    if f'{key_prefix}_sort_col' not in st.session_state:
        st.session_state[f'{key_prefix}_sort_col'] = default_sort_col if default_sort_col in df.columns else df.columns[0]
    if f'{key_prefix}_sort_order' not in st.session_state:
        st.session_state[f'{key_prefix}_sort_order'] = "Убывание" 

    # 2. Поиск
    search_query = st.text_input(f"🔍 Поиск ({title_text})", key=f"{key_prefix}_search")
    if search_query:
        mask = df.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        st.warning("Ничего не найдено.")
        return

    # 3. Применение сортировки
    with st.container():
        st.markdown("<div class='sort-container'>", unsafe_allow_html=True)
        col_s1, col_s2, col_sp = st.columns([2, 2, 4])
        with col_s1:
            sort_col = st.selectbox(
                "🗂 Сортировать по:", 
                df_filtered.columns, 
                key=f"{key_prefix}_sort_box",
                index=list(df_filtered.columns).index(st.session_state[f'{key_prefix}_sort_col']) if st.session_state[f'{key_prefix}_sort_col'] in df_filtered.columns else 0
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

    ascending = (sort_order == "Возрастание")
    
    # !FIX: ИСПРАВЛЕНА ЛОГИКА СОРТИРОВКИ
    # Теперь проверяем ТЕКУЩИЙ выбранный столбец (sort_col), а не дефолтный.
    if sort_col == "Рекомендация" and "sort_val" in df_filtered.columns:
         df_filtered = df_filtered.sort_values(by="sort_val", ascending=ascending)
    elif "Добавить" in sort_col or "+/-" in sort_col:
        df_filtered['_temp_sort'] = df_filtered[sort_col].abs()
        df_filtered = df_filtered.sort_values(by='_temp_sort', ascending=ascending).drop(columns=['_temp_sort'])
    else:
        df_filtered = df_filtered.sort_values(by=sort_col, ascending=ascending)

    # Обновление индекса
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered.index = df_filtered.index + 1
    
    # 4. Генерация Excel (СКАЧИВАЕТСЯ ПОЛНАЯ ТАБЛИЦА)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        export_df = df_filtered.copy()
        if "is_missing" in export_df.columns: del export_df["is_missing"]
        if "sort_val" in export_df.columns: del export_df["sort_val"]
        export_df.to_excel(writer, index=False, sheet_name='Data')
    excel_data = buffer.getvalue()
    
    with col_t2:
        st.download_button(
            label="📥 Скачать Excel (Все данные)",
            data=excel_data,
            file_name=f"{key_prefix}_export.xlsx",
            mime="application/vnd.ms-excel",
            key=f"{key_prefix}_down"
        )

    # 5. ПАГИНАЦИЯ (Отображаем по 20 строк)
    ROWS_PER_PAGE = 20
    if f'{key_prefix}_page' not in st.session_state:
        st.session_state[f'{key_prefix}_page'] = 1
        
    total_rows = len(df_filtered)
    total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
    if total_pages == 0: total_pages = 1
    
    current_page = st.session_state[f'{key_prefix}_page']
    if current_page > total_pages: current_page = total_pages
    if current_page < 1: current_page = 1
    st.session_state[f'{key_prefix}_page'] = current_page
    
    start_idx = (current_page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE
    
    df_view = df_filtered.iloc[start_idx:end_idx]

    def highlight_rows(row):
        base_style = 'background-color: #FFFFFF; color: #3D4858; border-bottom: 1px solid #DBEAFE;'
        styles = []
        status = row.get("Статус", "")
        
        for col_name in row.index:
            cell_style = base_style
            if col_name == "Статус":
                if status == "Недоспам":
                    cell_style += "color: #D32F2F; font-weight: bold;" 
                elif status == "Переспам":
                    cell_style += "color: #E65100; font-weight: bold;" 
                elif status == "Норма":
                    cell_style += "color: #2E7D32; font-weight: bold;" 
            styles.append(cell_style)
        return styles
    
    cols_to_hide = ["is_missing", "sort_val"]
    
    styled_df = df_view.style.apply(highlight_rows, axis=1)
    
    # Высота таблицы подстраивается под кол-во строк на странице (макс 20)
    dynamic_height = (len(df_view) * 35) + 40 
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=dynamic_height, 
        column_config={c: None for c in cols_to_hide}
    )
    
    # Кнопки навигации
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
# 6. ЛОГИКА ДЛЯ PERPLEXITY (AI GEN)
# ==========================================

STATIC_DATA_GEN = {
    'IP_PROP4817': "Условия поставки",
    'IP_PROP4818': "Оперативные отгрузки в регионы точно в срок",
    'IP_PROP4819': """<p>Надежная и быстрая доставка заказа в любую точку страны: "Стальметурал" отгружает товар 24 часа в сутки, 7 дней в неделю. Более 4 000 отгрузок в год. При оформлении заказа менеджер предложит вам оптимальный логистический маршрут.</p>""",
    
    'IP_PROP4820': """<p>Наши изделия успешно применяются на некоторых предприятиях Урала, центрального региона, Поволжья, Сибири. Партнеры по логистике предложат доставить заказ самым удобным способом – автомобильным, железнодорожным, даже авиационным транспортом. Для вас разработают транспортную схему под удобный способ получения. Погрузка выполняется полностью с соблюдением особенностей техники безопасности.</p>
<div class="h4">
<h4>Самовывоз</h4>
</div>
<p>Если обычно соглашаетесь самостоятельно забрать товар или даете это право уполномоченным, адрес и время работы склада в своем городе уточняйте у менеджера.</p>
<div class="h4">
<h4>Грузовой транспорт компании</h4>
</div>
<p>Отправим прокат на ваш объект собственным автопарком. Получение в упаковке для безопасной транспортировки, а именно на деревянном поддоне.</p>
<div class="h4">
<h4>Сотрудничаем с ТК</h4>
</div>
<p>Доставка с помощью транспортной компании по России и СНГ. Окончательная цена может измениться, так как ссылается на прайс-лист, который предоставляет контрагент, однако, сравним стоимость логистических служб и выберем лучшую.</p>""",

    'IP_PROP4821': "Оплата и реквизиты для постоянных клиентов:",
    'IP_PROP4822': """<p>Наша компания готова принять любые комфортные виды оплаты для юридических и физических лиц: по счету, наличная и безналичная, наложенный платеж, также возможны предоплата и отсрочка платежа.</p>""",
    
    'IP_PROP4823': """<div class="h4">
        <h3>Примеры возможной оплаты</h3>
</div>
<div class="an-col-12">
        <ul>
                <li style="font-weight: 400;">
                <p>
 <span style="font-weight: 400;">С помощью менеджера в центрах продаж</span>
                </p>
 </li>
        </ul>
        <p>
                 Важно! Цена не является публичной офертой. Приходите в наш офис, чтобы уточнить поступление, получить ответы на почти любой вопрос, согласовать возврат, счет, рассчитать логистику.
        </p>
        <ul>
                <li style="font-weight: 400;">
                <p>
 <span style="font-weight: 400;">На расчетный счет</span>
                </p>
 </li>
        </ul>
        <p>
                 По внутреннему счету в отделении банка или путем перечисления средств через личный кабинет (транзакции защищены, скорость зависит от отделения). Для права подтверждения нужно показать согласие на платежное поручение с отметкой банка.
        </p>
        <ul>
                <li style="font-weight: 400;">
                <p>
 <span style="font-weight: 400;">Наличными или банковской картой при получении</span>
                </p>
 </li>
        </ul>
        <p>
 <span style="font-weight: 400;">Поможем с оплатой: объем имеет значение. Крупным покупателям – деньги можно перевести после приемки товара.</span>
        </p>
        <p>
                 Менеджеры предоставят необходимую информацию.
        </p>
                <p>
                         Заказывайте через прайс-лист:
                </p>
                <p>
 <a class="btn btn-blue" href="/catalog/">Каталог (магазин-меню):</a>
                </p>
        </div>
</div>
 <br>""",

    'IP_PROP4824': "Описание, статьи, поиск, отзывы, новости, акции, журнал, info:",
    'IP_PROP4825': "Можем металлизировать, оцинковать, никелировать, проволочь",
    'IP_PROP4826': "Современный практический подход",
    
    'IP_PROP4834': "Надежность без примесей",
    'IP_PROP4835': "Популярный поставщик",
    'IP_PROP4836': "Качество и характер",
    'IP_PROP4837': "Порядок в ГОСТах"
}

def get_page_data_for_gen(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'utf-8'
    except Exception as e:
        return None, None, f"Ошибка соединения: {e}"
    
    if response.status_code != 200:
        return None, None, f"Ошибка статуса: {response.status_code}"

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Поиск текста описания
    description_div = soup.find('div', class_='description-container')
    base_text = description_div.get_text(separator="\n", strip=True) if description_div else ""
    
    if not base_text:
        # Резервный поиск, если класс другой
        base_text = soup.body.get_text(separator="\n", strip=True)[:5000]

    # Поиск тегов
    tags_container = soup.find(class_='popular-tags-inner')
    tags_data = []
    if tags_container:
        links = tags_container.find_all('a')
        for link in links:
            tag_name = link.get_text(strip=True)
            tag_url = link.get('href')
            if tag_url:
                tag_url = urljoin(url, tag_url)
            tags_data.append({'name': tag_name, 'url': tag_url})
    
    return base_text, tags_data, None

def generate_five_blocks(client, base_text, tag_name, seo_words=None):
    if not base_text: return ["Error: No base text"] * 5

    system_instruction = """
    Ты — профессиональный технический копирайтер и филолог русского языка.
    Твоя задача — написать 5 независимых текстовых блоков HTML.
    ВАЖНО: НЕ используй markdown обертки (```html). Пиши сразу чистый код.
    
    IMPORTANT: Do not include citations, references, or footnotes like [1], [2], [10] in the text.
    """

    keywords_instruction = ""
    if seo_words and len(seo_words) > 0:
        keywords_str = ", ".join(seo_words)
        keywords_instruction = f"""
        [КРИТИЧЕСКИ ВАЖНО: ГРАММАТИКА И СКЛОНЕНИЯ]
        Ниже список слов в ЛЕММАХ (начальной форме), которые нужно употребить: 
        {keywords_str}
        
        ТЫ ОБЯЗАН:
        1. Вставляя слова из списка, ИЗМЕНЯТЬ их окончания, падеж, число и род, чтобы они идеально согласовывались с соседними словами.
           - ПЛОХО: "Мы предлагаем быстрая доставка".
           - ХОРОШО: "Мы предлагаем <b>быструю доставку</b>".
           - ПЛОХО: "формат упаковка для безопасного транспортный раздела".
           - ХОРОШО: "формат <b>упаковки</b> для безопасного <b>транспортного</b> раздела".
        2. Каждое использованное слово из списка (в любой форме) выделяй тегом <b>.
        3. Достаточно 1 употребления каждого слова. Не спамь.
        4. Если слов много — пиши длинный, развернутый текст, добавляй вводные конструкции и полезную информацию, чтобы слова смотрелись органично, а не списком.
        """

    user_prompt = f"""
    ВВОДНЫЕ:
    Товар (Текущий тег): "{tag_name}".
    База знаний: \"\"\"{base_text[:3000]}\"\"\"

    {keywords_instruction}

    ЗАДАЧА:
    Сгенерируй ровно 5 текстовых блоков.

    СТРУКТУРА КАЖДОГО БЛОКА:
    1. Заголовок (<h2> для первого блока - СТРОГО "{tag_name}", без изменений. Для остальных <h3>).
    2. Абзац текста.
    3. Вводная фраза (заканчивается двоеточием).
    4. Список <ul> или <ol> (элементы заканчиваются точкой с запятой, последний точкой).
    5. Заключительный абзац.

    ГЛАВНОЕ ПРАВИЛО: Текст должен звучать естественно для человека. Роботизированные фразы запрещены. Склоняй слова!
    
    NO CITATIONS OR FOOTNOTES LIKE [1].

    ВЫВОД:
    Раздели блоки строго строкой: |||BLOCK_SEP|||
    """

    try:
        response = client.chat.completions.create(
            model="sonar-pro", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content
        
        # ------------------------------------
        # REMOVE CITATIONS (Regex cleaning)
        # ------------------------------------
        content = re.sub(r'\[\d+\]', '', content)
        
        # Чистка от маркдауна
        content = content.replace("```html", "").replace("```", "")
        
        blocks = content.split("|||BLOCK_SEP|||")
        clean_blocks = [b.strip() for b in blocks if b.strip()]
        
        while len(clean_blocks) < 5:
            clean_blocks.append("")
            
        return clean_blocks[:5]

    except Exception as e:
        return [f"API Error: {str(e)}"] * 5

def generate_html_table(client, user_prompt):
    system_instruction = """
    You are an HTML generator.
    Your task is to generate a semantic HTML table based on the user's request.
    
    IMPORTANT: Do not include citations, references, or footnotes like [1], [2] in the table content.
    
    CRITICAL: You MUST apply specific inline CSS styles to the table elements EXACTLY as follows:
    1. For the <table> tag, use: style="border-collapse: collapse; width: 100%; border: 2px solid black;"
    2. For every <th> tag, use: style="border: 2px solid black; padding: 5px;"
    3. For every <td> tag, use: style="border: 2px solid black; padding: 5px;"
    
    Do not use internal <style> blocks. Use only inline styles.
    Output ONLY the HTML code. Do not wrap it in markdown (```html).
    """
    
    try:
        response = client.chat.completions.create(
            model="sonar-pro", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5
        )
        content = response.choices[0].message.content
        
        # ------------------------------------
        # REMOVE CITATIONS (Regex cleaning)
        # ------------------------------------
        content = re.sub(r'\[\d+\]', '', content)
        
        # Чистка на всякий случай
        content = content.replace("```html", "").replace("```", "").strip()
        return content
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# 7. ИНТЕРФЕЙС (TABS)
# ==========================================

# ИСПОЛЬЗУЕМ ВКУЛАДКИ, ЧТОБЫ НЕ ЛОМАТЬ ДИЗАЙН ПЕРВОЙ ЧАСТИ
tab_seo, tab_ai, tab_tags, tab_tables = st.tabs(["📊 SEO Анализ (ГАР)", "🤖 AI Генерация (Perplexity)", "🏷️ Генератор плитки тегов", "🧩 Генератор таблиц"])

# ------------------------------------------
# Вклдака 1: ВЕСЬ СТАРЫЙ КОД (БЕЗ ИЗМЕНЕНИЙ СТРУКТУРЫ)
# ------------------------------------------
with tab_seo:
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
        source_type_new = st.radio("Источник конкурентов", ["Поиск через API Arsenkin (TOP-30)", "Список url-адресов ваших конкурентов"], horizontal=True, label_visibility="collapsed", key="competitor_source_radio")
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
        
        top_n = st.selectbox("Глубина сбора (ТОП)", [10, 20, 30], index=0, key="settings_top_n") 
        
        st.markdown("---")
        st.selectbox("Учитывать тип страниц по url", ["Все страницы", "Главные страницы", "Внутренние страницы"], key="settings_url_type")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.checkbox("Исключать содержимое <noindex>", True, key="settings_noindex")
            st.checkbox("Учитывать Alt/Title", False, key="settings_alt")
            st.checkbox("Учитывать числа", False, key="settings_numbers")
        with col_c2:
            st.checkbox("Нормировать по длине", True, key="settings_norm")
            st.checkbox("Исключать агрегаторы", True, key="settings_agg") 

    # ==========================================
    # ВЫПОЛНЕНИЕ (SEO ЛОГИКА)
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
        
        target_urls_raw = []
        my_data = None
        my_domain = ""
        my_serp_pos = 0 
        
        if my_input_type == "Релевантная страница на вашем сайте":
            with st.spinner("Скачивание вашей страницы..."):
                my_url_input = st.session_state.my_url_input
                my_data = parse_page(my_url_input, settings)
            
                if not my_data:
                    st.error("Не удалось скачать вашу страницу. Проверьте URL или настройки User-Agent.")
                    st.stop()
                my_domain = urlparse(my_url_input).netloc
        elif my_input_type == "Исходный код страницы или текст":
            my_data = {'url': 'Local', 'domain': 'local', 'body_text': st.session_state.my_content_input, 'anchor_text': ''}
            my_domain = "local" 

        if source_type == "API":
            TARGET_COMPETITORS = st.session_state.settings_top_n
            API_FETCH_DEPTH = 30 
            
            with st.spinner(f"Сбор ТОПа (глубина {API_FETCH_DEPTH}) через Arsenkin API..."):
                found_results = get_arsenkin_urls(
                    query=st.session_state.query_input, 
                    engine_type=st.session_state.settings_search_engine,
                    region_name=st.session_state.settings_region,
                    depth_val=API_FETCH_DEPTH
                )
                
            if not found_results:
                st.error("API не вернул ссылки. Проверьте **JSON-ответ сервера**.")
                st.stop()
                
            excl = [d.strip() for d in st.session_state.settings_excludes.split('\n') if d.strip()]
            if st.session_state.settings_agg: 
                excl.extend(["avito.ru", "ozon.ru", "wildberries.ru", "market.yandex.ru", 
                             "tiu.ru", "youtube.com", "vk.com", "yandex.ru", 
                             "leroymerlin.ru", "petrovich.ru"])
                
            filtered_results_all = []
            for result in found_results:
                url = result['url']
                pos = result['pos']
                domain = urlparse(url).netloc
                
                if my_domain and my_domain == domain:
                    if my_serp_pos == 0 or pos < my_serp_pos:
                        my_serp_pos = pos
                    continue 

                if any(x in domain for x in excl): 
                    continue 

                filtered_results_all.append(result)

            target_urls_raw = filtered_results_all[:TARGET_COMPETITORS]
            
            collected_competitors_count = len(target_urls_raw)
            st.info(f"Получено уникальных URL: {len(found_results)}. Выбрано **{collected_competitors_count}** релевантных конкурентов. Ваш сайт в ТОПе: **{'Да (Поз. ' + str(my_serp_pos) + ')' if my_serp_pos > 0 else 'Нет'}**.")

        else:
            raw_urls = st.session_state.get("manual_urls_ui", "")
            if raw_urls:
                urls = [u.strip() for u in raw_urls.split('\n') if u.strip()]
                target_urls_raw = [{'url': u, 'pos': i+1} for i, u in enumerate(urls)]
            else:
                target_urls_raw = []
                
            st.info(f"Загружено **{len(target_urls_raw)}** URL конкурентов вручную.")

        if not target_urls_raw and my_input_type != "Без страницы":
            st.error("Нет конкурентов для анализа после фильтрации.")
            st.stop()
            
        if not my_data and my_input_type != "Без страницы":
            st.error("Отсутствуют данные для вашего сайта.")
            st.stop()

        comp_data_full = []
        urls_to_fetch = [item['url'] for item in target_urls_raw]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(parse_page, u, settings): u for u in urls_to_fetch}
            done = 0
            total = len(urls_to_fetch)
            prog = st.progress(0)
            stat = st.empty()
            
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                if res: 
                    comp_data_full.append(res)
                done += 1
                prog.progress(done / total)
                stat.text(f"Скачивание страниц конкурентов: {done}/{total}")
        prog.empty()
        stat.empty()

        if not comp_data_full:
            st.warning("⚠️ Не удалось скачать контент со страниц конкурентов.")
        
        with st.spinner("Анализ данных..."):
            st.session_state.analysis_results = calculate_metrics(
                comp_data_full, 
                my_data, 
                settings, 
                my_serp_pos, 
                target_urls_raw 
            ) 
            st.session_state.analysis_done = True
            st.rerun()

    # --- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ (ИЗ SESSION STATE) ---
    if st.session_state.analysis_done and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.success("Анализ готов!")
        
        # КАРТОЧКА БАЛЛОВ
        st.markdown(f"""
            <div style='background-color: {LIGHT_BG_MAIN}; padding: 15px; border-radius: 8px; border: 1px solid {BORDER_COLOR}; margin-bottom: 20px;'>
                <h4 style='margin:0; color: {PRIMARY_COLOR};'>Результат вашего сайта (в баллах от 0 до 100)</h4>
                <p style='margin:5px 0 0 0;'>Ширина (охват семантики): <b>{results['my_score']['width']}</b> | Глубина (оптимизация): <b>{results['my_score']['depth']}</b></p>
            </div>
        """, unsafe_allow_html=True)

        # --- ОБНОВЛЕННЫЙ БЛОК: УПУЩЕННАЯ СЕМАНТИКА (TEXT BLOCK STYLE) ---
        high = results.get('missing_semantics_high', [])
        low = results.get('missing_semantics_low', [])
        
        count_total = len(high) + len(low)
        if count_total > 0:
            with st.expander(f"🧩 Упущенная семантика ({count_total} слов) — Нажмите, чтобы развернуть", expanded=False):
                
                # 1. ОСНОВНЫЕ СЛОВА (ВАЖНЫЕ)
                if high:
                    st.markdown("##### ⭐️ Основные связанные слова")
                    
                    words_list_h = [item['word'] for item in high]
                    # Формируем строку через запятую
                    text_cloud_h = ", ".join(words_list_h)
                    
                    st.markdown(
                        f"<div style='background-color:#EBF5FF; padding:15px; border-radius:8px; line-height: 1.6; border: 1px solid #BEE3F8; color: #2C5282; font-size: 14px; margin-bottom: 15px;'>"
                        f"{text_cloud_h}"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                
                # 2. ДОПОЛНИТЕЛЬНЫЕ СЛОВА (ХВОСТ)
                if low:
                    st.markdown("##### 🔹 Дополнительный список связанных слов")
                    st.markdown("Слова, встречающиеся реже, но присутствующие в ТОПе.")
                    
                    words_list_l = [item['word'] for item in low]
                    text_cloud_l = ", ".join(words_list_l)
                    
                    st.markdown(
                        f"<div style='background-color:#F7FAFC; padding:15px; border-radius:8px; line-height: 1.6; border: 1px solid #E2E8F0; color: #4A5568; font-size: 13px;'>"
                        f"{text_cloud_l}"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
        # ----------------------------------------

        st.markdown(f"""
            <div class="legend-box">
                <span class="text-red">Красный</span>: слова, которых нет у вас. <span class="text-bold">Жирный</span>: слова, участвующие в анализе.<br>
                Минимум: min(среднее, медиана). Переспам: % превышения макс. диапазона. <br>
                ℹ️ Для сортировки всего списка используйте меню над таблицей.
            </div>
        """, unsafe_allow_html=True)

        render_paginated_table(results['depth'], "1. Рекомендации по глубине", "tbl_depth_1", default_sort_col="Рекомендация", use_abs_sort_default=True)
        render_paginated_table(results['hybrid'], "3. Гибридный ТОП (TF-IDF)", "tbl_hybrid", default_sort_col="TF-IDF ТОП", use_abs_sort_default=False)
        render_paginated_table(results['relevance_top'], "4. ТОП релевантности (Баллы 0-100)", "tbl_rel", default_sort_col="Ширина (балл)", use_abs_sort_default=False)

# ------------------------------------------
# Вклдака 2: НОВЫЙ МОДУЛЬ (PERPLEXITY)
# ------------------------------------------
with tab_ai:
    st.title("AI Генератор Текстов (Perplexity)")
    st.markdown("Генерация HTML-блоков для подфильтров на основе контента родительской страницы.")

    with st.container():
        st.markdown("### 🔑 Настройки API")
        api_key_input = st.text_input("Введите ваш Perplexity API Key (начинается с pplx-)", type="password", key="pplx_key_input")
        
        st.markdown("### 📥 Ввод данных")
        target_url_gen = st.text_input("URL Страницы (где брать теги/товары)", placeholder="https://site.ru/catalog/category/", key="pplx_url_input")
    
    st.markdown("---")

    # --- ЛОГИКА ГЕНЕРАЦИИ (ПО КНОПКЕ) ---
    col_btn_start, col_btn_reset = st.columns([2,1])
    
    with col_btn_start:
        start_gen = st.button("🚀 Начать генерацию", type="primary", disabled=not api_key_input, key="btn_start_gen")
    
    # Кнопка ручного сброса
    with col_btn_reset:
        if st.button("🔄 Сбросить результат", key="btn_reset_gen"):
            st.session_state.ai_generated_df = None
            st.session_state.ai_excel_bytes = None
            st.rerun()

    if start_gen:
        # АВТОМАТИЧЕСКИЙ СБРОС ПЕРЕД НАЧАЛОМ НОВОЙ ГЕНЕРАЦИИ
        st.session_state.ai_generated_df = None
        st.session_state.ai_excel_bytes = None
        
        if not openai:
            st.error("Библиотека `openai` не установлена! `pip install openai`")
            st.stop()
            
        if not target_url_gen:
            st.error("Введите URL!")
            st.stop()
            
        try:
            client = openai.OpenAI(api_key=api_key_input, base_url="https://api.perplexity.ai")
        except Exception as e:
            st.error(f"Ошибка инициализации клиента: {e}")
            st.stop()

        with st.status("Скачивание данных со страницы...", expanded=True) as status:
            base_text, tags, error = get_page_data_for_gen(target_url_gen)
            
            if error:
                status.update(label="Ошибка!", state="error")
                st.error(error)
                st.stop()
                
            if not tags:
                status.update(label="Теги не найдены!", state="error")
                st.warning("На странице не найден блок `popular-tags-inner` или ссылки в нем.")
                st.stop()
            
            # --- СБОР КЛЮЧЕВЫХ СЛОВ ИЗ ВКЛАДКИ SEO (С ФИЛЬТРАЦИЕЙ) ---
            seo_keywords_list = []
            if st.session_state.analysis_results:
                high_list = st.session_state.analysis_results.get('missing_semantics_high', [])
                if high_list:
                    # 1. Фильтруем мусорные слова (whatsapp, zakaz и т.д.)
                    clean_candidates = []
                    for item in high_list:
                        word = item['word'].lower()
                        # Если слово НЕ в черном списке и длиннее 2 букв (чтобы убрать 'ok', 'pt')
                        if word not in GARBAGE_LATIN_STOPLIST and len(word) > 2:
                            clean_candidates.append(item['word'])
                    
                    # 2. БЕРЕМ ВСЕ, ЧТО ОСТАЛОСЬ (БЕЗ ЛИМИТА)
                    seo_keywords_list = clean_candidates
                    
                    st.info(f"Выбрано {len(seo_keywords_list)} слов для внедрения: {', '.join(seo_keywords_list)}")
                else:
                    st.warning("Список слов 'Ширина' пуст. Генерация пойдет без доп. ключей.")
            
            status.update(label=f"Найдено тегов: {len(tags)}. Начинаем генерацию...", state="running")
            
            all_rows = []
            prog_bar = st.progress(0)
            
            for i, tag in enumerate(tags):
                tag_name = tag['name']
                st.write(f"⏳ Обработка: **{tag_name}** ({i+1}/{len(tags)})")
                
                blocks = generate_five_blocks(client, base_text, tag_name, seo_keywords_list)
                
                row = {
                    'TagName': tag_name,
                    'URL': tag['url'],
                    'IP_PROP4839': blocks[0],
                    'IP_PROP4816': blocks[1],
                    'IP_PROP4838': blocks[2],
                    'IP_PROP4829': blocks[3],
                    'IP_PROP4831': blocks[4],
                    **STATIC_DATA_GEN
                }
                all_rows.append(row)
                prog_bar.progress((i + 1) / len(tags))
                time.sleep(0.5) 
            
            status.update(label="Готово!", state="complete")
            
            # --- СОХРАНЕНИЕ В SESSION STATE ---
            if all_rows:
                df = pd.DataFrame(all_rows)
                cols = [
                    'TagName', 'URL', 
                    'IP_PROP4839', 'IP_PROP4817', 'IP_PROP4818', 'IP_PROP4819', 'IP_PROP4820', 
                    'IP_PROP4821', 'IP_PROP4822', 'IP_PROP4823', 'IP_PROP4824',
                    'IP_PROP4816', 'IP_PROP4825', 'IP_PROP4826', 
                    'IP_PROP4834', 'IP_PROP4835', 'IP_PROP4836', 'IP_PROP4837',
                    'IP_PROP4838', 'IP_PROP4829', 'IP_PROP4831'
                ]
                final_cols = [c for c in cols if c in df.columns]
                df = df[final_cols]
                
                st.session_state.ai_generated_df = df
                
                # Создаем буфер байтов один раз и сохраняем его
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                st.session_state.ai_excel_bytes = buffer.getvalue()
                
                st.rerun() # Перезагрузка, чтобы обновить UI и показать кнопку скачивания

    # --- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТА (ВНЕ БЛОКА КНОПКИ) ---
    if st.session_state.ai_generated_df is not None:
        st.success("✅ Генерация завершена! Данные сохранены.")
        
        st.download_button(
            label="📥 Скачать Excel файл",
            data=st.session_state.ai_excel_bytes,
            file_name="seo_texts_result.xlsx",
            mime="application/vnd.ms-excel"
        )
        
        with st.expander("Просмотр данных (первые 5 строк)"):
            st.dataframe(st.session_state.ai_generated_df.head())

# ------------------------------------------
# Вклдака 3: ГЕНЕРАТОР ПЛИТКИ ТЕГОВ (NEW)
# ------------------------------------------
with tab_tags:
    st.title("🏷️ Генератор плитки тегов")
    st.markdown("Вставьте список ссылок (каждая с новой строки). Скрипт перейдет по ним, заберет название страницы (H1) и сформирует HTML-код плитки.")
    
    urls_input = st.text_area("Список ссылок", height=200, placeholder="https://site.ru/catalog/filter/1/\nhttps://site.ru/catalog/filter/2/", key="tag_urls_input")
    
    if st.button("Сгенерировать плитку", type="primary", key="btn_gen_tags"):
        if not urls_input.strip():
            st.error("Введите ссылки!")
            st.stop()
            
        urls_list = [u.strip() for u in urls_input.split('\n') if u.strip()]
        
        results_tags = []
        
        # Функция парсинга заголовка H1
        def fetch_h1_title(url):
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    r.encoding = 'utf-8' # Force utf-8 usually for Russian sites
                    soup = BeautifulSoup(r.text, 'html.parser')
                    
                    # 1. Пробуем H1
                    h1 = soup.find('h1')
                    if h1:
                        return h1.get_text(strip=True)
                    
                    # 2. Пробуем Title
                    if soup.title:
                        return soup.title.get_text(strip=True)
                        
                return "Нет заголовка"
            except:
                return "Ошибка доступа"

        # Многопоточный сбор
        with st.status("Сбор заголовков...", expanded=True) as status:
            progress_bar = st.progress(0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {executor.submit(fetch_h1_title, url): url for url in urls_list}
                
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        name = future.result()
                        results_tags.append({'url': url, 'name': name})
                    except Exception as exc:
                        results_tags.append({'url': url, 'name': "Ошибка"})
                    
                    completed_count += 1
                    progress_bar.progress(completed_count / len(urls_list))
            
            status.update(label="Готово!", state="complete")
        
        # Генерация HTML
        if results_tags:
            html_output = '<div class="popular-tags-text">\n<div class="popular-tags-inner-text">\n<div class="tag-items">\n'
            
            for item in results_tags:
                # Вставляем данные в шаблон
                html_output += f'<a href="{item["url"]}" class="tag-item">{item["name"]}</a>\n'
                
            html_output += '</div>\n</div>\n</div>'
            
            # Сохраняем в сессию
            st.session_state.tags_html_result = html_output
            st.rerun() # Обновляем страницу

    # Отображаем результат, если он есть в сессии
    if st.session_state.tags_html_result:
        st.success("HTML код сгенерирован:")
        st.code(st.session_state.tags_html_result, language='html')
        if st.button("Сбросить", key="reset_tags"):
            st.session_state.tags_html_result = None
            st.rerun()

# ------------------------------------------
# Вклдака 4: ГЕНЕРАТОР ТАБЛИЦ (NEW)
# ------------------------------------------
with tab_tables:
    st.title("🧩 Генератор HTML таблиц")
    st.markdown("Введите запрос, и ИИ создаст таблицу с жестко заданным оформлением (черные рамки, отступы).")
    
    # Повторяем ввод ключа здесь, чтобы не бегать между вкладками
    pplx_key_table = st.text_input("Perplexity API Key", type="password", key="pplx_key_table")
    
    table_prompt = st.text_area("Опишите, какую таблицу нужно создать", height=150, placeholder="Сравнительная таблица видов труб из ПВХ с характеристиками и применением")
    
    if st.button("Сгенерировать таблицу", type="primary", key="btn_gen_table"):
        if not pplx_key_table:
            st.error("Введите API ключ!")
            st.stop()
        if not table_prompt:
            st.error("Введите описание таблицы!")
            st.stop()
            
        if not openai:
            st.error("Библиотека `openai` не установлена.")
            st.stop()
            
        try:
            client_table = openai.OpenAI(api_key=pplx_key_table, base_url="https://api.perplexity.ai")
            
            with st.spinner("Генерация таблицы..."):
                html_result = generate_html_table(client_table, table_prompt)
            
            # Сохраняем в сессию
            st.session_state.table_html_result = html_result
            st.rerun()

        except Exception as e:
            st.error(f"Ошибка API: {e}")

    # Отображаем результат, если он есть в сессии
    if st.session_state.table_html_result:
        st.success("Готово!")
        st.code(st.session_state.table_html_result, language='html')
        
        st.markdown("### Предпросмотр (примерный):")
        st.markdown(st.session_state.table_html_result, unsafe_allow_html=True)
        
        if st.button("Сбросить", key="reset_table"):
            st.session_state.table_html_result = None
            st.rerun()







