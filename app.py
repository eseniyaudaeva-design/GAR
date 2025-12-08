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
import os # Добавлено для совместимости, если не было

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
                border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-top: 5rem;
            }
            </style>
            <div class="auth-container">
                <h3>📊 GAR PRO</h3>
                <h3>Вход в систему</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Используем отдельный ключ для предотвращения конфликтов с другими инпутами
        if 'password_input_auth' not in st.session_state:
            st.session_state.password_input_auth = ""
            
        password = st.text_input("Пароль", type="password", key="password_input_auth", label_visibility="collapsed")
        
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
    "rutube.ru", "vk.com", "facebook.com", "lemanapro.ru" # <-- ДОБАВЛЕНО ПО ЗАПРОСУ
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
        html, body, p, li, h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR} !important; }}
        .stButton button {{ background-color: {PRIMARY_COLOR} !important; color: white !important;
        border: none; border-radius: 6px; }}
        .stButton button:hover {{ background-color: {PRIMARY_DARK} !important;
        }}
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {LIGHT_BG_MAIN} !important;
            color: {TEXT_COLOR} !important; border: 1px solid {BORDER_COLOR} !important;
        }}
        div[data-testid="stDataFrame"] {{ border: 2px solid {PRIMARY_COLOR} !important;
        border-radius: 8px !important; }}
        div[data-testid="stDataFrame"] div[role="columnheader"] {{
            background-color: {HEADER_BG} !important;
            color: {PRIMARY_COLOR} !important; font-weight: 700 !important; border-bottom: 2px solid {PRIMARY_COLOR} !important;
        }}
        div[data-testid="stDataFrame"] div[role="gridcell"] {{
            background-color: #FFFFFF !important;
            color: {TEXT_COLOR} !important; border-bottom: 1px solid {ROW_BORDER_COLOR} !important;
        }}
        .legend-box {{ padding: 10px;
        background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 5px; font-size: 14px; margin-bottom: 10px;
        }}
        .text-red {{ color: #D32F2F; font-weight: bold;
        }}
        .text-bold {{ font-weight: 600;
        }}
        .sort-container {{ background-color: {LIGHT_BG_MAIN}; padding: 10px; border-radius: 8px; margin-bottom: 10px;
        border: 1px solid {BORDER_COLOR}; }}
        section[data-testid="stSidebar"] {{ background-color: #FFFFFF !important;
        border-left: 1px solid {BORDER_COLOR} !important; }}
        
        /* Стиль для истории проверок - выделение (Вкладка) */
        .stTabs [data-baseweb="tab-list"] button:nth-child(2) {{
            background-color: #ffe0b2; /* Светло-оранжевый фон */
            font-weight: bold;
        }}
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
    # st.sidebar.error(f"Ошибка загрузки NLP: {e}") # Убрано, чтобы не ломать интерфейс, если нет морфологии

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'history' not in st.session_state: # <-- Инициализация истории
    st.session_state.history = []
if 'comp_table_data' not in st.session_state: # <-- Инициализация данных для таблицы статуса
    st.session_state.comp_table_data = []


# --- ФУНКЦИЯ РАБОТЫ С API ARSENKIN (Оставлено без изменений) ---
def get_arsenkin_urls(query, engine_type, region_name, depth_val=10):
    url_set = "https://arsenkin.ru/api/tools/set"
    url_check = "https://arsenkin.ru/api/tools/check" 
    url_get = "https://arsenkin.ru/api/tools/get"    
    
    # ... (Остальной код get_arsenkin_urls остается без изменений) ...
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
        st.write("JSON-ответ сервера (если есть):")
        st.json(res_check_data)
        return []
        
    # 3. Получение результата (через /get)
    res_data = {}
    try:
        st.info("Статус 'finish' получен. Запрашиваем финальный результат...")
        r_final = requests.post(url_get, headers=headers, json={"task_id": task_id}, timeout=30)
        res_data = r_final.json()
      
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

    # 4. ФИНАЛЬНЫЙ ПАРСИНГ: 
    results_list = []
    try:
        if 'result' in res_data and 'result' in res_data['result'] and 'collect' in res_data['result']['result']:
            collect = res_data['result']['result']['collect']
        else:
            st.error("❌ Ошибка парсинга: Отсутствует поле 'collect' в ответе API.")
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
        st.error(f"❌ Критическая ошибка чтения и парсинга финального JSON-ответа: {e}")
        st.write("JSON, который не удалось разобрать:")
        st.json(res_data) 
        return []
        
    return results_list


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

# --- УСИЛЕННЫЙ ПАРСИНГ (Запрос 1) ---
def parse_page_robust(url, settings, retries=3, timeout=30):
    """Скачивает контент страницы с повторными попытками."""
    headers = {'User-Agent': settings['ua']}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status() # Вызывает HTTPError, если статус 4xx или 5xx
            
            # Если 200 OK, пытаемся парсить
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
                
            body_text_raw = soup.get_text(separator=' ') + " " + " ".join(extra_text)
            body_text = re.sub(r'\s+', ' ', body_text_raw).strip()
            
            if not body_text:
                return {'url': url, 'domain': urlparse(url).netloc, 'body_text': '', 'anchor_text': '', 'error': 'Нет текста после очистки'}
            
            return {'url': url, 'domain': urlparse(url).netloc, 'body_text': body_text, 'anchor_text': anchor_text, 'error': None}
        
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Ошибка: {e.response.status_code}"
            if attempt == retries - 1:
                return {'url': url, 'domain': urlparse(url).netloc, 'body_text': '', 'anchor_text': '', 'error': error_msg}
        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка сети/таймаут: {e.__class__.__name__}"
            if attempt == retries - 1:
                return {'url': url, 'domain': urlparse(url).netloc, 'body_text': '', 'anchor_text': '', 'error': error_msg}
        except Exception as e:
            error_msg = f"Неизвестная ошибка: {e}"
            if attempt == retries - 1:
                return {'url': url, 'domain': urlparse(url).netloc, 'body_text': '', 'anchor_text': '', 'error': error_msg}
        
        time.sleep(2 ** attempt) 
    
    return {'url': url, 'domain': urlparse(url).netloc, 'body_text': '', 'anchor_text': '', 'error': 'Неизвестная ошибка после всех попыток'}

# Оригинальная функция parse_page удалена/заменена на parse_page_robust

def calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, original_results):
    all_forms_map = defaultdict(set)
    
    # 1. Ваш сайт
    if not my_data or not my_data.get('body_text'):
        my_lemmas, my_forms, my_anchors, my_len = [], {}, [], 0
    else:
        my_lemmas, my_forms = process_text_detailed(my_data['body_text'], settings)
        my_anchors, _ = process_text_detailed(my_data['anchor_text'], settings)
        my_len = len(my_lemmas)
        for k, v in my_forms.items():
            all_forms_map[k].update(v)

    # 2. Конкуренты (только успешно скачанные)
    comp_docs = []
    for p in comp_data_full:
        body, c_forms = process_text_detailed(p['body_text'], settings)
        anchor, _ = process_text_detailed(p['anchor_text'], settings)
        comp_docs.append({'body': body, 'anchor': anchor, 'url': p['url'], 'pos': p['pos']})
        for k, v in c_forms.items():
            all_forms_map[k].update(v)
    
    if not comp_docs:
        # Тем не менее, нам нужна таблица релевантности, чтобы показать, кто был в ТОПе
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
        
        return {"depth": pd.DataFrame(), "hybrid": pd.DataFrame(), "ngrams": pd.DataFrame(), "relevance_top": table_rel_df, "my_score": {"width": 0, "depth": 0}}

    # Дальше расчеты идут только по успешно скачанным comp_docs
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

    # --- N-граммы (Фразы) - ИСПРАВЛЕНО (Запрос 3) ---
    table_ngrams = []
    if comp_docs and my_data:
        try:
            N_GRAM = 2
            my_ngrams, _ = process_text_detailed(my_data['body_text'], settings, N_GRAM)
            comp_ngrams_list = [process_text_detailed(p['body_text'], settings, N_GRAM)[0] for p in comp_docs]
            
            all_ngrams = set(my_ngrams)
            for c in comp_ngrams_list: all_ngrams.update(c)
            
            ngram_doc_freqs = Counter()
            for c in comp_ngrams_list: 
                for ng in set(c): ngram_doc_freqs[ng] += 1
                
            for ng in all_ngrams:
                df = ngram_doc_freqs[ng]
                # Фильтрация: как минимум 2 сайта в ТОПе или есть у нас
                if df < 2 and ng not in my_ngrams: continue
                
                my_c = my_ngrams.count(ng)
                comp_c = [c.count(ng) for c in comp_ngrams_list]
                
                sum_in_top = sum(comp_c)
                
                med_c = np.median(comp_c) if comp_c else 0
                max_c = np.max(comp_c) if comp_c else 0
                
                rec_min = int(round(med_c * norm_k))
                rec_max = int(round(max_c * norm_k))
                
                diff_ngram = 0
                if my_c < rec_min: diff_ngram = rec_min - my_c
                elif my_c > rec_max: diff_ngram = rec_max - my_c
                
                is_missing = (my_c == 0)
                
                if sum_in_top > 0 or my_c > 0:
                    table_ngrams.append({
                        "Слово/Фраза": ng, 
                        "Частота (Сумма)": sum_in_top,
                        "Мин. (рек)": rec_min, 
                        "Макс. (рек)": rec_max,
                        "Вхождений у вас": my_c,
                        "Добавить/Убрать": diff_ngram,
                        "Сайтов": df,
                        "is_missing": is_missing
                    })
        except Exception as e:
            # Предупреждение на случай, если расчет n-грамм не удался
            st.warning(f"Ошибка при расчете N-грамм: {e}") 
            table_ngrams = []


    # 3. Расчет ширины и глубины (баллы)
    competitor_stats_raw = []
    
    # Считаем метрики только по успешно скачанным
    for p in comp_docs: 
        p_lemmas = p['body']
        domain = p['domain']
        pos = p['pos']
        
        relevant_lemmas = [w for w in p_lemmas if w in vocab]
        raw_width = len(set(relevant_lemmas))
        raw_depth = len(relevant_lemmas)
        competitor_stats_raw.append({
            "domain": domain, "pos": pos, "raw_w": raw_width, "raw_d": raw_depth
        })

    # Определяем максимумы только по **успешно скачанным и проанализированным** конкурентам
    max_width_top = max([c['raw_w'] for c in competitor_stats_raw]) if competitor_stats_raw else 1
    max_depth_top = max([c['raw_d'] for c in competitor_stats_raw]) if competitor_stats_raw else 1
    
    table_rel = []
    
    # 3.1. Баллы конкурентов (рассчитываем по всем, кто был в original_results)
    for c in competitor_stats_raw:
        score_w = int(round((c['raw_w'] / max_width_top) * 100))
        score_d = int(round((c['raw_d'] / max_depth_top) * 100))
        
        # Добавляем данные в таблицу релевантности (ТОП)
        table_rel.append({
            "Домен": c['domain'], "Позиция": c['pos'],
            "Ширина (балл)": score_w, "Глубина (балл)": score_d
        })
        
    # 3.2. Баллы для ВАШЕГО сайта
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
        "Домен": my_label, "Позиция": my_serp_pos if my_serp_pos > 0 else len(original_results) + 1, 
        "Ширина (балл)": my_score_w, "Глубина (балл)": my_score_d
    })

    # Сортируем таблицу релевантности по позиции 
    table_rel_df = pd.DataFrame(table_rel)
    table_rel_df = table_rel_df.sort_values(by='Позиция', ascending=True).reset_index(drop=True)

    return {
        "depth": pd.DataFrame(table_depth), 
        "hybrid": pd.DataFrame(table_hybrid), 
        "ngrams": pd.DataFrame(table_ngrams), 
        "relevance_top": table_rel_df, 
        "my_score": {"width": my_score_w, "depth": my_score_d}
    }

# ==========================================
# 5. ФУНКЦИЯ ОТОБРАЖЕНИЯ (FINAL)
# ==========================================

# --- Функции Истории (Запрос 2) ---

def save_analysis_to_history(my_url, successful_urls, results, comp_data):
    """Сохраняет результаты анализа в историю сессии."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Краткий отчет
    history_entry = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'my_url': my_url,
        'successful_urls': successful_urls,
        'width': results['my_score']['width'],
        'depth': results['my_score']['depth'],
        'full_results': {
            'results': results,
            'comp_table_data': comp_data,
            'my_url_input': my_url,
            'competitors_input': "\n".join(successful_urls)
        }
    }
    st.session_state.history.insert(0, history_entry) 

def load_analysis_from_history(entry):
    """Загружает полный анализ из истории в текущую сессию для отображения."""
    
    # Сброс пагинации
    for key in list(st.session_state.keys()): 
        if key.endswith('_page'): st.session_state[key] = 1 
        
    st.session_state.analysis_results = entry['full_results']['results']
    st.session_state.comp_table_data = entry['full_results']['comp_table_data']
    st.session_state.my_url_input = entry['full_results']['my_url_input']
    st.session_state.manual_urls_ui = entry['full_results']['competitors_input']
    st.session_state.analysis_done = True
    st.toast(f"Загружен анализ от {entry['timestamp']}.")
    st.rerun()


def render_paginated_table(df, title_text, key_prefix, default_sort_col=None, use_abs_sort_default=False):
    if df.empty:
        st.info(f"{title_text}: Нет данных.")
        return
    st.markdown(f"### {title_text}")
    
    # БЛОК СОРТИРОВКИ 
    if f'{key_prefix}_sort_col' not in st.session_state:
        st.session_state[f'{key_prefix}_sort_col'] = default_sort_col if default_sort_col in df.columns else df.columns[0]
    if f'{key_prefix}_sort_dir' not in st.session_state:
        st.session_state[f'{key_prefix}_sort_dir'] = 'desc' if use_abs_sort_default else 'asc'

    col_sort, col_dir = st.columns([1, 1], key=f"{key_prefix}_sort_cols")
    
    with col_sort:
        sort_col = st.selectbox("Сортировать по колонке", options=df.columns, index=df.columns.get_loc(st.session_state[f'{key_prefix}_sort_col']), key=f"{key_prefix}_sort_col_select")
        if sort_col != st.session_state[f'{key_prefix}_sort_col']:
            st.session_state[f'{key_prefix}_sort_col'] = sort_col
            st.rerun()
            
    with col_dir:
        sort_dir = st.selectbox("Направление", options=['desc', 'asc'], index=['desc', 'asc'].index(st.session_state[f'{key_prefix}_sort_dir']), key=f"{key_prefix}_sort_dir_select")
        if sort_dir != st.session_state[f'{key_prefix}_sort_dir']:
            st.session_state[f'{key_prefix}_sort_dir'] = sort_dir
            st.rerun()
            
    ascending = sort_dir == 'asc'
    
    # Сортировка по абсолютной величине (если требуется)
    if use_abs_sort_default and sort_col in ['Добавить/Убрать', 'diff_abs']:
        df = df.sort_values(by=sort_col, ascending=ascending, key=lambda x: np.abs(x) if np.issubdtype(x.dtype, np.number) else x).copy()
    else:
        df = df.sort_values(by=sort_col, ascending=ascending).copy()

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
        styles = [base_style] * len(row)
        
        # Индекс для is_missing
        try:
            is_missing_idx = row.index.get_loc("is_missing")
        except KeyError:
            is_missing_idx = -1

        if is_missing_idx != -1 and row['is_missing']:
            # Стиль для пропущенных слов
            styles[0] += 'color: #D32F2F; font-weight: bold;'
        
        # Стиль для других колонок (убрано для простоты, если не нужно)
        # else:
        #     styles = [base_style + 'font-weight: 600;' if col_name not in ["diff_abs", "is_missing"] else base_style for col_name in row.index]

        return styles
    
    cols_to_hide = ["diff_abs", "is_missing"]
    
    # Если колонка 'Слово/Фраза' существует, используем ее для применения стиля
    col_config = {}
    if 'Слово' in df_view.columns:
        col_config['Слово'] = st.column_config.TextColumn("Слово", help="Слово или лемма")
    elif 'Слово/Фраза' in df_view.columns:
         col_config['Слово/Фраза'] = st.column_config.TextColumn("Слово/Фраза", help="Слово или фраза")

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

# --- НОВАЯ ФУНКЦИЯ ДЛЯ ТАБЛИЦЫ СТАТУСА (Запрос 5) ---
def render_competitor_status_table(comp_data):
    """
    Отображает таблицу статусов конкурентов.
    Домены теперь являются кликабельными ссылками на проанализированный URL.
    """
    st.markdown("### 2. Анализ конкурентов (статус)")
    
    if not comp_data:
        st.info("Нет данных о конкурентах.")
        return

    df = pd.DataFrame(comp_data)
    
    # Создаем кликабельные домены (Запрос 5)
    def make_clickable_domain(row):
        url = row['URL']
        domain = row['Домен']
        status = row['Статус']
        if "OK" in status:
            return f'<a href="{url}" target="_blank">{domain}</a>'
        return domain
        
    df['Домен'] = df.apply(make_clickable_domain, axis=1)
    
    # Подготовка DF для отображения
    display_df = df[['Домен', 'Статус', 'Ошибка']]
    
    # Отображаем таблицу с HTML-колонками
    st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)


# ==========================================
# 6. ИНТЕРФЕЙС
# ==========================================

st.title("SEO Анализатор Релевантности")

# --- ВКЛАДКИ (Запрос 2) ---
tab_analysis, tab_history = st.tabs(["📊 Анализ Семантики", "📚 ИСТОРИЯ ПРОВЕРОК"]) 

with tab_analysis:
    col_main, col_sidebar = st.columns([65, 35])
    
    with col_main:
        st.markdown("### URL или код страницы Вашего сайта")
        my_input_type = st.radio("Тип страницы", ["Релевантная страница на вашем сайте", "Исходный код страницы или текст", "Без страницы"], horizontal=True, label_visibility="collapsed", key="my_page_source_radio")
        my_url = ""
        my_page_content = ""
        
        if my_input_type == "Релевантная страница на вашем сайте":
            my_url = st.text_input("URL страницы", placeholder="https://site.ru/catalog/tovar", label_visibility="collapsed", key="my_url_input")
        elif my_input_type == "Исходный код страницы или текст":
            my_page_content = st.text_area("Исходный код...", height=300, placeholder="<html>...</html>", label_visibility="collapsed", key="my_content_input")

        st.markdown("### Источник конкурентов")
        source_type = st.radio("Тип источника", ["API (по запросу)", "Ручной список"], horizontal=True, label_visibility="collapsed", key="source_type_radio")
        
        query = ""
        if source_type == "API":
            query = st.text_input("Поисковой запрос (ключ)", placeholder="купить диван в москве", key="query_input")
            
        if source_type == "Ручной список":
            # Используем session_state.manual_urls_ui для хранения полных URL (Запрос 6)
            manual_urls_ui = st.text_area(
                "Список URL конкурентов (каждый с новой строки):", 
                height=300, 
                placeholder="https://comp1.ru/page/\nhttps://comp2.com/item/", 
                key="manual_urls_ui" 
            )

        # Кнопка анализа
        st.markdown("---")
        if st.button("🚀 Начать Анализ", type="primary", use_container_width=True):
            # Сброс пагинации и флага
            for key in list(st.session_state.keys()): 
                if key.endswith('_page'): st.session_state[key] = 1
            st.session_state.start_analysis_flag = True
            st.rerun()

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
            st.checkbox("Исключать noindex/script", True, key="settings_noindex")
            st.checkbox("Учитывать Alt/Title", False, key="settings_alt")
            st.checkbox("Учитывать числа", False, key="settings_numbers")
        with col_c2:
            st.checkbox("Нормировать по длине", True, key="settings_norm")
            st.checkbox("Исключать агрегаторы", True, key="settings_agg")
        
        st.markdown("---")
        st.markdown("##### ⛔ Стоп-слова")
        st.text_area("Стоп-слова (каждое с новой строки)", DEFAULT_STOPS, height=150, key="settings_stops")
    

# ==========================================
# 7. ВЫПОЛНЕНИЕ (СКОРРЕКТИРОВАННАЯ ЛОГИКА СБОРА)
# ==========================================

if st.session_state.get('start_analysis_flag'):
    st.session_state.start_analysis_flag = False

    # ... (Проверки входных данных) ...
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
    
    target_urls_raw = [] # Список URL:pos, которые прошли первичную фильтрацию
    my_data = None
    my_domain = ""
    my_serp_pos = 0 
    
    # 1. Сбор данных о ВАШЕМ сайте и домене
    if my_input_type == "Релевантная страница на вашем сайте":
        with st.spinner("Скачивание вашей страницы..."):
            my_url_input = st.session_state.my_url_input
            # Используем robust-функцию
            my_data = parse_page_robust(my_url_input, settings) 
            
            if my_data['error']:
                st.error(f"❌ Не удалось обработать Ваш URL: {my_data['error']}")
                st.stop()
            
            my_domain = my_data['domain']
            
    # 2. Сбор URL конкурентов
    found_results = []
    if source_type == "API":
        # ... (API logic remains the same) ...
        with st.spinner(f"Запрос ТОП-{st.session_state.settings_top_n} в {st.session_state.settings_search_engine} / {st.session_state.settings_region} по запросу '{st.session_state.query_input}'..."):
            found_results = get_arsenkin_urls(st.session_state.query_input, st.session_state.settings_search_engine, st.session_state.settings_region, st.session_state.settings_top_n)

        # 2.1. Фильтрация и трекинг позиции 
        filtered_results_all = []
        excl = set(DEFAULT_EXCLUDE_DOMAINS)
        
        if st.session_state.settings_agg:
            # Если агрегаторы исключаются, добавляем их к стоп-доменам
            excl.update(DEFAULT_EXCLUDE_DOMAINS)

        for result in found_results:
            url = result['url']
            pos = result['pos']
            domain = urlparse(url).netloc
            
            # 1. Трекинг нашего сайта
            if my_domain and my_domain == domain:
                if my_serp_pos == 0 or pos < my_serp_pos:
                    my_serp_pos = pos
                continue
            
            # 2. Исключаем домены из списка исключений
            if any(x in domain for x in excl):
                continue
            
            # Если прошел фильтры, добавляем в список всех чистых конкурентов
            filtered_results_all.append(result)

        # 2.2. Ограничение по TARGET_COMPETITORS (ВТОРЫМ ШАГОМ)
        # TARGET_COMPETITORS должно быть определено в оригинальном коде, 
        # используем 10 для безопасности.
        TARGET_COMPETITORS = st.session_state.settings_top_n
        target_urls_raw = filtered_results_all[:TARGET_COMPETITORS]
        collected_competitors_count = len(target_urls_raw)
        
        st.info(f"Получено уникальных URL от API: {len(found_results)}. После фильтрации **агрегаторов и стоп-доменов**, для анализа выбрано **{collected_competitors_count}** релевантных конкурентов (цель {TARGET_COMPETITORS}). Ваш сайт в ТОПе: **{'Да (Поз. ' + str(my_serp_pos) + ')' if my_serp_pos > 0 else 'Нет'}**.")
        
    else: # Ручной режим
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
        
    # 3. Скачивание всех конкурентов (обновлено для robust parsing)
    comp_data_full_raw = [] 
    
    with st.spinner(f"Скачивание {len(target_urls_raw)} конкурентов с повторными попытками..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            url_to_pos = {item['url']: item['pos'] for item in target_urls_raw}
            
            # Запускаем robust парсинг
            future_to_url = {executor.submit(parse_page_robust, url, settings): url for url in url_to_pos.keys()}
            
            for future in concurrent.futures.as_completed(future_to_url):
                result = future.result() 
                result['pos'] = url_to_pos[result['url']]
                comp_data_full_raw.append(result)

    # 4. Формирование финальных данных для анализа и таблицы
    comp_data_full = []
    comp_table_data = []
    successful_urls = []
    
    for item in target_urls_raw: 
        url = item['url']
        pos = item['pos']
        
        parsed_result = next((res for res in comp_data_full_raw if res['url'] == url), None)
        
        if parsed_result and parsed_result.get('body_text'):
            comp_data_full.append(parsed_result)
            comp_table_data.append({
                "URL": url,
                "Домен": parsed_result['domain'],
                "Статус": "OK (2)",
                "Ошибка": "",
                "Позиция": pos 
            })
            successful_urls.append(url)
        else:
            error = parsed_result['error'] if parsed_result and parsed_result.get('error') else "Не скачан/Исключен"
            comp_table_data.append({
                "URL": url,
                "Домен": urlparse(url).netloc,
                "Статус": f"Ошибка/Исключен", # Убрал 0/1, т.к. может быть ошибка, исключение или пустой текст
                "Ошибка": error,
                "Позиция": pos
            })
    
    # Обновление поля ввода конкурентов полными URL-адресами (Запрос 6)
    st.session_state.manual_urls_ui = "\n".join(successful_urls)
    
    # Сохранение данных для таблицы конкурентов
    st.session_state.comp_table_data = comp_table_data

    # 5. Расчет метрик
    with st.spinner("Анализ данных..."):
        results = calculate_metrics(
            comp_data_full, 
            my_data, 
            settings, 
            my_serp_pos, 
            target_urls_raw
        )
    st.session_state.analysis_results = results
    st.session_state.analysis_done = True
    
    # 6. Сохранение в историю (Запрос 2)
    save_analysis_to_history(st.session_state.my_url_input, successful_urls, results, comp_table_data)
    
    st.rerun()

# --- БЛОК ОТОБРАЖЕНИЯ РЕЗУЛЬТАТОВ ---
if st.session_state.analysis_done and st.session_state.analysis_results:
    with tab_analysis:
        results = st.session_state.analysis_results
        st.success("Анализ готов!")
        
        # 0. Результаты (Баллы)
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
        
        # 2. Анализ конкурентов (статус) - с кликабельными ссылками (Запрос 5)
        render_competitor_status_table(st.session_state.comp_table_data) 
        
        render_paginated_table(results['hybrid'], "3. Гибридный ТОП (TF-IDF)", "tbl_hybrid", default_sort_col="TF-IDF ТОП", use_abs_sort_default=False)
        
        # 4. N-граммы (Фразы) - теперь должны работать (Запрос 3)
        render_paginated_table(results['ngrams'], "4. N-граммы (Фразы)", "tbl_ngrams", default_sort_col="Частота (Сумма)", use_abs_sort_default=False)
        
        render_paginated_table(results['relevance_top'], "5. Релевантность ТОПа", "tbl_relevance_top", default_sort_col="Позиция", use_abs_sort_default=False)


# --- ВКЛАДКА ИСТОРИЯ (Запрос 2) ---
with tab_history:
    st.header("📚 История Проверок")
    
    if not st.session_state.history:
        st.info("История проверок пуста. Начните анализ на вкладке 'Анализ Семантики'.")
    else:
        for i, entry in enumerate(st.session_state.history):
            
            col_ts, col_btn = st.columns([4, 1])
            
            with col_ts:
                st.markdown(f"""
                    <div style='background-color: {LIGHT_BG_MAIN}; padding: 10px; border-radius: 5px; border: 1px solid {BORDER_COLOR}; margin-bottom: 10px;'>
                        <p style='margin:0; font-size: 1.1em; color: {PRIMARY_COLOR};'>
                            <b>{entry['timestamp']}</b>
                        </p>
                        <p style='margin:5px 0 0 0;'>
                            🔗 URL: <span style='word-break: break-all;'>{entry['my_url']}</span>
                        </p>
                        <p style='margin:5px 0 0 0;'>
                            Ширина: <b>{entry['width']}</b> | Глубина: <b>{entry['depth']}</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_btn:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                # Кнопка для загрузки полного анализа
                if st.button(f"Посмотреть", key=f"load_history_{i}", use_container_width=True):
                    load_analysis_from_history(entry)
