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
import os # Для работы с файловой системой

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
    "irr.ru", "onliner.by", "shop.by", "deal.by", "yell.ru", 
    "profi.ru", 
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
if 'app_page' not in st.session_state:
    st.session_state.app_page = "Анализ" # Добавление состояния для навигации

# --- ФУНКЦИИ ДЛЯ ИСТОРИИ ЗАДАЧ ---
RESULTS_FILE = "gar_pro_results.json" # Файл для сохранения истории

def load_results():
    """Загружает историю результатов из JSON файла."""
    if not os.path.exists(RESULTS_FILE):
        return []
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_results(data):
    """Сохраняет историю результатов в JSON файл."""
    try:
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        st.error(f"Ошибка сохранения результатов: {e}")
        return False
        
# --- ФУНКЦИИ ДЛЯ СКАЧИВАНИЯ ---
def convert_df_to_csv(df):
    """Преобразование DataFrame в CSV строку (с разделителем ';')."""
    return df.to_csv(index=False, sep=';', encoding='utf-8')

def convert_df_to_xml(df, root_name="Results", row_name="Item"):
    """Преобразование DataFrame в простую XML строку."""
    data = df.to_dict(orient='records')
    xml_string = f'<?xml version="1.0" encoding="utf8"?>\n<{root_name}>\n'
    
    for record in data:
        xml_string += f'  <{row_name}>\n'
        for key, value in record.items():
            # Замена недопустимых символов в именах тегов
            tag_name = re.sub(r'[^a-zA-Z0-9_]', '', key.replace(' ', '_'))
            # Экранирование значений
            safe_value = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
            xml_string += f'    <{tag_name}>{safe_value}</{tag_name}>\n'
        xml_string += f'  </{row_name}>\n'
    
    xml_string += f'</{root_name}>'
    return xml_string


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
            
        body_text_raw = soup.get_text(separator=' ') + " " + " ".join(extra_text)
        # Убираем лишние пробелы и новые строки
        body_text = re.sub(r'\s+', ' ', body_text_raw).strip()
        
        # Проверяем, что контент не пустой после обработки
        if not body_text:
            return None 

        return {'url': url, 'domain': urlparse(url).netloc, 'body_text': body_text, 'anchor_text': anchor_text}
    except: 
        return None

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

    # Разделяем успешно скачанные данные для анализа лемм и статистики
    comp_data_parsed = [d for d in comp_data_full if d.get('body_text')]
    
    # 2. Конкуренты (только успешно скачанные)
    comp_docs = []
    for p in comp_data_parsed:
        body, c_forms = process_text_detailed(p['body_text'], settings)
        anchor, _ = process_text_detailed(p['anchor_text'], settings)
        comp_docs.append({'body': body, 'anchor': anchor})
        for k, v in c_forms.items():
            all_forms_map[k].update(v)
    
    # Если нет успешно скачанных конкурентов, мы не можем рассчитать релевантность
    if not comp_docs:
        # Тем не менее, нам нужна таблица релевантности, чтобы показать, кто был в ТОПе
        
        table_rel_fallback = []
        # Добавляем все URL, которые пришли из API/ручного списка, чтобы показать их позиции
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
        
        # Добавляем Ваш сайт
        table_rel_fallback.append({
            "Домен": my_label, 
            "Позиция": my_serp_pos if my_serp_pos > 0 else len(original_results) + 1,
            "Ширина (балл)": 0, "Глубина (балл)": 0
        })
        
        table_rel_df = pd.DataFrame(table_rel_fallback).sort_values(by='Позиция', ascending=True).reset_index(drop=True)
        # --- ИЗМЕНЕНИЕ 1: Добавление относительного ранга (№) в начало ---
        table_rel_df.insert(0, '№', table_rel_df.index + 1)
        # -----------------------------------------------------------------
        
        return {"depth": pd.DataFrame(), "hybrid": pd.DataFrame(), "ngrams": pd.DataFrame(), "relevance_top": table_rel_df, "my_score": {"width": 0, "depth": 0}}


    # Дальше расчеты идут только по успешно скачанным comp_docs
    avg_len = np.mean([len(d['body']) for d in comp_docs])
    norm_k = (my_len / avg_len) if (settings['norm'] and my_len > 0 and avg_len > 0) else 1.0
    
    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    vocab = sorted(list(vocab))
    N = len(comp_docs) # N - количество успешно скачанных документов
    doc_freqs = Counter()
    for d in comp_docs:
        for w in set(d['body']): doc_freqs[w] += 1
        
    table_depth, table_hybrid = [], []
    
    # Инициализация для расчета баллов (ширина/глубина)
    competitor_stats_raw = []
    
    max_width_top = 1
    max_depth_top = 1

    # Логика для подсчета raw_width и raw_depth
    for d in comp_data_parsed:
        p_lemmas, _ = process_text_detailed(d['body_text'], settings)
        domain = urlparse(d['url']).netloc
        pos = d['pos']
        
        # Ширина: количество уникальных лемм, общих со всем словарем (vocab)
        relevant_lemmas = [w for w in p_lemmas if w in vocab]
        raw_width = len(set(relevant_lemmas))
        raw_depth = len(relevant_lemmas)
        
        competitor_stats_raw.append({
            "domain": domain,
            "pos": pos,
            "raw_w": raw_width,
            "raw_d": raw_depth
        })
    
    # Определяем максимумы только по **успешно скачанным и проанализированным** конкурентам
    if competitor_stats_raw:
        max_width_top = max([c['raw_w'] for c in competitor_stats_raw])
        max_depth_top = max([c['raw_d'] for c in competitor_stats_raw])
        
    # Расчет my_raw_w/d
    my_relevant = [w for w in my_lemmas if w in vocab]
    my_raw_w = len(set(my_relevant))
    my_raw_d = len(my_relevant)

    # Нормализация баллов для ВАШЕГО сайта
    my_score_w = int(round((my_raw_w / max_width_top) * 100))
    my_score_d = int(round((my_raw_d / max_depth_top) * 100))
    
    # 3. Расчет рекомендаций (Table Depth, Hybrid)
    
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

        # Вычисление рекомендаций
        rec_min = max(1, int(round(min(mean_total, med_total))))
        rec_max = max(1, int(round(max_total)))
        
        rec_anchor = max(0, int(round(np.median(c_anchor_tfs))))
        rec_text_min = max(0, rec_min - rec_anchor)
        
        # Разница (ключевой показатель)
        diff_total = int(round((rec_min - my_tf_total) * norm_k))
        diff_anchor = int(round((rec_anchor - my_tf_anchor) * norm_k))
        diff_text = int(round((rec_text_min - my_tf_text) * norm_k))
        
        # IDF
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
        idf = max(0.1, idf)
        
        # Переспам
        spam_percent = 0
        if my_tf_total > rec_max and rec_max > 0:
            spam_percent = round(((my_tf_total - rec_max) / rec_max) * 100, 1)
        elif my_tf_total > 0 and rec_max == 0:
            spam_percent = 100
        
        spam_idf = round(spam_percent * idf, 1)
        abs_diff = abs(diff_total)
        
        if med_total > 0.5 or my_tf_total > 0:
            table_depth.append({
                "Слово": word,
                "Словоформы": forms_str,
                "Повторы у вас": my_tf_total,
                "Повторов в ТОПе": sum_in_top,
                "Минимум (рек)": rec_min,
                "Максимум (рек)": rec_max,
                "Добавить/Убрать": diff_total,
                "Тег A у вас": my_tf_anchor,
                "Тег A (рек)": rec_anchor,
                "Тег A +/-": diff_anchor,
                "Текст у вас": my_tf_text,
                "Текст (рек)": rec_text_min,
                "Текст +/-": diff_text,
                "Переспам %": spam_percent,
                "Переспам*IDF": spam_idf,
                "diff_abs": abs_diff,
                "is_missing": (my_tf_total == 0)
            })
            
            table_hybrid.append({
                "Слово": word,
                "TF-IDF ТОП": round(med_total * idf, 2),
                "TF-IDF у вас": round(my_tf_total * idf, 2),
                "Сайтов": df,
                "Переспам": max_total 
            })
            
    # 4. Расчет N-грамм
    table_ngrams = []
    if comp_docs and my_data:
        try:
            my_bi, _ = process_text_detailed(my_data['body_text'], settings, 2)
            comp_bi = [process_text_detailed(p['body_text'], settings, 2)[0] for p in comp_docs]
            all_bi = set(my_bi)
            for c in comp_bi: all_bi.update(c)
            
            bi_freqs = Counter()
            for c in comp_bi:
                for b_ in set(c): bi_freqs[b_] += 1
                
            for bg in all_bi:
                df_bi = bi_freqs[bg]
                if df_bi < 2 and bg not in my_bi: continue
                
                my_c = my_bi.count(bg)
                comp_c = [d.count(bg) for d in comp_bi]
                
                sum_in_top_bi = sum(comp_c)
                mean_bi = np.mean(comp_c)
                med_bi = np.median(comp_c)
                max_bi = np.max(comp_c)

                rec_min_bi = max(1, int(round(min(mean_bi, med_bi))))
                diff_bi = int(round((rec_min_bi - my_c)))
                
                table_ngrams.append({
                    "Биграмма (леммы)": bg,
                    "Повторы у вас": my_c,
                    "Повторов в ТОПе": sum_in_top_bi,
                    "Медиана (рек)": rec_min_bi,
                    "Добавить/Убрать": diff_bi,
                    "Сайтов": df_bi,
                    "Максимум": max_bi
                })
        except Exception as e:
            st.warning(f"Ошибка при расчете N-грамм: {e}")
            
    # 5. Таблица релевантности (TOP)
    table_rel = []
    # Баллы конкурентов
    for c in competitor_stats_raw:
        score_w = int(round((c['raw_w'] / max_width_top) * 100))
        score_d = int(round((c['raw_d'] / max_depth_top) * 100))
        table_rel.append({
            "Домен": c['domain'],
            "Позиция": c['pos'], # Это фактическая позиция в SERP
            "Ширина (балл)": score_w,
            "Глубина (балл)": score_d
        })
        
    # Добавляем ВАШ сайт в таблицу
    if my_data and my_data.get('domain'):
        my_label = f"{my_data['domain']} (Вы)"
    else:
        my_label = "Ваш сайт"
        
    table_rel.append({
        "Домен": my_label,
        "Позиция": my_serp_pos if my_serp_pos > 0 else len(original_results) + 1, # Ставим после последнего конкурента
        "Ширина (балл)": my_score_w,
        "Глубина (балл)": my_score_d
    })

    # Сортируем таблицу релевантности по позиции
    table_rel_df = pd.DataFrame(table_rel)
    table_rel_df = table_rel_df.sort_values(by='Позиция', ascending=True).reset_index(drop=True)

    # --- ИЗМЕНЕНИЕ 1: Добавление относительного ранга (№) в начало ---
    table_rel_df.insert(0, '№', table_rel_df.index + 1)
    # -----------------------------------------------------------------
    
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

def render_paginated_table(df, title_text, key_prefix, default_sort_col=None, use_abs_sort_default=False):
    # --- ИЗМЕНЕНИЕ 3: Добавление кнопок скачивания CSV/XML ---
    st.markdown(f"#### {title_text}")

    df_for_download = df.copy() 
    # Удаляем служебные колонки перед скачиванием
    if 'diff_abs' in df_for_download.columns:
        df_for_download = df_for_download.drop(columns=['diff_abs'])
    if 'is_missing' in df_for_download.columns:
        df_for_download = df_for_download.drop(columns=['is_missing'])
        
    csv_data = convert_df_to_csv(df_for_download)
    xml_data = convert_df_to_xml(df_for_download, root_name=key_prefix, row_name="item")

    c_dl1, c_dl2, c_dl_spacer = st.columns([1, 1, 8])

    with c_dl1:
        st.download_button(
            label="⬇️ CSV",
            data=csv_data,
            file_name=f"{key_prefix}.csv",
            mime="text/csv",
            key=f"{key_prefix}_dl_csv",
            use_container_width=True
        )

    with c_dl2:
        st.download_button(
            label="⬇️ XML",
            data=xml_data,
            file_name=f"{key_prefix}.xml",
            mime="text/xml",
            key=f"{key_prefix}_dl_xml",
            use_container_width=True
        )
    # -------------------------------------------------------------------------
    
    # Оригинальная логика сортировки и пагинации
    if default_sort_col and default_sort_col in df.columns:
        # Убедимся, что колонка для сортировки не является частью служебных, если мы используем `use_abs_sort_default`
        if use_abs_sort_default and 'diff_abs' in df.columns:
            df = df.sort_values(by='diff_abs', ascending=False).reset_index(drop=True)
        else:
            df = df.sort_values(by=default_sort_col, ascending=False).reset_index(drop=True)
            
    df = df.reset_index(drop=True)
    df.index = df.index + 1 # Стандартный индекс для отображения в Streamlit (1, 2, 3...)
    ROWS_PER_PAGE = 20 
    
    if f'{key_prefix}_page' not in st.session_state:
        st.session_state[f'{key_prefix}_page'] = 1
        
    total_rows = len(df)
    total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
    
    current_page = st.session_state[f'{key_prefix}_page']
    
    # Корректировка страницы, если она вышла за границы
    if current_page > total_pages: current_page = total_pages
    if current_page < 1: current_page = 1
    
    start_idx = (current_page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE
    df_view = df.iloc[start_idx:end_idx]

    # ПОКРАСКА ЯЧЕЕК
    def highlight_rows(row):
        base_style = 'background-color: #FFFFFF; color: #3D4858; border-bottom: 1px solid #DBEAFE;'
        styles = []
        
        # Список колонок, для которых не нужно применять стили жирности/красного
        cols_to_skip = ["diff_abs", "is_missing", "№", "Позиция"] 
        
        for col_name in row.index:
            # Проверка, что 'is_missing' существует и его значение True
            if col_name == 'is_missing' and 'is_missing' in row and row['is_missing']:
                # Красный цвет для строк, где слова нет у вас
                styles.append(base_style + 'color: #D32F2F; font-weight: bold;') 
            elif col_name not in cols_to_skip:
                # Жирный для остальных, кроме служебных и специальных колонок
                styles.append(base_style + 'font-weight: 600;') 
            else:
                styles.append(base_style)
        return styles

    cols_to_hide = ["diff_abs", "is_missing"]
    
    # Применяем стили только если DataFrame не пуст
    if not df_view.empty:
        styled_df = df_view.style.apply(highlight_rows, axis=1)
    else:
        styled_df = df_view 

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

# --- ФУНКЦИЯ ОТОБРАЖЕНИЯ СТРАНИЦЫ ИСТОРИИ ---
def render_history_page():
    st.title("📊 История Анализов")
    st.markdown("Здесь хранятся результаты всех ваших предыдущих задач.")
    
    all_results = load_results()
    
    if not all_results:
        st.info("История задач пуста.")
        return
        
    for idx, task in enumerate(all_results):
        # Используем expanser для каждого результата
        header = f"[{task['date_str']}] {task['query']} ({task['url']} / {task['region']})"
        with st.expander(header):
            st.markdown(f"**Запрос:** {task['query']}")
            st.markdown(f"**URL:** {task['url']}")
            st.markdown(f"**Регион/ПС:** {task['region']} / {task['engine']}")
            
            # Отображение ключевых метрик
            st.markdown(f""" 
                <div style='background-color: {LIGHT_BG_MAIN}; padding: 10px; border-radius: 6px; border: 1px solid {BORDER_COLOR};'>
                    <h5 style='margin:0; color: {PRIMARY_COLOR};'>Результат вашего сайта (в баллах)</h5>
                    <p style='margin:5px 0 0 0;'>Ширина (охват): <b>{task['my_score']['width']}</b> | Глубина (оптимизация): <b>{task['my_score']['depth']}</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Кнопка для загрузки данных из истории в текущий анализ
            if st.button(f"Показать детальные таблицы", key=f"show_details_{idx}"):
                # Конвертируем JSON-структуру обратно в DataFrames
                st.session_state.analysis_results = {
                    'depth': pd.DataFrame.from_records(task['depth']),
                    'hybrid': pd.DataFrame.from_records(task['hybrid']),
                    'ngrams': pd.DataFrame.from_records(task['ngrams']),
                    'relevance_top': pd.DataFrame.from_records(task['relevance_top']),
                    'my_score': task['my_score']
                }
                st.session_state.analysis_done = True
                st.session_state.app_page = "Анализ" # Переключаем на страницу анализа
                st.rerun()

# ==========================================
# 6. ИНТЕРФЕЙС
# ==========================================

col_main, col_sidebar = st.columns([65, 35])

with col_sidebar:
    st.session_state.app_page = st.radio(
        "Навигация",
        ["Анализ", "История"],
        index=0 if st.session_state.app_page == "Анализ" else 1,
        key="app_page_select"
    )
    
if st.session_state.app_page == "История":
    render_history_page()
    
elif st.session_state.app_page == "Анализ":
    
    with col_main:
        st.title("SEO Анализатор Релевантности")
        
        # --- БЛОК ВВОДА ДАННЫХ В АНАЛИЗЕ ---
        st.markdown("### URL или код страницы Вашего сайта")
        my_input_type = st.radio("Тип страницы", ["Релевантная страница на вашем сайте", "Исходный код страницы или текст"], key="my_input_type", horizontal=True)

        if my_input_type == "Релевантная страница на вашем сайте":
            st.text_input("URL страницы", placeholder="https://ваш-домен.ru/page/", key="my_url_input")
            st.session_state['my_content_input'] = '' 

        else:
            st.text_area("Исходный код или текст", height=200, placeholder="<html>...</html> или просто текст для анализа", key="my_content_input")
            st.session_state['my_url_input'] = '' 

        st.markdown("### Поисковой запрос и конкуренты")
        source_type = st.radio("Источник конкурентов", ["API Arsenkin", "Ручной список URL"], key="source_type", horizontal=True)

        if source_type == "API Arsenkin":
            st.text_input("Поисковой запрос", placeholder="купить товар в москве", key="query_input")
            st.session_state['manual_urls_ui'] = '' 

        else:
            st.text_area("Список URL конкурентов (каждый с новой строки)", height=150, key="manual_urls_ui", placeholder="https://comp1.ru/page\nhttps://comp2.ru/page\n...")
            st.session_state['query_input'] = '' 

        if st.button("Начать анализ", type="primary", use_container_width=True):
            # Сброс пагинации
            for key in list(st.session_state.keys()):
                if key.endswith('_page'):
                    st.session_state[key] = 1
            st.session_state.start_analysis_flag = True
        
        # --- КОНЕЦ БЛОКА ВВОДА ДАННЫХ ---
        
    with col_sidebar:
        # --- БЛОК НАСТРОЕК В САЙДБАРЕ ---
        st.markdown("#####⚙️ Настройки")
        ua = st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0"], key="settings_ua")
        search_engine = st.selectbox("Поисковая система", ["Яндекс", "Google", "Яндекс + Google"], key="settings_search_engine")
        region = st.selectbox("Регион поиска", list(REGION_MAP.keys()), key="settings_region")
        device = st.selectbox("Устройство", ["Desktop", "Mobile"], key="settings_device")
        # Максимальная глубина, которую позволяет API - 30. 
        top_n = st.selectbox("Глубина сбора (ТОП)", [10, 20, 30], index=0, key="settings_top_n")
        st.markdown("---")
        
        # Настройки парсинга
        st.selectbox("Учитывать тип страниц по url", ["Все страницы", "Главные страницы", "Внутренние страницы"], key="settings_url_type")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.checkbox("Исключать noindex/script", True, key="settings_noindex")
            st.checkbox("Учитывать Alt/Title", False, key="settings_alt")
            st.checkbox("Учитывать числа", False, key="settings_numbers")
        with col_c2:
            st.checkbox("Нормировать по длине", True, key="settings_norm")
            agg_default = st.checkbox("Исключать агрегаторы", True, key="settings_agg") 
        
        custom_stops = st.text_area("Стоп-слова (каждое с новой строки)", DEFAULT_STOPS, height=100, key="settings_stops_ui")
        # --- КОНЕЦ БЛОКА НАСТРОЕК ---
        
    # ==========================================
    # 7. ВЫПОЛНЕНИЕ (СКОРРЕКТИРОВАННАЯ ЛОГИКА СБОРА)
    # ==========================================
    if st.session_state.get('start_analysis_flag'):
        st.session_state.start_analysis_flag = False
        
        # --- ПРОВЕРКИ ---
        if st.session_state.my_input_type == "Релевантная страница на вашем сайте" and not st.session_state.get('my_url_input'):
            st.error("Введите URL!")
            st.stop()
        if st.session_state.my_input_type == "Исходный код страницы или текст" and not st.session_state.get('my_content_input', '').strip():
            st.error("Введите исходный код!")
            st.stop()
        if st.session_state.source_type == "API Arsenkin" and not st.session_state.get('query_input'):
            st.error("Введите поисковой запрос!")
            st.stop()
        if st.session_state.source_type == "Ручной список URL" and not st.session_state.get("manual_urls_ui", "").strip():
            st.error("Введите список URL конкурентов!")
            st.stop()
            
        settings = {
            'noindex': st.session_state.settings_noindex,
            'alt_title': st.session_state.settings_alt,
            'numbers': st.session_state.settings_numbers,
            'norm': st.session_state.settings_norm,
            'ua': st.session_state.settings_ua,
            'custom_stops': [w.strip() for w in st.session_state.settings_stops_ui.split('\n') if w.strip()],
            'agg': st.session_state.settings_agg
        }
        
        my_domain = urlparse(st.session_state.get('my_url_input', '')).netloc
        my_serp_pos = 0 
        excl = settings['custom_stops'] + (DEFAULT_EXCLUDE_DOMAINS if settings['agg'] else [])
        TARGET_COMPETITORS = st.session_state.settings_top_n

        # --- СБОР URL КОНКУРЕНТОВ ---
        found_results = []
        if st.session_state.source_type == "API Arsenkin":
            with st.spinner(f"Запрос ТОПа {st.session_state.settings_top_n} к API Arsenkin..."):
                found_results = get_arsenkin_urls(
                    st.session_state.query_input, 
                    st.session_state.settings_search_engine, 
                    st.session_state.settings_region, 
                    st.session_state.settings_top_n
                )

            # 2.1. Фильтрация и трекинг позиции (ПЕРВЫМ ШАГОМ)
            filtered_results_all = []
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
            # Берем только то количество, которое указано в TARGET_COMPETITORS
            target_urls_raw = filtered_results_all[:TARGET_COMPETITORS]
            collected_competitors_count = len(target_urls_raw)
            st.info(f"Получено уникальных URL от API: {len(found_results)}. После фильтрации **агрегаторов и стоп-доменов**, для анализа выбрано **{collected_competitors_count}** релевантных конкурентов (цель {TARGET_COMPETITORS}). Ваш сайт в ТОПе: **{'Да (Поз. ' + str(my_serp_pos) + ')' if my_serp_pos > 0 else 'Нет'}**.")

        else:
            # Ручной режим
            raw_urls = st.session_state.get("manual_urls_ui", "")
            if raw_urls:
                # В ручном режиме позиция не важна, просто список URL
                urls = [u.strip() for u in raw_urls.split('\n') if u.strip()]
                target_urls_raw = [{'url': u, 'pos': i+1} for i, u in enumerate(urls)]
                my_serp_pos = 0 
                st.info(f"Для анализа выбрано **{len(target_urls_raw)}** конкурентов из ручного списка.")
            else:
                target_urls_raw = []
        
        if not target_urls_raw and my_input_type != "Исходный код страницы или текст":
            st.error("Не удалось собрать конкурентов для анализа!")
            st.stop()
            
        # --- ПАРСИНГ ---
        comp_data_full = []
        if target_urls_raw:
            with st.spinner(f"Парсинг {len(target_urls_raw)} страниц конкурентов..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(parse_page, item['url'], settings): item for item in target_urls_raw}
                    for future in concurrent.futures.as_completed(futures):
                        item = futures[future]
                        try:
                            data = future.result()
                            if data:
                                data['pos'] = item['pos'] 
                                comp_data_full.append(data)
                        except Exception as e:
                            st.warning(f"Ошибка парсинга {item['url']}: {e}")
        
        my_data = None
        if st.session_state.my_input_type == "Релевантная страница на вашем сайте" and st.session_state.my_url_input:
            with st.spinner("Парсинг Вашей страницы..."):
                my_data = parse_page(st.session_state.my_url_input, settings)
                if not my_data:
                    st.error("Не удалось спарсить Вашу страницу. Проверьте URL или настройки (User-Agent).")
        
        elif st.session_state.my_input_type == "Исходный код страницы или текст" and st.session_state.my_content_input:
            my_data = {
                'url': 'Local Content',
                'domain': 'local-content.ru',
                'body_text': st.session_state.my_content_input,
                'anchor_text': '' 
            }
            
        if not my_data:
            st.error("Не удалось получить данные для Вашего сайта. Анализ невозможен.")
            st.stop()

        # --- АНАЛИЗ ---
        with st.spinner("Анализ данных..."):
            results = calculate_metrics(
                comp_data_full, my_data, settings, my_serp_pos, target_urls_raw
            )
            
        st.session_state.analysis_results = results
        st.session_state.analysis_done = True
        
        # --- ИЗМЕНЕНИЕ 2: Сохранение результата в историю ---
        if st.session_state.analysis_results:
            # Конвертируем DataFrames в JSON-совместимый формат (список словарей)
            new_result = {
                "timestamp": time.time(),
                # Формат времени изменен, чтобы не вызывать ошибку форматирования
                "date_str": time.strftime("%Y-%m-%d %H:%M:%S"), 
                "query": st.session_state.get('query_input', 'N/A'),
                "url": st.session_state.get('my_url_input', 'N/A'),
                "region": st.session_state.settings_region,
                "engine": st.session_state.settings_search_engine,
                # Сохраняем как список словарей
                "depth": results['depth'].to_dict(orient='records'),
                "hybrid": results['hybrid'].to_dict(orient='records'),
                "ngrams": results['ngrams'].to_dict(orient='records'),
                "relevance_top": results['relevance_top'].to_dict(orient='records'),
                "my_score": results['my_score']
            }
            
            all_results = load_results()
            all_results.insert(0, new_result) 
            save_results(all_results)
        # ---------------------------------------------------
        
        st.rerun()

    if st.session_state.analysis_done and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        with col_main:
            st.success("Анализ готов!")
            
            st.markdown(f"""
                <div style='background-color: {LIGHT_BG_MAIN}; padding: 15px; border-radius: 8px; border: 1px solid {BORDER_COLOR}; margin-bottom: 20px;'>
                    <h4 style='margin:0; color: {PRIMARY_COLOR};'>Результат вашего сайта (в баллах от 0 до 100)</h4>
                    <p style='margin:5px 0 0 0;'>Ширина (охват семантики): <b>{results['my_score']['width']}</b> | Глубина (оптимизация): <b>{results['my_score']['depth']}</b></p>
                </div>
                <div class=\"legend-box\">
                    <span class=\"text-red\">Красный</span>: слова, которых нет у вас. <span class=\"text-bold\">Жирный</span>: слова, участвующие в анализе.<br>
                    Минимум: min(среднее, медиана). Переспам: % превышения макс. диапазона. <br>
                    ℹ️ Для сортировки всего списка используйте меню над таблицей.
                </div>
            """, unsafe_allow_html=True)
        
            # Таблица ТОП/Релевантность (№ и Позиция)
            render_paginated_table(results['relevance_top'], "1. Релевантность ТОПа (Общий замер)", "tbl_relevance_top", default_sort_col="Позиция", use_abs_sort_default=False)
            
            # Таблица Глубина (основная)
            render_paginated_table(results['depth'], "2. Рекомендации по глубине (Минимум/Максимум)", "tbl_depth_1", default_sort_col="Добавить/Убрать", use_abs_sort_default=True)
            
            # Таблица Гибридный ТОП (TF-IDF)
            render_paginated_table(results['hybrid'], "3. Гибридный ТОП (TF-IDF)", "tbl_hybrid", default_sort_col="TF-IDF ТОП", use_abs_sort_default=False)

            # Таблица N-граммы
            render_paginated_table(results['ngrams'], "4. Рекомендации по N-граммам", "tbl_ngrams", default_sort_col="Добавить/Убрать", use_abs_sort_default=True)
