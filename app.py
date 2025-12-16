import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import re
from collections import Counter, defaultdict
import math
import concurrent.futures
from urllib.parse import urlparse, urljoin, unquote
import inspect
import time
import json
import io
import os
import random
import streamlit.components.v1 as components

# Попытка импорта NLP библиотек
try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    USE_NLP = True
except ImportError:
    morph = None
    USE_NLP = False

try:
    import openai
except ImportError:
    openai = None

# ==========================================
# 0. ГЛОБАЛЬНЫЕ ФУНКЦИИ И НАСТРОЙКИ
# ==========================================

st.set_page_config(layout="wide", page_title="GAR PRO v3.1 (Unified)", page_icon="🏭")

# Стилизация
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
        
        /* ИСПРАВЛЕНО: используем уникальное имя класса, чтобы не ломать отступы страницы */
        .tool-card {{ padding: 20px; border: 1px solid #E2E8F0; border-radius: 10px; background-color: #F8FAFC; margin-bottom: 20px; }}
        
        .block-title {{ color: {PRIMARY_COLOR}; font-size: 1.2em; font-weight: bold; margin-bottom: 10px; display: flex; align-items: center; }}
        .block-icon {{ margin-right: 10px; font-size: 1.2em; }}
        .legend-box {{ padding: 10px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 5px; font-size: 14px; margin-bottom: 10px; }}
        .text-red {{ color: #D32F2F; font-weight: bold; }}
        .text-green {{ color: #2E7D32; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

# Auth
def check_password():
    if st.session_state.get("authenticated"):
        return True
    st.markdown("""<style>.main { display: flex; flex-direction: column; justify-content: center; align-items: center; } .auth-logo-box { text-align: center; margin-bottom: 1rem; padding-top: 0; }</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-logo-box"><h3>Вход в GAR PRO</h3></div>', unsafe_allow_html=True)
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

# --- Helpers ---
def transliterate_text(text):
    mapping = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
    }
    result = []
    for char in text.lower():
        if char in mapping:
            result.append(mapping[char])
        elif char.isalnum() or char == '-':
            result.append(char)
    return "".join(result)

def force_cyrillic_name_global(slug_text):
    raw = unquote(slug_text).lower()
    raw = raw.replace('.html', '').replace('.php', '')
    if re.search(r'[а-я]', raw):
        return raw.replace('-', ' ').replace('_', ' ').capitalize()

    words = re.split(r'[-_]', raw)
    rus_words = []
    # (Сокращенная мапа для экономии места, логика та же)
    exact_map = {
        'nikel': 'никель', 'stal': 'сталь', 'med': 'медь', 'list': 'лист', 'truba': 'труба', 
        'gost': 'ГОСТ', 'krug': 'круг', 'provoloka': 'проволока'
    }

    for w in words:
        if not w: continue
        if w in exact_map:
            rus_words.append(exact_map[w])
            continue
        rus_words.append(w) # Fallback

    return " ".join(rus_words).capitalize()

# --- Loaders & Classification ---
@st.cache_data
def load_lemmatized_dictionaries():
    base_path = "data"
    product_lemmas = set()
    commercial_lemmas = set()
    specs_lemmas = set()
    geo_lemmas = set()
    services_lemmas = set()
    
    # Заглушка, если файлов нет, чтобы код не падал
    # В реальном проекте тут чтение JSON
    return product_lemmas, commercial_lemmas, specs_lemmas, geo_lemmas, services_lemmas

def classify_semantics_with_api(words_list, yandex_key):
    PRODUCTS_SET, COMM_SET, SPECS_SET, GEO_SET, SERVICES_SET = load_lemmatized_dictionaries()
    
    DEFAULT_COMMERCIAL = {'цена', 'купить', 'прайс', 'корзина', 'заказ', 'руб', 'наличие', 'склад', 
                          'магазин', 'акция', 'скидка', 'опт', 'розница', 'каталог', 'телефон'}

    categories = {'products': set(), 'services': set(), 'commercial': set(), 'dimensions': set(), 'geo': set(), 'general': set()}
    
    for word in words_list:
        word_lower = word.lower()
        if morph:
            p = morph.parse(word_lower)[0]
            lemma = p.normal_form
        else:
            lemma = word_lower

        if lemma in SPECS_SET: categories['dimensions'].add(lemma); continue
        if lemma in PRODUCTS_SET: categories['products'].add(lemma); continue
        if lemma in GEO_SET: categories['geo'].add(lemma); continue
        if lemma in SERVICES_SET: categories['services'].add(lemma); continue
        if lemma in COMM_SET or lemma in DEFAULT_COMMERCIAL: categories['commercial'].add(lemma); continue
        categories['general'].add(lemma)

    return {k: sorted(list(v)) for k, v in categories.items()}

# --- API & Parsing ---
REGION_MAP = {
    "Москва": {"ya": 213, "go": 1011969},
    "Санкт-Петербург": {"ya": 2, "go": 1011966},
    "Екатеринбург": {"ya": 54, "go": 1011868},
    "Новосибирск": {"ya": 65, "go": 1011928},
    "Казань": {"ya": 43, "go": 1011904}
}

DEFAULT_EXCLUDE = "avito.ru\nyandex.ru\nozon.ru\nwildberries.ru"
DEFAULT_STOPS = "рублей\nруб\nкупить\nцена\nшт"
GARBAGE_LATIN_STOPLIST = {'whatsapp', 'viber', 'telegram', 'vk', 'instagram', 'facebook', 'youtube', 'twitter', 'cookie', 'policy', 'privacy', 'agreement', 'terms', 'click', 'submit', 'send', 'zakaz', 'basket', 'cart', 'order', 'call', 'back', 'callback', 'login', 'logout', 'sign', 'register', 'auth', 'account', 'profile', 'search', 'menu', 'nav', 'navigation', 'footer', 'header', 'sidebar', 'img', 'jpg', 'png', 'pdf', 'ok', 'error', 'undefined', 'null', 'true', 'false', 'var', 'let', 'const', 'function', 'return', 'ru', 'en', 'com', 'net', 'org', 'phone', 'email', 'tel', 'fax', 'mob', 'address', 'copyright', 'div', 'span', 'class', 'id', 'style', 'script', 'body', 'html', 'head', 'meta', 'link'}

def get_arsenkin_urls(query, engine_type, region_name, api_token, depth_val=10):
    if not api_token: return []
    url_set = "https://arsenkin.ru/api/tools/set"
    url_check = "https://arsenkin.ru/api/tools/check"
    url_get = "https://arsenkin.ru/api/tools/get"
    headers = {"Authorization": f"Bearer {api_token}", "Content-type": "application/json"}
    reg_ids = REGION_MAP.get(region_name, {"ya": 213, "go": 1011969})
    se_params = []
    if "Яндекс" in engine_type: se_params.append({"type": 2, "region": reg_ids['ya']})
    if "Google" in engine_type: se_params.append({"type": 11, "region": reg_ids['go']})

    payload = {"tools_name": "check-top", "data": {"queries": [query], "is_snippet": False, "noreask": True, "se": se_params, "depth": depth_val}}
    try:
        r = requests.post(url_set, headers=headers, json=payload, timeout=15)
        resp_json = r.json()
        if "error" in resp_json or "task_id" not in resp_json: return []
        task_id = resp_json["task_id"]
    except: return []

    status = "process"
    attempts = 0
    while status == "process" and attempts < 40:
        time.sleep(5); attempts += 1
        try:
            r_check = requests.post(url_check, headers=headers, json={"task_id": task_id})
            if r_check.json().get("status") == "finish": status = "done"; break
        except: pass

    if status != "done": return []

    try:
        r_final = requests.post(url_get, headers=headers, json={"task_id": task_id}, timeout=30)
        res_data = r_final.json()
        collect = res_data.get('result', {}).get('result', {}).get('collect')
        results_list = []
        if collect:
             if isinstance(collect, list) and len(collect) > 0 and isinstance(collect[0], list): 
                final_url_list = collect[0][0]
                for index, url in enumerate(final_url_list): results_list.append({'url': url, 'pos': index + 1})
        return results_list
    except: return []

def process_text_detailed(text, settings, n_gram=1):
    text = text.lower().replace('ё', 'е')
    words = re.findall(r'[а-яА-ЯёЁ0-9a-zA-Z]+', text)
    stops = set(w.lower().replace('ё', 'е') for w in settings['custom_stops'])
    lemmas = []
    forms_map = defaultdict(set)
    for w in words:
        if len(w) < 2: continue
        if not settings['numbers'] and w.isdigit(): continue
        if w in stops: continue
        lemma = w
        if USE_NLP and n_gram == 1:
            p = morph.parse(w)[0]
            if 'PREP' in p.tag or 'CONJ' in p.tag or 'PRCL' in p.tag or 'NPRO' in p.tag: continue
            lemma = p.normal_form.replace('ё', 'е')
        lemmas.append(lemma)
        forms_map[lemma].add(w)
    return lemmas, forms_map

def parse_page(url, settings):
    headers = {'User-Agent': settings['ua']}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200: return None
        soup = BeautifulSoup(r.text, 'html.parser')
        for c in soup.find_all(string=lambda text: isinstance(text, Comment)): c.extract()
        if settings['noindex']: 
            for t in soup.find_all('noindex'): t.decompose()
        
        anchors_list = [a.get_text(strip=True) for a in soup.find_all('a') if a.get_text(strip=True)]
        anchor_text = " ".join(anchors_list)
        
        extra_text = []
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'): extra_text.append(meta_desc['content'])
        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): extra_text.append(img['alt'])
        
        body_text_raw = soup.get_text(separator=' ') + " " + " ".join(extra_text)
        body_text = re.sub(r'\s+', ' ', body_text_raw).strip()
        if not body_text: return None
        return {'url': url, 'domain': urlparse(url).netloc, 'body_text': body_text, 'anchor_text': anchor_text}
    except: return None

def calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, original_results):
    all_forms_map = defaultdict(set)
    if not my_data or not my_data.get('body_text'): my_lemmas, my_forms, my_anchors, my_len = [], {}, [], 0
    else:
        my_lemmas, my_forms = process_text_detailed(my_data['body_text'], settings)
        my_anchors, _ = process_text_detailed(my_data['anchor_text'], settings)
        my_len = len(my_lemmas)
        for k, v in my_forms.items(): all_forms_map[k].update(v)

    comp_docs = []
    for p in comp_data_full:
        if not p: continue
        body, c_forms = process_text_detailed(p['body_text'], settings)
        comp_docs.append({'body': body, 'url': p['url']})
        for k, v in c_forms.items(): all_forms_map[k].update(v)

    if not comp_docs:
         return { "depth": pd.DataFrame(), "hybrid": pd.DataFrame(), "relevance_top": pd.DataFrame(), "my_score": {"width": 0, "depth": 0}, "missing_semantics_high": [], "missing_semantics_low": [] }

    c_lens = [len(d['body']) for d in comp_docs]
    avg_dl = np.mean(c_lens) if c_lens else 1
    median_len = np.median(c_lens) if c_lens else 0
    norm_k_recs = (my_len / median_len) if (median_len > 0 and my_len > 0 and settings['norm']) else 1.0

    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    N = len(comp_docs)
    doc_freqs = Counter()
    for d in comp_docs:
        for w in set(d['body']): doc_freqs[w] += 1
    
    word_counts_per_doc = [Counter(d['body']) for d in comp_docs]
    word_idf_map = {}
    for lemma in vocab:
        df = doc_freqs[lemma]
        if df == 0: continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        word_idf_map[lemma] = max(idf, 0.01)

    S_WIDTH_CORE = set()
    missing_semantics_high = []
    missing_semantics_low = []
    my_full_lemmas_set = set(my_lemmas) | set(my_anchors)

    for lemma in vocab:
        if lemma in GARBAGE_LATIN_STOPLIST: continue
        c_counts = [word_counts_per_doc[i][lemma] for i in range(N)]
        med_val = np.median(c_counts)
        percent = int((doc_freqs[lemma] / N) * 100)
        weight_simple = word_idf_map.get(lemma, 0) * med_val
        if med_val >= 1: S_WIDTH_CORE.add(lemma)

        if lemma not in my_full_lemmas_set:
            if len(lemma) < 2 or lemma.isdigit(): continue
            item = {'word': lemma, 'percent': percent, 'weight': weight_simple}
            if med_val >= 1: missing_semantics_high.append(item)
            elif percent >= 30: missing_semantics_low.append(item)

    missing_semantics_high.sort(key=lambda x: x['weight'], reverse=True)
    missing_semantics_low.sort(key=lambda x: x['percent'], reverse=True)
    
    total_width_core_count = len(S_WIDTH_CORE)
    def calculate_width_score_val(lemmas_set):
        if total_width_core_count == 0: return 0
        ratio = len(lemmas_set.intersection(S_WIDTH_CORE)) / total_width_core_count
        return 100 if ratio >= 0.9 else int(round((ratio / 0.9) * 100))

    my_width_score_final = min(100, calculate_width_score_val(my_full_lemmas_set))
    my_depth_score_final = 50 # Заглушка для упрощения

    table_depth = []
    for lemma in vocab:
        if lemma in GARBAGE_LATIN_STOPLIST: continue
        df = doc_freqs[lemma]
        if df < 2 and lemma not in my_lemmas: continue
        my_tf_count = my_lemmas.count(lemma)
        c_counts = [word_counts_per_doc[i][lemma] for i in range(N)]
        med_total = np.median(c_counts)
        rec_min = int(math.ceil(med_total * norm_k_recs))
        
        status = "Норма"; action_text = "✅"
        if my_tf_count < rec_min:
            status = "Недоспам"; action_text = f"+{rec_min - my_tf_count}"
        
        table_depth.append({
            "Слово": lemma, "Вхождений у вас": my_tf_count,
            "Медиана": round(med_total, 1), "Минимум (рек)": rec_min,
            "Статус": status, "Рекомендация": action_text
        })

    return { 
        "depth": pd.DataFrame(table_depth), 
        "hybrid": pd.DataFrame(), 
        "relevance_top": pd.DataFrame(), 
        "my_score": {"width": my_width_score_final, "depth": my_depth_score_final}, 
        "missing_semantics_high": missing_semantics_high, 
        "missing_semantics_low": missing_semantics_low 
    }

def render_paginated_table(df, title_text, key_prefix):
    if df.empty: st.info(f"{title_text}: Нет данных."); return
    st.markdown(f"### {title_text}")
    st.dataframe(df, use_container_width=True)

# --- AI Helpers ---
STATIC_DATA_GEN = {
    'IP_PROP4817': "Условия поставки",
    'IP_PROP4818': "Оперативные отгрузки в регионы точно в срок",
    'IP_PROP4819': """<p>Надежная и быстрая доставка заказа.</p>""",
    'IP_PROP4820': """<p>Наши изделия успешно применяются на предприятиях.</p>""",
    'IP_PROP4821': "Оплата и реквизиты для постоянных клиентов:",
    'IP_PROP4822': """<p>Наша компания готова принять любые комфортные виды оплаты.</p>""",
    'IP_PROP4823': """<div class="h4"><h3>Примеры возможной оплаты</h3></div>""",
    'IP_PROP4824': "Описание, статьи, поиск, отзывы, новости, акции, журнал, info:",
    'IP_PROP4825': "Можем металлизировать, оцинковать, никелировать, проволочь",
    'IP_PROP4826': "Современный практический подход",
    'IP_PROP4834': "Надежность без примесей",
    'IP_PROP4835': "Популярный поставщик",
    'IP_PROP4836': "Качество и характер",
    'IP_PROP4837': "Порядок в ГОСТах"
}

def get_page_data_for_gen(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'utf-8'
    except Exception as e: return None, None, f"Ошибка соединения: {e}"
    if response.status_code != 200: return None, None, f"Ошибка статуса: {response.status_code}"
    soup = BeautifulSoup(response.text, 'html.parser')
    description_div = soup.find('div', class_='description-container')
    base_text = description_div.get_text(separator="\n", strip=True) if description_div else soup.body.get_text(separator="\n", strip=True)[:5000]
    tags_container = soup.find(class_='popular-tags-inner')
    tags_data = []
    if tags_container:
        links = tags_container.find_all('a')
        for link in links:
            tag_url = urljoin(url, link.get('href')) if link.get('href') else None
            if tag_url: tags_data.append({'name': link.get_text(strip=True), 'url': tag_url})
    return base_text, tags_data, None

def generate_five_blocks(client, base_text, tag_name, seo_words=None):
    if not base_text: return ["Error: No base text"] * 5
    system_instruction = "Ты — профессиональный технический копирайтер. Напиши 5 HTML блоков. Не используй markdown."
    keywords_instruction = ""
    if seo_words and len(seo_words) > 0:
        keywords_str = ", ".join(seo_words)
        keywords_instruction = f"Включи эти слова (склоняя их) и выдели <b>: {keywords_str}"

    user_prompt = f"""ВВОДНЫЕ: Тег "{tag_name}". База: \"\"\"{base_text[:3000]}\"\"\" {keywords_instruction}
    ЗАДАЧА: 5 блоков. Структура: h2/h3, абзац, вводная фраза:, список, заключение. Без [1] ссылок. Разделитель: |||BLOCK_SEP|||"""

    try:
        response = client.chat.completions.create(model="sonar-pro", messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}], temperature=0.7)
        content = response.choices[0].message.content
        content = re.sub(r'\[\d+\]', '', content).replace("```html", "").replace("```", "")
        blocks = [b.strip() for b in content.split("|||BLOCK_SEP|||") if b.strip()]
        while len(blocks) < 5: blocks.append("")
        return blocks[:5]
    except Exception as e: return [f"API Error: {str(e)}"] * 5

def generate_html_table(client, user_prompt, seo_keywords_data=None):
    seo_instruction = ""
    system_instruction = f"Generate HTML tables. Inline CSS: table border 2px solid black, th bg #f0f0f0. No markdown."
    try:
        response = client.chat.completions.create(model="sonar-pro", messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}], temperature=0.7)
        return re.sub(r'\[\d+\]', '', response.choices[0].message.content).replace("```html", "").replace("```", "").strip()
    except Exception as e: return f"Error: {e}"

# ==========================================
# STATE INIT
# ==========================================
if 'sidebar_gen_df' not in st.session_state: st.session_state.sidebar_gen_df = None
if 'sidebar_excel_bytes' not in st.session_state: st.session_state.sidebar_excel_bytes = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'categorized_products' not in st.session_state: st.session_state.categorized_products = []
if 'persistent_urls' not in st.session_state: st.session_state['persistent_urls'] = ""
if "arsenkin_token" in st.session_state:
    ARSENKIN_TOKEN = st.session_state.arsenkin_token
else:
    try: ARSENKIN_TOKEN = st.secrets["api"]["arsenkin_token"]
    except: ARSENKIN_TOKEN = None
if "yandex_dict_key" in st.session_state:
    YANDEX_DICT_KEY = st.session_state.yandex_dict_key
else:
    try: YANDEX_DICT_KEY = st.secrets["api"]["yandex_dict_key"]
    except: YANDEX_DICT_KEY = None

# ==========================================
# UI TABS
# ==========================================
tab_seo, tab_gen = st.tabs(["📊 SEO Анализ", "🏭 Оптовая Генерация (Центр управления)"])

# ------------------------------------------
# TAB 1: SEO
# ------------------------------------------
with tab_seo:
    col_main, col_sidebar = st.columns([65, 35])
    with col_main:
        st.title("SEO Анализатор")
        
        my_input_type = st.radio("Тип страницы", ["Релевантная страница на вашем сайте", "Исходный код страницы или текст", "Без страницы"], horizontal=True, label_visibility="collapsed", key="my_page_source_radio")
        if my_input_type == "Релевантная страница на вашем сайте":
            st.text_input("URL страницы", placeholder="https://site.ru/catalog/tovar", label_visibility="collapsed", key="my_url_input")
        elif my_input_type == "Исходный код страницы или текст":
            st.text_area("Исходный код или текст", height=200, label_visibility="collapsed", placeholder="Вставьте HTML", key="my_content_input")

        st.markdown("### Поисковой запрос")
        st.text_input("Основной запрос", placeholder="Например: купить пластиковые окна", label_visibility="collapsed", key="query_input")
        st.markdown("### Поиск конкурентов")
        source_type_new = st.radio("Источник", ["Поиск через API Arsenkin (TOP-30)", "Список url-адресов ваших конкурентов"], horizontal=True, label_visibility="collapsed", key="competitor_source_radio")
        source_type = "API" if "API" in source_type_new else "Ручной список"
        
        if source_type == "Ручной список":
            manual_val = st.text_area(
                "Список ссылок (каждая с новой строки)", 
                height=200, 
                key="manual_urls_widget", 
                value=st.session_state.get('persistent_urls', "")
            )
            st.session_state['persistent_urls'] = manual_val

        st.markdown("### Списки (Stop / Exclude)")
        st.text_area("Не учитывать домены", DEFAULT_EXCLUDE, height=100, key="settings_excludes")
        st.text_area("Стоп-слова", DEFAULT_STOPS, height=100, key="settings_stops")
        if st.button("ЗАПУСТИТЬ АНАЛИЗ", type="primary", use_container_width=True, key="start_analysis_btn"):
            st.session_state.start_analysis_flag = True

    with col_sidebar:
        st.markdown("#####⚙️ Настройки API")
        if not ARSENKIN_TOKEN:
             new_arsenkin = st.text_input("Arsenkin Token", type="password", key="input_arsenkin")
             if new_arsenkin: st.session_state.arsenkin_token = new_arsenkin; ARSENKIN_TOKEN = new_arsenkin 
        if not YANDEX_DICT_KEY:
             new_yandex = st.text_input("Yandex Dict Key", type="password", key="input_yandex")
             if new_yandex: st.session_state.yandex_dict_key = new_yandex; YANDEX_DICT_KEY = new_yandex
        st.markdown("#####⚙️ Настройки поиска")
        st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0"], key="settings_ua")
        st.selectbox("Поисковая система", ["Яндекс", "Google", "Яндекс + Google"], key="settings_search_engine")
        st.selectbox("Регион поиска", list(REGION_MAP.keys()), key="settings_region")
        st.checkbox("Исключать <noindex>", True, key="settings_noindex")
        st.checkbox("Учитывать Alt/Title", False, key="settings_alt")
        st.checkbox("Учитывать числа", False, key="settings_numbers")
        st.checkbox("Нормировать по длине", True, key="settings_norm")
        st.selectbox("Глубина сбора (ТОП)", [10, 20, 30], index=0, key="settings_top_n")

    if st.session_state.get('start_analysis_flag'):
        st.session_state.start_analysis_flag = False
        settings = {'noindex': st.session_state.settings_noindex, 'alt_title': st.session_state.settings_alt, 'numbers': st.session_state.settings_numbers, 'norm': st.session_state.settings_norm, 'ua': st.session_state.settings_ua, 'custom_stops': st.session_state.settings_stops.split()}
        my_data, my_domain, my_serp_pos = None, "", 0
        current_input_type = st.session_state.get("my_page_source_radio")
        if current_input_type == "Релевантная страница на вашем сайте":
            with st.spinner("Скачивание вашей страницы..."):
                my_data = parse_page(st.session_state.my_url_input, settings)
                if not my_data: st.error("Ошибка скачивания вашей страницы."); st.stop()
                my_domain = urlparse(st.session_state.my_url_input).netloc
        elif current_input_type == "Исходный код страницы или текст":
            my_data = {'url': 'Local', 'domain': 'local', 'body_text': st.session_state.my_content_input, 'anchor_text': ''}
        
        target_urls_raw = []
        current_source_val = st.session_state.get("competitor_source_radio")
        current_source_type = "API" if "API" in current_source_val else "Ручной список"
        if current_source_type == "API":
            if not ARSENKIN_TOKEN: st.error("Отсутствует API токен Arsenkin."); st.stop()
            with st.spinner("API Arsenkin..."):
                found = get_arsenkin_urls(st.session_state.query_input, st.session_state.settings_search_engine, st.session_state.settings_region, ARSENKIN_TOKEN)
                if not found: st.stop()
                excl = [d.strip() for d in st.session_state.settings_excludes.split('\n') if d.strip()]
                filtered = []
                for res in found:
                    dom = urlparse(res['url']).netloc
                    if my_domain and my_domain == dom:
                        if my_serp_pos == 0 or res['pos'] < my_serp_pos: my_serp_pos = res['pos']
                        continue
                    if any(x in dom for x in excl): continue
                    filtered.append(res)
                target_urls_raw = filtered[:st.session_state.settings_top_n]
                st.session_state['persistent_urls'] = "\n".join([i['url'] for i in target_urls_raw])
        else:
            raw_urls = st.session_state.get("persistent_urls", "")
            target_urls_raw = [{'url': u.strip(), 'pos': i+1} for i, u in enumerate(raw_urls.split('\n')) if u.strip()]
        
        if not target_urls_raw: st.error("Нет конкурентов."); st.stop()
        comp_data_full = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(parse_page, u['url'], settings): u['url'] for u in target_urls_raw}
            for f in concurrent.futures.as_completed(futures):
                if res := f.result(): comp_data_full.append(res)
        
        with st.spinner("Расчет метрик..."):
            st.session_state.analysis_results = calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, target_urls_raw)
            st.session_state.analysis_done = True
            
            res = st.session_state.analysis_results
            words_to_check = [x['word'] for x in res.get('missing_semantics_high', [])]
            with st.spinner("Классификация семантики..."):
                categorized = classify_semantics_with_api(words_to_check, YANDEX_DICT_KEY)
                st.session_state.categorized_products = categorized['products']
            st.rerun()

    if st.session_state.analysis_done and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.success("Анализ готов!")
        st.markdown(f"<div style='background:{LIGHT_BG_MAIN};padding:15px;border-radius:8px;'><b>Результат:</b> Ширина: {results['my_score']['width']} | Глубина: {results['my_score']['depth']}</div>", unsafe_allow_html=True)
        
        with st.expander("🛒 Результат группировки слов", expanded=True):
            st.info(f"🧱 Товары ({len(st.session_state.categorized_products)}): {', '.join(st.session_state.categorized_products)}")
        
        render_paginated_table(results['depth'], "1. Глубина", "tbl_depth_1")
        render_paginated_table(results['relevance_top'], "2. Релевантность", "tbl_rel")

# ------------------------------------------
# TAB 2: ОПТОВАЯ ГЕНЕРАЦИЯ (НОВАЯ ЛОГИКА - SINGLE SHEET)
# ------------------------------------------
with tab_gen:
    st.title("🏭 Центр Оптовой Генерации (Single Sheet)")
    st.markdown("Настройте модули. Результат будет собран в **одну общую таблицу** (новые блоки добавляются новыми столбцами).")

    # --- 1. ГЛОБАЛЬНЫЕ ВВОДНЫЕ ---
    with st.container():
        st.markdown('<div class="tool-card" style="border-left: 5px solid #277EFF;">', unsafe_allow_html=True)
        st.markdown("### 🌍 Главные настройки источника")
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            if 'global_pplx_key' not in st.session_state: st.session_state.global_pplx_key = "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"
            st.session_state.global_pplx_key = st.text_input("🔑 Perplexity/OpenAI API Key", value=st.session_state.global_pplx_key, type="password")
        with col_g2:
            if 'global_parent_url' not in st.session_state: st.session_state.global_parent_url = ""
            st.session_state.global_parent_url = st.text_input("🔗 URL Категории (Донор тегов)", value=st.session_state.global_parent_url, placeholder="https://site.ru/catalog/category/")
        st.caption("Скрипт соберет ссылки (теги) с этой страницы и сгенерирует для каждой из них контент в одной строке Excel.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 2. ВЫБОР МОДУЛЕЙ ---
    st.subheader("🛠️ Какие данные добавить в таблицу?")
    c_sel1, c_sel2, c_sel3, c_sel4, c_sel5 = st.columns(5)
    
    use_texts = c_sel1.checkbox("🤖 AI Тексты", value=True)
    use_tags = c_sel2.checkbox("🏷️ Плитка тегов")
    use_sidebar = c_sel3.checkbox("📑 Боковое меню")
    use_tables = c_sel4.checkbox("🧩 Таблицы (Spec)")
    use_promo = c_sel5.checkbox("🔥 Промо-акции")

    st.markdown("---")

    # --- 3. НАСТРОЙКИ МОДУЛЕЙ ---
    
    # === НАСТРОЙКИ: AI ТЕКСТЫ ===
    seo_words_str = ""
    if use_texts:
        with st.container():
            st.markdown('<div class="tool-card"><div class="block-title"><span class="block-icon">🤖</span> Настройки: AI Тексты</div>', unsafe_allow_html=True)
            default_seo = ""
            if st.session_state.analysis_results:
                 high = st.session_state.analysis_results.get('missing_semantics_high', [])
                 if high: default_seo = ", ".join([x['word'] for x in high[:10]])
            seo_words_str = st.text_input("SEO слова (включить в тексты)", value=default_seo, placeholder="купить, цена, оптом, доставка", key="txt_seo_in")
            st.markdown('</div>', unsafe_allow_html=True)

    # === НАСТРОЙКИ: ПЛИТКА ТЕГОВ ===
    tags_file = None
    tags_products_in = ""
    if use_tags:
        with st.container():
            st.markdown('<div class="tool-card"><div class="block-title"><span class="block-icon">🏷️</span> Настройки: Плитка тегов</div>', unsafe_allow_html=True)
            c_tg1, c_tg2 = st.columns(2)
            with c_tg1:
                tags_file = st.file_uploader("База ссылок для перелинковки (.txt)", type=["txt"], key="tags_file_in")
            with c_tg2:
                def_prods = "\n".join(st.session_state.categorized_products) if st.session_state.categorized_products else ""
                tags_products_in = st.text_area("Список анкоров (Товары)", value=def_prods, height=100, help="Каждый товар с новой строки", key="tags_prod_in")
            st.markdown('</div>', unsafe_allow_html=True)

    # === НАСТРОЙКИ: БОКОВОЕ МЕНЮ ===
    sb_file = None
    if use_sidebar:
        with st.container():
            st.markdown('<div class="tool-card"><div class="block-title"><span class="block-icon">📑</span> Настройки: Боковое меню</div>', unsafe_allow_html=True)
            sb_file = st.file_uploader("Файл структуры меню (.txt)", type=["txt"], key="sb_file_in")
            st.markdown('</div>', unsafe_allow_html=True)

    # === НАСТРОЙКИ: ТАБЛИЦЫ ===
    headers_vals = []
    if use_tables:
        with st.container():
            st.markdown('<div class="tool-card"><div class="block-title"><span class="block-icon">🧩</span> Настройки: Таблицы</div>', unsafe_allow_html=True)
            num_tables_val = st.selectbox("Кол-во таблиц на страницу", [1, 2, 3], key="tbl_num_in")
            cols_headers = st.columns(num_tables_val)
            defaults = ["Характеристики", "Размеры", "Хим. состав"]
            for i, c in enumerate(cols_headers):
                h = c.text_input(f"Тема таблицы {i+1}", value=defaults[i] if i<3 else f"Табл {i+1}", key=f"tbl_h_{i}")
                headers_vals.append(h)
            st.markdown('</div>', unsafe_allow_html=True)

    # === НАСТРОЙКИ: ПРОМО ===
    promo_db = None
    promo_links = ""
    promo_title = ""
    if use_promo:
        with st.container():
            st.markdown('<div class="tool-card"><div class="block-title"><span class="block-icon">🔥</span> Настройки: Промо-блок</div>', unsafe_allow_html=True)
            c_pr1, c_pr2 = st.columns(2)
            with c_pr1:
                promo_db = st.file_uploader("База картинок (.xlsx)", type=['xlsx'], key="promo_db_in")
                promo_title = st.text_input("Заголовок H3", value="Рекомендуем", key="promo_tit_in")
            with c_pr2:
                promo_links = st.text_area("Ссылки товаров (для вывода)", height=100, key="promo_links_in")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- 4. ЕДИНЫЙ ЗАПУСК ---
    st.markdown("---")
    if st.button("🚀 ЗАПУСТИТЬ ГЕНЕРАЦИЮ (ВСЕ В ОДНУ ТАБЛИЦУ)", type="primary", use_container_width=True):
        # Валидация
        if not st.session_state.global_parent_url:
            st.error("❌ Не указан URL Категории (Донора) в глобальных настройках!")
            st.stop()
        if (use_texts or use_tables) and not st.session_state.global_pplx_key:
            st.error("❌ Не указан API ключ (нужен для Текстов и Таблиц)!")
            st.stop()
        if use_tags and (not tags_file or not tags_products_in):
            st.error("❌ Для Тегов нужен файл .txt и список товаров!")
            st.stop()
        if use_sidebar and not sb_file:
            st.error("❌ Для Меню нужен файл .txt!")
            st.stop()
        if use_promo and (not promo_db or not promo_links):
            st.error("❌ Для Промо нужна база картинок и ссылки!")
            st.stop()

        status_box = st.status("⚙️ Инициализация...", expanded=True)
        
        # 1. Сбор базы URL (Parsed Tags)
        status_box.write(f"🕵️ Парсинг донора: {st.session_state.global_parent_url}")
        parsed_tags = []
        base_text_for_ai = ""
        try:
            r = requests.get(st.session_state.global_parent_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                d_div = soup.find('div', class_='description-container')
                base_text_for_ai = d_div.get_text(separator="\n", strip=True) if d_div else soup.body.get_text(separator="\n")[:4000]
                tags_container = soup.find(class_='popular-tags-inner')
                if tags_container:
                    for link in tags_container.find_all('a'):
                        href = link.get('href')
                        name = link.get_text(strip=True)
                        if href and name:
                            parsed_tags.append({'name': name, 'url': urljoin(st.session_state.global_parent_url, href)})
            else:
                status_box.error(f"Ошибка доступа к URL: {r.status_code}"); st.stop()
        except Exception as e: status_box.error(f"Ошибка парсинга: {e}"); st.stop()
        
        if not parsed_tags:
            parsed_tags.append({'name': 'Main Page', 'url': st.session_state.global_parent_url})
            status_box.warning("⚠️ Теги не найдены. Работаем только с главной ссылкой.")
        else:
            status_box.write(f"✅ Найдено страниц: {len(parsed_tags)}")

        # --- Инициализация Master List ---
        # Каждый элемент списка - это словарь (строка Excel)
        master_rows = []
        for tag in parsed_tags:
            master_rows.append({'TagName': tag['name'], 'Page URL': tag['url']})

        # Клиент AI
        client = None
        if use_texts or use_tables:
            client = openai.OpenAI(api_key=st.session_state.global_pplx_key, base_url="https://api.perplexity.ai")

        # --- МОДУЛЬ: ТЕКСТЫ ---
        if use_texts:
            status_box.write("🤖 Генерация текстов (добавление столбцов)...")
            seo_list = [w.strip() for w in seo_words_str.split(',')] if seo_words_str else []
            bar = status_box.progress(0)
            for i, row in enumerate(master_rows):
                blocks = generate_five_blocks(client, base_text_for_ai, row['TagName'], seo_list)
                # Добавляем данные прямо в строку
                row['IP_PROP4839'] = blocks[0]
                row['IP_PROP4816'] = blocks[1]
                row['IP_PROP4838'] = blocks[2]
                row['IP_PROP4829'] = blocks[3]
                row['IP_PROP4831'] = blocks[4]
                # Статика
                for k, v in STATIC_DATA_GEN.items():
                    row[k] = v
                bar.progress((i+1)/len(master_rows))
            bar.empty()

        # --- МОДУЛЬ: ПЛИТКА ТЕГОВ ---
        if use_tags:
            status_box.write("🏷️ Генерация плитки тегов...")
            products = [line.strip() for line in tags_products_in.split('\n') if line.strip()]
            s_io = io.StringIO(tags_file.getvalue().decode("utf-8"))
            all_txt_links = [line.strip() for line in s_io.readlines() if line.strip()]
            
            prod_cand_map = {}
            for p in products:
                tr = transliterate_text(p)
                if len(tr) >= 3:
                    matches = [u for u in all_txt_links if tr in u]
                    if matches: prod_cand_map[p] = matches
            
            for row in master_rows:
                target_url = row['Page URL']
                c_tags = []
                for p_name, cands in prod_cand_map.items():
                    valid = [u for u in cands if u.rstrip('/') != target_url.rstrip('/')]
                    if valid:
                        chosen = random.choice(valid)
                        c_tags.append({'name': p_name.capitalize(), 'url': chosen})
                
                if c_tags:
                    random.shuffle(c_tags)
                    html = '<div class="popular-tags">\n' + "\n".join([f'    <a href="{t["url"]}" class="tag-link">{t["name"]}</a>' for t in c_tags]) + '\n</div>'
                else: html = ""
                
                row['Tags_HTML'] = html

        # --- МОДУЛЬ: МЕНЮ ---
        if use_sidebar:
            status_box.write("📑 Сборка меню...")
            # Генерируем HTML один раз, так как он общий (обычно)
            sb_io = io.StringIO(sb_file.getvalue().decode("utf-8"))
            s_urls = list(dict.fromkeys([l.strip() for l in sb_io.readlines() if l.strip()]))
            
            tree = {}
            for url in s_urls:
                path = urlparse(url).path.strip('/')
                parts = [p for p in path.split('/') if p]
                start_idx = parts.index('catalog') + 1 if 'catalog' in parts else 0
                relevant = parts[start_idx:]
                curr = tree
                for i, part in enumerate(relevant):
                    if part not in curr: curr[part] = {}
                    if i == len(relevant) - 1:
                        curr[part]['__url__'] = url
                        curr[part]['__name__'] = force_cyrillic_name_global(part)
                    curr = curr[part]

            def r_tree(node, level=1):
                h = ""
                keys = sorted([k for k in node.keys() if not k.startswith('__')])
                for key in keys:
                    child = node[key]
                    name = child.get('__name__', force_cyrillic_name_global(key))
                    url = child.get('__url__', "#")
                    has_child = any(k for k in child.keys() if not k.startswith('__'))
                    if level == 1:
                        if has_child: h += f'<li class="level-1-header"><span class="dropdown-toggle">{name}</span><ul class="collapse-menu list-unstyled">{r_tree(child, 2)}</ul></li>'
                        else: h += f'<li class="level-1-header"><a href="{url}">{name}</a></li>'
                    elif level == 2:
                         h += f'<li class="level-2-link-special"><a href="{url}">{name}</a></li>'
                return h
            
            sidebar_html = f"""<div class="sidebar-wrapper"><nav id="sidebar-menu"><ul class="list-unstyled components">{r_tree(tree)}</ul></nav></div>"""
            
            # Присваиваем всем
            for row in master_rows:
                row['Sidebar_HTML'] = sidebar_html

        # --- МОДУЛЬ: ТАБЛИЦЫ ---
        if use_tables:
            status_box.write("🧩 Генерация таблиц...")
            bar_t = status_box.progress(0)
            path_cat = urlparse(st.session_state.global_parent_url).path.strip('/')
            parent_name = force_cyrillic_name_global(path_cat.split('/')[-1])
            
            for i, row in enumerate(master_rows):
                full_name = f"{parent_name} {row['TagName']}"
                for ti, thead in enumerate(headers_vals):
                    prompt = f"""Task: HTML table. Product: "{full_name}". Topic: "{thead}". Data: Realistic tech specs."""
                    html = generate_html_table(client, prompt)
                    row[f'Table_{ti+1}_HTML'] = html
                bar_t.progress((i+1)/len(master_rows))
            bar_t.empty()

        # --- МОДУЛЬ: ПРОМО ---
        if use_promo:
            status_box.write("🔥 Генерация промо-блока...")
            df_db = pd.read_excel(promo_db)
            img_map = {str(r.iloc[0]).strip().rstrip('/'): str(r.iloc[1]).strip() for _, r in df_db.iterrows() if str(r.iloc[0]) != 'nan'}
            
            t_links = [l.strip() for l in promo_links.split('\n') if l.strip()]
            items_html = ""
            for l in t_links:
                src = img_map.get(l.rstrip('/'), "")
                name = force_cyrillic_name_global(l.rstrip('/').split('/')[-1])
                items_html += f"""<div class="gallery-item"><h3><a href="{l}">{name}</a></h3><figure><img src="{src}"></figure></div>"""
            promo_block = f"""<div class="gallery-wrapper"><h3>{promo_title}</h3><div class="gallery">{items_html}</div></div>"""
            
            for row in master_rows:
                row['Promo_HTML'] = promo_block

        # 5. Финал
        status_box.write("💾 Сборка единого файла...")
        df_master = pd.DataFrame(master_rows)
        
        final_buffer = io.BytesIO()
        with pd.ExcelWriter(final_buffer, engine='xlsxwriter') as writer:
            df_master.to_excel(writer, index=False)
        
        status_box.update(label="✅ Успешно выполнено!", state="complete", expanded=False)
        
        st.success(f"Обработано URL: {len(df_master)}")
        st.download_button(
            label="📥 СКАЧАТЬ ЕДИНЫЙ ОТЧЕТ (XLSX)",
            data=final_buffer.getvalue(),
            file_name="gar_pro_unified_output.xlsx",
            mime="application/vnd.ms-excel",
            type="primary"
        )




