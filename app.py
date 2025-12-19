import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import re
from collections import Counter, defaultdict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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

# ==========================================
# FIX FOR PYTHON 3.11+
# ==========================================
if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

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
# 0. ГЛОБАЛЬНЫЕ ФУНКЦИИ
# ==========================================

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
    
    exact_map = {
        'nikel': 'никель', 'stal': 'сталь', 'med': 'медь', 'latun': 'латунь',
        'bronza': 'бронза', 'svinec': 'свинец', 'titan': 'титан', 'tsink': 'цинк',
        'dural': 'дюраль', 'dyural': 'дюраль', 'chugun': 'чугун',
        'alyuminiy': 'алюминий', 'al': 'алюминиевая', 'alyuminievaya': 'алюминиевая',
        'nerzhaveyushchiy': 'нержавеющий', 'nerzhaveyka': 'нержавейка',
        'profil': 'профиль', 'shveller': 'швеллер', 'ugolok': 'уголок',
        'polosa': 'полоса', 'krug': 'круг', 'kvadrat': 'квадрат',
        'list': 'лист', 'truba': 'труба', 'setka': 'сетка',
        'provoloka': 'проволока', 'armatura': 'арматура', 'balka': 'балка',
        'katanka': 'катанка', 'otvod': 'отвод', 'perehod': 'переход',
        'flanec': 'фланец', 'zaglushka': 'заглушка', 'metiz': 'метизы',
        'profnastil': 'профнастил', 'shtrips': 'штрипс', 'lenta': 'лента',
        'shina': 'шина', 'prutok': 'пруток', 'shestigrannik': 'шестигранник',
        'vtulka': 'втулка', 'kabel': 'кабель', 'panel': 'панель',
        'detal': 'деталь', 'set': 'сеть', 'cep': 'цепь', 'svyaz': 'связь',
        'rezba': 'резьба', 'gost': 'ГОСТ',
        'polipropilenovye': 'полипропиленовые', 'truby': 'трубы',
        'ocinkovannaya': 'оцинкованная', 'riflenyy': 'рифленый'
    }

    for w in words:
        if not w: continue
        if w in exact_map:
            rus_words.append(exact_map[w])
            continue
        
        processed_w = w
        if processed_w.endswith('yy'): processed_w = processed_w[:-2] + 'ый'
        elif processed_w.endswith('iy'): processed_w = processed_w[:-2] + 'ий'
        elif processed_w.endswith('ij'): processed_w = processed_w[:-2] + 'ий'
        elif processed_w.endswith('yi'): processed_w = processed_w[:-2] + 'ий'
        elif processed_w.endswith('aya'): processed_w = processed_w[:-3] + 'ая'
        elif processed_w.endswith('oye'): processed_w = processed_w[:-3] + 'ое'
        elif processed_w.endswith('ye'): processed_w = processed_w[:-2] + 'ые'

        replacements = [
            ('shch', 'щ'), ('sch', 'щ'), ('yo', 'ё'), ('zh', 'ж'), ('ch', 'ч'), ('sh', 'ш'), 
            ('yu', 'ю'), ('ya', 'я'), ('kh', 'х'), ('ts', 'ц'), ('ph', 'ф'),
            ('a', 'а'), ('b', 'б'), ('v', 'в'), ('g', 'г'), ('d', 'д'), ('e', 'е'), 
            ('z', 'з'), ('i', 'и'), ('j', 'й'), ('k', 'к'), ('l', 'л'), ('m', 'м'), 
            ('n', 'н'), ('o', 'о'), ('p', 'п'), ('r', 'р'), ('s', 'с'), ('t', 'т'), 
            ('u', 'у'), ('f', 'ф'), ('h', 'х'), ('c', 'к'), ('w', 'в'), ('y', 'ы'), ('x', 'кс')
        ]
        
        temp_res = processed_w
        for eng, rus in replacements:
            temp_res = temp_res.replace(eng, rus)
        
        rus_words.append(temp_res)

    draft_phrase = " ".join(rus_words)
    draft_phrase = draft_phrase.replace('профил', 'профиль').replace('профильн', 'профильн')
    draft_phrase = draft_phrase.replace('елный', 'ельный').replace('алный', 'альный')
    draft_phrase = draft_phrase.replace('елная', 'ельная').replace('алная', 'альная')
    draft_phrase = draft_phrase.replace('сталн', 'стальн').replace('медьн', 'медн')
    draft_phrase = draft_phrase.replace('йа', 'я').replace('йо', 'ё')

    return draft_phrase.capitalize()

def get_breadcrumb_only(url, ua_settings="Mozilla/5.0"):
    """
    Заходит по URL и достает название ТОЛЬКО из последнего элемента хлебных крошек.
    H1 игнорируется.
    """
    try:
        headers = {'User-Agent': ua_settings}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200: 
            return None
        
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # 1. Ищем контейнер хлебных крошек по популярным классам/id
        breadcrumbs = soup.find(class_=re.compile(r'breadcrumb|breadcrumbs|nav-path|nav-chain|bx-breadcrumb', re.I))
        if not breadcrumbs:
            breadcrumbs = soup.find(id=re.compile(r'breadcrumb|breadcrumbs|nav-path', re.I))

        if breadcrumbs:
            full_text = breadcrumbs.get_text(separator='|||', strip=True)
            parts = [p.strip() for p in full_text.split('|||') if p.strip()]
            clean_parts = [p for p in parts if p not in ['/', '\\', '>', '»', '•', '-', '|']]
            
            if clean_parts:
                last_item = clean_parts[-1]
                if len(last_item) > 2 and last_item.lower() != "главная":
                    return last_item
    except:
        return None
    return None

# ==========================================
# ЗАГРУЗКА СЛОВАРЕЙ
# ==========================================
@st.cache_data
def load_lemmatized_dictionaries():
    base_path = "data"
    
    product_lemmas = set()
    commercial_lemmas = set()
    specs_lemmas = set()
    geo_lemmas = set()
    services_lemmas = set()

    # 1. ТОВАРЫ
    path_prod = os.path.join(base_path, "metal_products.json")
    if os.path.exists(path_prod):
        try:
            with open(path_prod, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_raw_words = []
                if isinstance(data, dict):
                    for cat_list in data.values():
                        all_raw_words.extend(cat_list)
                elif isinstance(data, list):
                    all_raw_words = data
                
                for phrase in all_raw_words:
                    words = str(phrase).lower().split() 
                    for w in words:
                        clean_w = re.sub(r'[^a-zа-яё0-9-]', '', w)
                        if not clean_w: continue
                        product_lemmas.add(clean_w) # Оригинал
                        if morph: product_lemmas.add(morph.parse(clean_w)[0].normal_form) # Лемма
        except Exception as e:
            st.error(f"Ошибка в metal_products.json: {e}")

    # 2. КОММЕРЦИЯ
    path_comm = os.path.join(base_path, "commercial_triggers.json")
    if os.path.exists(path_comm):
        try:
            with open(path_comm, 'r', encoding='utf-8') as f:
                raw_comm = json.load(f)
                if isinstance(raw_comm, list):
                    for w in raw_comm:
                        w_clean = str(w).lower().strip()
                        commercial_lemmas.add(w_clean)
                        if morph: 
                            commercial_lemmas.add(morph.parse(w_clean)[0].normal_form)
        except: pass

    # 3. ГЕО
    path_geo = os.path.join(base_path, "geo_locations.json")
    if os.path.exists(path_geo):
        try:
            with open(path_geo, 'r', encoding='utf-8') as f:
                raw_geo = json.load(f)
                for w in raw_geo:
                    w_clean = str(w).lower().strip()
                    geo_lemmas.add(w_clean)
                    if morph: geo_lemmas.add(morph.parse(w_clean)[0].normal_form)
        except Exception as e:
            st.error(f"Ошибка в geo_locations.json: {e}")

    # 4. УСЛУГИ
    path_serv = os.path.join(base_path, "services_triggers.json")
    if os.path.exists(path_serv):
        try:
            with open(path_serv, 'r', encoding='utf-8') as f:
                raw_serv = json.load(f)
                if isinstance(raw_serv, list):
                    for w in raw_serv:
                        parts = str(w).replace('-', ' ').lower().split()
                        for part in parts:
                            services_lemmas.add(part)
                            if morph: services_lemmas.add(morph.parse(part)[0].normal_form)
        except Exception as e:
            st.error(f"Ошибка в services_triggers.json: {e}")

    # 5. ХАРАКТЕРИСТИКИ
    path_specs = os.path.join(base_path, "tech_specs.json")
    if os.path.exists(path_specs):
        try:
            with open(path_specs, 'r', encoding='utf-8') as f:
                raw_specs = json.load(f)
                if isinstance(raw_specs, list):
                    for w in raw_specs:
                        w_clean = str(w).lower().strip()
                        specs_lemmas.add(w_clean)
                        if morph: specs_lemmas.add(morph.parse(w_clean)[0].normal_form)
        except Exception as e:
            st.error(f"Ошибка в tech_specs.json: {e}")

    return product_lemmas, commercial_lemmas, specs_lemmas, geo_lemmas, services_lemmas

# ==========================================
# КЛАССИФИКАТОР
# ==========================================
def classify_semantics_with_api(words_list, yandex_key):
    PRODUCTS_SET, COMM_SET, SPECS_SET, GEO_SET, SERVICES_SET = load_lemmatized_dictionaries()
    
    if 'debug_geo_count' not in st.session_state:
        st.session_state.debug_geo_count = len(GEO_SET)
    
    st.sidebar.info(f"Статус баз:\n📦 Товары: {len(PRODUCTS_SET)}\n🛠️ Услуги: {len(SERVICES_SET)}\n⚙️ Марки/ГОСТ: {len(SPECS_SET)}\n🌍 Города: {len(GEO_SET)}")

    dim_pattern = re.compile(r'\d+(?:[\.\,]\d+)?\s?[хx\*×]\s?\d+', re.IGNORECASE)
    grade_pattern = re.compile(r'^([а-яa-z]{1,4}\-?\d+[а-яa-z0-9]*)$', re.IGNORECASE)
    
    DEFAULT_COMMERCIAL = {'цена', 'купить', 'прайс', 'корзина', 'заказ', 'руб', 'наличие', 'склад', 
                          'магазин', 'акция', 'скидка', 'опт', 'розница', 'каталог', 'телефон', 
                          'менеджер', 'сайт', 'главная', 'вход', 'регистрация', 'отзыв', 'гарантия'}

    categories = {'products': set(), 'services': set(), 'commercial': set(), 'dimensions': set(), 'geo': set(), 'general': set()}
    
    for word in words_list:
        word_lower = word.lower()
        
        # 1. Сначала проверяем на технические параметры
        if word_lower in SPECS_SET:
            categories['dimensions'].add(word_lower)
            continue
            
        if morph:
            p = morph.parse(word_lower)[0]
            lemma = p.normal_form
        else:
            lemma = word_lower

        if lemma in SPECS_SET:
            categories['dimensions'].add(lemma)
            continue

        # 2. Регулярки для размеров
        if dim_pattern.search(word_lower) or grade_pattern.match(word_lower) or word_lower.isdigit():
            categories['dimensions'].add(word_lower)
            continue

        # 3. ТОВАРЫ
        if lemma in PRODUCTS_SET:
            categories['products'].add(lemma)
            continue 

        # 4. ГЕО
        if lemma in GEO_SET:
            categories['geo'].add(lemma)
            continue
        
        is_geo_derivative = False
        if len(lemma) > 5: 
            for city in GEO_SET:
                if len(city) > 4 and lemma.startswith(city[:-1]): 
                    categories['geo'].add(lemma)
                    is_geo_derivative = True
                    break
        if is_geo_derivative: continue

        # 5. УСЛУГИ
        if lemma in SERVICES_SET or lemma.endswith('обработка') or lemma.endswith('изготовление'):
            categories['services'].add(lemma)
            continue

        # 6. КОММЕРЦИЯ
        if lemma in COMM_SET or lemma in DEFAULT_COMMERCIAL:
            categories['commercial'].add(lemma)
            continue
            
        # 7. ОБЩИЕ
        categories['general'].add(lemma)

    return {k: sorted(list(v)) for k, v in categories.items()}

# ==========================================
# STATE INIT
# ==========================================
if 'sidebar_gen_df' not in st.session_state: st.session_state.sidebar_gen_df = None
if 'sidebar_excel_bytes' not in st.session_state: st.session_state.sidebar_excel_bytes = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'ai_generated_df' not in st.session_state: st.session_state.ai_generated_df = None
if 'ai_excel_bytes' not in st.session_state: st.session_state.ai_excel_bytes = None
if 'tags_html_result' not in st.session_state: st.session_state.tags_html_result = None
if 'table_html_result' not in st.session_state: st.session_state.table_html_result = None
if 'tags_generated_df' not in st.session_state: st.session_state.tags_generated_df = None
if 'tags_excel_data' not in st.session_state: st.session_state.tags_excel_data = None
if 'categorized_products' not in st.session_state: st.session_state.categorized_products = []
if 'categorized_services' not in st.session_state: st.session_state.categorized_services = []
if 'categorized_commercial' not in st.session_state: st.session_state.categorized_commercial = []
if 'categorized_dimensions' not in st.session_state: st.session_state.categorized_dimensions = []
if 'categorized_geo' not in st.session_state: st.session_state.categorized_geo = []
if 'categorized_general' not in st.session_state: st.session_state.categorized_general = []
if 'auto_tags_words' not in st.session_state: st.session_state.auto_tags_words = []
if 'auto_promo_words' not in st.session_state: st.session_state.auto_promo_words = []
if 'persistent_urls' not in st.session_state: st.session_state['persistent_urls'] = ""

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO v2.6 (Mass Promo)", page_icon="📊")

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

def check_password():
    if st.session_state.get("authenticated"):
        return True
    st.markdown("""<style>.main { display: flex; flex-direction: column; justify-content: center; align-items: center; } .auth-logo-box { text-align: center; margin-bottom: 1rem; padding-top: 0; }</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-logo-box"><h3>Вход в систему</h3></div>', unsafe_allow_html=True)
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

if "arsenkin_token" in st.session_state:
    ARSENKIN_TOKEN = st.session_state.arsenkin_token
else:
    try: ARSENKIN_TOKEN = st.secrets["api"]["arsenkin_token"]
    except (FileNotFoundError, KeyError): ARSENKIN_TOKEN = None

if "yandex_dict_key" in st.session_state:
    YANDEX_DICT_KEY = st.session_state.yandex_dict_key
else:
    try: YANDEX_DICT_KEY = st.secrets["api"]["yandex_dict_key"]
    except (FileNotFoundError, KeyError): YANDEX_DICT_KEY = None

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

DEFAULT_EXCLUDE_DOMAINS = ["yandex.ru", "avito.ru", "beru.ru", "tiu.ru", "aliexpress.com", "ebay.com", "auto.ru", "2gis.ru", "sravni.ru", "toshop.ru", "price.ru", "pandao.ru", "instagram.com", "wikipedia.org", "rambler.ru", "hh.ru", "banki.ru", "regmarkets.ru", "zoon.ru", "pulscen.ru", "prodoctorov.ru", "blizko.ru", "domclick.ru", "satom.ru", "quto.ru", "edadeal.ru", "cataloxy.ru", "irr.ru", "onliner.by", "shop.by", "deal.by", "yell.ru", "profi.ru", "irecommend.ru", "otzovik.com", "ozon.ru", "ozon.by", "market.yandex.ru", "youtube.com", "gosuslugi.ru", "dzen.ru", "2gis.by", "wildberries.ru", "rutube.ru", "vk.com", "facebook.com"]
DEFAULT_EXCLUDE = "\n".join(DEFAULT_EXCLUDE_DOMAINS)
DEFAULT_STOPS = "рублей\nруб\nкупить\nцена\nшт\nсм\nмм\nкг\nкв\nм2\nстр\nул"

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
        
        .stApp > header {{ background-color: transparent !important; }}
        .stTextInput input:disabled, .stTextArea textarea:disabled, .stSelectbox div[aria-disabled="true"] {{
            opacity: 1 !important; background-color: {LIGHT_BG_MAIN} !important; color: {TEXT_COLOR} !important; cursor: text !important; -webkit-text-fill-color: {TEXT_COLOR} !important; border-color: {BORDER_COLOR} !important;
        }}
        .stButton button:disabled {{ opacity: 1 !important; background-color: {PRIMARY_COLOR} !important; color: white !important; cursor: progress !important; }}
        div[data-testid="stAppViewContainer"] {{ filter: none !important; opacity: 1 !important; transition: none !important; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# PARSING & METRICS
# ==========================================

def get_yandex_dict_info(text, api_key):
    if not api_key: return {'lemma': text, 'pos': 'unknown'}
    url = "https://dictionary.yandex.net/api/v1/dicservice.json/lookup"
    params = {'key': api_key, 'lang': 'ru-ru', 'text': text, 'ui': 'ru'}
    try:
        r = requests.get(url, params=params, timeout=2)
        if r.status_code == 200:
            data = r.json()
            if data.get('def'):
                first_def = data['def'][0]
                return {'lemma': first_def.get('text', text), 'pos': first_def.get('pos', 'unknown')}
    except: pass
    return {'lemma': text, 'pos': 'unknown'}

def get_arsenkin_urls(query, engine_type, region_name, api_token, depth_val=10):
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
        if "error" in resp_json or "task_id" not in resp_json: st.error(f"❌ Ошибка API: {resp_json}"); return []
        task_id = resp_json["task_id"]
        st.toast(f"Задача ID {task_id} запущена")
    except Exception as e: st.error(f"❌ Ошибка сети: {e}"); return []

    status = "process"
    attempts = 0
    while status == "process" and attempts < 40:
        time.sleep(5); attempts += 1
        try:
            r_check = requests.post(url_check, headers=headers, json={"task_id": task_id})
            res_check_data = r_check.json()
            if res_check_data.get("status") == "finish": status = "done"; break
        except: pass

    if status != "done": st.error(f"⏳ Тайм-аут API"); return []

    try:
        r_final = requests.post(url_get, headers=headers, json={"task_id": task_id}, timeout=30)
        res_data = r_final.json()
    except Exception as e: st.error(f"❌ Ошибка получения результата: {e}"); return []

    results_list = []
    try:
        collect = res_data.get('result', {}).get('result', {}).get('collect')
        if not collect: return []
        final_url_list = []
        if isinstance(collect, list) and len(collect) > 0 and isinstance(collect[0], list): final_url_list = collect[0][0]
        else:
             unique_urls = set()
             for engine_data in collect:
                 if isinstance(engine_data, dict):
                     for _, serps in engine_data.items():
                         for item in serps:
                             if item.get('url') and item.get('url') not in unique_urls:
                                 results_list.append({'url': item['url'], 'pos': item['pos']})
                                 unique_urls.add(item['url'])
             return results_list

        if final_url_list:
            for index, url in enumerate(final_url_list): results_list.append({'url': url, 'pos': index + 1})
    except Exception as e: st.error(f"❌ Ошибка парсинга JSON: {e}"); return []
    return results_list

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
    # Настройка повторных попыток (Retries)
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    headers = {'User-Agent': settings['ua']}
    try:
        # Увеличили timeout до 20 секунд
        r = session.get(url, headers=headers, timeout=20)
        if r.status_code != 200: return None
        
        soup = BeautifulSoup(r.text, 'html.parser')
        
        tags_to_remove = []
        if settings['noindex']: tags_to_remove.append('noindex')
        
        # Удаляем комментарии и лишние теги
        for c in soup.find_all(string=lambda text: isinstance(text, Comment)): c.extract()
        if tags_to_remove:
            for t in soup.find_all(tags_to_remove): t.decompose()
            
        # Сбор анкоров
        anchors_list = [a.get_text(strip=True) for a in soup.find_all('a') if a.get_text(strip=True)]
        anchor_text = " ".join(anchors_list)
        
        # Сбор мета и alt
        extra_text = []
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'): extra_text.append(meta_desc['content'])
        
        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): extra_text.append(img['alt'])
            for t in soup.find_all(title=True): extra_text.append(t['title'])
            
        # Основной текст
        body_text_raw = soup.get_text(separator=' ') + " " + " ".join(extra_text)
        body_text = re.sub(r'\s+', ' ', body_text_raw).strip()
        
        if not body_text: return None
        return {'url': url, 'domain': urlparse(url).netloc, 'body_text': body_text, 'anchor_text': anchor_text}
    except: 
        return None

def calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, original_results):
    all_forms_map = defaultdict(set)
    if not my_data or not my_data.get('body_text'): my_lemmas, my_forms, my_anchors, my_len = [], {}, [], 0
    else:
        my_lemmas, my_forms = process_text_detailed(my_data['body_text'], settings)
        my_anchors, _ = process_text_detailed(my_data['anchor_text'], settings)
        my_len = len(my_lemmas)
        for k, v in my_forms.items(): all_forms_map[k].update(v)

    comp_data_parsed = [d for d in comp_data_full if d.get('body_text')]
    comp_docs = []
    for p in comp_data_parsed:
        body, c_forms = process_text_detailed(p['body_text'], settings)
        anchor, _ = process_text_detailed(p['anchor_text'], settings)
        comp_docs.append({'body': body, 'anchor': anchor, 'url': p['url'], 'domain': p['domain']})
        for k, v in c_forms.items(): all_forms_map[k].update(v)

    if not comp_docs:
        return { "depth": pd.DataFrame(), "hybrid": pd.DataFrame(), "relevance_top": pd.DataFrame(), "my_score": {"width": 0, "depth": 0}, "missing_semantics_high": [], "missing_semantics_low": [] }

    c_lens = [len(d['body']) for d in comp_docs]
    avg_dl = np.mean(c_lens) if c_lens else 1
    median_len = np.median(c_lens) if c_lens else 0
    norm_k_recs = (my_len / median_len) if (median_len > 0 and my_len > 0 and settings['norm']) else 1.0

    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    vocab = sorted(list(vocab))
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
    lsi_candidates_weighted = []

    for lemma in vocab:
        if lemma in GARBAGE_LATIN_STOPLIST: continue
        c_counts = [word_counts_per_doc[i][lemma] for i in range(N)]
        med_val = np.median(c_counts)
        percent = int((doc_freqs[lemma] / N) * 100)
        weight_simple = word_idf_map.get(lemma, 0) * med_val
        if med_val > 0: lsi_candidates_weighted.append((lemma, weight_simple))
        is_width_word = False
        if med_val >= 1: S_WIDTH_CORE.add(lemma); is_width_word = True

        if lemma not in my_full_lemmas_set:
            if len(lemma) < 2 or lemma.isdigit(): continue
            item = {'word': lemma, 'percent': percent, 'weight': weight_simple}
            if is_width_word: missing_semantics_high.append(item)
            elif percent >= 30: missing_semantics_low.append(item)

    missing_semantics_high.sort(key=lambda x: x['weight'], reverse=True)
    missing_semantics_low.sort(key=lambda x: x['percent'], reverse=True)
    lsi_candidates_weighted.sort(key=lambda x: x[1], reverse=True)
    S_DEPTH_TOP70 = set([x[0] for x in lsi_candidates_weighted[:70]])
    total_width_core_count = len(S_WIDTH_CORE)

    def calculate_bm25_okapi(doc_tokens, doc_len):
        if avg_dl == 0 or doc_len == 0: return 0
        score = 0
        counts = Counter(doc_tokens)
        k1 = 1.2; b = 0.75
        target_words = S_WIDTH_CORE if S_WIDTH_CORE else S_DEPTH_TOP70
        for word in target_words:
            if word not in counts: continue
            tf = counts[word]
            idf = word_idf_map.get(word, 0)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_dl)))
        return score

    def calculate_width_score_val(lemmas_set):
        if total_width_core_count == 0: return 0
        ratio = len(lemmas_set.intersection(S_WIDTH_CORE)) / total_width_core_count
        return 100 if ratio >= 0.9 else int(round((ratio / 0.9) * 100))

    competitor_scores_map = {}
    comp_bm25_list = []
    for i, doc in enumerate(comp_docs):
        raw_bm25 = calculate_bm25_okapi(doc['body'], c_lens[i])
        comp_bm25_list.append(raw_bm25)
        width_val = calculate_width_score_val(set(doc['body']))
        competitor_scores_map[doc['url']] = {'width_final': min(100, width_val), 'bm25_val': raw_bm25}

    median_bm25_top = np.median(comp_bm25_list) if comp_bm25_list else 0
    spam_limit = median_bm25_top * 1.25 if median_bm25_top > 0 else 1

    for url, scores in competitor_scores_map.items():
        depth_val = int(round((scores['bm25_val'] / spam_limit) * 100))
        scores['depth_final'] = min(100, depth_val)

    my_bm25 = calculate_bm25_okapi(my_lemmas, my_len)
    my_depth_score_final = min(100, int(round((my_bm25 / spam_limit) * 100)))
    my_width_score_final = min(100, calculate_width_score_val(my_full_lemmas_set))

    table_depth, table_hybrid = [], []
    for lemma in vocab:
        if lemma in GARBAGE_LATIN_STOPLIST: continue
        df = doc_freqs[lemma]
        if df < 2 and lemma not in my_lemmas: continue
        my_tf_count = my_lemmas.count(lemma)
        forms_str = ", ".join(sorted(list(all_forms_map.get(lemma, set())))) if all_forms_map.get(lemma) else lemma
        c_counts = [word_counts_per_doc[i][lemma] for i in range(N)]
        med_total = np.median(c_counts); max_total = np.max(c_counts)
        base_min = min(np.mean(c_counts), med_total)
        rec_min = int(math.ceil(base_min * norm_k_recs))
        rec_max = int(round(max_total * norm_k_recs))
        if rec_max < rec_min: rec_max = rec_min
        rec_median = med_total * norm_k_recs
        
        status = "Норма"; action_diff = 0; action_text = "✅"
        if my_tf_count < rec_min:
            status = "Недоспам"; action_diff = int(round(rec_min - my_tf_count))
            if action_diff == 0: action_diff = 1
            action_text = f"+{action_diff}"
        elif my_tf_count > rec_max:
            status = "Переспам"; action_diff = int(round(my_tf_count - rec_max))
            if action_diff == 0: action_diff = 1
            action_text = f"-{action_diff}"

        depth_percent = int(round((my_tf_count / rec_median) * 100)) if rec_median > 0.1 else (0 if my_tf_count == 0 else 100)
        weight_hybrid = word_idf_map.get(lemma, 0) * (my_tf_count / my_len if my_len > 0 else 0)
        table_depth.append({
            "Слово": lemma, "Словоформы": forms_str, "Вхождений у вас": my_tf_count,
            "Медиана": round(med_total, 1), "Минимум (рек)": rec_min, "Максимум (рек)": rec_max,
            "Глубина %": min(100, depth_percent), "Статус": status, "Рекомендация": action_text,
            "is_missing": (status == "Недоспам" and my_tf_count == 0), "sort_val": abs(action_diff) if status != "Норма" else 0
        })
        table_hybrid.append({
            "Слово": lemma, "TF-IDF ТОП": round(word_idf_map.get(lemma, 0) * (med_total / avg_dl if avg_dl > 0 else 0), 4),
            "TF-IDF у вас": round(weight_hybrid, 4), "Сайтов": df, "Переспам": max_total
        })

    table_rel = []
    for item in original_results:
        url = item['url']
        scores = competitor_scores_map.get(url, {'width_final':0, 'depth_final':0})
        table_rel.append({ "Домен": urlparse(url).netloc, "Позиция": item['pos'], "Ширина (балл)": scores['width_final'], "Глубина (балл)": scores['depth_final'] })
    my_label = f"{my_data['domain']} (Вы)" if (my_data and my_data.get('domain')) else "Ваш сайт"
    table_rel.append({ "Домен": my_label, "Позиция": my_serp_pos if my_serp_pos > 0 else len(original_results) + 1, "Ширина (балл)": my_width_score_final, "Глубина (балл)": my_depth_score_final })

    return { "depth": pd.DataFrame(table_depth), "hybrid": pd.DataFrame(table_hybrid), "relevance_top": pd.DataFrame(table_rel).sort_values(by='Позиция', ascending=True).reset_index(drop=True), "my_score": {"width": my_width_score_final, "depth": my_depth_score_final}, "missing_semantics_high": missing_semantics_high, "missing_semantics_low": missing_semantics_low }

def render_paginated_table(df, title_text, key_prefix, default_sort_col=None, use_abs_sort_default=False):
    if df.empty: st.info(f"{title_text}: Нет данных."); return
    col_t1, col_t2 = st.columns([7, 3])
    with col_t1: st.markdown(f"### {title_text}")
    if f'{key_prefix}_sort_col' not in st.session_state: st.session_state[f'{key_prefix}_sort_col'] = default_sort_col if (default_sort_col and default_sort_col in df.columns) else df.columns[0]
    if f'{key_prefix}_sort_order' not in st.session_state: st.session_state[f'{key_prefix}_sort_order'] = "Убывание"

    search_query = st.text_input(f"🔍 Поиск ({title_text})", key=f"{key_prefix}_search")
    if search_query:
        mask = df.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
        df_filtered = df[mask].copy()
    else: df_filtered = df.copy()

    if df_filtered.empty: st.warning("Ничего не найдено."); return

    with st.container():
        st.markdown("<div class='sort-container'>", unsafe_allow_html=True)
        col_s1, col_s2, col_sp = st.columns([2, 2, 4])
        with col_s1:
            current_sort = st.session_state[f'{key_prefix}_sort_col']
            if current_sort not in df_filtered.columns: current_sort = df_filtered.columns[0]
            sort_col = st.selectbox("🗂 Сортировать по:", df_filtered.columns, key=f"{key_prefix}_sort_box", index=list(df_filtered.columns).index(current_sort))
            st.session_state[f'{key_prefix}_sort_col'] = sort_col
        with col_s2:
            sort_order = st.radio("Порядок:", ["Убывание", "Возрастание"], horizontal=True, key=f"{key_prefix}_order_box", index=0 if st.session_state[f'{key_prefix}_sort_order'] == "Убывание" else 1)
            st.session_state[f'{key_prefix}_sort_order'] = sort_order
        st.markdown("</div>", unsafe_allow_html=True)

    ascending = (sort_order == "Возрастание")
    if use_abs_sort_default and sort_col == "Рекомендация" and "sort_val" in df_filtered.columns: df_filtered = df_filtered.sort_values(by="sort_val", ascending=ascending)
    elif ("Добавить" in sort_col or "+/-" in sort_col) and df_filtered[sort_col].dtype == object:
        try:
            df_filtered['_temp_sort'] = df_filtered[sort_col].astype(str).str.replace(r'[^\d]', '', regex=True)
            df_filtered['_temp_sort'] = pd.to_numeric(df_filtered['_temp_sort'], errors='coerce').fillna(0)
            df_filtered = df_filtered.sort_values(by='_temp_sort', ascending=ascending).drop(columns=['_temp_sort'])
        except: df_filtered = df_filtered.sort_values(by=sort_col, ascending=ascending)
    else: df_filtered = df_filtered.sort_values(by=sort_col, ascending=ascending)

    df_filtered = df_filtered.reset_index(drop=True); df_filtered.index = df_filtered.index + 1
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        export_df = df_filtered.copy()
        if "is_missing" in export_df.columns: del export_df["is_missing"]
        if "sort_val" in export_df.columns: del export_df["sort_val"]
        export_df.to_excel(writer, index=False, sheet_name='Data')
    excel_data = buffer.getvalue()
    with col_t2: st.download_button(label="📥 Скачать Excel", data=excel_data, file_name=f"{key_prefix}_export.xlsx", mime="application/vnd.ms-excel", key=f"{key_prefix}_down")

    ROWS_PER_PAGE = 20
    if f'{key_prefix}_page' not in st.session_state: st.session_state[f'{key_prefix}_page'] = 1
    total_rows = len(df_filtered); total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
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
                if status == "Недоспам": cell_style += "color: #D32F2F; font-weight: bold;"
                elif status == "Переспам": cell_style += "color: #E65100; font-weight: bold;"
                elif status == "Норма": cell_style += "color: #2E7D32; font-weight: bold;"
            styles.append(cell_style)
        return styles

    cols_to_hide = [c for c in ["is_missing", "sort_val"] if c in df_view.columns]
    try: styled_df = df_view.style.apply(highlight_rows, axis=1)
    except: styled_df = df_view
    st.dataframe(styled_df, use_container_width=True, height=(len(df_view) * 35) + 40, column_config={c: None for c in cols_to_hide})
    c_spacer, c_btn_prev, c_info, c_btn_next = st.columns([6, 1, 1, 1])
    with c_btn_prev:
        if st.button("⬅️", key=f"{key_prefix}_prev", disabled=(current_page <= 1), use_container_width=True):
            st.session_state[f'{key_prefix}_page'] -= 1
            st.rerun()
    with c_info: st.markdown(f"<div style='text-align: center; margin-top: 10px;'><b>{current_page}</b> / {total_pages}</div>", unsafe_allow_html=True)
    with c_btn_next:
        if st.button("➡️", key=f"{key_prefix}_next", disabled=(current_page >= total_pages), use_container_width=True):
            st.session_state[f'{key_prefix}_page'] += 1
            st.rerun()
    st.markdown("---")

# ==========================================
# PERPLEXITY GEN
# ==========================================
STATIC_DATA_GEN = {
    'IP_PROP4817': "Условия поставки",
    'IP_PROP4818': "Оперативные отгрузки в регионы точно в срок",
    'IP_PROP4819': """<p>Надежная и быстрая доставка заказа в любую точку страны: "Стальметурал" отгружает товар 24 часа в сутки, 7 дней в неделю. Более 4 000 отгрузок в год. При оформлении заказа менеджер предложит вам оптимальный логистический маршрут.</p>""",
    'IP_PROP4820': """<p>Наши изделия успешно применяются на некоторых предприятиях Урала, центрального региона, Поволжья, Сибири. Партнеры по логистике предложат доставить заказ самым удобным способом – автомобильным, железнодорожным, даже авиационным транспортом. Для вас разработают транспортную схему под удобный способ получения. Погрузка выполняется полностью с соблюдением особенностей техники безопасности.</p><div class="h4"><h4>Самовывоз</h4></div><p>Если обычно соглашаетесь самостоятельно забрать товар или даете это право уполномоченным, адрес и время работы склада в своем городе уточняйте у менеджера.</p><div class="h4"><h4>Грузовой транспорт компании</h4></div><p>Отправим прокат на ваш объект собственным автопарком. Получение в упаковке для безопасной транспортировки, а именно на деревянном поддоне.</p><div class="h4"><h4>Сотрудничаем с ТК</h4></div><p>Доставка с помощью транспортной компании по России и СНГ. Окончательная цена может измениться, так как ссылается на прайс-лист, который предоставляет контрагент, однако, сравним стоимость логистических служб и выберем лучшую.</p>""",
    'IP_PROP4821': "Оплата и реквизиты для постоянных клиентов:",
    'IP_PROP4822': """<p>Наша компания готова принять любые комфортные виды оплаты для юридических и физических лиц: по счету, наличная и безналичная, наложенный платеж, также возможны предоплата и отсрочка платежа.</p>""",
    'IP_PROP4823': """<div class="h4"><h3>Примеры возможной оплаты</h3></div><div class="an-col-12"><ul><li style="font-weight: 400;"><p><span style="font-weight: 400;">С помощью менеджера в центрах продаж</span></p></li></ul><p>Важно! Цена не является публичной офертой. Приходите в наш офис, чтобы уточнить поступление, получить ответы на почти любой вопрос, согласовать возврат, счет, рассчитать логистику.</p><ul><li style="font-weight: 400;"><p><span style="font-weight: 400;">На расчетный счет</span></p></li></ul><p>По внутреннему счету в отделении банка или путем перечисления средств через личный кабинет (транзакции защищены, скорость зависит от отделения). Для права подтверждения нужно показать согласие на платежное поручение с отметкой банка.</p><ul><li style="font-weight: 400;"><p><span style="font-weight: 400;">Наличными или банковской картой при получении</span></p></li></ul><p><span style="font-weight: 400;">Поможем с оплатой: объем имеет значение. Крупным покупателям – деньги можно перевести после приемки товара.</span></p><p>Менеджеры предоставят необходимую информацию.</p><p>Заказывайте через прайс-лист:</p><p><a class="btn btn-blue" href="/catalog/">Каталог (магазин-меню):</a></p></div></div><br>""",
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
    if seo_keywords_data:
        words_desc = [f"- '{item['word']}': {item['count']} times" for item in seo_keywords_data]
        seo_instruction = f"MANDATORY SEO: Use these words ({', '.join(words_desc)}). Wrap in <b>."
    system_instruction = f"Generate HTML tables. Inline CSS: table border 2px solid black, th bg #f0f0f0. {seo_instruction} No markdown."
    try:
        response = client.chat.completions.create(model="sonar-pro", messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}], temperature=0.7)
        return re.sub(r'\[\d+\]', '', response.choices[0].message.content).replace("```html", "").replace("```", "").strip()
    except Exception as e: return f"Error: {e}"

# ==========================================
# 7. UI TABS RESTRUCTURED
# ==========================================
tab_seo_main, tab_wholesale_main = st.tabs(["📊 SEO Анализ", "🏭 Оптовый генератор"])

# ------------------------------------------
# TAB 1: SEO ANALYSIS (KEPT AS IS)
# ------------------------------------------
with tab_seo_main:
    col_main, col_sidebar = st.columns([65, 35])
    with col_main:
        st.title("SEO Анализатор")
        
        # Сброс кэша для словарей
        if st.button("🧹 Обновить словари (Кэш)", key="clear_cache_btn"):
            st.cache_data.clear()
            st.rerun()

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
            for key in list(st.session_state.keys()):
                if key.endswith('_page'): st.session_state[key] = 1
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
        st.selectbox("Глубина сбора (ТОП)", [10, 20, 30], index=0, key="settings_top_n")
        st.checkbox("Исключать <noindex>", True, key="settings_noindex")
        st.checkbox("Учитывать Alt/Title", False, key="settings_alt")
        st.checkbox("Учитывать числа", False, key="settings_numbers")
        st.checkbox("Нормировать по длине", True, key="settings_norm")
        st.checkbox("Исключать агрегаторы", True, key="settings_agg")

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
                if st.session_state.settings_agg: excl.extend(["avito", "ozon", "wildberries", "market.yandex", "tiu", "youtube", "vk.com", "yandex", "leroymerlin", "petrovich"])
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
        with st.status("🕵️ Сканирование конкурентов...", expanded=True) as status:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(parse_page, u['url'], settings): u['url'] for u in target_urls_raw}
                done_count = 0
                total = len(target_urls_raw)
                
                for f in concurrent.futures.as_completed(futures):
                    res = f.result()
                    if res: 
                        comp_data_full.append(res)
                    done_count += 1
                    # Обновляем статус, чтобы было видно прогресс
                    status.update(label=f"Сканирование: {done_count}/{total} (Успешно: {len(comp_data_full)})")
            
            # ВАЖНО: Сортируем данные по URL, чтобы порядок всегда был одинаковым!
            # Это устраняет "плавающие" баги при расчетах, зависящих от порядка.
            comp_data_full.sort(key=lambda x: x['url'])

            if len(comp_data_full) < len(target_urls_raw):
                st.warning(f"⚠️ Не удалось скачать {len(target_urls_raw) - len(comp_data_full)} сайтов. Рекомендации могут быть неточными.")
            else:
                st.success(f"✅ Успешно скачано {len(comp_data_full)} из {len(target_urls_raw)} конкурентов.")

        with st.spinner("Расчет метрик..."):
            st.session_state.analysis_results = calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, target_urls_raw)
            st.session_state.analysis_done = True
            res = st.session_state.analysis_results
            words_to_check = [x['word'] for x in res.get('missing_semantics_high', [])]
            if not words_to_check:
                st.session_state.categorized_products = []; st.session_state.categorized_services = []; st.session_state.categorized_commercial = []; st.session_state.categorized_dimensions = []
            else:
                with st.spinner("Классификация семантики..."):
                    categorized = classify_semantics_with_api(words_to_check, YANDEX_DICT_KEY)
                st.session_state.categorized_products = categorized['products']
                st.session_state.categorized_services = categorized['services']
                st.session_state.categorized_commercial = categorized['commercial']
                st.session_state.categorized_geo = categorized['geo']
                st.session_state.categorized_dimensions = categorized['dimensions']
                st.session_state.categorized_general = categorized['general']

            # --- ЛОГИКА РАСПРЕДЕЛЕНИЯ (ВШИТА В АНАЛИЗ) ---
            all_found_products = st.session_state.categorized_products
            count_prods = len(all_found_products)
            
            if count_prods < 20:
                st.session_state.auto_tags_words = all_found_products
                st.session_state.auto_promo_words = []
            else:
                half_count = int(math.ceil(count_prods / 2)) # Forced int
                st.session_state.auto_tags_words = all_found_products[:half_count]
                st.session_state.auto_promo_words = all_found_products[half_count:]
            
            # --- ПРИНУДИТЕЛЬНОЕ ОБНОВЛЕНИЕ WIDGET KEYS ---
            # Это решает проблему пустых полей даже при rerun
            st.session_state['tags_products_edit_final'] = "\n".join(st.session_state.auto_tags_words)
            st.session_state['promo_keywords_area_final'] = "\n".join(st.session_state.auto_promo_words)

            st.rerun()

    if st.session_state.analysis_done and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.success("Анализ готов!")
        st.markdown(f"<div style='background:{LIGHT_BG_MAIN};padding:15px;border-radius:8px;'><b>Результат:</b> Ширина: {results['my_score']['width']} | Глубина: {results['my_score']['depth']}</div>", unsafe_allow_html=True)
        with st.expander("🛒 Результат группировки слов", expanded=True):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1: st.info(f"🧱 Товары ({len(st.session_state.categorized_products)})"); st.caption(", ".join(st.session_state.categorized_products))
            with c2: st.error(f"🛠️ Услуги ({len(st.session_state.categorized_services)})"); st.caption(", ".join(st.session_state.categorized_services))
            with c3: st.warning(f"💰 Коммерц ({len(st.session_state.categorized_commercial)})"); st.caption(", ".join(st.session_state.categorized_commercial))
            with c4: st.markdown(f"**🌍 Гео ({len(st.session_state.categorized_geo)})**"); st.caption(", ".join(st.session_state.categorized_geo))
            with c5: dims = st.session_state.get('categorized_dimensions', []); st.success(f"📏 Размеры, марки, ГОСТ ({len(dims)})"); st.caption(", ".join(dims))
            with c6: gen_words = st.session_state.get('categorized_general', []); st.markdown(f"**📂 Общие ({len(gen_words)})**"); st.caption(", ".join(gen_words))
        high = results.get('missing_semantics_high', [])
        low = results.get('missing_semantics_low', [])
        if high or low:
            with st.expander(f"🧩 Упущенная семантика ({len(high)+len(low)})", expanded=False):
                if high: st.markdown(f"<div style='background:#EBF5FF;padding:10px;border-radius:5px;'><b>Важные:</b> {', '.join([x['word'] for x in high])}</div>", unsafe_allow_html=True)
                if low: st.markdown(f"<div style='background:#F7FAFC;padding:10px;border-radius:5px;margin-top:5px;'><b>Доп:</b> {', '.join([x['word'] for x in low])}</div>", unsafe_allow_html=True)
        render_paginated_table(results['depth'], "1. Глубина", "tbl_depth_1", default_sort_col="Рекомендация", use_abs_sort_default=True)
        render_paginated_table(results['hybrid'], "3. TF-IDF", "tbl_hybrid", default_sort_col="TF-IDF ТОП")
        render_paginated_table(results['relevance_top'], "4. Релевантность", "tbl_rel", default_sort_col="Ширина (балл)")

# ------------------------------------------
# TAB 2: WHOLESALE GENERATOR (COMBINED)
# ------------------------------------------
with tab_wholesale_main:
    st.header("🏭 Единый генератор контента")
    
    # ==========================================
    # 0. СБОР И РАСПРЕДЕЛЕНИЕ СЕМАНТИКИ
    # ==========================================
    cat_products = st.session_state.get('categorized_products', [])
    cat_services = st.session_state.get('categorized_services', [])
    structure_keywords = cat_products + cat_services
    count_struct = len(structure_keywords)

    tags_default_text = ""
    promo_default_text = ""
    sidebar_default_text = ""

    if count_struct > 0:
        if count_struct < 10:
            tags_default_text = "\n".join(structure_keywords)
        elif count_struct < 30:
            mid = math.ceil(count_struct / 2)
            tags_default_text = "\n".join(structure_keywords[:mid])
            promo_default_text = "\n".join(structure_keywords[mid:])
        else:
            part = math.ceil(count_struct / 3)
            tags_default_text = "\n".join(structure_keywords[:part])
            promo_default_text = "\n".join(structure_keywords[part:part*2])
            sidebar_default_text = "\n".join(structure_keywords[part*2:])

    # Техничка и Контекст
    cat_dimensions = st.session_state.get('categorized_dimensions', [])
    tech_context_text = ", ".join(cat_dimensions) if cat_dimensions else ""
    cat_commercial = st.session_state.get('categorized_commercial', [])
    cat_general = st.session_state.get('categorized_general', [])
    cat_geo = st.session_state.get('categorized_geo', [])
    text_context_list = cat_commercial + cat_general + cat_geo
    text_context_str = ", ".join(text_context_list)

    # ==========================================
    # 1. ВВОДНЫЕ ДАННЫЕ
    # ==========================================
    with st.container(border=True):
        st.subheader("1. Источник и Доступы")
        col_top_1, col_top_2 = st.columns([3, 1])
        with col_top_1:
            main_category_url = st.text_input("URL Категории", 
                placeholder="https://site.ru/catalog/...", 
                help="Скрипт соберет товары с этой страницы")
        with col_top_2:
            default_key = st.session_state.get('pplx_key_cache', "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")
            pplx_api_key = st.text_input("AI API Key", value=default_key, type="password")
            if pplx_api_key: st.session_state.pplx_key_cache = pplx_api_key
        
        if count_struct > 0:
            st.info(f"📊 **SEO-данные:** Структура ({count_struct}), Техничка ({len(cat_dimensions)}), Текст ({len(text_context_list)}).")

    # ==========================================
    # 2. ВЫБОР МОДУЛЕЙ
    # ==========================================
    st.subheader("2. Какие блоки генерируем?")
    col_ch1, col_ch2, col_ch3, col_ch4, col_ch5 = st.columns(5)
    with col_ch1: use_text = st.checkbox("🤖 AI Тексты", value=True)
    with col_ch2: use_tags = st.checkbox("🏷️ Теги")
    with col_ch3: use_tables = st.checkbox("🧩 Таблицы")
    with col_ch4: use_promo = st.checkbox("🔥 Промо")
    with col_ch5: use_sidebar = st.checkbox("📑 Сайдбар")

    # ==========================================
    # 3. НАСТРОЙКИ МОДУЛЕЙ
    # ==========================================
    
    global_tags_list = []
    global_promo_list = []
    global_sidebar_list = []
    
    tags_file_content = ""
    table_prompts = []
    df_db_promo = None
    promo_title = "Рекомендуем"
    sidebar_content = ""
    
    if any([use_text, use_tags, use_tables, use_promo, use_sidebar]):
        st.subheader("3. Настройки модулей")

        # --- [1] AI ТЕКСТЫ ---
        if use_text:
            with st.container(border=True):
                st.markdown("#### 🤖 1. AI Тексты")
                if text_context_list:
                    st.success(f"✅ В текст будут внедрены слова: {text_context_str[:50]}...")
                else:
                    st.warning("⚠️ Нет доп. слов для текста.")

        # --- [2] ТЕГИ ---
        if use_tags:
            with st.container(border=True):
                st.markdown("#### 🏷️ 2. Теги")
                kws_input_tags = st.text_area("Список (Товары + Услуги)", value=tags_default_text, height=100, key="kws_tags_auto")
                global_tags_list = [x.strip() for x in kws_input_tags.split('\n') if x.strip()]
                
                if not global_tags_list: st.warning("⚠️ Список пуст!")
                st.markdown("---")
                col_t1, col_t2 = st.columns([1, 2])
                with col_t1: u_manual = st.checkbox("Своя база ссылок (.txt)", key="cb_tags_vert")
                with col_t2:
                    default_tags_path = "data/links_base.txt"
                    if not u_manual and os.path.exists(default_tags_path):
                        st.success(f"✅ База репозитория (`links_base.txt`)")
                        with open(default_tags_path, "r", encoding="utf-8") as f: tags_file_content = f.read()
                    elif u_manual:
                        up_t = st.file_uploader("Файл .txt", type=["txt"], key="up_tags_vert", label_visibility="collapsed")
                        if up_t: tags_file_content = up_t.getvalue().decode("utf-8")
                    else: st.error("❌ Файл базы не найден!")

        # --- [3] ТАБЛИЦЫ ---
        if use_tables:
            with st.container(border=True):
                st.markdown("#### 🧩 3. Таблицы")
                if tech_context_text:
                    st.caption(f"Контекст: {tech_context_text[:100]}...")
                cnt = st.number_input("Кол-во таблиц", 1, 5, 2, key="num_tbl_vert")
                defaults = ["Характеристики", "Размеры", "Хим. состав"]
                for i in range(cnt):
                    val = defaults[i] if i < len(defaults) else f"Таблица {i+1}"
                    t_p = st.text_input(f"Тема {i+1}", value=val, key=f"tbl_topic_vert_{i}")
                    table_prompts.append(t_p)

        # --- [4] ПРОМО ---
        if use_promo:
            with st.container(border=True):
                st.markdown("#### 🔥 4. Промо-блок")
                kws_input_promo = st.text_area("Список (Товары + Услуги)", value=promo_default_text, height=100, key="kws_promo_auto")
                global_promo_list = [x.strip() for x in kws_input_promo.split('\n') if x.strip()]

                if not global_promo_list: st.warning("⚠️ Список пуст!")
                st.markdown("---")
                col_p1, col_p2 = st.columns([1, 2])
                with col_p1:
                    promo_title = st.text_input("Заголовок", "Смотрите также", key="pr_tit_vert")
                    u_img_man = st.checkbox("Своя база картинок", key="cb_img_vert")
                with col_p2:
                    default_img_db = "data/images_db.xlsx"
                    if not u_img_man and os.path.exists(default_img_db):
                        st.success("✅ База картинок (`images_db.xlsx`)")
                        try: df_db_promo = pd.read_excel(default_img_db)
                        except: pass
                    elif u_img_man:
                        up_i = st.file_uploader("Файл .xlsx", type=['xlsx'], key="up_img_vert", label_visibility="collapsed")
                        if up_i: df_db_promo = pd.read_excel(up_i)
                    else: st.error("❌ База картинок не найдена!")

        # --- [5] САЙДБАР ---
        if use_sidebar:
            with st.container(border=True):
                st.markdown("#### 📑 5. Сайдбар")
                kws_input_sidebar = st.text_area("Список (Товары + Услуги)", value=sidebar_default_text, height=100, key="kws_sidebar_auto")
                global_sidebar_list = [x.strip() for x in kws_input_sidebar.split('\n') if x.strip()]
                
                if not global_sidebar_list: st.warning("⚠️ Список пуст!")
                st.markdown("---")
                col_s1, col_s2 = st.columns([1, 2])
                with col_s1: u_sb_man = st.checkbox("Свой файл меню (.txt)", key="cb_sb_vert")
                with col_s2:
                    def_menu = "data/menu_structure.txt"
                    if not u_sb_man and os.path.exists(def_menu):
                        st.success("✅ Меню репозитория (`menu_structure.txt`)")
                        with open(def_menu, "r", encoding="utf-8") as f: sidebar_content = f.read()
                    elif u_sb_man:
                        up_s = st.file_uploader("Файл .txt", type=['txt'], key="up_sb_vert", label_visibility="collapsed")
                        if up_s: sidebar_content = up_s.getvalue().decode("utf-8")
                    else: st.error("❌ Файл меню не найден!")

    st.markdown("---")
    
    # ==========================================
    # 4. ЗАПУСК
    # ==========================================
    
    ready_to_go = True
    if not main_category_url: ready_to_go = False
    if (use_text or use_tables) and not pplx_api_key: ready_to_go = False
    if use_tags and not tags_file_content: ready_to_go = False
    if use_promo and df_db_promo is None: ready_to_go = False
    if use_sidebar and not sidebar_content: ready_to_go = False
    
    if st.button("🚀 ЗАПУСТИТЬ ГЕНЕРАЦИЮ (ОДНА КНОПКА)", type="primary", disabled=not ready_to_go, use_container_width=True):
        status_box = st.status("🛠️ Начинаем работу...", expanded=True)
        final_data = [] 
        
        # 1. Сбор целевых страниц
        target_pages = []
        try:
            status_box.write(f"🕵️ Сканируем категорию: {main_category_url}")
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(main_category_url, headers=headers, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                tags_container = soup.find(class_='popular-tags-inner')
                if tags_container:
                    for link in tags_container.find_all('a'):
                        href = link.get('href')
                        name = link.get_text(strip=True)
                        if href and name:
                            full_url = urljoin(main_category_url, href)
                            target_pages.append({'url': full_url, 'name': name})
                
                if not target_pages:
                    status_box.warning("Теги не найдены. Обрабатываем только указанный URL.")
                    h1 = soup.find('h1')
                    name = h1.get_text(strip=True) if h1 else "Товар"
                    target_pages.append({'url': main_category_url, 'name': name})
            else: status_box.error(f"Ошибка доступа: {r.status_code}"); st.stop()
        except Exception as e: status_box.error(f"Ошибка соединения: {e}"); st.stop()
        status_box.write(f"✅ Найдено страниц для обработки: {len(target_pages)}")
        
        # ==========================================
        # ПОДГОТОВКА И СБОР URL ДЛЯ ПАРСИНГА ИМЕН
        # ==========================================
        
        urls_to_fetch_names = set() # Собираем сюда все URL, для которых нужны имена
        
        # [TAGS]
        tags_map = {}
        if use_tags:
            s_io = io.StringIO(tags_file_content)
            all_links = [l.strip() for l in s_io.readlines() if l.strip()]
            for kw in global_tags_list:
                tr = transliterate_text(kw).replace(' ', '-').replace('_', '-')
                if len(tr) >= 3:
                    matches = [u for u in all_links if tr in u]
                    if matches: 
                        tags_map[kw] = matches
                        urls_to_fetch_names.update(matches)

        # [PROMO]
        promo_items_pool = [] 
        if use_promo:
            p_img_map = {}
            for _, row in df_db_promo.iterrows():
                u = str(row.iloc[0]).strip(); img = str(row.iloc[1]).strip()
                if u and u != 'nan' and img and img != 'nan': p_img_map[u.rstrip('/')] = img
            
            used_urls = set()
            for kw in global_promo_list:
                tr = transliterate_text(kw).replace(' ', '-').replace('_', '-')
                if len(tr) < 3: continue
                matches = [u for u in p_img_map.keys() if tr in u]
                for m in matches:
                    if m not in used_urls:
                        urls_to_fetch_names.add(m)
                        promo_items_pool.append({'url': m, 'img': p_img_map[m]})
                        used_urls.add(m)

        # [SIDEBAR]
        sidebar_matched_urls = []
        if use_sidebar:
            s_io = io.StringIO(sidebar_content)
            all_menu_urls = [l.strip() for l in s_io.readlines() if l.strip()]
            
            if global_sidebar_list:
                for kw in global_sidebar_list:
                    tr = transliterate_text(kw).replace(' ', '-').replace('_', '-')
                    if len(tr) < 3: continue
                    found = [u for u in all_menu_urls if tr in u]
                    sidebar_matched_urls.extend(found)
                sidebar_matched_urls = list(set(sidebar_matched_urls))
            else:
                sidebar_matched_urls = all_menu_urls
            
            urls_to_fetch_names.update(sidebar_matched_urls)

        # ==========================================
        # МАССОВЫЙ ПАРСИНГ ИМЕН
        # ==========================================
        url_name_cache = {}
        if urls_to_fetch_names:
            status_box.write(f"🌍 Получаем реальные названия для {len(urls_to_fetch_names)} ссылок...")
            
            def fetch_name_worker(u): 
                return u, get_breadcrumb_only(u) 
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                future_to_url = {executor.submit(fetch_name_worker, u): u for u in urls_to_fetch_names}
                done_cnt = 0
                prog_fetch = status_box.progress(0)
                for future in concurrent.futures.as_completed(future_to_url):
                    u_res, name_res = future.result()
                    # Ключ кэша — URL без слеша на конце, для надежности
                    norm_key = u_res.rstrip('/')
                    
                    if name_res:
                        url_name_cache[norm_key] = name_res
                    else:
                        slug = norm_key.split('/')[-1]
                        url_name_cache[norm_key] = force_cyrillic_name_global(slug)
                    
                    done_cnt += 1
                    prog_fetch.progress(done_cnt / len(urls_to_fetch_names))
            
            status_box.write("✅ Названия собраны!")

        # ==========================================
        # СБОРКА КОНТЕНТА
        # ==========================================
        
        # 1. SIDEBAR (Сборка с использованием кэша имен)
        full_sidebar_code = ""
        if use_sidebar:
            status_box.write("🔨 Сборка меню...")
            tree = {}
            for url in sidebar_matched_urls:
                path = urlparse(url).path.strip('/')
                parts = [p for p in path.split('/') if p]
                idx_start = 0
                if 'catalog' in parts: idx_start = parts.index('catalog') + 1
                rel_parts = parts[idx_start:] if parts[idx_start:] else parts
                
                curr = tree
                for i, part in enumerate(rel_parts):
                    if part not in curr: curr[part] = {}
                    if i == len(rel_parts) - 1:
                        curr[part]['__url__'] = url
                        # БЕРЕМ ИМЯ ИЗ КЭША (по нормализованному ключу)
                        cache_key = url.rstrip('/')
                        curr[part]['__name__'] = url_name_cache.get(cache_key, force_cyrillic_name_global(part))
                    curr = curr[part]
            
            def render_tree_internal(node, level=1):
                html = ""
                keys = sorted([k for k in node.keys() if not k.startswith('__')])
                for key in keys:
                    child = node[key]
                    name = child.get('__name__', force_cyrillic_name_global(key))
                    url = child.get('__url__')
                    has_children = any(k for k in child.keys() if not k.startswith('__'))
                    
                    if level == 1:
                        html += '<li class="level-1-header">\n'
                        if has_children:
                            html += f'    <span class="dropdown-toggle">{name}</span>\n'
                            html += '    <ul class="collapse-menu list-unstyled">\n'
                            html += render_tree_internal(child, level=2)
                            html += '    </ul>\n'
                        else:
                            target = url if url else "#"
                            html += f'    <a href="{target}">{name}</a>\n'
                        html += '</li>\n'
                    elif level == 2:
                        if has_children:
                            html += '<li class="level-2-header">\n'
                            html += f'    <span class="dropdown-toggle">{name}</span>\n'
                            html += '    <ul class="collapse-menu list-unstyled">\n'
                            html += render_tree_internal(child, level=3)
                            html += '    </ul>\n'
                        else:
                            target = url if url else "#"
                            html += f'<li class="level-2-link-special"><a href="{target}">{name}</a></li>\n'
                    elif level >= 3:
                        target = url if url else "#"
                        html += f'<li class="level-3-link"><a href="{target}">{name}</a></li>\n'
                return html

            inner_html = render_tree_internal(tree, level=1)
            full_sidebar_code = f"""<div class="page-content-with-sidebar"><button id="mobile-menu-toggle" class="menu-toggle-button">☰</button><div class="sidebar-wrapper"><nav id="sidebar-menu"><ul class="list-unstyled components">{inner_html}</ul></nav></div></div>{SIDEBAR_ASSETS}"""

        # 2. CLIENT
        client = None
        if openai and (use_text or use_tables):
            client = openai.OpenAI(api_key=pplx_api_key, base_url="https://api.perplexity.ai")

        # 3. ЦИКЛ ПО СТРАНИЦАМ
        progress_bar = status_box.progress(0)
        total_steps = len(target_pages)
        
        for idx, page in enumerate(target_pages):
            row_data = {'Page URL': page['url'], 'Product Name': page['name']}
            
            # --- AI TEXT ---
            if use_text and client:
                try:
                    blocks = generate_five_blocks(client, f"Контент для {page['name']}", page['name'], seo_words=text_context_list)
                    row_data['Text_Block_1'] = blocks[0]
                    row_data['Text_Block_2'] = blocks[1]
                    row_data['Text_Block_3'] = blocks[2]
                    row_data['Text_Block_4'] = blocks[3]
                    row_data['Text_Block_5'] = blocks[4]
                    for k, v in STATIC_DATA_GEN.items(): row_data[k] = v
                except Exception as e: row_data['Text_Error'] = str(e)

            # --- TAGS ---
            if use_tags:
                possible_candidates = []
                for kw, urls in tags_map.items():
                    valid = [u for u in urls if u.rstrip('/') != page['url'].rstrip('/')]
                    if valid: possible_candidates.append(random.choice(valid))
                random.shuffle(possible_candidates)
                selected = list(set(possible_candidates))[:20]
                if selected:
                    html_parts = ['<div class="popular-tags">']
                    for l in selected:
                        cache_key = l.rstrip('/')
                        nm = url_name_cache.get(cache_key, "Товар")
                        html_parts.append(f'<a href="{l}" class="tag-link">{nm}</a>')
                    html_parts.append('</div>')
                    row_data['Tags HTML'] = "\n".join(html_parts)
                else: row_data['Tags HTML'] = ""

            # --- AI TABLES ---
            if use_tables and client:
                for t_i, t_topic in enumerate(table_prompts):
                    sys_p = "Generate HTML table only. Inline CSS borders."
                    context_hint = ""
                    if tech_context_text:
                        context_hint = f" Use specs: {tech_context_text}."
                    usr_p = f"Product: {page['name']}. Topic: {t_topic}. Realistic table.{context_hint}"
                    try:
                        resp = client.chat.completions.create(model="sonar-pro", messages=[{"role":"system","content":sys_p},{"role":"user","content":usr_p}], temperature=0.5)
                        t_html = resp.choices[0].message.content.replace("```html","").replace("```","")
                        row_data[f'Table_{t_i+1}_HTML'] = t_html
                    except: row_data[f'Table_{t_i+1}_HTML'] = "Error"

            # --- PROMO ---
            if use_promo:
                candidates = [x for x in promo_items_pool if x['url'].rstrip('/') != page['url'].rstrip('/')]
                if len(candidates) > 5: chosen = random.sample(candidates, 5)
                else: chosen = candidates
                if chosen:
                    items_html = ""
                    for item in chosen:
                        cache_key = item['url'].rstrip('/')
                        real_name = url_name_cache.get(cache_key, "Товар") # БЕРЕМ ИЗ КЭША
                        items_html += f"""<div class="gallery-item"><h3><a href="{item['url']}">{real_name}</a></h3><figure><a href="{item['url']}"><img src="{item['img']}" loading="lazy"></a></figure></div>"""
                    css = "<style>.five-col-gallery{display:flex;gap:15px;}</style>"
                    full_promo = f"""{css}<div class="gallery-wrapper"><h3>{promo_title}</h3><div class="five-col-gallery">{items_html}</div></div>"""
                    row_data['Promo HTML'] = full_promo
                else: row_data['Promo HTML'] = ""

            # --- SIDEBAR ---
            if use_sidebar:
                row_data['Sidebar HTML'] = full_sidebar_code

            final_data.append(row_data)
            progress_bar.progress((idx + 1) / total_steps)

        df_result = pd.DataFrame(final_data)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_result.to_excel(writer, index=False)
        
        st.session_state.unified_excel_data = buffer.getvalue()
        status_box.update(label="✅ Конвейер завершен! Файл готов.", state="complete", expanded=False)

    if 'unified_excel_data' in st.session_state:
        st.success("Файл успешно сгенерирован!")
        st.download_button(
            label="📥 СКАЧАТЬ ЕДИНЫЙ EXCEL",
            data=st.session_state.unified_excel_data,
            file_name="unified_content_gen.xlsx",
            mime="application/vnd.ms-excel",
            key="btn_dl_unified"
        )

