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
import random

def transliterate_text(text):
    """
    Превращает 'Швеллер' в 'shveller', 'Анод' в 'anod'.
    Используется для нечеткого поиска товара в URL.
    """
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

# Состояния для AI генерации
if 'ai_generated_df' not in st.session_state:
    st.session_state.ai_generated_df = None
if 'ai_excel_bytes' not in st.session_state:
    st.session_state.ai_excel_bytes = None

# Состояния для Тегов и Таблиц
if 'tags_html_result' not in st.session_state:
    st.session_state.tags_html_result = None
if 'table_html_result' not in st.session_state:
    st.session_state.table_html_result = None

# --- НОВЫЕ СОСТОЯНИЯ ДЛЯ КЛАССИФИКАЦИИ ---
if 'categorized_products' not in st.session_state:
    st.session_state.categorized_products = []
if 'categorized_services' not in st.session_state:
    st.session_state.categorized_services = []
if 'categorized_commercial' not in st.session_state:
    st.session_state.categorized_commercial = []
if 'categorized_dimensions' not in st.session_state:
    st.session_state.categorized_dimensions = []

# Переменная для хранения ссылок
if 'persistent_urls' not in st.session_state:
    st.session_state['persistent_urls'] = ""

if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

# ==========================================
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ И СПИСКИ
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO v2.3 (Secure)", page_icon="📊")

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
        .main { display: flex; flex-direction: column; justify-content: center; align-items: center; }
        .auth-logo-box { text-align: center; margin-bottom: 1rem; padding-top: 0; }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-logo-box"><h3>Вход в систему</h3></div>', unsafe_allow_html=True)
        password = st.text_input("Пароль", type="password", key="password_input", label_visibility="collapsed")
        if st.button("ВОЙТИ", type="primary", use_container_width=True):
            # Простой пароль для примера
            if password == "jfV6Xel-Q7vp-_s2UYPO":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Неверный пароль")
    return False

if not check_password():
    st.stop()

# ==========================================
# 3. ПОЛУЧЕНИЕ API КЛЮЧЕЙ (БЕЗОПАСНО)
# ==========================================
# Попытка получить Arsenkin Token
if "arsenkin_token" in st.session_state:
    ARSENKIN_TOKEN = st.session_state.arsenkin_token
else:
    try:
        ARSENKIN_TOKEN = st.secrets["api"]["arsenkin_token"]
    except (FileNotFoundError, KeyError):
        ARSENKIN_TOKEN = None

# Попытка получить Yandex Key
if "yandex_dict_key" in st.session_state:
    YANDEX_DICT_KEY = st.session_state.yandex_dict_key
else:
    try:
        YANDEX_DICT_KEY = st.secrets["api"]["yandex_dict_key"]
    except (FileNotFoundError, KeyError):
        YANDEX_DICT_KEY = None


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

# Цвета и стили
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
# 4. ЛОГИКА (БЭКЕНД)
# ==========================================

try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    USE_NLP = True
except Exception as e:
    morph = None
    USE_NLP = False
    st.sidebar.error(f"Ошибка загрузки NLP: {e}")

# --- ФУНКЦИЯ ЗАПРОСА К ЯНДЕКС СЛОВАРЮ ---
def get_yandex_dict_info(text, api_key):
    """
    Возвращает нормальную форму (лемму) и часть речи (pos) через API Яндекса.
    """
    if not api_key:
        return {'lemma': text, 'pos': 'unknown'}
        
    url = "https://dictionary.yandex.net/api/v1/dicservice.json/lookup"
    params = {
        'key': api_key,
        'lang': 'ru-ru', 
        'text': text,
        'ui': 'ru'
    }
    try:
        r = requests.get(url, params=params, timeout=2)
        if r.status_code == 200:
            data = r.json()
            if data.get('def'):
                first_def = data['def'][0]
                lemma = first_def.get('text', text)
                pos = first_def.get('pos', 'unknown')
                return {'lemma': lemma, 'pos': pos}
    except:
        pass
    return {'lemma': text, 'pos': 'unknown'}

# --- ФУНКЦИЯ КЛАССИФИКАЦИИ С API ЯНДЕКСА ---
def classify_semantics_with_api(words_list, yandex_key):
    # --- 1. БАЗОВЫЕ СПИСКИ (ТОЛЬКО МУСОР И КОММЕРЦИЯ) ---
    # Мы убираем "Белые списки" товаров. Оставляем только то, что точно НЕ товар.
    
    # Регулярки для размеров/марок
    dim_pattern = re.compile(r'\d+(?:[\.\,]\d+)?\s?[хx\*×]\s?\d+', re.IGNORECASE)
    grade_pattern = re.compile(r'^([а-яa-z]{1,4}\-?\d+[а-яa-z0-9]*)$', re.IGNORECASE)
    gost_pattern = re.compile(r'(гост|din|ту|iso|ст|сп)\s?\d+', re.IGNORECASE)

    # UI мусор (меню, корзина и т.д.)
    SITE_UI_GARBAGE = {
        'меню', 'поиск', 'главная', 'карта', 'сайт', 'кабинет', 'вход', 'регистрация', 
        'корзина', 'избранное', 'профиль', 'телефон', 'адрес', 'контакты', 'email', 
        'звонок', 'callback', 'отзыв', 'отзывы', 'вопрос', 'ответ', 'менеджер', 
        'политика', 'конфиденциальность', 'соглашение', 'оферта', 'cookie', 
        'отправить', 'ошибка', 'кнопка', 'форма', 'поле', 'обзор', 'новости', 'статьи',
        'характеристика', 'описание', 'параметр', 'свойство', 'артикул', 'код',
        'калькулятор', 'фильтр', 'сортировка', 'показать', 'сбросить', 'категория', 
        'раздел', 'список', 'вид', 'тип', 'класс', 'серия', 'рейтинг', 'наличие', 
        'склад', 'производитель', 'бренд', 'марка', 'вес', 'длина', 'ширина', 
        'высота', 'толщина', 'диаметр', 'размер', 'объем', 'масса', 'тонна', 'метр', 
        'шт', 'кг', 'упаковка', 'интернет', 'магазин', 'каталог', 'год', 'день', 'час'
    }

    # Явные коммерческие слова (которые морфология считает существительными)
    COMMERCIAL_STOP_WORDS = {
        'купить', 'цена', 'цены', 'прайс', 'стоимость', 'продажа', 'недорого', 
        'дешево', 'дорого', 'скидка', 'акция', 'распродажа', 'оптом', 'розница', 
        'руб', 'рублей', 'заказ', 'оплата', 'платеж', 'рассрочка', 'кредит', 
        'лизинг', 'доставка', 'самовывоз', 'отгрузка', 'поставка', 'транспорт', 
        'логистика', 'гарантия', 'возврат', 'обмен', 'снабжение', 'выгодный', 
        'качественный', 'надежный', 'большой', 'малый', 'удобный', 'быстрый', 
        'бесплатный', 'хороший', 'доступный', 'индивидуальный', 'профессиональный', 
        'собственный', 'официальный', 'уникальный', 'широкий', 'партнер', 
        'преимущество', 'связь', 'звонить'
    }

    SERVICE_KEYWORDS = {
        'резка', 'гибка', 'сварка', 'оцинковка', 'рубка', 'монтаж', 'укладка', 
        'проектирование', 'изоляция', 'сверление', 'грунтовка', 'покраска', 
        'металлообработка', 'обработка', 'строительство', 'ремонт', 'производство', 
        'изготовление', 'размотка', 'протяжка', 'цинкование', 'покрытие'
    }

    categories = {
        'products': set(),
        'services': set(),
        'commercial': set(),
        'dimensions': set()
    }

    # Список слов, которые нужно проверить через API (если Pymorphy не справился)
    # Но в 95% случаев Pymorphy справится сам.
    
    for word in words_list:
        word_lower = word.lower()

        # 1. Размеры и цифры
        if dim_pattern.search(word_lower) or grade_pattern.match(word_lower) or gost_pattern.search(word_lower) or word_lower.isdigit():
            categories['dimensions'].add(word_lower)
            continue

        # 2. Морфологический анализ (PyMorphy2)
        # Это ключевой момент. Мы смотрим на теги.
        if morph:
            p = morph.parse(word_lower)[0]
            lemma = p.normal_form
            tags = p.tag
        else:
            # Если NLP не загрузился - фоллбэк на простую логику
            lemma = word_lower
            tags = set()

        # --- ФИЛЬТРЫ МУСОРА ---
        
        # Если слово в стоп-листах
        if lemma in SITE_UI_GARBAGE or lemma in COMMERCIAL_STOP_WORDS:
            categories['commercial'].add(lemma)
            continue
        
        # Если это География (Geox), Имя (Name), Фамилия (Surn)
        if 'Geox' in tags or 'Name' in tags or 'Surn' in tags or 'Patr' in tags:
            categories['commercial'].add(lemma) # Города и имена - в коммерцию/общее
            continue

        # Если это Глагол (INFN - инфинитив, VERB - глагол, GRND - деепричастие)
        # "Использоваться", "Звонить", "Находиться"
        if 'INFN' in tags or 'VERB' in tags or 'GRND' in tags or 'PRTF' in tags:
            categories['commercial'].add(lemma)
            continue
            
        # Если это местоимение, предлог, союз (на всякий случай)
        if 'PREP' in tags or 'CONJ' in tags or 'PRCL' in tags or 'NPRO' in tags:
            categories['commercial'].add(lemma)
            continue

        # --- ОПРЕДЕЛЕНИЕ УСЛУГ ---
        
        # 1. По словарю
        if lemma in SERVICE_KEYWORDS or lemma.endswith('обработка'):
            categories['services'].add(lemma)
            continue
            
        # 2. По суффиксам (Эвристика)
        # Слова на -ние (кроме оборудования/крепления) и -ка (гибка, резка) часто услуги
        if lemma.endswith('ние') and lemma not in ['оборудование', 'крепление', 'соединение', 'приспособление']:
             # Часто это отглагольные существительные -> Услуги
             categories['services'].add(lemma)
             continue
             
        if lemma.endswith('ка') and ('NOUN' in tags) and lemma not in ['балка', 'проволока', 'сетка', 'трубка', 'гайка', 'шайба', 'поковка', 'упаковка', 'рейка']:
            # Рискованная эвристика, но для резки/гибки работает. 
            # Если сомневаемся - лучше пусть упадет в услуги, чем в товары.
            # Но список исключений для металла важен (балка, сетка).
            if lemma in ['резка', 'гибка', 'рубка', 'ковка', 'сварка', 'доставка', 'нарезка', 'укладка', 'покраска']:
                 categories['services'].add(lemma)
                 continue

        # --- ОПРЕДЕЛЕНИЕ ТОВАРОВ (ОСНОВНАЯ ЛОГИКА) ---
        
        # Если мы дошли до сюда:
        # 1. Это не мусор
        # 2. Это не город
        # 3. Это не глагол
        # 4. Это не явная услуга
        
        # Если это Существительное (NOUN) -> Скорее всего ТОВАР
        if 'NOUN' in tags:
            if len(lemma) > 2:
                categories['products'].add(lemma)
            continue
            
        # Если это Прилагательное (ADJF) -> Материал (Медный, Стальной) -> ТОВАР
        # Обычные прилагательные (хороший, быстрый) отсеялись в COMMERCIAL_STOP_WORDS
        if 'ADJF' in tags:
             if len(lemma) > 2:
                categories['products'].add(lemma)
             continue
             
        # Если это Латиница (которая не попала в размеры), считаем товаром (бренды, марки)
        if re.search(r'[a-zA-Z]', lemma):
            categories['products'].add(lemma)
            continue

        # Если ничего не подошло (Unknown), по умолчанию кидаем в товары, если слово длинное.
        # Лучше получить "странный товар", чем потерять "редкий металл".
        if len(lemma) > 3:
            categories['products'].add(lemma)
        else:
            categories['commercial'].add(lemma)

    return {k: sorted(list(v)) for k, v in categories.items()}

# --- ФУНКЦИЯ API ARSENKIN ---
def get_arsenkin_urls(query, engine_type, region_name, api_token, depth_val=10):
    url_set = "https://arsenkin.ru/api/tools/set"
    url_check = "https://arsenkin.ru/api/tools/check"
    url_get = "https://arsenkin.ru/api/tools/get"

    headers = {
        "Authorization": f"Bearer {api_token}",
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
        return []

    res_data = {}
    try:
        r_final = requests.post(url_get, headers=headers, json={"task_id": task_id}, timeout=30)
        res_data = r_final.json()
        if res_data.get("code") != "TASK_RESULT":
            st.error(f"❌ Ошибка: API не вернул финальный результат.")
            return []
    except Exception as e:
        st.error(f"❌ Ошибка сети при получении результата: {e}")
        return []

    results_list = []
    try:
        if 'result' in res_data and 'result' in res_data['result'] and 'collect' in res_data['result']['result']:
            collect = res_data['result']['result']['collect']
        else:
            unique_urls = set()
            if 'result' in res_data and isinstance(res_data['result'], list):
                return res_data['result']
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
            return results_list

        if final_url_list:
            for index, url in enumerate(final_url_list):
                results_list.append({'url': url, 'pos': index + 1})
    except Exception as e:
        st.error(f"❌ Ошибка парсинга JSON: {e}")
        return []
    return results_list

def process_text_detailed(text, settings, n_gram=1):
    text = text.lower().replace('ё', 'е')
    pattern = r'[а-яА-ЯёЁ0-9a-zA-Z]+'
    words = re.findall(pattern, text)
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
        if settings['noindex']: tags_to_remove.append('noindex')
        for c in soup.find_all(string=lambda text: isinstance(text, Comment)): c.extract()
        if tags_to_remove:
            for t in soup.find_all(tags_to_remove): t.decompose()

        anchors_list = [a.get_text(strip=True) for a in soup.find_all('a') if a.get_text(strip=True)]
        anchor_text = " ".join(anchors_list)

        extra_text = []
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'): extra_text.append(meta_desc['content'])
        meta_kw = soup.find('meta', attrs={'name': 'keywords'})
        if meta_kw and meta_kw.get('content'): extra_text.append(meta_kw['content'])

        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): extra_text.append(img['alt'])
            for t in soup.find_all(title=True): extra_text.append(t['title'])

        body_text_raw = soup.get_text(separator=' ') + " " + " ".join(extra_text)
        body_text = re.sub(r'\s+', ' ', body_text_raw).strip()

        if not body_text: return None
        return {'url': url, 'domain': urlparse(url).netloc, 'body_text': body_text, 'anchor_text': anchor_text}
    except:
        return None

def calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, original_results):
    all_forms_map = defaultdict(set)

    if not my_data or not my_data.get('body_text'):
        my_lemmas, my_forms, my_anchors, my_len = [], {}, [], 0
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
        return { "depth": pd.DataFrame(), "hybrid": pd.DataFrame(), "relevance_top": pd.DataFrame(),
            "my_score": {"width": 0, "depth": 0}, "missing_semantics_high": [], "missing_semantics_low": [] }

    c_lens = [len(d['body']) for d in comp_docs]
    avg_dl = np.mean(c_lens) if c_lens else 1
    if avg_dl == 0: avg_dl = 1
    median_len = np.median(c_lens) if c_lens else 0
    norm_k_recs = (my_len / median_len) if (median_len > 0 and my_len > 0 and settings['norm']) else 1.0

    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    vocab = sorted(list(vocab))
    N = len(comp_docs)

    doc_freqs = Counter()
    for d in comp_docs:
        for w in set(d['body']): doc_freqs[w] += 1
    word_counts_per_doc = []
    for d in comp_docs: word_counts_per_doc.append(Counter(d['body']))

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
        if med_val >= 1:
            S_WIDTH_CORE.add(lemma)
            is_width_word = True

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
        k1 = 1.2
        b = 0.75
        target_words = S_WIDTH_CORE if S_WIDTH_CORE else S_DEPTH_TOP70
        for word in target_words:
            if word not in counts: continue
            tf = counts[word]
            idf = word_idf_map.get(word, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
            score += idf * (numerator / denominator)
        return score

    def calculate_width_score_val(lemmas_set):
        if total_width_core_count == 0: return 0
        intersection_count = len(lemmas_set.intersection(S_WIDTH_CORE))
        ratio = intersection_count / total_width_core_count
        if ratio >= 0.9: return 100
        else: return int(round((ratio / 0.9) * 100))

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
        forms_set = all_forms_map.get(lemma, set())
        forms_str = ", ".join(sorted(list(forms_set))) if forms_set else lemma
        c_counts = [word_counts_per_doc[i][lemma] for i in range(N)]
        med_total = np.median(c_counts)
        max_total = np.max(c_counts)

        base_min = min(np.mean(c_counts), med_total)
        rec_min = int(math.ceil(base_min * norm_k_recs))
        rec_max = int(round(max_total * norm_k_recs))
        if rec_max < rec_min: rec_max = rec_min
        rec_median = med_total * norm_k_recs

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

        depth_percent = 0
        if rec_median > 0.1: depth_percent = int(round((my_tf_count / rec_median) * 100))
        else: depth_percent = 0 if my_tf_count == 0 else 100

        weight_hybrid = word_idf_map.get(lemma, 0) * (my_tf_count / my_len if my_len > 0 else 0)
        table_depth.append({
            "Слово": lemma, "Словоформы": forms_str, "Вхождений у вас": my_tf_count,
            "Медиана": round(med_total, 1), "Минимум (рек)": rec_min, "Максимум (рек)": rec_max,
            "Глубина %": min(100, depth_percent), "Статус": status, "Рекомендация": action_text,
            "is_missing": (status == "Недоспам" and my_tf_count == 0),
            "sort_val": abs(action_diff) if status != "Норма" else 0
        })
        table_hybrid.append({
            "Слово": lemma, "TF-IDF ТОП": round(word_idf_map.get(lemma, 0) * (med_total / avg_dl if avg_dl > 0 else 0), 4),
            "TF-IDF у вас": round(weight_hybrid, 4), "Сайтов": df, "Переспам": max_total
        })

    table_rel = []
    for item in original_results:
        url = item['url']
        scores = competitor_scores_map.get(url, {'width_final':0, 'depth_final':0})
        table_rel.append({
            "Домен": urlparse(url).netloc, "Позиция": item['pos'],
            "Ширина (балл)": scores['width_final'], "Глубина (балл)": scores['depth_final']
        })

    my_label = f"{my_data['domain']} (Вы)" if (my_data and my_data.get('domain')) else "Ваш сайт"
    table_rel.append({ "Домен": my_label, "Позиция": my_serp_pos if my_serp_pos > 0 else len(original_results) + 1,
        "Ширина (балл)": my_width_score_final, "Глубина (балл)": my_depth_score_final })

    return {
        "depth": pd.DataFrame(table_depth), "hybrid": pd.DataFrame(table_hybrid),
        "relevance_top": pd.DataFrame(table_rel).sort_values(by='Позиция', ascending=True).reset_index(drop=True),
        "my_score": {"width": my_width_score_final, "depth": my_depth_score_final},
        "missing_semantics_high": missing_semantics_high, "missing_semantics_low": missing_semantics_low
    }

# ==========================================
# 5. ФУНКЦИЯ ОТОБРАЖЕНИЯ (PAGINATION)
# ==========================================
def render_paginated_table(df, title_text, key_prefix, default_sort_col=None, use_abs_sort_default=False):
    if df.empty:
        st.info(f"{title_text}: Нет данных.")
        return

    col_t1, col_t2 = st.columns([7, 3])
    with col_t1: st.markdown(f"### {title_text}")

    if f'{key_prefix}_sort_col' not in st.session_state:
        st.session_state[f'{key_prefix}_sort_col'] = default_sort_col if (default_sort_col and default_sort_col in df.columns) else df.columns[0]
    if f'{key_prefix}_sort_order' not in st.session_state:
        st.session_state[f'{key_prefix}_sort_order'] = "Убывание"

    search_query = st.text_input(f"🔍 Поиск ({title_text})", key=f"{key_prefix}_search")
    if search_query:
        mask = df.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        st.warning("Ничего не найдено.")
        return

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
    if use_abs_sort_default and sort_col == "Рекомендация" and "sort_val" in df_filtered.columns:
         df_filtered = df_filtered.sort_values(by="sort_val", ascending=ascending)
    elif ("Добавить" in sort_col or "+/-" in sort_col) and df_filtered[sort_col].dtype == object:
        try:
            df_filtered['_temp_sort'] = df_filtered[sort_col].astype(str).str.replace(r'[^\d]', '', regex=True)
            df_filtered['_temp_sort'] = pd.to_numeric(df_filtered['_temp_sort'], errors='coerce').fillna(0)
            df_filtered = df_filtered.sort_values(by='_temp_sort', ascending=ascending).drop(columns=['_temp_sort'])
        except:
             df_filtered = df_filtered.sort_values(by=sort_col, ascending=ascending)
    else:
        df_filtered = df_filtered.sort_values(by=sort_col, ascending=ascending)

    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered.index = df_filtered.index + 1

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        export_df = df_filtered.copy()
        if "is_missing" in export_df.columns: del export_df["is_missing"]
        if "sort_val" in export_df.columns: del export_df["sort_val"]
        export_df.to_excel(writer, index=False, sheet_name='Data')
    excel_data = buffer.getvalue()

    with col_t2:
        st.download_button(label="📥 Скачать Excel", data=excel_data, file_name=f"{key_prefix}_export.xlsx", mime="application/vnd.ms-excel", key=f"{key_prefix}_down")

    ROWS_PER_PAGE = 20
    if f'{key_prefix}_page' not in st.session_state: st.session_state[f'{key_prefix}_page'] = 1
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
                if status == "Недоспам": cell_style += "color: #D32F2F; font-weight: bold;"
                elif status == "Переспам": cell_style += "color: #E65100; font-weight: bold;"
                elif status == "Норма": cell_style += "color: #2E7D32; font-weight: bold;"
            styles.append(cell_style)
        return styles

    cols_to_hide = ["is_missing", "sort_val"]
    cols_to_hide = [c for c in cols_to_hide if c in df_view.columns]

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
# 6. ЛОГИКА ДЛЯ PERPLEXITY (AI GEN)
# ==========================================
STATIC_DATA_GEN = {
    'IP_PROP4817': "Условия поставки",
    'IP_PROP4818': "Оперативные отгрузки в регионы точно в срок",
    'IP_PROP4819': """<p>Надежная и быстрая доставка заказа в любую точку страны: "Стальметурал" отгружает товар 24 часа в сутки, 7 дней в неделю. Более 4 000 отгрузок в год. При оформлении заказа менеджер предложит вам оптимальный логистический маршрут.</p>""",
    'IP_PROP4820': """<p>Наши изделия успешно применяются на некоторых предприятиях Урала, центрального региона, Поволжья, Сибири. Партнеры по логистике предложат доставить заказ самым удобным способом – автомобильным, железнодорожным, даже авиационным транспортом. Для вас разработают транспортную схему под удобный способ получения. Погрузка выполняется полностью с соблюдением особенностей техники безопасности.</p>
<div class="h4"><h4>Самовывоз</h4></div><p>Если обычно соглашаетесь самостоятельно забрать товар или даете это право уполномоченным, адрес и время работы склада в своем городе уточняйте у менеджера.</p>
<div class="h4"><h4>Грузовой транспорт компании</h4></div><p>Отправим прокат на ваш объект собственным автопарком. Получение в упаковке для безопасной транспортировки, а именно на деревянном поддоне.</p>
<div class="h4"><h4>Сотрудничаем с ТК</h4></div><p>Доставка с помощью транспортной компании по России и СНГ. Окончательная цена может измениться, так как ссылается на прайс-лист, который предоставляет контрагент, однако, сравним стоимость логистических служб и выберем лучшую.</p>""",
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

def generate_five_blocks(client, base_text, tag_name):
    if not base_text: return ["Error: No base text"] * 5
    system_instruction = "Ты — профессиональный технический копирайтер. Напиши 5 HTML блоков. Не используй markdown."
    
    user_prompt = f"""ВВОДНЫЕ: Тег "{tag_name}". База: \"\"\"{base_text[:3000]}\"\"\"
    ЗАДАЧА: 5 блоков. Структура: h2/h3, абзац, вводная фраза:, список, заключение. Без [1] ссылок. Разделитель: |||BLOCK_SEP|||"""

    try:
        # Убрали seo_words из аргументов
        response = client.chat.completions.create(model="sonar-pro", messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}], temperature=0.7)
        content = response.choices[0].message.content
        content = re.sub(r'\[\d+\]', '', content).replace("```html", "").replace("```", "")
        blocks = [b.strip() for b in content.split("|||BLOCK_SEP|||") if b.strip()]
        while len(blocks) < 5: blocks.append("")
        return blocks[:5]
    except Exception as e: return [f"API Error: {str(e)}"] * 5

def generate_html_table(client, user_prompt):
    # Убрали seo_instruction про MANDATORY SEO и жирный шрифт
    
    system_instruction = f"Generate HTML tables. Inline CSS: table border 2px solid black, th bg #f0f0f0. No markdown."
    try:
        response = client.chat.completions.create(model="sonar-pro", messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}], temperature=0.7)
        return re.sub(r'\[\d+\]', '', response.choices[0].message.content).replace("```html", "").replace("```", "").strip()
    except Exception as e: return f"Error: {e}"

# ==========================================
# 7. ИНТЕРФЕЙС (TABS)
# ==========================================
tab_seo, tab_ai, tab_tags, tab_tables = st.tabs(["📊 SEO Анализ", "🤖 AI Генерация", "🏷️ Генератор тегов", "🧩 Таблицы"])

# ------------------------------------------
# Вкладка 1: SEO
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
            def update_manual_urls(): st.session_state['persistent_urls'] = st.session_state.manual_urls_widget
            st.text_area("Список ссылок (каждая с новой строки)", height=200, key="manual_urls_widget", value=st.session_state['persistent_urls'], on_change=update_manual_urls)

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
             if new_arsenkin:
                 st.session_state.arsenkin_token = new_arsenkin
                 ARSENKIN_TOKEN = new_arsenkin 
        
        if not YANDEX_DICT_KEY:
             new_yandex = st.text_input("Yandex Dict Key", type="password", key="input_yandex")
             if new_yandex:
                 st.session_state.yandex_dict_key = new_yandex
                 YANDEX_DICT_KEY = new_yandex

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

    # --- ЛОГИКА АНАЛИЗА (ВНУТРИ ВКЛАДКИ) ---
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
            if not ARSENKIN_TOKEN:
                st.error("Отсутствует API токен Arsenkin. Введите его в настройках или в secrets.toml")
                st.stop()
                
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(parse_page, u['url'], settings): u['url'] for u in target_urls_raw}
            done, total = 0, len(target_urls_raw)
            prog = st.progress(0)
            for f in concurrent.futures.as_completed(futures):
                if res := f.result(): comp_data_full.append(res)
                done += 1
                prog.progress(done / total)
        prog.empty()

        with st.spinner("Расчет метрик..."):
            st.session_state.analysis_results = calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, target_urls_raw)
            st.session_state.analysis_done = True

            # ==========================================
            # КЛАССИФИКАЦИЯ
            # ==========================================
            res = st.session_state.analysis_results
            words_to_check = [x['word'] for x in res.get('missing_semantics_high', [])]

            if not words_to_check:
                st.session_state.categorized_products = []
                st.session_state.categorized_services = []
                st.session_state.categorized_commercial = []
                st.session_state.categorized_dimensions = []
            else:
                # Здесь вызывается ваша новая функция классификации
                with st.spinner("Уточнение семантики (Pymorphy)..."):
                    categorized = classify_semantics_with_api(words_to_check, YANDEX_DICT_KEY)

                st.session_state.categorized_products = categorized['products']
                st.session_state.categorized_services = categorized['services']
                st.session_state.categorized_commercial = categorized['commercial']
                st.session_state.categorized_dimensions = categorized['dimensions']
                
                # ==========================================================
                # ВСТАВИТЬ ЭТИ СТРОКИ СЮДА:
                # Принудительно обновляем виджет во вкладке "Генератор тегов"
                # ==========================================================
                products_str = "\n".join(st.session_state.categorized_products)
                st.session_state['tags_products_edit_smart'] = products_str  # <--- ДОБАВИТЬ ВОТ ЭТО

            st.rerun()

    if st.session_state.analysis_done and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.success("Анализ готов!")
        st.markdown(f"<div style='background:{LIGHT_BG_MAIN};padding:15px;border-radius:8px;'><b>Результат:</b> Ширина: {results['my_score']['width']} | Глубина: {results['my_score']['depth']}</div>", unsafe_allow_html=True)

        with st.expander("🛒 Результат группировки слов (С учетом Яндекс API)", expanded=True):
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            with c1:
                st.info(f"🧱 Товары ({len(st.session_state.categorized_products)})")
                st.caption(", ".join(st.session_state.categorized_products))

            with c2:
                st.error(f"🛠️ Услуги ({len(st.session_state.categorized_services)})")
                st.caption(", ".join(st.session_state.categorized_services))
            
            with c3:
                st.warning(f"💰 Коммерция / Гео / Общее ({len(st.session_state.categorized_commercial)})")
                st.caption(", ".join(st.session_state.categorized_commercial))

            with c4:
                dims = st.session_state.get('categorized_dimensions', [])
                st.success(f"📏 Размеры и марки ({len(dims)})")
                st.caption(", ".join(dims))

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
# Вкладка 2: AI
# ------------------------------------------
with tab_ai:
    st.title("AI Генератор (Perplexity)")
    pplx_key = st.text_input("Perplexity API Key", type="password", key="pplx_key_input")
    target_url_gen = st.text_input("URL Страницы (донор тегов)", key="pplx_url_input")

    if st.button("🚀 Начать генерацию", key="btn_start_gen", disabled=not pplx_key):
        st.session_state.ai_generated_df = None
        if not openai: st.error("Нет openai"); st.stop()
        client = openai.OpenAI(api_key=pplx_key, base_url="https://api.perplexity.ai")

        with st.status("Генерация...", expanded=True) as status:
            base_text, tags, err = get_page_data_for_gen(target_url_gen)
            if err or not tags: st.error(err or "Нет тегов"); st.stop()

            seo_list = [x['word'] for x in st.session_state.analysis_results.get('missing_semantics_high', []) if x['word'] not in GARBAGE_LATIN_STOPLIST][:15] if st.session_state.analysis_results else []

            all_rows = []
            bar = st.progress(0)
            for i, tag in enumerate(tags):
                blocks = generate_five_blocks(client, base_text, tag['name'], seo_list)
                all_rows.append({'TagName': tag['name'], 'URL': tag['url'], 'IP_PROP4839': blocks[0], 'IP_PROP4816': blocks[1], 'IP_PROP4838': blocks[2], 'IP_PROP4829': blocks[3], 'IP_PROP4831': blocks[4], **STATIC_DATA_GEN})
                bar.progress((i+1)/len(tags))

            df = pd.DataFrame(all_rows)
            st.session_state.ai_generated_df = df
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer: df.to_excel(writer, index=False)
            st.session_state.ai_excel_bytes = buffer.getvalue()
            st.rerun()

    if st.session_state.ai_generated_df is not None:
        st.download_button("📥 Скачать Excel", st.session_state.ai_excel_bytes, "seo_texts.xlsx", "application/vnd.ms-excel")
        st.dataframe(st.session_state.ai_generated_df.head())

# ------------------------------------------
# Вкладка 3: ТЕГИ (SMART MASS PRODUCTION v15)
# ------------------------------------------
with tab_tags:
    st.title("🏷️ Генератор плитки тегов (Smart SEO)")

    # --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
    
    # Кэш для спеллера
    if 'speller_cache' not in st.session_state:
        st.session_state.speller_cache = {}

    def spell_check_yandex_cached(text):
        """Проверка орфографии с кэшированием"""
        if not text: return ""
        if text in st.session_state.speller_cache:
            return st.session_state.speller_cache[text]
            
        url = "https://speller.yandex.net/services/spellservice.json/checkText"
        params = {"text": text, "lang": "ru", "options": 518}
        try:
            r = requests.get(url, params=params, timeout=1.0)
            if r.status_code == 200:
                data = r.json()
                fixed_text = text
                for error in data:
                    if error.get('s'):
                        fixed_text = fixed_text.replace(error['word'], error['s'][0])
                
                # Сохраняем в кэш
                st.session_state.speller_cache[text] = fixed_text
                return fixed_text
        except:
            pass
        return text

    def smart_reverse_translit(slug):
        """
        Умный парсинг Slug -> Человеческое название (v15: Hybrid GOST/Latin)
        """
        # --- 1. ТОЧНЫЙ СЛОВАРЬ (Переопределяет любую логику) ---
        TECHNICAL_DICT = {
            # Самые ходовые ГОСТы (транслит -> кириллица)
            'gost': 'ГОСТ', 'tu': 'ТУ',
            'st3': 'Ст3', 'st3sp': 'Ст3сп', 'st3ps': 'Ст3пс',
            '09g2s': '09Г2С', '17g1s': '17Г1С',
            'a500c': 'А500С', 'a500s': 'А500С', 'v500s': 'В500С', # Важно! с -> С (кириллица)
            'a240': 'А240', 'a400': 'А400', 'a500': 'А500',
            '12x18n10t': '12Х18Н10Т', '08x18n10': '08Х18Н10',
            '40x': '40Х', '20x': '20Х', '65g': '65Г',
            'd16t': 'Д16Т', 'amg': 'АМг', 'ad31': 'АД31',
            # Полимеры
            'pvc': 'ПВХ', 'pnd': 'ПНД', 'pvd': 'ПВД',
            # Сокращения
            'hk': 'Х/К', 'gk': 'Г/К', 'bp': 'ВР'
        }

        # --- 2. ЕДИНИЦЫ ИЗМЕРЕНИЯ ---
        UNITS_MAP = {
            'mm': 'мм', 'cm': 'см', 'm': 'м', 'kg': 'кг', 't': 'т', 
            'sht': 'шт', 'rub': 'руб'
        }

        # --- 3. МАРКЕРЫ ЗАПАДНЫХ МАРОК (Оставляем латиницей) ---
        # Если слово начинается с этого -> Uppercase (без транслита)
        LATIN_STARTS = ('aisi', 'astm', 'din', 'en', 'hardox', 'weldox', 'magnelis', 'ruukki', 'ssab')
        
        # Если слово содержит эти буквы (маркеры евро-стандартов), то скорее всего это латиница
        # J (S355J2), W (Weldox), Q (S460Q), R (S235JR) - в транслите ГОСТа J и Q почти не юзают.
        LATIN_CHARS_MARKERS = ['j', 'q', 'w'] 
        
        # Специфичные европейские марки стали (S + цифры, P + цифры и т.д.)
        # Регулярка ловит: s355, p265, l450 и т.д.
        EURO_GRADE_PATTERN = re.compile(r'^[sple]\d{3}[a-z0-9]*$', re.IGNORECASE)

        # --- НАЧАЛО ОБРАБОТКИ ---
        slug = slug.lower().strip()
        slug = re.sub(r'\.html|\.php|\.htm', '', slug)
        slug = slug.replace('_', '-').replace('/', '-')
        
        parts = [p for p in slug.split('-') if p]
        final_words = []

        for part in parts:
            # A. ПРОВЕРКА ПО СЛОВАРЮ (Приоритет №1)
            if part in TECHNICAL_DICT:
                final_words.append(TECHNICAL_DICT[part])
                continue

            # B. ПРОВЕРКА НА РАЗМЕР (100mm -> 100мм)
            is_unit = False
            for eng_unit, rus_unit in UNITS_MAP.items():
                if part.endswith(eng_unit) and part[:-len(eng_unit)].replace('.', '').isdigit():
                    num = part[:-len(eng_unit)]
                    final_words.append(f"{num}{rus_unit}")
                    is_unit = True
                    break
            if is_unit: continue

            # C. ПРОВЕРКА НА ЛАТИНСКУЮ МАРКУ (Приоритет №2)
            # 1. Известные бренды/стандарты (hardox, aisi...)
            if part.startswith(LATIN_STARTS):
                final_words.append(part.upper())
                continue
            
            # 2. Наличие специфичных латинских букв (S355J2, AISI)
            if any(marker in part for marker in LATIN_CHARS_MARKERS):
                final_words.append(part.upper())
                continue
            
            # 3. Евро-паттерны (S355, P265...)
            if EURO_GRADE_PATTERN.match(part):
                final_words.append(part.upper())
                continue

            # D. ТРАНСЛИТЕРАЦИЯ (ГОСТ и Обычные слова)
            
            # Подготовка текста (Cyrillic mapping)
            text = part
            replacements = [
                ('shch', 'щ'), ('sch', 'щ'), ('sh', 'ш'), ('ch', 'ч'), ('zh', 'ж'),
                ('yu', 'ю'), ('ya', 'я'), ('yo', 'ё'), ('ts', 'ц'), ('tc', 'ц'), ('kh', 'х')
            ]
            for eng, rus in replacements:
                text = text.replace(eng, rus)

            text = re.sub(r'iy(?=\s|$)', 'ий', text)
            text = re.sub(r'yy(?=\s|$)', 'ый', text)
            text = text.replace('ij', 'ий')
            text = re.sub(r'y(?=\s|$)', 'ы', text)

            mapping = {
                'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
                'h': 'х', 'i': 'и', 'j': 'й', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
                'o': 'о', 'p': 'п', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
                'v': 'в', 'w': 'в', 'x': 'х', 'z': 'з', 'y': 'ы', 'q': 'к'
            }
            chars = []
            for c in text:
                chars.append(mapping.get(c, c))
            rus_text = "".join(chars)

            # E. ЭВРИСТИКА ГОСТА (Приоритет №3)
            # Если после транслита в слове остались цифры -> это ГОСТ марка -> CAPS
            # Пример: 09g2s -> 09г2с -> 09Г2С
            if any(char.isdigit() for char in rus_text) and len(rus_text) < 10:
                 final_words.append(rus_text.upper())
            else:
                # Обычное слово -> Прогоняем через Спеллер (если длинное)
                if len(rus_text) > 3:
                     rus_text = spell_check_yandex_cached(rus_text)
                final_words.append(rus_text)

        # Сборка фразы (Первая буква заглавная)
        result = " ".join(final_words)
        return result[0].upper() + result[1:] if result else ""

    # --- ИНТЕРФЕЙС ---
    col_t1, col_t2 = st.columns([1, 1])
    
    with col_t1:
        st.markdown("##### 🔗 Источник (Откуда парсим)")
        category_url = st.text_input("URL Категории (где собрать список подкатегорий)", placeholder="https://site.ru/catalog/truba/")
        
        st.markdown("##### 📂 База ссылок")
        uploaded_file = st.file_uploader("Файл со ссылками (.txt)", type=["txt"], key="urls_uploader_smart")

    with col_t2:
        st.markdown("##### 📝 Список товаров (Ключи поиска)")
        
        # 1. Проверяем, существует ли ключ в session_state. 
        # Если нет (первый запуск) — заполняем его данными из categorized_products.
        # Если да (после анализа или ввода вручную) — используем существующее значение.
        if "tags_products_edit_smart" not in st.session_state:
            raw_products = st.session_state.get('categorized_products', [])
            st.session_state.tags_products_edit_smart = "\n".join(raw_products) if raw_products else ""

        # 2. Создаем виджет БЕЗ аргумента 'value'. 
        # Streamlit сам возьмет значение из st.session_state['tags_products_edit_smart'].
        products_input = st.text_area(
            "Список товаров (будут искаться в базе):", 
            height=200, 
            key="tags_products_edit_smart",
            help="Скрипт будет искать ссылки, содержащие эти слова (в транслите)."
        )
        
        # Превращаем текст обратно в список для скрипта
        products = [line.strip() for line in products_input.split('\n') if line.strip()]

    # --- ЗАПУСК ---
    st.markdown("---")
    if st.button("🚀 Запустить Smart-генерацию", key="btn_tags_smart_gen", disabled=(not products or not uploaded_file or not category_url)):
        
        status_box = st.status("🚀 Запуск процесса...", expanded=True)
        
        # 1. Парсинг
        status_box.write(f"🕵️ Парсим категорию: {category_url}")
        target_urls_list = []
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            r = requests.get(category_url, headers=headers, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                tags_container = soup.find(class_='popular-tags-inner')
                if tags_container:
                    for link in tags_container.find_all('a'):
                        href = link.get('href')
                        if href: target_urls_list.append(urljoin(category_url, href))
                else:
                    status_box.warning("Не найден блок .popular-tags-inner. Ищем ссылки в контенте...")
                    main_area = soup.find('main') or soup.body
                    if main_area:
                        for link in main_area.find_all('a'):
                            href = link.get('href')
                            if href and '/catalog/' in href:
                                target_urls_list.append(urljoin(category_url, href))
        except Exception as e:
            status_box.error(f"Ошибка парсинга: {e}")
            st.stop()
            
        target_urls_list = list(set(target_urls_list))
        
        if not target_urls_list:
            status_box.error("Целевые страницы не найдены.")
            st.stop()
            
        status_box.write(f"✅ Найдено страниц: {len(target_urls_list)}")

        # 2. Индексация базы
        status_box.write("📂 Индексация базы ссылок...")
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        all_txt_links = [line.strip() for line in stringio.readlines() if line.strip()]
        
        product_candidates_map = {}
        for p in products:
            tr = transliterate_text(p)
            if len(tr) >= 3:
                matches = [u for u in all_txt_links if tr in u.lower()]
                if matches: product_candidates_map[p] = matches
        
        status_box.write(f"✅ Товары сопоставлены: {len(product_candidates_map)} шт.")

        # 3. Генерация
        status_box.write("🧠 Генерация анкоров (Smart Translit v15)...")
        final_rows = []
        prog_bar = st.progress(0)
        
        with requests.Session() as session:
            for i, target_url in enumerate(target_urls_list):
                current_page_tags = []
                
                available_products = list(product_candidates_map.keys())
                random.shuffle(available_products)
                limit = random.randint(12, 20)
                selected_products = available_products[:limit]
                
                for prod_name in selected_products:
                    candidates = product_candidates_map[prod_name]
                    norm_target = target_url.rstrip('/')
                    valid_candidates = [u for u in candidates if u.rstrip('/') != norm_target]
                    
                    if valid_candidates:
                        chosen_url = random.choice(valid_candidates)
                        
                        # SMART NAME GENERATION
                        try:
                            parsed = urlparse(chosen_url)
                            path_parts = parsed.path.strip('/').split('/')
                            slug = path_parts[-1] if path_parts[-1] else (path_parts[-2] if len(path_parts)>1 else "")
                            
                            if not slug or len(slug) < 3:
                                anchor_text = prod_name.capitalize()
                            else:
                                anchor_text = smart_reverse_translit(slug)
                        except:
                            anchor_text = prod_name.capitalize()
                        
                        current_page_tags.append({
                            'name': anchor_text,
                            'url': chosen_url
                        })
                
                if current_page_tags:
                    random.shuffle(current_page_tags)
                    html_block = '<div class="popular-tags">\n' + \
                                 "\n".join([f'    <a href="{item["url"]}" class="tag-link">{item["name"]}</a>' for item in current_page_tags]) + \
                                 '\n</div>'
                else:
                    html_block = "<!-- Нет тегов -->"
                
                final_rows.append({
                    'Page URL': target_url,
                    'Tags HTML': html_block
                })
                
                prog_bar.progress((i + 1) / len(target_urls_list))

        prog_bar.empty()
        status_box.update(label="Готово!", state="complete", expanded=False)

        # 4. Скачивание
        df_tags_result = pd.DataFrame(final_rows)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_tags_result.to_excel(writer, index=False)
            worksheet = writer.sheets['Sheet1']
            worksheet.set_column('A:A', 60)
            worksheet.set_column('B:B', 100)
            
        excel_bytes = buffer.getvalue()
        
        st.success(f"🎉 Сгенерировано {len(final_rows)} блоков.")
        st.download_button(
            label="📥 Скачать Excel",
            data=excel_bytes,
            file_name="smart_tags_tiles.xlsx",
            mime="application/vnd.ms-excel"
        )

# ------------------------------------------
# Вкладка 4: ТАБЛИЦЫ
# ------------------------------------------
# ------------------------------------------
# Вкладка 4: ТАБЛИЦЫ (MASS GEN v2)
# ------------------------------------------
with tab_tables:
    st.title("🧩 Генератор таблиц (Mass Production)")
    
    # 1. Настройки доступа и источника
    c_set1, c_set2 = st.columns(2)
    with c_set1:
        pplx_key_tbl = st.text_input("Perplexity API Key", type="password", key="pplx_key_tbl")
    with c_set2:
        target_url_tbl = st.text_input("URL Родительской категории (источник тегов)", placeholder="https://site.ru/catalog/armatura/", key="url_source_tbl")

    st.markdown("---")

    # 2. Настройка количества и описания таблиц
    st.subheader("⚙️ Конфигурация таблиц")
    
    num_tables = st.selectbox("Сколько таблиц генерировать для каждого урла?", [1, 2, 3, 4, 5], key="num_tables_select")
    
    table_prompts = []
    
    # Динамическое создание полей ввода
    st.info("📝 Опишите, что должно быть в каждой таблице. ИИ наполнит их данными, специфичными для конкретного товара.")
    
    cols_prompts = st.columns(num_tables)
    for i in range(num_tables):
        with cols_prompts[i]:
            def_val = f"Технические характеристики" if i == 0 else f"Размеры и вес"
            prompt_text = st.text_area(f"Таблица №{i+1} (Описание)", value=def_val, height=150, key=f"tbl_prompt_{i}")
            table_prompts.append(prompt_text)

    # 3. Логика запуска
    if st.button("🚀 Запустить генерацию таблиц", key="btn_gen_tbl_mass", disabled=(not pplx_key_tbl or not target_url_tbl)):
        
        # Инициализация клиента
        if not openai: 
            st.error("Библиотека openai не установлена/не найдена.")
            st.stop()
            
        client = openai.OpenAI(api_key=pplx_key_tbl, base_url="https://api.perplexity.ai")
        
        status_box = st.status("⏳ Анализ категории...", expanded=True)
        
        # А. Парсинг тегов (используем ту же функцию, что и в AI вкладке)
        try:
            _, tags_data, err_msg = get_page_data_for_gen(target_url_tbl)
            if err_msg or not tags_data:
                status_box.error(f"Ошибка сбора тегов: {err_msg if err_msg else 'Теги не найдены'}")
                st.stop()
            
            # Ограничитель на случай сбоя парсинга (чтобы не сжечь бюджет, если тегов 1000)
            status_box.write(f"✅ Найдено товаров (тегов): {len(tags_data)}")
        except Exception as e:
            status_box.error(f"Критическая ошибка: {e}")
            st.stop()

        # Б. Генерация
        all_table_rows = []
        progress_bar = st.progress(0)
        
        total_steps = len(tags_data)
        
        for idx, tag_item in enumerate(tags_data):
            tag_name = tag_item['name']
            tag_url = tag_item['url']
            
            status_box.write(f"⚙️ Обработка: {tag_name}...")
            
            # Формируем строку данных
            row_data = {'URL': tag_url, 'Название': tag_name}
            
            # Генерируем каждую таблицу по очереди
            for tbl_i, prompt_desc in enumerate(table_prompts):
                
                # Специальный промт для уникальности
                system_instruction = "You are a senior technical data specialist. Output ONLY HTML code. No markdown formatting, no backticks, no introduction."
                user_prompt = f"""
                CONTEXT: The specific product/sub-category is "{tag_name}".
                TASK: Generate a technical HTML table based on this description: "{prompt_desc}".
                CRITICAL REQUIREMENTS:
                1. The data inside the table MUST be specific to "{tag_name}", not generic.
                2. Style: <table style="width:100%; border-collapse: collapse; border: 2px solid black;">, headers with background #f0f0f0.
                3. Return ONLY the HTML <table>...</table> code.
                """
                
                try:
                    response = client.chat.completions.create(
                        model="sonar-pro", 
                        messages=[
                            {"role": "system", "content": system_instruction}, 
                            {"role": "user", "content": user_prompt}
                        ], 
                        temperature=0.5 # Чуть ниже температура для большей точности данных
                    )
                    html_content = response.choices[0].message.content
                    
                    # Очистка от маркдауна, если ИИ все же добавил его
                    html_content = re.sub(r'```html', '', html_content)
                    html_content = re.sub(r'```', '', html_content).strip()
                    
                    row_data[f'Table {tbl_i+1}'] = html_content
                    
                except Exception as e:
                    row_data[f'Table {tbl_i+1}'] = f"Error: {e}"
            
            all_table_rows.append(row_data)
            progress_bar.progress((idx + 1) / total_steps)

        # В. Завершение и сохранение
        status_box.update(label="✅ Генерация завершена!", state="complete", expanded=False)
        
        df_tables = pd.DataFrame(all_table_rows)
        
        # Сохранение в Session State для отображения кнопки скачивания
        st.session_state.tables_gen_df = df_tables
        
        # Создание Excel в памяти
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_tables.to_excel(writer, index=False)
        st.session_state.tables_excel_bytes = buffer.getvalue()
        
        st.success(f"Готово! Сгенерировано таблиц для {len(df_tables)} страниц.")

    # 4. Вывод результатов и скачивание
    if 'tables_gen_df' in st.session_state and st.session_state.tables_gen_df is not None:
        st.markdown("### 📥 Результаты")
        
        st.download_button(
            label="Скачать Excel файл",
            data=st.session_state.tables_excel_bytes,
            file_name="generated_tables.xlsx",
            mime="application/vnd.ms-excel",
            type="primary"
        )
        
        st.dataframe(st.session_state.tables_gen_df.head(), use_container_width=True)
