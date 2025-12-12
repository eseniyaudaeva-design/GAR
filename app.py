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
# Мы НЕ пишем ключи здесь. Мы читаем их из st.secrets

# Попытка получить Arsenkin Token
if "arsenkin_token" in st.session_state:
    ARSENKIN_TOKEN = st.session_state.arsenkin_token
else:
    # Пробуем достать из secrets.toml
    try:
        ARSENKIN_TOKEN = st.secrets["api"]["arsenkin_token"]
    except (FileNotFoundError, KeyError):
        ARSENKIN_TOKEN = None

# Попытка получить Yandex Key
if "yandex_dict_key" in st.session_state:
    YANDEX_DICT_KEY = st.session_state.yandex_dict_key
else:
    # Пробуем достать из secrets.toml
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
        # Если ключа нет, просто возвращаем слово как есть
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
                # Берем первое определение
                first_def = data['def'][0]
                lemma = first_def.get('text', text)
                pos = first_def.get('pos', 'unknown')
                return {'lemma': lemma, 'pos': pos}
    except:
        pass
    return {'lemma': text, 'pos': 'unknown'}

# --- ФУНКЦИЯ КЛАССИФИКАЦИИ С API ЯНДЕКСА (ИСПРАВЛЕННАЯ v2.5 - Технические прилагательные) ---
def classify_semantics_with_api(words_list, yandex_key):
    # 1. СПИСКИ-ИСКЛЮЧЕНИЯ И ПАТТЕРНЫ
    
    dim_pattern = re.compile(r'\d+(?:[\.\,]\d+)?\s?[хx\*×]\s?\d+', re.IGNORECASE)
    grade_pattern = re.compile(r'^([а-яa-z]{1,4}\-?\d+[а-яa-z0-9]*)$', re.IGNORECASE)
    gost_pattern = re.compile(r'(гост|din|ту|iso|ст|сп)\s?\d+', re.IGNORECASE)

    # UI и Навигация
    SITE_UI_GARBAGE = {
        'меню', 'поиск', 'главная', 'карта', 'сайт', 'личный', 'кабинет', 
        'вход', 'регистрация', 'корзина', 'избранное', 'сравнение', 'профиль',
        'телефон', 'адрес', 'контакты', 'email', 'звонок', 'callback', 
        'отзыв', 'отзывы', 'вопрос', 'ответ', 'менеджер', 'консультация',
        'политика', 'конфиденциальность', 'соглашение', 'оферта', 'cookie',
        'соглашаться', 'согласие', 'принимать', 'отправить',
        'ошибка', 'успешно', 'кнопка', 'форма', 'поле', 'обзор', 'новости', 'статьи',
        'характеристика', 'описание', 'параметр', 'свойство', 'артикул', 'код',
        'калькулятор', 'фильтр', 'сортировка', 'показать', 'сбросить',
        'имя', 'фамилия', 'сообщение', 'файл', 'документ', 'сертификат',
        'категория', 'раздел', 'список', 'вид', 'тип', 'класс', 'серия',
        'рейтинг', 'наличие', 'склад', 'производитель', 'бренд', 'марка',
        'вес', 'длина', 'ширина', 'высота', 'толщина', 'диаметр', 'размер',
        'объем', 'масса', 'тонна', 'метр', 'шт', 'кг', 'упаковка', 'цена',
        'интернет', 'магазин', 'каталог', 'год'
    }

    # Коммерция + Маркетинговые прилагательные (Мусорные для товаров)
    COMMERCIAL_WORDS = {
        'купить', 'заказать', 'цена', 'цены', 'прайс', 'стоимость', 
        'продажа', 'недорого', 'дешево', 'дорого', 'скидка', 'акция', 
        'распродажа', 'оптом', 'розница', 'руб', 'рублей', 'уе',
        'заказ', 'оплата', 'платеж', 'рассрочка', 'кредит', 'лизинг',
        'доставка', 'самовывоз', 'отгрузка', 'поставка', 'транспорт', 'логистика',
        'гарантия', 'возврат', 'обмен',
        # Прилагательные, которые НЕ являются товаром
        'выгодный', 'низкий', 'высокий', 'лучший', 'качественный', 'надежный',
        'большой', 'малый', 'удобный', 'быстрый', 'бесплатный', 'хороший',
        'доступный', 'индивидуальный', 'профессиональный', 'собственный',
        'официальный', 'уникальный', 'широкий', 'огромный', 'различный'
    }

    GEO_ROOTS = [
        'москв', 'питер', 'спб', 'екб', 'екатерин', 'росси', 'рф', 'город', 'област',
        'новгород', 'казан', 'киев', 'минск', 'алматы', 'самара', 'омск', 'челябин',
        'ростов', 'уфа', 'волгоград', 'перм', 'краснояр', 'воронеж', 'саратов', 'краснодар',
        'тюмен', 'ижевск', 'тольятти', 'барнаул', 'иркутск', 'ульяновск', 'хабаровск'
    ]
    
    SERVICE_KEYWORDS = {
        'резка', 'гибка', 'сварка', 'оцинковка', 'рубка', 'монтаж', 'укладка', 
        'проектирование', 'изоляция', 'сверление', 'грунтовка', 'покраска', 'услуга',
        'металлообработка', 'обработка', 'строительство', 'ремонт', 'производство', 'изготовление'
    }

    categories = {
        'products': set(),
        'services': set(),
        'commercial': set(),
        'dimensions': set()
    }

    # 2. ПРЕДВАРИТЕЛЬНАЯ ФИЛЬТРАЦИЯ
    api_candidates = []

    for word in words_list:
        word_lower = word.lower()

        if dim_pattern.search(word_lower) or grade_pattern.match(word_lower) or gost_pattern.search(word_lower) or word_lower.isdigit():
            categories['dimensions'].add(word_lower)
            continue
        
        if morph:
            p = morph.parse(word_lower)[0]
            lemma = p.normal_form
        else:
            lemma = word_lower

        if lemma in SITE_UI_GARBAGE or lemma in COMMERCIAL_WORDS:
            categories['commercial'].add(lemma)
            continue
        
        if any(root in lemma for root in GEO_ROOTS):
            categories['commercial'].add(lemma)
            continue

        if lemma in SERVICE_KEYWORDS or lemma.endswith('обработка'):
            categories['services'].add(lemma)
            continue

        api_candidates.append(word_lower)

    # 3. ЗАПРОС К API YANDEX
    yandex_results = {} 
    
    if api_candidates and yandex_key:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_word = {executor.submit(get_yandex_dict_info, w, yandex_key): w for w in api_candidates}
            for future in concurrent.futures.as_completed(future_to_word):
                orig_word = future_to_word[future]
                try:
                    res = future.result()
                    yandex_results[orig_word] = res
                except:
                    yandex_results[orig_word] = {'lemma': orig_word, 'pos': 'unknown'}
    else:
        for w in api_candidates:
             yandex_results[w] = {'lemma': w, 'pos': 'unknown'}

    # 4. РАСПРЕДЕЛЕНИЕ (ИСПРАВЛЕННАЯ ЛОГИКА)
    for word in api_candidates:
        info = yandex_results.get(word, {'lemma': word, 'pos': 'unknown'})
        lemma = info['lemma']
        pos = info['pos']
        
        # Проверка на стоп-слова
        if lemma in SITE_UI_GARBAGE or lemma in COMMERCIAL_WORDS:
            categories['commercial'].add(lemma)
            continue

        # Услуги
        is_service = False
        if lemma.endswith('ние') or lemma.endswith('ение'):
            is_service = True
        elif lemma.endswith('обработка'):
            is_service = True
        elif lemma in SERVICE_KEYWORDS:
            is_service = True
        
        if is_service:
            categories['services'].add(lemma)
            continue

        # --- ГЛАВНОЕ ИЗМЕНЕНИЕ ---
        # Логика ТОВАРОВ:
        # 1. Существительное (noun) -> Товар (труба, сетка)
        # 2. Прилагательное (adjective) -> Товар (профильный, дюралевый), ЕСЛИ его нет в списке COMMERCIAL_WORDS
        # 3. Причастие (participle) -> Товар (сварной, оцинкованный)
        
        if pos == 'noun':
            if len(lemma) > 2:
                categories['products'].add(lemma)
        elif pos == 'adjective' or pos == 'participle': 
            # Если прилагательное прошло фильтр COMMERCIAL_WORDS выше, значит это техническое слово
            if len(lemma) > 2:
                categories['products'].add(lemma)
        elif pos == 'unknown':
            # Если не знаем, но похоже на прилагательное (материальное)
            if len(lemma) > 2:
                # В этом месте мы доверяем фильтру COMMERCIAL_WORDS. 
                # Если слово "дюралевый" не попало в стоп-лист, оно идет в товары.
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
                return res_data['result']  # Fallback
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
        
        # ЛОГИКА ВВОДА КЛЮЧЕЙ ЕСЛИ ИХ НЕТ В СЕКРЕТАХ
        # Если ключи не найдены в st.secrets (например, локальный запуск без файла secrets.toml),
        # предложим ввести их вручную.
        
        if not ARSENKIN_TOKEN:
             new_arsenkin = st.text_input("Arsenkin Token", type="password", key="input_arsenkin")
             if new_arsenkin:
                 st.session_state.arsenkin_token = new_arsenkin
                 ARSENKIN_TOKEN = new_arsenkin # Обновляем для текущего прогона
        
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
            # Проверка наличия ключа перед запуском
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
                with st.spinner("Уточнение семантики через Яндекс Словарь..."):
                    # Передаем текущий (возможно введенный вручную) ключ
                    categorized = classify_semantics_with_api(words_to_check, YANDEX_DICT_KEY)

                st.session_state.categorized_products = categorized['products']
                st.session_state.categorized_services = categorized['services']
                st.session_state.categorized_commercial = categorized['commercial']
                st.session_state.categorized_dimensions = categorized['dimensions']

            st.rerun()

    if st.session_state.analysis_done and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.success("Анализ готов!")
        st.markdown(f"<div style='background:{LIGHT_BG_MAIN};padding:15px;border-radius:8px;'><b>Результат:</b> Ширина: {results['my_score']['width']} | Глубина: {results['my_score']['depth']}</div>", unsafe_allow_html=True)

        # --- БЛОК ВИЗУАЛИЗАЦИИ КАТЕГОРИЙ (4 БЛОКА) ---
        with st.expander("🛒 Результат группировки слов (С учетом Яндекс API)", expanded=True):
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            # БЛОК 1: ТОВАРЫ
            with c1:
                st.info(f"🧱 Товары ({len(st.session_state.categorized_products)})")
                st.caption(", ".join(st.session_state.categorized_products))

            # БЛОК 2: УСЛУГИ
            with c2:
                st.error(f"🛠️ Услуги ({len(st.session_state.categorized_services)})")
                st.caption(", ".join(st.session_state.categorized_services))
            
            # БЛОК 3: КОММЕРЦИЯ (Глаголы, Деньги, Города)
            with c3:
                st.warning(f"💰 Коммерция / Гео / Общее ({len(st.session_state.categorized_commercial)})")
                st.caption(", ".join(st.session_state.categorized_commercial))

            # БЛОК 4: РАЗМЕРЫ И МАРКИ
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
# Вкладка 3: ТЕГИ (ОБНОВЛЕННАЯ ВЕРСИЯ v3)
# ------------------------------------------
with tab_tags:
    st.title("🏷️ Генератор плитки тегов")
    
    st.markdown("""
    **Режим: Умная перелинковка (SEO Matching)**
    Скрипт берет слова из поля ниже и ищет подходящие страницы в вашем файле ссылок.
    """)

    # 1. Подготовка списка слов (из Session State или пустой)
    auto_products = st.session_state.get('categorized_products', [])
    default_text_value = ", ".join(auto_products) if auto_products else ""

    # 2. Редактируемое поле ввода (как просили на скрине)
    st.markdown("### Список товарных слов (можно редактировать)")
    tags_input_text = st.text_area(
        "Введите слова через запятую", 
        value=default_text_value, 
        height=150, 
        key="tags_manual_input",
        help="Эти слова будут искаться в URL адресах файла."
    )

    # 3. Загрузчик файла
    uploaded_file = st.file_uploader("Загрузите файл со ссылками (.txt)", type=["txt"], key="urls_uploader")
    
    with st.expander("Какой формат файла нужен?", expanded=False):
        st.code("""https://site.ru/catalog/anod-mednyy/
https://site.ru/catalog/balka-bu/
https://site.ru/catalog/vtulki-rti/
... (каждая ссылка с новой строки)""", language="text")

    if st.button("🚀 Найти совпадения и создать плитку", key="btn_match_tags_txt"):
        # Парсим слова из текстового поля
        if not tags_input_text.strip():
            st.warning("Список слов пуст.")
            st.stop()
            
        # Разбиваем по запятым и чистим
        products_to_process = [x.strip() for x in tags_input_text.split(',') if x.strip()]

        if not uploaded_file:
            st.error("Пожалуйста, загрузите файл со ссылками.")
            st.stop()

        # Читаем файл ссылок
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        all_urls = [line.strip() for line in stringio.readlines() if line.strip()]

        if not all_urls:
            st.error("Файл со ссылками пуст.")
            st.stop()

        st.success(f"Обрабатываем слов: {len(products_to_process)} | Ссылок в файле: {len(all_urls)}")

        # 4. ЛОГИКА СОПОСТАВЛЕНИЯ
        matched_tags = []
        progress_bar = st.progress(0)
        
        for idx, word in enumerate(products_to_process):
            translit_word = transliterate_text(word)
            # Пропускаем слишком короткие слова, чтобы избежать мусора
            if len(translit_word) < 2: 
                continue 

            candidates = []
            
            for url in all_urls:
                # --- ЛОГИКА "ПОСЛЕ catalog/" ---
                url_lower = url.lower()
                
                # Если есть /catalog/, ищем только после него
                if '/catalog/' in url_lower:
                    search_scope = url_lower.split('/catalog/', 1)[1]
                else:
                    # Если нет catalog, берем путь (path) без домена
                    try:
                        parsed = urlparse(url_lower)
                        search_scope = parsed.path
                    except:
                        search_scope = url_lower # Фолбэк

                # Проверяем вхождение транслита в обрезанную часть URL
                if translit_word in search_scope:
                    candidates.append(url)
            
            # --- ФОРМИРОВАНИЕ РЕЗУЛЬТАТА ---
            if candidates:
                # Если нашли ссылки - берем случайную
                best_match = random.choice(candidates)
                matched_tags.append({
                    'name': word.capitalize(),
                    'url': best_match,
                    'has_link': True
                })
            else:
                # Если ссылок не нашли - все равно добавляем слово, но без ссылки
                matched_tags.append({
                    'name': word.capitalize(),
                    'url': '#',
                    'has_link': False
                })
            
            progress_bar.progress((idx + 1) / len(products_to_process))
        
        progress_bar.empty()

        # 5. ВЫВОД РЕЗУЛЬТАТА (HTML)
        st.subheader(f"✅ Сгенерировано тегов: {len(matched_tags)}")
        
        # Генерируем HTML
        # Если есть ссылка -> <a>, если нет -> <span> (визуально похожий, но неактивный)
        html_parts = []
        for item in matched_tags:
            if item['has_link']:
                html_parts.append(f'    <a href="{item["url"]}" class="tag-link">{item["name"]}</a>')
            else:
                # Спан для тегов без ссылок
                html_parts.append(f'    <span class="tag-link tag-empty">{item["name"]}</span>')

        html_output = '<div class="popular-tags">\n' + "\n".join(html_parts) + '\n</div>'
        
        st.session_state.tags_html_result = html_output
        st.rerun()

    # Блок отображения результата
    if st.session_state.tags_html_result:
        t_view, t_code = st.tabs(["👁️ Предпросмотр", "💻 HTML Код"])
        
        with t_view:
            st.markdown(st.session_state.tags_html_result, unsafe_allow_html=True)
            
        with t_code:
            st.text_area("Скопируйте HTML:", st.session_state.tags_html_result, height=300)
        
        if st.button("Сброс результата", key="reset_tags_txt"):
            st.session_state.tags_html_result = None
            st.rerun()

# ------------------------------------------
# Вкладка 4: ТАБЛИЦЫ
# ------------------------------------------
with tab_tables:
    st.title("🧩 Генератор таблиц")
    pplx_key_tbl = st.text_input("Perplexity API Key", type="password", key="pplx_key_tbl")
    prompt_tbl = st.text_area("Описание таблицы")

    # Авто-выбор топ-4 слов для добавления
    top_missing = []
    if st.session_state.analysis_results:
        df = st.session_state.analysis_results['depth']
        mask = df['Рекомендация'].astype(str).str.startswith('+')
        df_miss = df[mask].copy()
        df_miss['val'] = df_miss['Рекомендация'].apply(lambda x: int(str(x).replace('+','')))
        top_missing = [{'word': r['Слово'], 'count': r['val']} for _, r in df_miss.sort_values('val', ascending=False).head(4).iterrows()]

    if top_missing:
        st.info(f"Слова для внедрения: {', '.join([x['word'] for x in top_missing])}")

    if st.button("Сгенерировать", key="btn_gen_tbl"):
        if not openai: st.error("Нет openai"); st.stop()
        client = openai.OpenAI(api_key=pplx_key_tbl, base_url="https://api.perplexity.ai")
        with st.spinner("Генерация..."):
            res = generate_html_table(client, prompt_tbl, top_missing)
            st.session_state.table_html_result = res
            st.rerun()

    if st.session_state.table_html_result:
        t1, t2 = st.tabs(["👁️ View", "💻 Code"])
        with t1: st.markdown(st.session_state.table_html_result, unsafe_allow_html=True)
        with t2: st.code(st.session_state.table_html_result, language='html')








