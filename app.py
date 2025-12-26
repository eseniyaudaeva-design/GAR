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
import copy
import plotly.graph_objects as go

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
    import google.generativeai as genai
except ImportError:
    genai = None

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
    try:
        session = requests.Session()
        retry = Retry(connect=3, read=3, redirect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        headers = {'User-Agent': ua_settings}
        r = session.get(url, headers=headers, timeout=25)
        if r.status_code != 200: 
            return None
        
        soup = BeautifulSoup(r.text, 'html.parser')
        
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

def render_clean_block(title, icon, words_list):
    unique_words = sorted(list(set(words_list))) if words_list else []
    count = len(unique_words)
    
    if count > 0:
        content_html = ", ".join(unique_words)
        # Карточка раскрывается
        html_code = f"""
        <details class="details-card">
            <summary class="card-summary">
                <div>
                    <span class="arrow-icon">▶</span>
                    {icon} {title}
                </div>
                <span class="count-tag">{count}</span>
            </summary>
            <div class="card-content">
                {content_html}
            </div>
        </details>
        """
    else:
        # Если пусто - карточка неактивна (без контента)
        html_code = f"""
        <div class="details-card">
            <div class="card-summary" style="cursor: default; color: #9ca3af;">
                <div>{icon} {title}</div>
                <span class="count-tag">0</span>
            </div>
        </div>
        """
    
    st.markdown(html_code, unsafe_allow_html=True)

def render_relevance_chart(df_rel, unique_key="default"):
    if df_rel.empty: return
    df = df_rel[df_rel['Позиция'] > 0].copy()
    if df.empty: return

    df = df.sort_values(by='Позиция')
    x_indices = np.arange(len(df))
    tick_links = []
    
    for _, row in df.iterrows():
        raw_name = row['Домен'].replace(' (Вы)', '').strip()
        clean_domain = raw_name.replace('www.', '').split('/')[0]
        label_text = f"{row['Позиция']}. {clean_domain}"
        if len(label_text) > 20: label_text = label_text[:18] + ".."
        url_target = row.get('URL', f"https://{raw_name}")
        link_html = f"<a href='{url_target}' target='_blank' class='chart-link'>{label_text}</a>"
        tick_links.append(link_html)

    df['Total_Rel'] = (df['Ширина (балл)'] + df['Глубина (балл)']) / 2
    z = np.polyfit(x_indices, df['Total_Rel'], 1)
    p = np.poly1d(z)
    df['Trend'] = p(x_indices)

    fig = go.Figure()
    COLOR_MAIN = '#4F46E5'; COLOR_WIDTH = '#0EA5E9'; COLOR_DEPTH = '#E11D48'; COLOR_TREND = '#15803d'
    COMMON_CONFIG = dict(mode='lines+markers', line=dict(width=3, shape='spline'), marker=dict(size=8, line=dict(width=2, color='white'), symbol='circle'))

    fig.add_trace(go.Scatter(x=x_indices, y=df['Total_Rel'], name='Общая', line=dict(color=COLOR_MAIN, **COMMON_CONFIG['line']), marker=dict(color=COLOR_MAIN, **COMMON_CONFIG['marker']), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=x_indices, y=df['Ширина (балл)'], name='Ширина', line=dict(color=COLOR_WIDTH, **COMMON_CONFIG['line']), marker=dict(color=COLOR_WIDTH, **COMMON_CONFIG['marker']), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=x_indices, y=df['Глубина (балл)'], name='Глубина', line=dict(color=COLOR_DEPTH, **COMMON_CONFIG['line']), marker=dict(color=COLOR_DEPTH, **COMMON_CONFIG['marker']), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=x_indices, y=df['Trend'], name='Тренд', line=dict(color=COLOR_TREND, **COMMON_CONFIG['line']), marker=dict(color=COLOR_TREND, **COMMON_CONFIG['marker']), mode='lines+markers', opacity=0.8))

    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=12, color="#111827", family="Inter, sans-serif")),
        xaxis=dict(showgrid=False, linecolor='#E5E7EB', tickmode='array', tickvals=x_indices, ticktext=tick_links, tickfont=dict(size=12), fixedrange=True, range=[-0.5, len(df) - 0.5], automargin=True),
        yaxis=dict(range=[0, 115], showgrid=True, gridcolor='#F3F4F6', gridwidth=1, zeroline=False, fixedrange=True),
        margin=dict(l=10, r=10, t=50, b=40), hovermode="x unified", height=380
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"rel_chart_{unique_key}")

def analyze_serp_anomalies(df_rel):
    if df_rel.empty: return [], [], {"type": "none", "msg": ""}
    df = df_rel[~df_rel['Домен'].str.contains("\(Вы\)", na=False)].copy()
    if df.empty: return [], [], {"type": "none", "msg": ""}

    df['Ширина (балл)'] = pd.to_numeric(df['Ширина (балл)'], errors='coerce').fillna(0)
    df['Глубина (балл)'] = pd.to_numeric(df['Глубина (балл)'], errors='coerce').fillna(0)
    df['Total'] = (df['Ширина (балл)'] + df['Глубина (балл)']) / 2
    
    max_score = df['Total'].max()
    if max_score < 1: max_score = 1
    threshold = max(max_score * 0.75, 40) 
    
    anomalies = []
    normal_urls = []
    
    for _, row in df.iterrows():
        current_url = str(row.get('URL', '')).strip()
        if not current_url or current_url.lower() == 'nan': current_url = f"https://{row['Домен']}" 
        score = row['Total']
        if score < threshold:
            reason = f"Скор {int(score)} < {int(threshold)} (Лидер {int(max_score)})"
            anomalies.append({'url': current_url, 'reason': reason, 'score': score})
        else:
            normal_urls.append(current_url)

    if anomalies: st.toast(f"🗑️ Фильтр (Лидер {int(max_score)} / Порог {int(threshold)}). Исключено: {len(anomalies)}", icon="⚠️")
    else: st.toast(f"✅ Все конкуренты ок. (Лидер {int(max_score)} / Порог {int(threshold)}). Мин. балл: {int(df['Total'].min())}", icon="ℹ️")
    
    x = np.arange(len(df)); y = df['Total'].values
    slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
    trend_msg = "📉 Нормальный топ" if slope < -1 else ("📈 Перевернутый топ" if slope > 1 else "➡️ Ровный топ")

    return normal_urls, anomalies, {"type": "info", "msg": trend_msg}

@st.cache_data
def load_lemmatized_dictionaries():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "data")
    sets = {"products": set(), "commercial": set(), "specs": set(), "geo": set(), "services": set(), "sensitive": set()}
    files_map = {"metal_products.json": "products", "commercial_triggers.json": "commercial", "geo_locations.json": "geo", "services_triggers.json": "services", "tech_specs.json": "specs", "SENSITIVE_STOPLIST.json": "sensitive"}

    for filename, set_key in files_map.items():
        full_path = os.path.join(base_path, filename)
        if not os.path.exists(full_path): continue
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f) 
                words_bucket = []
                if isinstance(data, dict):
                    for cat_list in data.values(): words_bucket.extend(cat_list)
                elif isinstance(data, list): words_bucket = data
                for phrase in words_bucket:
                    w_clean = str(phrase).lower().strip().replace('ё', 'е')
                    if not w_clean: continue
                    sets[set_key].add(w_clean)
                    if morph:
                        normal_form = morph.parse(w_clean)[0].normal_form.replace('ё', 'е')
                        sets[set_key].add(normal_form)
                    if ' ' in w_clean:
                        parts = w_clean.split()
                        for p in parts:
                            sets[set_key].add(p)
                            if morph: sets[set_key].add(morph.parse(p)[0].normal_form.replace('ё', 'е'))
        except Exception: pass
    return sets["products"], sets["commercial"], sets["specs"], sets["geo"], sets["services"], sets["sensitive"]

def classify_semantics_with_api(words_list, yandex_key):
    PRODUCTS_SET, COMM_SET, SPECS_SET, GEO_SET, SERVICES_SET, SENS_SET = load_lemmatized_dictionaries()
    FULL_SENSITIVE = SENS_SET.union(SENSITIVE_STOPLIST)

    if 'debug_geo_count' not in st.session_state: st.session_state.debug_geo_count = len(GEO_SET)
    st.sidebar.info(f"Словари (из файлов):\n📦 Товары: {len(PRODUCTS_SET)}\n💰 Коммерция: {len(COMM_SET)}\n🛠️ Услуги: {len(SERVICES_SET)}\n🌍 Города: {len(GEO_SET)}")

    dim_pattern = re.compile(r'\d+(?:[\.\,]\d+)?\s?[хx\*×]\s?\d+', re.IGNORECASE)
    grade_pattern = re.compile(r'^([а-яa-z]{1,4}\-?\d+[а-яa-z0-9]*)$', re.IGNORECASE)
    categories = {'products': set(), 'services': set(), 'commercial': set(), 'dimensions': set(), 'geo': set(), 'general': set(), 'sensitive': set()}
    
    for word in words_list:
        word_lower = word.lower()
        is_sensitive = False
        if word_lower in FULL_SENSITIVE: is_sensitive = True
        else:
            for stop_w in FULL_SENSITIVE:
                if len(stop_w) > 3 and stop_w in word_lower: is_sensitive = True; break
        if is_sensitive: categories['sensitive'].add(word_lower); continue
        
        lemma = word_lower
        if morph:
            p = morph.parse(word_lower)[0]
            lemma = p.normal_form

        if word_lower in SPECS_SET or lemma in SPECS_SET: categories['dimensions'].add(word_lower); continue
        if dim_pattern.search(word_lower) or grade_pattern.match(word_lower) or word_lower.isdigit(): categories['dimensions'].add(word_lower); continue

        if word_lower in PRODUCTS_SET or lemma in PRODUCTS_SET: categories['products'].add(word_lower); continue
        is_product_root = False
        for prod in PRODUCTS_SET:
            check_root = prod[:-1] if len(prod) > 4 else prod
            if len(check_root) > 3 and check_root in word_lower:
                categories['products'].add(word_lower)
                is_product_root = True
                break
        if is_product_root: continue

        if lemma in GEO_SET or word_lower in GEO_SET: categories['geo'].add(word_lower); continue
        if lemma in SERVICES_SET or word_lower in SERVICES_SET: categories['services'].add(word_lower); continue
        if lemma.endswith('обработка') or lemma.endswith('изготовление') or lemma == "резка": categories['services'].add(word_lower); continue
        if lemma in COMM_SET or word_lower in COMM_SET: categories['commercial'].add(word_lower); continue
        categories['general'].add(word_lower)

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
if 'categorized_sensitive' not in st.session_state: st.session_state.categorized_sensitive = []

if 'orig_products' not in st.session_state: st.session_state.orig_products = []
if 'orig_services' not in st.session_state: st.session_state.orig_services = []
if 'orig_commercial' not in st.session_state: st.session_state.orig_commercial = []
if 'orig_dimensions' not in st.session_state: st.session_state.orig_dimensions = []
if 'orig_geo' not in st.session_state: st.session_state.orig_geo = []
if 'orig_general' not in st.session_state: st.session_state.orig_general = []

if 'auto_tags_words' not in st.session_state: st.session_state.auto_tags_words = []
if 'auto_promo_words' not in st.session_state: st.session_state.auto_promo_words = []
if 'persistent_urls' not in st.session_state: st.session_state['persistent_urls'] = ""

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

SENSITIVE_STOPLIST_RAW = {
    "украина", "ukraine", "ua", "всу", "зсу", "ато",
    "киев", "львов", "харьков", "одесса", "днепр", "мариуполь",
    "донецк", "луганск", "днр", "лнр", "донбасс", 
    "мелитополь", "бердянск", "бахмут", "запорожье", "херсон",
    "крым", "севастополь", "симферополь"
}
SENSITIVE_STOPLIST = {w.lower() for w in SENSITIVE_STOPLIST_RAW}

def check_password():
    if st.session_state.get("authenticated"): return True
    st.markdown("""<style>.main { display: flex; flex-direction: column; justify-content: center; align-items: center; } .auth-logo-box { text-align: center; margin-bottom: 1rem; padding-top: 0; }</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-logo-box"><h3>Вход в систему</h3></div>', unsafe_allow_html=True)
        password = st.text_input("Пароль", type="password", key="password_input", label_visibility="collapsed")
        if st.button("ВОЙТИ", type="primary", use_container_width=True):
            if password == "ZVC01w4_pIquj0bMiaAu":
                st.session_state.authenticated = True
                st.rerun()
            else: st.error("❌ Неверный пароль")
    return False

if not check_password(): st.stop()

if "arsenkin_token" in st.session_state: ARSENKIN_TOKEN = st.session_state.arsenkin_token
else:
    try: ARSENKIN_TOKEN = st.secrets["api"]["arsenkin_token"]
    except (FileNotFoundError, KeyError): ARSENKIN_TOKEN = None

if "yandex_dict_key" in st.session_state: YANDEX_DICT_KEY = st.session_state.yandex_dict_key
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

DEFAULT_EXCLUDE_DOMAINS = {
    "yandex.ru", "avito.ru", "beru.ru", "tiu.ru", "aliexpress.com", "aliexpress.ru", 
    "ebay.com", "auto.ru", "2gis.ru", "sravni.ru", "toshop.ru", "price.ru", 
    "pandao.ru", "instagram.com", "wikipedia.org", "rambler.ru", "hh.ru", 
    "banki.ru", "regmarkets.ru", "zoon.ru", "pulscen.ru", "prodoctorov.ru", 
    "blizko.ru", "domclick.ru", "satom.ru", "quto.ru", "edadeal.ru", 
    "cataloxy.ru", "irr.ru", "onliner.by", "shop.by", "deal.by", "yell.ru", 
    "profi.ru", "irecommend.ru", "otzovik.com", "ozon.ru", "ozon.by", 
    "market.yandex.ru", "youtube.com", "www.youtube.com", "gosuslugi.ru", 
    "www.gosuslugi.ru", "dzen.ru", "2gis.by", "wildberries.ru", "rutube.ru", 
    "vk.com", "facebook.com", "chipdip.ru"
    }

DEFAULT_EXCLUDE = "\n".join(DEFAULT_EXCLUDE_DOMAINS)
DEFAULT_STOPS = "рублей\nруб\nстр\nул\nшт\nсм\nмм\nмл\nкг\nкв\nм²\nсм²\nм2\nсм2"

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
        /* Стили для ссылок внутри графика Plotly */
        .chart-link {{
            color: #277EFF !important;
            font-weight: 600 !important;
            text-decoration: none !important;
            border-bottom: 4px solid #CBD5E1 !important; 
            display: inline-block !important;
            transition: border-color 0.2s ease !important;
        }}
        .chart-link:hover {{
            border-bottom-color: #277EFF !important;
            cursor: pointer !important;
        }}
    </style>
""", unsafe_allow_html=True)

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
    while status == "process" and attempts < 120:
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

def parse_page(url, settings, query_context=""):
    try:
        from curl_cffi import requests as cffi_requests
        headers = {
            'User-Agent': settings['ua'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        }
        r = cffi_requests.get(url, headers=headers, timeout=20, impersonate="chrome110")
        if r.status_code == 403: raise Exception("CURL_CFFI получил 403 Forbidden")
        if r.status_code != 200: return None
        content = r.content
        encoding = r.encoding if r.encoding else 'utf-8'
    except Exception:
        try:
            import requests
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            session = requests.Session()
            headers = {'User-Agent': settings['ua']}
            r = session.get(url, headers=headers, timeout=20, verify=False)
            if r.status_code != 200: return None
            content = r.content
            encoding = r.apparent_encoding
        except Exception: return None

    try:
        soup = BeautifulSoup(content, 'html.parser', from_encoding=encoding)
        product_titles = []
        search_roots = set()
        if query_context:
            clean_q = query_context.lower().replace('купить', '').replace('цена', '').replace(' в ', ' ')
            words = re.findall(r'[а-яa-z]+', clean_q)
            for w in words:
                if len(w) > 3: search_roots.add(w[:-1])
                else: search_roots.add(w)
        
        parsed_current = urlparse(url)
        current_path_clean = parsed_current.path.rstrip('/')
        seen_titles = set()
        
        for a in soup.find_all('a', href=True):
            txt = a.get_text(strip=True)
            raw_href = a['href']
            if len(txt) < 5 or len(txt) > 300: continue
            if raw_href.startswith('#') or raw_href.startswith('javascript'): continue
            
            abs_href = urljoin(url, raw_href)
            parsed_href = urlparse(abs_href)
            href_path_clean = parsed_href.path.rstrip('/')
            
            is_child_path = href_path_clean.startswith(current_path_clean)
            is_deeper = len(href_path_clean) > len(current_path_clean)
            is_not_query_param_only = (href_path_clean != current_path_clean)

            if is_child_path and is_deeper and is_not_query_param_only:
                txt_lower = txt.lower()
                href_lower = abs_href.lower()
                has_keywords = False
                if search_roots:
                    for root in search_roots:
                        if root in txt_lower or root in href_lower:
                            has_keywords = True; break
                else:
                    if re.search(r'\d', txt): has_keywords = True

                is_buy_button = txt_lower in {'купить', 'подробнее', 'в корзину', 'заказать', 'цена'}
                if has_keywords and not is_buy_button:
                    if txt not in seen_titles:
                        product_titles.append(txt)
                        seen_titles.add(txt)
        
        h1_tag = soup.find('h1')
        h1_text = h1_tag.get_text(strip=True) if h1_tag else ""

        soup_no_grid = BeautifulSoup(content, 'html.parser', from_encoding=encoding)
        grid_div = soup_no_grid.find('div', class_='an-container-fluid an-container-xl')
        if grid_div: grid_div.decompose()
        
        tags_to_remove = []
        if settings['noindex']: tags_to_remove.append('noindex')
        
        for s in [soup, soup_no_grid]:
            for c in s.find_all(string=lambda text: isinstance(text, Comment)): c.extract()
            if tags_to_remove:
                for t in s.find_all(tags_to_remove): t.decompose()
            for script in s(["script", "style", "svg", "path", "noscript"]): script.decompose()

        anchors_list = [a.get_text(strip=True) for a in soup.find_all('a') if a.get_text(strip=True)]
        anchor_text = " ".join(anchors_list)
        
        extra_text = []
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'): extra_text.append(meta_desc['content'])

        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): extra_text.append(img['alt'])
            for t in soup.find_all(title=True): extra_text.append(t['title'])

        body_text_raw = soup.get_text(separator=' ') + " " + " ".join(extra_text)
        body_text = re.sub(r'\s+', ' ', body_text_raw).strip()
        body_text_no_grid_raw = soup_no_grid.get_text(separator=' ') + " " + " ".join(extra_text)
        body_text_no_grid = re.sub(r'\s+', ' ', body_text_no_grid_raw).strip()

        if not body_text: return None
        return {
            'url': url, 'domain': urlparse(url).netloc, 
            'body_text': body_text, 'body_text_no_grid': body_text_no_grid,
            'anchor_text': anchor_text, 'h1': h1_text, 'product_titles': product_titles
        }
    except Exception: return None
        
def calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, original_results):
    def math_round(number): return int(number + 0.5)

    all_forms_map = defaultdict(set)
    global_forms_counter = defaultdict(Counter) 
    
    if not my_data or not my_data.get('body_text'): 
        my_lemmas, my_forms, my_anchors, my_len = [], {}, [], 0
        my_clean_domain = "local"
    else:
        my_lemmas, my_forms = process_text_detailed(my_data['body_text'], settings)
        my_anchors, _ = process_text_detailed(my_data['anchor_text'], settings)
        my_len = len(my_lemmas)
        for k, v in my_forms.items(): all_forms_map[k].update(v)
        my_clean_domain = my_data['domain'].lower().replace('www.', '').split(':')[0]

    comp_docs = []
    for p in comp_data_full:
        if not p.get('body_text'): continue
        p_domain = p['domain'].lower().replace('www.', '').split(':')[0]
        if my_clean_domain != "local" and p_domain == my_clean_domain: continue
        body, c_forms = process_text_detailed(p['body_text'], settings)
        raw_words_for_stats = re.findall(r'[а-яА-ЯёЁ0-9a-zA-Z]+', p['body_text'].lower())
        for rw in raw_words_for_stats:
            if len(rw) < 2: continue
            if morph:
                parsed = morph.parse(rw)[0]
                if 'PREP' not in parsed.tag and 'CONJ' not in parsed.tag:
                    rw_lemma = parsed.normal_form.replace('ё', 'е')
                    global_forms_counter[rw_lemma][rw] += 1
        anchor, _ = process_text_detailed(p['anchor_text'], settings)
        comp_docs.append({'body': body, 'anchor': anchor, 'url': p['url'], 'domain': p['domain']})
        for k, v in c_forms.items(): all_forms_map[k].update(v)

    if not comp_docs:
        return { "depth": pd.DataFrame(), "hybrid": pd.DataFrame(), "relevance_top": pd.DataFrame(), "my_score": {"width": 0, "depth": 0}, "missing_semantics_high": [], "missing_semantics_low": [] }

    c_lens = [len(d['body']) for d in comp_docs]
    avg_dl = np.mean(c_lens) if c_lens else 1
    N = len(comp_docs)
    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    vocab = sorted(list(vocab))
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

    table_depth = []
    table_hybrid = []
    missing_semantics_high = []
    missing_semantics_low = []
    words_with_median_gt_0 = set() 
    my_found_words_from_median = set() 
    
    for lemma in vocab:
        if lemma in GARBAGE_LATIN_STOPLIST: continue
        raw_counts = [word_counts_per_doc[i][lemma] for i in range(N)]
        sorted_raw = sorted(raw_counts)
        if sorted_raw:
            rec_median_absolute = math_round(np.median(sorted_raw))
            obs_min = math_round(sorted_raw[0])
            obs_max = math_round(sorted_raw[-1])
        else:
            rec_median_absolute = 0; obs_min = 0; obs_max = 0

        if settings['norm'] and my_len > 0:
            norm_counts = []
            for i in range(N):
                raw_cnt = raw_counts[i]
                comp_len = c_lens[i]
                if comp_len > 0:
                    normalized_val = raw_cnt * (my_len / comp_len)
                    norm_counts.append(normalized_val)
                else: norm_counts.append(0)
            if norm_counts: rec_median_target = math_round(np.median(sorted(norm_counts)))
            else: rec_median_target = 0
        else: rec_median_target = rec_median_absolute

        my_tf_count = my_lemmas.count(lemma)
        if obs_max == 0 and my_tf_count == 0: continue

        if rec_median_target >= 1:
            words_with_median_gt_0.add(lemma)
            if my_tf_count > 0: my_found_words_from_median.add(lemma)

        display_word = lemma
        if global_forms_counter[lemma]: display_word = global_forms_counter[lemma].most_common(1)[0][0]

        if my_tf_count == 0:
            weight = word_idf_map.get(lemma, 0) * (rec_median_target if rec_median_target > 0 else 1)
            item = {'word': display_word, 'weight': weight}
            if rec_median_target >= 1: missing_semantics_high.append(item)
            else: missing_semantics_low.append(item)

        diff = rec_median_target - my_tf_count
        if diff == 0: status = "Норма"; action_text = "✅"; sort_val = 0
        elif diff > 0: status = "Недоспам"; action_text = f"+{diff}"; sort_val = diff
        else: status = "Переспам"; action_text = f"{diff}"; sort_val = abs(diff)

        forms_str = ", ".join(sorted(list(all_forms_map.get(lemma, set())))) if all_forms_map.get(lemma) else lemma
        table_depth.append({
            "Слово": display_word, "Словоформы": forms_str, "Вхождений у вас": my_tf_count,
            "Медиана": rec_median_absolute, "Минимум (конкур.)": obs_min, "Максимум (конкур.)": obs_max,
            "Статус": status, "Рекомендация": action_text, "is_missing": (my_tf_count == 0), "sort_val": sort_val
        })
        table_hybrid.append({
            "Слово": display_word,
            "TF-IDF ТОП": round(word_idf_map.get(lemma, 0) * (rec_median_absolute / avg_dl if avg_dl > 0 else 0), 4),
            "TF-IDF у вас": round(word_idf_map.get(lemma, 0) * (my_tf_count / my_len if my_len > 0 else 0), 4),
            "Сайтов": doc_freqs[lemma], "Переспам": obs_max
        })
    
    total_needed = len(words_with_median_gt_0)
    total_found = len(my_found_words_from_median)
    if total_needed > 0:
        ratio = total_found / total_needed
        my_width_score_final = int(min(100, ratio * 120))
    else: my_width_score_final = 0

    S_WIDTH_CORE = words_with_median_gt_0 
    
    def calculate_raw_power(doc_tokens, doc_len):
        if avg_dl == 0 or doc_len == 0: return 0
        score = 0
        counts = Counter(doc_tokens)
        k1 = 1.2; b = 0.75
        for word in S_WIDTH_CORE:
            if word not in counts: continue
            tf = counts[word]
            idf = word_idf_map.get(word, 0)
            term_weight = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_dl)))
            score += term_weight
        return score

    comp_raw_scores = []
    competitor_scores_map = {}
    for i, doc in enumerate(comp_docs):
        c_found = len(set(doc['body']).intersection(S_WIDTH_CORE))
        if total_needed > 0: c_width_val = int(min(100, (c_found / total_needed) * 120))
        else: c_width_val = 0
        raw_val = calculate_raw_power(doc['body'], c_lens[i])
        comp_raw_scores.append(raw_val)
        competitor_scores_map[doc['url']] = {'width_final': c_width_val, 'raw_depth': raw_val}

    if comp_raw_scores:
        median_raw = np.median(comp_raw_scores)
        ref_val = median_raw if median_raw > 0.1 else 1.0
    else: ref_val = 1.0
    
    k_norm = 80.0 / ref_val
    my_raw_bm25 = calculate_raw_power(my_lemmas, my_len)
    my_depth_score_final = int(round(min(100, my_raw_bm25 * k_norm)))

    for url, data in competitor_scores_map.items(): data['depth_final'] = int(round(min(100, data['raw_depth'] * k_norm)))

    missing_semantics_high.sort(key=lambda x: x['weight'], reverse=True)
    missing_semantics_low.sort(key=lambda x: x['weight'], reverse=True)
    missing_semantics_low = missing_semantics_low[:500]

    table_rel = []
    my_site_found_in_selection = False
    for item in original_results:
        url = item['url']
        if url not in competitor_scores_map: continue
        row_domain = urlparse(url).netloc.lower().replace('www.', '')
        is_my_site = False
        if my_clean_domain and my_clean_domain != "local" and my_clean_domain in row_domain:
            is_my_site = True
            my_site_found_in_selection = True
            display_name = f"{urlparse(url).netloc} (Вы)"
        else: display_name = urlparse(url).netloc
        scores = competitor_scores_map[url]
        table_rel.append({ "Домен": display_name, "URL": url, "Позиция": item['pos'], "Ширина (балл)": scores['width_final'], "Глубина (балл)": scores['depth_final'] })
        
    if not my_site_found_in_selection:
        pos_to_show = my_serp_pos if my_serp_pos > 0 else 0
        my_label = f"{my_data['domain']} (Вы)" if (my_data and my_data.get('domain')) else "Ваш сайт"
        my_full_url = my_data['url'] if (my_data and 'url' in my_data) else "#"
        table_rel.append({ "Домен": my_label, "URL": my_full_url, "Позиция": pos_to_show, "Ширина (балл)": my_width_score_final, "Глубина (балл)": my_depth_score_final })

    df_rel_for_analysis = pd.DataFrame(table_rel)
    good_urls, bad_urls_dicts, trend_info = analyze_serp_anomalies(df_rel_for_analysis)
    
    st.session_state['detected_anomalies'] = bad_urls_dicts
    st.session_state['serp_trend_info'] = trend_info
    
    if bad_urls_dicts:
        st.session_state['persistent_urls'] = "\n".join(good_urls)
        st.session_state['excluded_urls_auto'] = "\n".join([item['url'] for item in bad_urls_dicts])
    else:
        st.session_state['persistent_urls'] = "\n".join([r.get('URL', r['Домен']) for r in table_rel if "(Вы)" not in r['Домен']])
        st.session_state['excluded_urls_auto'] = ""

    return { 
        "depth": pd.DataFrame(table_depth), "hybrid": pd.DataFrame(table_hybrid), 
        "relevance_top": pd.DataFrame(table_rel).sort_values(by='Позиция', ascending=True).reset_index(drop=True), 
        "my_score": {"width": my_width_score_final, "depth": my_depth_score_final}, 
        "missing_semantics_high": missing_semantics_high, "missing_semantics_low": missing_semantics_low 
    }
    
def get_hybrid_word_type(word, main_marker_root, specs_dict=None):
    w = word.lower()
    specs_dict = specs_dict or set()
    if w == main_marker_root: return "1. 💎 Маркер (Товар)"
    if morph:
        norm = morph.parse(w)[0].normal_form
        if norm == main_marker_root: return "1. 💎 Маркер (Товар)"
    if re.search(r'(gost|din|iso|en|tu|astm|aisi|гост|ост|ту|дин)', w): return "6. 📜 Стандарт"
    if re.fullmatch(r'\d+([.,]\d+)?', w): return "5. 🔢 Размеры/Прочее"
    if re.search(r'^\d+[xх*\-/]\d+', w): return "5. 🔢 Размеры/Прочее"
    if re.search(r'\d+(мм|mm|м|m|kg|кг|bar|бар|атм)$', w): return "5. 🔢 Размеры/Прочее"
    if re.match(r'^(d|dn|pn|sn|sdr|ду|ру|ø)\d+', w): return "5. 🔢 Размеры/Прочее"
    if w in specs_dict: return "3. 🏗️ Марка/Сплав"
    if re.search(r'\d', w): return "3. 🏗️ Марка/Сплав"
    if re.search(r'^[a-z\-]+$', w): return "7. 🔠 Латиница/Бренд"
    if morph:
        p = morph.parse(w)[0]
        tag = p.tag
        if {'PREP'} in tag or {'CONJ'} in tag: return "SKIP"
        if {'ADJF'} in tag or {'PRTF'} in tag or {'ADJS'} in tag: return "2. 🎨 Свойства"
        if {'NOUN'} in tag: return "4. 🔗 Дополнения"
    if w.endswith(('ий', 'ый', 'ая', 'ое', 'ые', 'ая')): return "2. 🎨 Свойства"
    return "4. 🔗 Дополнения"
    
def calculate_naming_metrics(comp_data_full, my_data, settings):
    SPECS_SET = st.session_state.get('categorized_dimensions', set())
    if not SPECS_SET: _, _, SPECS_SET, _, _, _ = load_lemmatized_dictionaries()
    my_tokens = []
    if my_data and my_data.get('body_text_no_grid'):
        raw_w = re.findall(r'[а-яА-Яa-zA-Z0-9\-]+', my_data['body_text_no_grid'].lower())
        for w in raw_w:
            if not re.search(r'\d', w) and morph: my_tokens.append(morph.parse(w)[0].normal_form)
            else: my_tokens.append(w)
    all_words_flat = []
    site_vocab_map = []
    for p in comp_data_full:
        titles = p.get('product_titles', [])
        valid_titles = [t for t in titles if 5 < len(t) < 150]
        if not valid_titles: site_vocab_map.append(set()); continue
        curr_site_tokens = set()
        for t in valid_titles:
            words = re.findall(r'[а-яА-Яa-zA-Z0-9\-]+', t.lower())
            for w in words:
                if len(w) < 2: continue
                if re.search(r'\d', w): token = w
                elif re.search(r'^[a-z]+$', w): token = w
                elif morph: token = morph.parse(w)[0].normal_form
                else: token = w
                all_words_flat.append(token)
                curr_site_tokens.add(token)
        site_vocab_map.append(curr_site_tokens)
    if not all_words_flat: return pd.DataFrame()
    N_sites = len(site_vocab_map)
    counts = Counter([w for w in all_words_flat if not re.search(r'\d', w)])
    main_marker_root = ""
    for w, c in counts.most_common(10):
        if morph and 'NOUN' in morph.parse(w)[0].tag: main_marker_root = w; break
    if not main_marker_root and counts: main_marker_root = counts.most_common(1)[0][0]
    vocab = sorted(list(set(all_words_flat)))
    table_rows = []
    for token in vocab:
        if token in GARBAGE_LATIN_STOPLIST: continue
        sites_with_word = sum(1 for s_set in site_vocab_map if token in s_set)
        freq_percent = int((sites_with_word / N_sites) * 100)
        cat = get_hybrid_word_type(token, main_marker_root, SPECS_SET)
        if cat == "SKIP": continue
        is_spec = "Марка" in cat or "Стандарт" in cat
        if is_spec and freq_percent < 5: continue
        if not is_spec and "Размеры" not in cat and freq_percent < 15: continue
        if "Размеры" in cat and freq_percent < 15: continue
        rec_median = 1 if freq_percent > 30 else 0
        my_tf = my_tokens.count(token)
        diff = rec_median - my_tf
        action_text = f"+{diff}" if diff > 0 else ("✅" if diff == 0 else f"{diff}")
        table_rows.append({
            "Тип хар-ки": cat[3:], "Слово": token, "Частотность (%)": f"{freq_percent}%",
            "У Вас": my_tf, "Медиана": rec_median, "Добавить": action_text, "raw_freq": freq_percent, "cat_sort": int(cat[0])
        })
    df = pd.DataFrame(table_rows)
    if not df.empty: df = df.sort_values(by=["cat_sort", "raw_freq"], ascending=[True, False])
    return df

def analyze_ideal_name(comp_data_full):
    SPECS_SET = st.session_state.get('categorized_dimensions', set())
    if not SPECS_SET: _, _, SPECS_SET, _, _, _ = load_lemmatized_dictionaries()
    titles = []
    for d in comp_data_full:
        ts = d.get('product_titles', [])
        titles.extend([t for t in ts if 5 < len(t) < 150])
    if not titles: return "Нет данных", []
    all_w = []
    for t in titles: all_w.extend(re.findall(r'[а-яА-Яa-zA-Z0-9\-]+', t.lower()))
    c = Counter(all_w)
    main_marker_root = ""
    for w, _ in c.most_common(5):
        if not re.search(r'\d', w):
             if morph and 'NOUN' in morph.parse(w)[0].tag: main_marker_root = w; break
             elif not morph: main_marker_root = w; break
    if not main_marker_root and c: main_marker_root = c.most_common(1)[0][0]
    structure_counter = Counter()
    vocab_by_type = defaultdict(Counter)
    sample = titles[:500]
    for t in sample:
        words = re.findall(r'[а-яА-Яa-zA-Z0-9\-]+', t.lower())
        pattern = []
        for w in words:
            if len(w) < 2: continue
            cat_full = get_hybrid_word_type(w, main_marker_root, SPECS_SET)
            if cat_full == "SKIP": continue
            try: cat_short = cat_full.split('.', 1)[1].strip().split(' ', 1)[1]
            except: cat_short = cat_full 
            vocab_by_type[cat_short][w] += 1
            if not pattern or pattern[-1] != cat_short: pattern.append(cat_short)
        if pattern:
            structure_str = " + ".join(pattern)
            structure_counter[structure_str] += 1
    if not structure_counter: return "Структура не найдена", []
    best_struct_str, _ = structure_counter.most_common(1)[0]
    best_struct_list = best_struct_str.split(" + ")
    final_parts = []
    used_words = set()
    for block in best_struct_list:
        if "Размеры" in block or "Стандарт" in block or "Марка" in block:
            top_cand = vocab_by_type[block].most_common(1)
            if top_cand and top_cand[0][1] > (len(sample) * 0.3): final_parts.append(top_cand[0][0])
            else: final_parts.append(f"[{block.upper()}]")
            continue
        candidates = vocab_by_type[block].most_common(3)
        for w, cnt in candidates:
            if w not in used_words:
                if "Маркер" in block: w = w.capitalize()
                final_parts.append(w)
                used_words.add(w)
                break
    ideal_name = " ".join(final_parts)
    report = []
    report.append(f"**Схема:** {best_struct_str}")
    report.append("")
    report.append("**Популярные значения:**")
    for block in best_struct_list:
        if "Размеры" in block: continue
        top = [f"{w}" for w, c in vocab_by_type[block].most_common(3)]
        report.append(f"- **{block}**: {', '.join(top)}")
    return ideal_name, report

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
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.encoding = 'utf-8'
    except Exception as e: return None, None, None, f"Ошибка соединения: {e}"
    if response.status_code != 200: return None, None, None, f"Ошибка статуса: {response.status_code}"
    soup = BeautifulSoup(response.text, 'html.parser')
    description_div = soup.find('div', class_='description-container')
    target_h2 = None
    if description_div: target_h2 = description_div.find('h2')
    if not target_h2: target_h2 = soup.find('h2')
    page_header = target_h2.get_text(strip=True) if target_h2 else "Описание товара"
    base_text = description_div.get_text(separator="\n", strip=True) if description_div else soup.body.get_text(separator="\n", strip=True)[:5000]
    tags_container = soup.find(class_='popular-tags-inner')
    tags_data = []
    if tags_container:
        links = tags_container.find_all('a')
        for link in links:
            tag_url = urljoin(url, link.get('href')) if link.get('href') else None
            if tag_url: tags_data.append({'name': link.get_text(strip=True), 'url': tag_url})
    return base_text, tags_data, page_header, None

def call_gemini_safe(model, prompt, retries=3):
    base_wait = 5 # Пауза между запросами
    for attempt in range(retries):
        try:
            time.sleep(base_wait) 
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                wait_time = 60
                st.warning(f"⏳ Превышен лимит Google API (429). Ждем {wait_time} сек...")
                time.sleep(wait_time)
            else: return f"Error: {err_str}"
    return "Error: Timeout (Too many 429s)"

def generate_ai_content_blocks(api_key, base_text, tag_name, forced_header, num_blocks=5, seo_words=None):
    if not base_text: return ["Error: No base text"] * 5
    if not genai: return ["Error: google-generativeai lib not installed"] * 5
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel('gemini-2.0-flash-exp') 
    except: model = genai.GenerativeModel('gemini-1.5-flash')
    seo_words = seo_words or []
    seo_instruction_block = ""
    if seo_words:
        seo_list_str = ", ".join(seo_words)
        seo_instruction_block = f"""
        Внедрить слова: {seo_list_str}. 
        Обязательно выдели их тегом <b>.
        Раскидай по всем блокам.
        """
    system_instruction = "Ты технический копирайтер. Выдаешь ТОЛЬКО HTML. Без Markdown. Без ссылок [1]."
    user_prompt = f"""
    Товар: "{tag_name}"
    Фактура: \"\"\"{base_text[:4000]}\"\"\"
    {seo_instruction_block}
    Напиши {num_blocks} блоков HTML, разделитель: |||BLOCK_SEP|||
    Темы:
    1. <h2>{forced_header}</h2> (Введение).
    2-{num_blocks}. <h3> (Характеристики, Применение).
    Структура блока: Заголовок -> Текст -> Список -> Текст.
    """
    full_prompt = f"{system_instruction}\n\n{user_prompt}"
    content = call_gemini_safe(model, full_prompt)
    if "Error:" in content and "BLOCK_SEP" not in content: return [content] * num_blocks
    content = re.sub(r'\[\d+\]', '', content)
    content = content.replace("```html", "").replace("```", "").strip()
    blocks = [b.strip() for b in content.split("|||BLOCK_SEP|||") if b.strip()]
    while len(blocks) < num_blocks: blocks.append("")
    return blocks[:num_blocks]

# ==========================================
# 7. UI TABS RESTRUCTURED
# ==========================================
tab_seo_main, tab_wholesale_main = st.tabs(["📊 SEO Анализ", "🏭 Оптовый генератор"])

with tab_seo_main:
    col_main, col_sidebar = st.columns([65, 35])
    with col_main:
        st.title("SEO Анализатор")
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
        if st.session_state.get('force_radio_switch'):
            st.session_state["competitor_source_radio"] = "Список url-адресов ваших конкурентов"
            st.session_state['force_radio_switch'] = False
        source_type_new = st.radio("Источник", ["Поиск через API Arsenkin (TOP-30)", "Список url-адресов ваших конкурентов"], horizontal=True, label_visibility="collapsed", key="competitor_source_radio")
        source_type = "API" if "API" in source_type_new else "Ручной список"
        if source_type == "Ручной список":
            if st.session_state.get('analysis_done'):
                col_reset, _ = st.columns([1, 4])
                with col_reset:
                    if st.button("🔄 Новый поиск (Сброс)", type="secondary", help="Сбросить все результаты и ввести новый список"):
                        keys_to_clear = ['analysis_done', 'analysis_results', 'excluded_urls_auto', 'detected_anomalies', 'serp_trend_info', 'persistent_urls', 'naming_table_df', 'ideal_h1_result']
                        for k in keys_to_clear:
                            if k in st.session_state: del st.session_state[k]
                        st.rerun()
            has_exclusions = st.session_state.get('excluded_urls_auto') and len(st.session_state.get('excluded_urls_auto')) > 5
            if has_exclusions:
                c_url_1, c_url_2 = st.columns(2)
                with c_url_1:
                    manual_val = st.text_area("✅ Активные конкуренты (Для анализа)", height=200, key="manual_urls_widget", value=st.session_state.get('persistent_urls', ""))
                    st.session_state['persistent_urls'] = manual_val
                with c_url_2:
                    st.text_area("🚫 Авто-исключенные (Вы можете вернуть их влево)", height=200, key="excluded_urls_widget_display", value=st.session_state.get('excluded_urls_auto', ""), help="Сюда попали слабые сайты. Если считаете, что сайт нормальный - скопируйте его и вставьте обратно в левое окно.")
            else:
                manual_val = st.text_area("Список ссылок (каждая с новой строки)", height=200, key="manual_urls_widget", value=st.session_state.get('persistent_urls', ""))
                st.session_state['persistent_urls'] = manual_val
        if st.session_state.get('analysis_done') and st.session_state.get('analysis_results'):
            results = st.session_state.analysis_results
            if 'relevance_top' in results and not results['relevance_top'].empty:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 График релевантности (Нажмите, чтобы раскрыть)", expanded=False):
                  graph_data = st.session_state.get('full_graph_data', results['relevance_top'])
                  render_relevance_chart(graph_data, unique_key="main")
                st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Списки (Stop / Exclude)")
        st.text_area("Не учитывать домены", DEFAULT_EXCLUDE, height=100, key="settings_excludes")
        st.text_area("Стоп-слова", DEFAULT_STOPS, height=100, key="settings_stops")
        if st.button("ЗАПУСТИТЬ АНАЛИЗ", type="primary", use_container_width=True, key="start_analysis_btn"):
            st.session_state.analysis_results = None
            st.session_state.analysis_done = False
            st.session_state.naming_table_df = None
            st.session_state.ideal_h1_result = None
            st.session_state.gen_result_df = None
            st.session_state.unified_excel_data = None
            if 'excluded_urls_auto' in st.session_state: del st.session_state['excluded_urls_auto']
            if 'detected_anomalies' in st.session_state: del st.session_state['detected_anomalies']
            if 'serp_trend_info' in st.session_state: del st.session_state['serp_trend_info']
            for key in list(st.session_state.keys()):
                if key.endswith('_page'): st.session_state[key] = 1
            st.session_state.start_analysis_flag = True
            st.rerun()

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
        st.selectbox("Кол-во конкурентов для анализа", [10, 20], index=0, key="settings_top_n")
        st.checkbox("Исключать <noindex>", True, key="settings_noindex")
        st.checkbox("Учитывать Alt/Title", False, key="settings_alt")
        st.checkbox("Учитывать числа", False, key="settings_numbers")
        st.checkbox("Нормировать по длине", True, key="settings_norm")

    if st.session_state.analysis_done and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        d_score = results['my_score']['depth']
        w_score = results['my_score']['width']
        w_color = "#2E7D32" if w_score >= 80 else ("#E65100" if w_score >= 50 else "#D32F2F")
        if 75 <= d_score <= 88: d_color = "#2E7D32"; d_status = "ИДЕАЛ (Топ)"
        elif 88 < d_score <= 100: d_color = "#D32F2F"; d_status = "ПЕРЕСПАМ (Риск)"
        elif 55 <= d_score < 75: d_color = "#F9A825"; d_status = "Средняя"
        else: d_color = "#D32F2F"; d_status = "Низкая"
        st.success("Анализ готов!")
        st.markdown(f"""
        <div style='display: flex; gap: 20px; flex-wrap: wrap;'>
            <div style='flex: 1; background:{LIGHT_BG_MAIN}; padding:15px; border-radius:8px; border-left: 5px solid {w_color};'>
                <div style='font-size: 12px; color: #666;'>ШИРИНА (Охват тем)</div>
                <div style='font-size: 24px; font-weight: bold; color: {w_color};'>{w_score}/100</div>
            </div>
            <div style='flex: 1; background:{LIGHT_BG_MAIN}; padding:15px; border-radius:8px; border-left: 5px solid {d_color};'>
                <div style='font-size: 12px; color: #666;'>ГЛУБИНА (Цель: ~80)</div>
                <div style='font-size: 24px; font-weight: bold; color: {d_color};'>{d_score}/100 <span style='font-size:14px; font-weight:normal;'>({d_status})</span></div>
            </div>
        </div><br>""", unsafe_allow_html=True)
        with st.expander("🛒 Семантическое ядро и Фильтрация", expanded=True):
            if not st.session_state.get('orig_products'): st.info("⚠️ Данные отсутствуют. Запустите анализ.")
            else:
                c1, c2, c3 = st.columns(3)
                with c1: render_clean_block("Товары", "🧱", st.session_state.categorized_products)
                with c2: render_clean_block("Гео", "🌍", st.session_state.categorized_geo)
                with c3: render_clean_block("Коммерция", "💰", st.session_state.categorized_commercial)
                c4, c5, c6 = st.columns(3)
                with c4: render_clean_block("Услуги", "🛠️", st.session_state.categorized_services)
                with c5: render_clean_block("Размеры/ГОСТ", "📏", st.session_state.categorized_dimensions)
                with c6: render_clean_block("Общие", "📂", st.session_state.categorized_general)
                st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
                cs1, cs2 = st.columns([1, 3])
                if 'sensitive_words_input_final' not in st.session_state:
                    current_list = st.session_state.get('categorized_sensitive', [])
                    st.session_state['sensitive_words_input_final'] = "\n".join(current_list)
                current_text_value = st.session_state['sensitive_words_input_final']
                with cs1:
                    count_excluded = len([x for x in current_text_value.split('\n') if x.strip()])
                    st.markdown(f"**⛔ Стоп-слова**"); st.markdown(f"Исключено: **{count_excluded}**")
                with cs2:
                    new_sens_str = st.text_area("hidden_label", height=100, key="sensitive_words_input_final", label_visibility="collapsed", placeholder="Слова для исключения...")
                    if st.button("🔄 Обновить фильтр", type="primary", use_container_width=True):
                        raw_input = st.session_state.get("sensitive_words_input_final", "")
                        new_stop_set = set([w.strip().lower() for w in raw_input.split('\n') if w.strip()])
                        st.session_state.categorized_sensitive = sorted(list(new_stop_set))
                        def apply_filter(orig_list_key, stop_set):
                            original = st.session_state.get(orig_list_key, [])
                            return [w for w in original if w.lower() not in stop_set]
                        st.session_state.categorized_products = apply_filter('orig_products', new_stop_set)
                        st.session_state.categorized_services = apply_filter('orig_services', new_stop_set)
                        st.session_state.categorized_commercial = apply_filter('orig_commercial', new_stop_set)
                        st.session_state.categorized_geo = apply_filter('orig_geo', new_stop_set)
                        st.session_state.categorized_dimensions = apply_filter('orig_dimensions', new_stop_set)
                        st.session_state.categorized_general = apply_filter('orig_general', new_stop_set)
                        all_prods = st.session_state.categorized_products
                        count_prods = len(all_prods)
                        if count_prods < 20:
                            st.session_state.auto_tags_words = all_prods
                            st.session_state.auto_promo_words = []
                        else:
                            half = int(math.ceil(count_prods / 2))
                            st.session_state.auto_tags_words = all_prods[:half]
                            st.session_state.auto_promo_words = all_prods[half:]
                        st.session_state['kws_tags_auto'] = "\n".join(st.session_state.auto_tags_words)
                        st.session_state['kws_promo_auto'] = "\n".join(st.session_state.auto_promo_words)
                        st.toast("Фильтр обновлен!", icon="✅"); time.sleep(0.5); st.rerun()

        high = results.get('missing_semantics_high', [])
        low = results.get('missing_semantics_low', [])
        if high or low:
            with st.expander(f"🧩 Упущенная семантика ({len(high)+len(low)})", expanded=False):
                if high: st.markdown(f"<div style='background:#EBF5FF;padding:10px;border-radius:5px;'><b>Важные:</b> {', '.join([x['word'] for x in high])}</div>", unsafe_allow_html=True)
                if low: st.markdown(f"<div style='background:#F7FAFC;padding:10px;border-radius:5px;margin-top:5px;'><b>Дополнительные слова:</b> {', '.join([x['word'] for x in low])}</div>", unsafe_allow_html=True)
        render_paginated_table(results['depth'], "1. Глубина", "tbl_depth_1", default_sort_col="Рекомендация", use_abs_sort_default=True)
        if 'naming_table_df' in st.session_state and st.session_state.naming_table_df is not None:
            df_naming = st.session_state.naming_table_df
            st.markdown("### 2. Рекомендации по названию товаров")
            if 'ideal_h1_result' in st.session_state:
                res_ideal = st.session_state.ideal_h1_result
                if isinstance(res_ideal, (tuple, list)) and len(res_ideal) >= 2:
                    example_name = res_ideal[0]
                    report_list = res_ideal[1]
                    formula_str = "Формула не определена"
                    for line in report_list:
                        if "структура" in line or "Схема" in line:
                            formula_str = line.replace("**Самая частая структура:**", "").replace("**Схема:**", "").strip()
                            break
                    with st.container(border=True):
                        st.markdown("#### 🧪 Идеальная формула названия")
                        st.info(f"**{formula_str}**", icon="🧩")
                        st.markdown(f"**Пример генерации:** _{example_name}_")
            st.markdown("##### Детальный анализ характеристик")
            if not df_naming.empty:
                col_ctrl1, col_ctrl2 = st.columns([1, 3])
                with col_ctrl1: show_tech = st.toggle("Показать размеры и цифры", value=False, key="toggle_show_tech_specs_unique")
                df_display = df_naming.copy()
                if not show_tech: df_display = df_display[~df_display['Тип хар-ки'].str.contains("Размеры", na=False)]
                if 'cat_sort' in df_display.columns: df_display = df_display.sort_values(by=["cat_sort", "raw_freq"], ascending=[True, False])
                cols_to_show = ["Тип хар-ки", "Слово", "Частотность (%)", "У Вас", "Медиана", "Добавить"]
                existing_cols = [c for c in cols_to_show if c in df_display.columns]
                df_display = df_display[existing_cols]
                def style_rows(row):
                    val = str(row.get('Добавить', ''))
                    if "+" in val: return ['background-color: #fff1f2; color: #9f1239'] * len(row)
                    if "✅" in val: return ['background-color: #f0fdf4; color: #166534'] * len(row)
                    return [''] * len(row)
                st.dataframe(df_display.style.apply(style_rows, axis=1), use_container_width=True, hide_index=True, height=(len(df_display) * 35) + 38 if len(df_display) < 15 else 500)
            else: st.warning("Нет данных для отображения.")
        render_paginated_table(results['hybrid'], "3. TF-IDF", "tbl_hybrid", default_sort_col="TF-IDF ТОП")
        render_paginated_table(results['relevance_top'], "4. Релевантность", "tbl_rel", default_sort_col="Ширина (балл)")

    if st.session_state.get('start_analysis_flag'):
        st.session_state.start_analysis_flag = False
        settings = {'noindex': st.session_state.settings_noindex, 'alt_title': st.session_state.settings_alt, 'numbers': st.session_state.settings_numbers, 'norm': st.session_state.settings_norm, 'ua': st.session_state.settings_ua, 'custom_stops': st.session_state.settings_stops.split()}
        my_data, my_domain, my_serp_pos = None, "", 0
        current_input_type = st.session_state.get("my_page_source_radio")
        if current_input_type == "Релевантная страница на вашем сайте":
            with st.spinner("Скачивание вашей страницы..."):
                my_data = parse_page(st.session_state.my_url_input, settings, st.session_state.query_input)
                if not my_data: st.error("Ошибка скачивания вашей страницы."); st.stop()
                my_domain = urlparse(st.session_state.my_url_input).netloc
        elif current_input_type == "Исходный код страницы или текст":
            my_data = {'url': 'Local', 'domain': 'local', 'body_text': st.session_state.my_content_input, 'anchor_text': ''}
        candidates_pool = []
        current_source_val = st.session_state.get("competitor_source_radio")
        user_target_top_n = st.session_state.settings_top_n
        download_limit = 30 
        if "API" in current_source_val:
            if not ARSENKIN_TOKEN: st.error("Отсутствует API токен Arsenkin."); st.stop()
            with st.spinner(f"API Arsenkin (Запрос Топ-30)..."):
                raw_top = get_arsenkin_urls(st.session_state.query_input, st.session_state.settings_search_engine, st.session_state.settings_region, ARSENKIN_TOKEN, depth_val=30)
                if not raw_top: st.stop()
                excl = [d.strip() for d in st.session_state.settings_excludes.split('\n') if d.strip()]
                agg_list = ["avito", "ozon", "wildberries", "market.yandex", "tiu", "youtube", "vk.com", "yandex", "leroymerlin", "petrovich", "satom", "pulscen", "blizko", "deal.by", "satu.kz", "prom.ua", "wikipedia", "dzen", "rutube", "kino", "otzovik", "irecommend", "profi.ru", "zoon", "2gis", "megamarket.ru", "lamoda.ru", "utkonos.ru", "vprok.ru", "allbiz.ru", "all-companies.ru", "orgpage.ru", "list-org.com", "rusprofile.ru", "e-katalog.ru", "kufar.by", "wildberries.kz", "ozon.kz", "kaspi.kz", "pulscen.kz", "allbiz.kz", "wildberries.uz", "olx.uz", "pulscen.uz", "allbiz.uz", "wildberries.kg", "pulscen.kg", "allbiz.kg", "all.biz", "b2b-center.ru"]
                excl.extend(agg_list)
                for res in raw_top:
                    dom = urlparse(res['url']).netloc.lower()
                    if my_domain and (my_domain in dom or dom in my_domain):
                        if my_serp_pos == 0 or res['pos'] < my_serp_pos: my_serp_pos = res['pos']
                    is_garbage = False
                    for x in excl:
                        if x.lower() in dom: is_garbage = True; break
                    if is_garbage: continue
                    candidates_pool.append(res)
        else:
            raw_input_urls = st.session_state.get("persistent_urls", "")
            candidates_pool = [{'url': u.strip(), 'pos': i+1} for i, u in enumerate(raw_input_urls.split('\n')) if u.strip()]

        if not candidates_pool: st.error("После фильтрации не осталось кандидатов."); st.stop()
        comp_data_valid = []
        with st.status(f"🕵️ Глубокое сканирование (Всего кандидатов: {len(candidates_pool)})...", expanded=True) as status:
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                futures = {executor.submit(parse_page, item['url'], settings, st.session_state.query_input): item for item in candidates_pool}
                done_count = 0
                for f in concurrent.futures.as_completed(futures):
                    original_item = futures[f]
                    try:
                        res = f.result()
                        if res:
                            res['pos'] = original_item['pos']
                            comp_data_valid.append(res)
                    except: pass
                    done_count += 1
                    status.update(label=f"Обработано: {done_count}/{len(candidates_pool)} | Успешно скачано: {len(comp_data_valid)}")

            comp_data_valid.sort(key=lambda x: x['pos'])
            data_for_graph = comp_data_valid[:download_limit]
            targets_for_graph = [{'url': d['url'], 'pos': d['pos']} for d in data_for_graph]

        with st.spinner("Анализ и фильтрация..."):
            results_full = calculate_metrics(data_for_graph, my_data, settings, my_serp_pos, targets_for_graph)
            st.session_state['full_graph_data'] = results_full['relevance_top']
            df_rel_check = results_full['relevance_top']
            good_urls, bad_urls_dicts, trend = analyze_serp_anomalies(df_rel_check)
            st.session_state['serp_trend_info'] = trend
            
            bad_urls_set = set(item['url'] for item in bad_urls_dicts)
            if "API" in current_source_val:
                clean_data_pool = [d for d in data_for_graph if d['url'] not in bad_urls_set]
                final_clean_data = clean_data_pool[:user_target_top_n]
            else: final_clean_data = data_for_graph 
            
            final_clean_targets = [{'url': d['url'], 'pos': d['pos']} for d in final_clean_data]
            results_final = calculate_metrics(final_clean_data, my_data, settings, my_serp_pos, final_clean_targets)
            st.session_state.analysis_results = results_final
            naming_df = calculate_naming_metrics(final_clean_data, my_data, settings)
            st.session_state.naming_table_df = naming_df 
            st.session_state.ideal_h1_result = analyze_ideal_name(final_clean_data)
            st.session_state.analysis_done = True
            
            if "API" in current_source_val and 'full_graph_data' in st.session_state: df_rel_check = st.session_state['full_graph_data']
            else: df_rel_check = st.session_state.analysis_results['relevance_top']
            should_auto_filter = True
            is_manual_mode = "Ручной" in current_source_val
            has_previous_exclusions = 'excluded_urls_auto' in st.session_state and len(st.session_state.get('excluded_urls_auto', '')) > 5
            if is_manual_mode and has_previous_exclusions: should_auto_filter = False 
            good_urls, bad_urls_dicts, trend = analyze_serp_anomalies(df_rel_check)
            st.session_state['serp_trend_info'] = trend
            
            if should_auto_filter and bad_urls_dicts:
                st.session_state['detected_anomalies'] = bad_urls_dicts
                st.session_state['persistent_urls'] = "\n".join(good_urls)
                excluded_list = [item['url'] for item in bad_urls_dicts]
                st.session_state['excluded_urls_auto'] = "\n".join(excluded_list)
                st.toast(f"🧹 Авто-фильтр: Исключено {len(bad_urls_dicts)} слабых сайтов.", icon="🗑️")
            elif not should_auto_filter and bad_urls_dicts:
                all_current_urls = [d['url'] for d in final_clean_data]
                st.session_state['persistent_urls'] = "\n".join(all_current_urls)
                st.toast(f"🛡️ Ручной режим: Слабые сайты ({len(bad_urls_dicts)} шт.) оставлены в анализе.", icon="🔓")
            else:
                st.session_state['persistent_urls'] = "\n".join(good_urls)
                if should_auto_filter:
                    if 'excluded_urls_auto' in st.session_state: del st.session_state['excluded_urls_auto']
                    if 'detected_anomalies' in st.session_state: del st.session_state['detected_anomalies']

            res = st.session_state.analysis_results
            words_to_check = [x['word'] for x in res.get('missing_semantics_high', [])]
            if not words_to_check:
                st.session_state.categorized_products = []; st.session_state.categorized_services = []
                st.session_state.categorized_commercial = []; st.session_state.categorized_dimensions = []
            else:
                with st.spinner("Классификация семантики..."):
                    categorized = classify_semantics_with_api(words_to_check, YANDEX_DICT_KEY)
                st.session_state.categorized_products = categorized['products']
                st.session_state.categorized_services = categorized['services']
                st.session_state.categorized_commercial = categorized['commercial']
                st.session_state.categorized_geo = categorized['geo']
                st.session_state.categorized_dimensions = categorized['dimensions']
                st.session_state.categorized_general = categorized['general']
                st.session_state.categorized_sensitive = categorized['sensitive']
                st.session_state.orig_products = categorized['products'] + categorized['sensitive']
                st.session_state.orig_services = categorized['services'] + categorized['sensitive']
                st.session_state.orig_commercial = categorized['commercial'] + categorized['sensitive']
                st.session_state.orig_geo = categorized['geo'] + categorized['sensitive']
                st.session_state.orig_dimensions = categorized['dimensions'] + categorized['sensitive']
                st.session_state.orig_general = categorized['general'] + categorized['sensitive']
                st.session_state['sensitive_words_input_final'] = "\n".join(categorized['sensitive'])

            all_found_products = st.session_state.categorized_products
            count_prods = len(all_found_products)
            if count_prods < 20:
                st.session_state.auto_tags_words = all_found_products
                st.session_state.auto_promo_words = []
            else:
                half_count = int(math.ceil(count_prods / 2))
                st.session_state.auto_tags_words = all_found_products[:half_count]
                st.session_state.auto_promo_words = all_found_products[half_count:]
            
            st.session_state['tags_products_edit_final'] = "\n".join(st.session_state.auto_tags_words)
            st.session_state['promo_keywords_area_final'] = "\n".join(st.session_state.auto_promo_words)
            st.session_state['force_radio_switch'] = True
            if "API" in current_source_val: st.session_state['persistent_urls'] = "\n".join(good_urls)
            st.rerun()

with tab_wholesale_main:
    st.header("🏭 Единый генератор контента")
    cat_products = st.session_state.get('categorized_products', [])
    cat_services = st.session_state.get('categorized_services', [])
    structure_keywords = cat_products + cat_services
    count_struct = len(structure_keywords)
    if 'auto_tags_words' in st.session_state and st.session_state.auto_tags_words:
         tags_list_source = st.session_state.auto_tags_words
         promo_list_source = st.session_state.auto_promo_words
    else:
         if count_struct > 0:
            if count_struct < 10: tags_list_source = structure_keywords; promo_list_source = []
            elif count_struct < 30:
                mid = math.ceil(count_struct / 2)
                tags_list_source = structure_keywords[:mid]
                promo_list_source = structure_keywords[mid:]
            else:
                part = math.ceil(count_struct / 3)
                tags_list_source = structure_keywords[:part]
                promo_list_source = structure_keywords[part:part*2]
         else: tags_list_source = []; promo_list_source = []
    
    sidebar_default_text = ""
    if count_struct >= 30 and 'auto_tags_words' not in st.session_state:
         part = math.ceil(count_struct / 3)
         sidebar_default_text = "\n".join(structure_keywords[part*2:])

    tags_default_text = ", ".join(tags_list_source)
    promo_default_text = ", ".join(promo_list_source)
    cat_dimensions = st.session_state.get('categorized_dimensions', [])
    tech_context_default = ", ".join(cat_dimensions) if cat_dimensions else ""
    cat_commercial = st.session_state.get('categorized_commercial', [])
    cat_general = st.session_state.get('categorized_general', [])
    cat_geo = st.session_state.get('categorized_geo', [])
    text_context_list_raw = cat_commercial + cat_general
    text_context_default = ", ".join(text_context_list_raw)
    geo_context_default = ", ".join(cat_geo)

    auto_check_text = bool(text_context_list_raw)
    auto_check_tags = bool(tags_list_source)
    auto_check_tables = bool(cat_dimensions)
    auto_check_promo = bool(promo_list_source)
    auto_check_sidebar = bool(sidebar_default_text.strip())
    auto_check_geo = bool(cat_geo)

    with st.container(border=True):
        st.subheader("1. Источник и Доступы")
        col_source, col_key = st.columns([3, 1])
        use_manual_html = st.checkbox("📝 Вставить HTML код страницы", key="cb_manual_html_mode", value=False)
        with col_source:
            if use_manual_html:
                manual_html_source = st.text_area("Исходный код страницы (HTML)", height=200, placeholder="<html>...</html>", help="Скопируйте сюда исходный код страницы.")
                main_category_url = None
            else:
                main_category_url = st.text_input("URL Категории", placeholder="https://site.ru/catalog/...", help="Скрипт соберет товары с этой страницы")
                manual_html_source = None
        with col_key:
            default_key = st.session_state.get('pplx_key_cache', "AIzaSyALrGas6cbkwNN9lko-Ag4_czO58NS4HIc")
            pplx_api_key = st.text_input("Google AI API Key", value=default_key, type="password")
            if pplx_api_key: st.session_state.pplx_key_cache = pplx_api_key

    st.subheader("2. Какие блоки генерируем?")
    st.info("ℹ️ **Авто-настройка:** Галочки активированы автоматически там, где после анализа нашлись подходящие слова. Вы можете изменить выбор вручную.")
    col_ch1, col_ch2, col_ch3, col_ch4, col_ch5, col_ch6 = st.columns(6)
    with col_ch1: use_text = st.checkbox("🤖 AI Тексты", value=auto_check_text)
    with col_ch2: use_tags = st.checkbox("🏷️ Теги", value=auto_check_tags)
    with col_ch3: use_tables = st.checkbox("🧩 Таблицы", value=auto_check_tables)
    with col_ch4: use_promo = st.checkbox("🔥 Промо", value=auto_check_promo)
    with col_ch5: use_sidebar = st.checkbox("📑 Сайдбар", value=auto_check_sidebar)
    with col_ch6: use_geo = st.checkbox("🌍 Гео-блок", value=auto_check_geo)

    global_tags_list = []
    global_promo_list = []
    global_sidebar_list = []
    global_geo_list = []
    tags_file_content = ""
    table_prompts = []
    df_db_promo = None
    promo_title = "Рекомендуем"
    sidebar_content = ""
    text_context_final_list = []
    tech_context_final_str = ""
    num_text_blocks_val = 5 

    if any([use_text, use_tags, use_tables, use_promo, use_sidebar, use_geo]):
        st.subheader("3. Настройки модулей")
        if use_text:
            with st.container(border=True):
                st.markdown("#### 🤖 1. AI Тексты")
                col_txt1, col_txt2 = st.columns([1, 4])
                with col_txt1: num_text_blocks_val = st.selectbox("Кол-во блоков", [1, 2, 3, 4, 5], index=4, key="sb_num_blocks")
                with col_txt2:
                    ai_words_input = st.text_area("Слова для внедрения (Коммерция + Общие)", value=text_context_default, height=100, key="ai_text_context_editable", help="Эти слова нейросеть постарается внедрить в текст.")
                text_context_final_list = [x.strip() for x in re.split(r'[,\n]+', ai_words_input) if x.strip()]

        if use_tags:
            with st.container(border=True):
                st.markdown("#### 🏷️ 2. Теги")
                kws_input_tags = st.text_area("Список (Товары + Услуги) - через запятую", value=tags_default_text, height=100, key="kws_tags_auto")
                global_tags_list = [x.strip() for x in re.split(r'[,\n]+', kws_input_tags) if x.strip()]
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

        def generate_context_aware_headers(count, query, dimensions_list, general_list):
            query_lower = query.lower()
            dims_str = " ".join(dimensions_list).lower()
            gen_str = " ".join(general_list).lower()
            full_context = f"{dims_str} {gen_str} {query_lower}"
            has_sizes_signal = (len(dimensions_list) > 0 or bool(re.search(r'\d+[xх*]\d+', full_context)) or any(x in full_context for x in ['размер', 'габарит', 'толщин', 'диаметр', 'раскрой', 'вес', 'масс']))
            has_gost_signal = any(x in full_context for x in ['гост', 'din', 'aisi', 'astm', 'ту ', 'стандарт'])
            has_grade_signal = any(x in full_context for x in ['марк', 'сплав', 'сталь', 'ст.', 'материал', 'химич', 'состав'])
            has_usage_signal = any(x in full_context for x in ['применен', 'сфер', 'назначен', 'использ'])
            priority_stack = []
            if has_grade_signal: priority_stack.append("Марки и сплавы")
            if has_sizes_signal: priority_stack.append("Таблица размеров")
            if has_gost_signal: priority_stack.append("ГОСТы и стандарты")
            if "хим" in full_context and "состав" in full_context:
                 if "Марки и сплавы" in priority_stack:
                     idx = priority_stack.index("Марки и сплавы")
                     priority_stack.insert(idx+1, "Химический состав")
                 else: priority_stack.append("Химический состав")
            defaults = ["Технические характеристики", "Свойства", "Сферы использования", "Параметры изделия", "Аналоги"]
            final_headers = []
            for p in priority_stack:
                if p not in final_headers: final_headers.append(p)
            for d in defaults:
                if d not in final_headers: final_headers.append(d)
            while len(final_headers) < count: final_headers.append("Характеристики")
            return final_headers[:count]

        if use_tables:
            with st.container(border=True):
                st.markdown("#### 🧩 3. Таблицы")
                raw_query = st.session_state.get('query_input', '')
                found_dims = st.session_state.get('categorized_dimensions', [])
                found_general = st.session_state.get('categorized_general', [])
                col_ctx, col_cnt = st.columns([3, 1]) 
                with col_ctx:
                    tech_context_final_str = st.text_area("Контекст для таблиц (Марки, ГОСТ, Размеры)", value=tech_context_default, height=68, key="table_context_editable", help="Эти данные помогут AI составить правильную таблицу.")
                with col_cnt:
                    cnt_options = [1, 2, 3, 4, 5]
                    cnt = st.selectbox("Кол-во таблиц", cnt_options, index=1, key="num_tbl_vert_select")
                smart_headers_list = generate_context_aware_headers(cnt, raw_query, found_dims, found_general)
                table_presets = ["Технические характеристики", "Свойства", "Параметры изделия", "Общее описание", "Таблица размеров", "Сортамент", "Химический состав", "Физические свойства", "Механические свойства", "Марки и сплавы", "Состав материала", "ГОСТы и стандарты", "Техническая документация", "Требования ГОСТ", "Назначение", "Сферы использования", "Условия эксплуатации", "Где используется", "Классификация", "Модификации", "Аналоги", "Сравнение моделей", "Разновидности"]
                table_prompts = []
                st.write(""); cols = st.columns(cnt)
                for i, col in enumerate(cols):
                    with col:
                        st.caption(f"**Таблица {i+1}**")
                        suggested_topic = smart_headers_list[i]
                        try: default_idx = table_presets.index(suggested_topic)
                        except: default_idx = 0
                        is_manual = st.checkbox("Свой заголовок", key=f"cb_tbl_manual_{i}")
                        if is_manual:
                            selected_topic = st.text_input(f"Название табл. {i+1}", value="", key=f"tbl_topic_custom_{i}", label_visibility="collapsed")
                            if not selected_topic.strip(): selected_topic = "Характеристики" 
                        else:
                            selected_topic = st.selectbox(f"Тема табл. {i+1}", table_presets, index=default_idx, key=f"tbl_topic_select_{i}", label_visibility="collapsed")
                        table_prompts.append(selected_topic)

        if use_promo:
            with st.container(border=True):
                st.markdown("#### 🔥 4. Промо-блок")
                kws_input_promo = st.text_area("Список (Товары + Услуги) - через запятую", value=promo_default_text, height=100, key="kws_promo_auto")
                global_promo_list = [x.strip() for x in re.split(r'[,\n]+', kws_input_promo) if x.strip()]
                if not global_promo_list: st.warning("⚠️ Список пуст!")
                st.markdown("---")
                col_p1, col_p2 = st.columns([1, 2])
                with col_p1:
                    promo_presets = ["Смотрите также", "Похожие товары", "Вас может заинтересовать", "Рекомендуем", "Другие предложения", "Вам может пригодиться", "Также в этом разделе", "С этим товаром покупают", "Часто покупают вместе", "Сопутствующие товары", "Хиты продаж", "Выбор покупателей", "Лидеры спроса", "Популярное сейчас", "Топ товаров категории", "Лучшая цена", "Спецпредложения", "Успейте заказать", "Не забудьте добавить", "Вы недавно смотрели"]
                    raw_query = st.session_state.get('query_input', '').lower()
                    comm_words = st.session_state.get('categorized_commercial', [])
                    comm_context = f"{raw_query} {' '.join(comm_words)}".lower()
                    target_header = "Смотрите также"
                    is_commercial = any(x in comm_context for x in ["купить", "цена", "заказ", "стоимость", "прайс", "магазин", "корзина"])
                    is_promo = any(x in comm_context for x in ["акция", "скидк", "распродаж", "выгодн"])
                    is_top = any(x in comm_context for x in ["топ", "лучш", "рейтинг", "популярн"])
                    if is_promo: target_header = "Спецпредложения"
                    elif is_top: target_header = "Лидеры спроса"
                    elif is_commercial: target_header = "С этим товаром покупают"
                    try: promo_smart_idx = promo_presets.index(target_header)
                    except: promo_smart_idx = 0
                    use_custom_header = st.checkbox("Ввести свой заголовок", key="cb_custom_header")
                    if use_custom_header: promo_title = st.text_input("Ваш заголовок", placeholder="Смотрите также", key="pr_tit_vert")
                    else: promo_title = st.selectbox("Варианты заголовка", promo_presets, index=promo_smart_idx, key="promo_header_select")
                    st.markdown("<br>", unsafe_allow_html=True)
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

        if use_sidebar:
            with st.container(border=True):
                st.markdown("#### 📑 5. Сайдбар")
                kws_input_sidebar = st.text_area("Список (Товары + Услуги) - с новой строки", value=sidebar_default_text, height=100, key="kws_sidebar_auto")
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

        if use_geo:
            with st.container(border=True):
                st.markdown("#### 🌍 6. Гео-блок")
                kws_input_geo = st.text_area("Список городов/регионов (из вкладки Анализ) - через запятую", value=geo_context_default, height=100, key="kws_geo_auto")
                global_geo_list = [x.strip() for x in re.split(r'[,\n]+', kws_input_geo) if x.strip()]
                if not global_geo_list: st.warning("⚠️ Список городов пуст!")
                else: st.info(f"Будет сгенерирован текст доставки для поля IP_PROP4819 с упоминанием этих городов.")
    st.markdown("---")
    
    ready_to_go = True
    if use_manual_html:
        if not manual_html_source: ready_to_go = False
    else:
        if not main_category_url: ready_to_go = False
    if (use_text or use_tables) and not pplx_api_key: ready_to_go = False
    if use_promo and df_db_promo is None: ready_to_go = False
    if use_geo and not pplx_api_key: ready_to_go = False
    
    if st.button("🚀 ЗАПУСТИТЬ ГЕНЕРАЦИЮ", type="primary", disabled=not ready_to_go, use_container_width=True):
        st.session_state.gen_result_df = None
        st.session_state.unified_excel_data = None
        status_box = st.status("🛠️ Подготовка данных...", expanded=True)
        final_data = [] 
        
        tags_map = {}
        all_tags_links = []
        if use_tags:
            if tags_file_content:
                s_io = io.StringIO(tags_file_content)
                all_tags_links = [l.strip() for l in s_io.readlines() if l.strip()]
            elif os.path.exists("data/links_base.txt"):
                with open("data/links_base.txt", "r", encoding="utf-8") as f:
                    all_tags_links = [l.strip() for l in f.readlines() if l.strip()]
            for kw in global_tags_list:
                tr = transliterate_text(kw).replace(' ', '-').replace('_', '-')
                search_roots = {tr}
                if len(tr) > 5: search_roots.add(tr[:-1]); search_roots.add(tr[:-2])
                elif len(tr) > 4: search_roots.add(tr[:-1])
                matches = []
                for u in all_tags_links:
                    u_lower = u.lower()
                    for root in search_roots:
                        if root in u_lower: matches.append(u); break
                if matches: tags_map[kw] = matches

        p_img_map = {}
        if use_promo and df_db_promo is not None:
            for _, row in df_db_promo.iterrows():
                u = str(row.iloc[0]).strip(); img = str(row.iloc[1]).strip()
                if u and u != 'nan' and img and img != 'nan': p_img_map[u.rstrip('/')] = img
        
        all_menu_urls = []
        if use_sidebar:
            if sidebar_content:
                s_io = io.StringIO(sidebar_content)
                all_menu_urls = [l.strip() for l in s_io.readlines() if l.strip()]
            elif os.path.exists("data/menu_structure.txt"):
                with open("data/menu_structure.txt", "r", encoding="utf-8") as f:
                    all_menu_urls = [l.strip() for l in f.readlines() if l.strip()]

        missing_words_log = set()
        if use_tags:
            for kw in global_tags_list:
                if kw not in tags_map: missing_words_log.add(kw)
        if use_promo:
            for kw in global_promo_list:
                tr = transliterate_text(kw).replace(' ', '-').replace('_', '-')
                roots = [tr]
                if len(tr) > 5: roots.extend([tr[:-1], tr[:-2]])
                has_match = False
                for u in p_img_map.keys():
                    if any(r in u for r in roots): has_match = True; break
                if not has_match: missing_words_log.add(kw)
        if use_sidebar and global_sidebar_list:
            for kw in global_sidebar_list:
                tr = transliterate_text(kw).replace(' ', '-').replace('_', '-')
                roots = [tr]
                if len(tr) > 5: roots.extend([tr[:-1], tr[:-2]])
                has_match = False
                for u in all_menu_urls:
                    if any(r in u for r in roots): has_match = True; break
                if not has_match: missing_words_log.add(kw)

        if missing_words_log:
            missing_list = sorted(list(missing_words_log))
            for w in missing_list:
                if w not in text_context_final_list: text_context_final_list.append(w)
            tech_additions = []
            for w in missing_list:
                if any(char.isdigit() for char in w) or any(x in w.lower() for x in ['гост', 'тип', 'форма', 'мм', 'кг']):
                    tech_additions.append(w)
            if tech_additions: tech_context_final_str += "\n" + ", ".join(tech_additions)
            status_box.markdown(f"""<div style="background-color: #FFF4E5; border-left: 5px solid #FF9800; padding: 15px; border-radius: 4px; margin-bottom: 15px; color: #663C00;"><strong>⚠️ Внимание: Часть ссылок не найдена</strong><br><span style="font-size: 0.9em;">Мы не нашли точного совпадения в структуре для: <b>{', '.join(missing_list)}</b>.<br>✅ <u>Они были перенесены в ТЗ для Нейросети (будут в тексте/таблицах).</u></span></div>""", unsafe_allow_html=True)
            time.sleep(2)

        target_pages = []
        soup = None
        current_base_url = main_category_url if main_category_url else "http://localhost"

        try:
            if use_manual_html:
                status_box.write("📂 Обрабатываем HTML код...")
                soup = BeautifulSoup(manual_html_source, 'html.parser')
            else:
                status_box.write(f"🕵️ Сканируем категорию: {main_category_url}")
                session = requests.Session()
                retry = Retry(connect=3, read=3, redirect=3, backoff_factor=0.5)
                adapter = HTTPAdapter(max_retries=retry)
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                r = session.get(main_category_url, headers=headers, timeout=30, verify=False)
                if r.status_code == 200: soup = BeautifulSoup(r.text, 'html.parser')
                else: 
                    status_box.error(f"Ошибка доступа: {r.status_code}")
                    st.stop()
            if soup:
                tags_container = soup.find(class_='popular-tags-inner')
                if tags_container:
                    for link in tags_container.find_all('a'):
                        href = link.get('href')
                        name = link.get_text(strip=True)
                        if href and name:
                            full_url = urljoin(current_base_url, href)
                            target_pages.append({'url': full_url, 'name': name})
                if not target_pages:
                    status_box.warning("Теги товаров не найдены (класс .popular-tags-inner). Генерируем для одной страницы.")
                    h1 = soup.find('h1')
                    name = h1.get_text(strip=True) if h1 else "Товар"
                    target_pages.append({'url': current_base_url, 'name': name})
        except Exception as e: 
            status_box.error(f"Критическая ошибка: {e}")
            st.stop()
            
        urls_to_fetch_names = set()
        promo_items_pool = [] 
        if use_tags:
            for kw, matches in tags_map.items(): urls_to_fetch_names.update(matches)
        if use_promo:
            used_urls = set()
            for kw in global_promo_list:
                if kw in missing_words_log: continue
                tr = transliterate_text(kw).replace(' ', '-').replace('_', '-')
                roots = [tr]
                if len(tr) > 5: roots.extend([tr[:-1], tr[:-2]])
                matches = []
                for u in p_img_map.keys():
                    if any(r in u for r in roots): matches.append(u)
                for m in matches:
                    if m not in used_urls:
                        urls_to_fetch_names.add(m)
                        promo_items_pool.append({'url': m, 'img': p_img_map[m]})
                        used_urls.add(m)
        sidebar_matched_urls = []
        if use_sidebar:
            if global_sidebar_list:
                for kw in global_sidebar_list:
                    if kw in missing_words_log: continue
                    tr = transliterate_text(kw).replace(' ', '-').replace('_', '-')
                    roots = [tr]
                    if len(tr) > 5: roots.extend([tr[:-1], tr[:-2]])
                    found = []
                    for u in all_menu_urls:
                        if any(r in u for r in roots): found.append(u)
                    sidebar_matched_urls.extend(found)
                sidebar_matched_urls = list(set(sidebar_matched_urls))
            else: sidebar_matched_urls = all_menu_urls
            urls_to_fetch_names.update(sidebar_matched_urls)

        url_name_cache = {}
        if urls_to_fetch_names:
            status_box.write(f"🌍 Получаем названия для {len(urls_to_fetch_names)} ссылок...")
            def fetch_name_worker(u): return u, get_breadcrumb_only(u) 
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                future_to_url = {executor.submit(fetch_name_worker, u): u for u in urls_to_fetch_names}
                done_cnt = 0
                prog_fetch = status_box.progress(0)
                for future in concurrent.futures.as_completed(future_to_url):
                    u_res, name_res = future.result()
                    norm_key = u_res.rstrip('/')
                    if name_res: url_name_cache[norm_key] = name_res
                    else:
                        slug = norm_key.split('/')[-1]
                        url_name_cache[norm_key] = force_cyrillic_name_global(slug)
                    done_cnt += 1
                    prog_fetch.progress(done_cnt / len(urls_to_fetch_names))
            status_box.write("✅ Названия собраны!")

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
            full_sidebar_code = f"""<div class="page-content-with-sidebar"><button id="mobile-menu-toggle" class="menu-toggle-button">☰</button><div class="sidebar-wrapper"><nav id="sidebar-menu"><ul class="list-unstyled components">{inner_html}</ul></nav></div></div>"""

        has_ai_key = bool(pplx_api_key) and (genai is not None)
        progress_bar = status_box.progress(0)
        total_steps = len(target_pages)
        model_flash = None
        if has_ai_key:
            try:
                genai.configure(api_key=pplx_api_key)
                model_flash = genai.GenerativeModel('gemini-2.0-flash-exp')
            except:
                try: model_flash = genai.GenerativeModel('gemini-1.5-flash')
                except: pass
        
        for idx, page in enumerate(target_pages):
            base_text_raw, tags_on_page, real_header_h2, err = get_page_data_for_gen(page['url'])
            header_for_ai = real_header_h2 if real_header_h2 else page['name']
            row_data = {'Page URL': page['url'], 'Product Name': header_for_ai}
            for k, v in STATIC_DATA_GEN.items(): row_data[k] = v
            current_page_seo_words = list(text_context_final_list)
            
            row_data['Tags HTML'] = "" 
            row_data['Promo HTML'] = ""
            if use_tags:
                tags_html_parts = []
                html_collector = []
                for kw in global_tags_list:
                    if kw in tags_map:
                        valid = [u for u in tags_map[kw] if u.rstrip('/') != page['url'].rstrip('/')]
                        if valid:
                            sel = random.choice(valid)
                            nm = url_name_cache.get(sel.rstrip('/'), kw)
                            html_collector.append(f'<a href="{sel}" class="tag-link">{nm}</a>')
                        else:
                             if kw not in current_page_seo_words: current_page_seo_words.append(kw)
                if html_collector: row_data['Tags HTML'] = '<div class="popular-tags">' + "\n".join(html_collector) + '</div>'
            if use_promo:
                cands = [p for p in promo_items_pool if p['url'].rstrip('/') != page['url'].rstrip('/')]
                random.shuffle(cands)
                if cands:
                    p_html = f'<div class="promo-section"><h3>{promo_title}</h3><div class="promo-grid" style="display:flex;gap:15px;overflow-x:auto;">'
                    for item in cands:
                        p_name = url_name_cache.get(item['url'].rstrip('/'), "Товар")
                        p_html += f'<div class="promo-card" style="min-width:220px;"><a href="{item["url"]}"><img src="{item["img"]}" style="max-height:100px;"><br>{p_name}</a></div>'
                    p_html += '</div></div>'
                    row_data['Promo HTML'] = p_html

            if use_text and has_ai_key:
                blocks = generate_ai_content_blocks(api_key=pplx_api_key, base_text=base_text_raw if base_text_raw else "", tag_name=page['name'], forced_header=header_for_ai, num_blocks=num_text_blocks_val, seo_words=current_page_seo_words)
                for i, b in enumerate(blocks): row_data[f'Text_Block_{i+1}'] = b
            if use_tables and has_ai_key and model_flash:
                for t_i, t_topic in enumerate(table_prompts):
                    ctx = f"Данные: {tech_context_final_str}" if tech_context_final_str else ""
                    prompt = f"Create HTML table for '{header_for_ai}'. Topic: {t_topic}. {ctx}. No Markdown."
                    html = call_gemini_safe(model_flash, prompt) 
                    html = html.replace("```html", "").replace("```", "").strip()
                    if "<table" not in html and "Error" not in html: html = f"<table>{html}</table>"
                    row_data[f'Table_{t_i+1}_HTML'] = html
            if use_geo and has_ai_key and global_geo_list and model_flash:
                cities = ", ".join(random.sample(global_geo_list, min(20, len(global_geo_list))))
                prompt = f"Write HTML <p> regarding delivery of '{header_for_ai}' to: {cities}."
                html = call_gemini_safe(model_flash, prompt) 
                row_data['IP_PROP4819'] = html.replace("```html", "").replace("```", "").strip()
            if use_sidebar: row_data['Sidebar HTML'] = full_sidebar_code
            final_data.append(row_data)
            progress_bar.progress((idx + 1) / total_steps)

        df_result = pd.DataFrame(final_data)
        st.session_state.gen_result_df = df_result 
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer: df_result.to_excel(writer, index=False)
        st.session_state.unified_excel_data = buffer.getvalue()
        status_box.update(label="✅ Конвейер завершен! Данные готовы.", state="complete", expanded=False)

    if st.session_state.get('unified_excel_data') is not None:
        st.success("Файл успешно сгенерирован!")
        st.download_button(label="📥 СКАЧАТЬ ЕДИНЫЙ EXCEL", data=st.session_state.unified_excel_data, file_name="unified_content_gen.xlsx", mime="application/vnd.ms-excel", key="btn_dl_unified")

with tab_wholesale_main: 
    if 'gen_result_df' in st.session_state and st.session_state.gen_result_df is not None:
        st.markdown("---")
        st.header("👀 Предпросмотр результата")
        df = st.session_state.gen_result_df
        page_options = df['Product Name'].tolist()
        selected_page_name = st.selectbox("Выберите страницу для просмотра:", page_options, key="preview_selector")
        row = df[df['Product Name'] == selected_page_name].iloc[0]
        has_text = any((f'Text_Block_{i}' in row and pd.notna(row[f'Text_Block_{i}']) and str(row[f'Text_Block_{i}']).strip()) for i in range(1, 6))
        table_cols = [c for c in df.columns if 'Table_' in c and '_HTML' in c and pd.notna(row[c]) and str(row[c]).strip()]
        has_tables = len(table_cols) > 0
        has_tags = 'Tags HTML' in row and pd.notna(row['Tags HTML']) and str(row['Tags HTML']).strip()
        has_sidebar = 'Sidebar HTML' in row and pd.notna(row['Sidebar HTML']) and str(row['Sidebar HTML']).strip()
        has_geo = 'IP_PROP4819' in row and pd.notna(row['IP_PROP4819']) and str(row['IP_PROP4819']).strip()
        has_promo = 'Promo HTML' in row and pd.notna(row['Promo HTML']) and str(row['Promo HTML']).strip()
        has_visual = has_tags or has_sidebar or has_geo or has_promo 
        active_tabs = []
        if has_text: active_tabs.append("📝 Текст")
        if has_tables: active_tabs.append("🧩 Таблицы")
        if has_visual: active_tabs.append("🎨 Визуал")

        st.markdown("""<style>.preview-box { border: 1px solid #e0e0e0; padding: 20px; border-radius: 8px; background: #fff; margin-bottom: 20px; }.preview-label { font-size: 12px; font-weight: bold; color: #888; text-transform: uppercase; margin-bottom: 5px; }.popular-tags { display: flex; flex-wrap: wrap; gap: 8px; }.tag-link { background: #f0f2f5; color: #333; padding: 5px 10px; border-radius: 4px; text-decoration: none; font-size: 13px; }table { width: 100%; border-collapse: collapse; font-size: 14px; }th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }th { background-color: #f2f2f2; font-weight: bold; }.sidebar-wrapper ul { list-style-type: none; padding-left: 10px; }.level-1-header { font-weight: bold; margin-top: 10px; color: #277EFF; }.promo-grid { display: flex !important; flex-wrap: wrap; gap: 10px; }.promo-card { width: 23%; box-sizing: border-box; }.promo-card img { max-width: 100%; height: auto; }</style>""", unsafe_allow_html=True)

        if not active_tabs: st.warning("⚠️ Контент пуст.")
        else:
            tabs_objects = st.tabs(active_tabs)
            tabs_map = dict(zip(active_tabs, tabs_objects))
            if "📝 Текст" in tabs_map:
                with tabs_map["📝 Текст"]:
                    st.subheader(row['Product Name'])
                    for i in range(1, 6):
                        col_key = f'Text_Block_{i}'
                        if col_key in row and pd.notna(row[col_key]):
                            content = str(row[col_key]).strip()
                            if content:
                                with st.container():
                                    st.caption(f"Блок {i}")
                                    st.markdown(f"<div class='preview-box'>{content}</div>", unsafe_allow_html=True)
            if "🧩 Таблицы" in tabs_map:
                with tabs_map["🧩 Таблицы"]:
                    for t_col in table_cols:
                        content = row[t_col]
                        clean_title = t_col.replace('_HTML', '').replace('_', ' ')
                        st.caption(clean_title)
                        st.markdown(content, unsafe_allow_html=True)
            if "🎨 Визуал" in tabs_map:
                with tabs_map["🎨 Визуал"]:
                    if has_promo:
                         st.markdown('<div class="preview-label">Промо-блок (Рекомендации)</div>', unsafe_allow_html=True)
                         st.markdown(f"<div class='preview-box'>{row['Promo HTML']}</div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        if has_tags:
                            st.markdown('<div class="preview-label">Теги</div>', unsafe_allow_html=True)
                            st.markdown(f"<div class='preview-box'>{row['Tags HTML']}</div>", unsafe_allow_html=True)
                        if has_geo:
                            st.markdown('<div class="preview-label">Гео-блок</div>', unsafe_allow_html=True)
                            st.markdown(f"<div class='preview-box'>{row['IP_PROP4819']}</div>", unsafe_allow_html=True)
                    with c2:
                        if has_sidebar:
                            st.markdown('<div class="preview-label">Сайдбар</div>', unsafe_allow_html=True)
                            st.markdown(f"<div class='preview-box' style='max-height: 400px; overflow-y: auto;'>{row['Sidebar HTML']}</div>", unsafe_allow_html=True)
