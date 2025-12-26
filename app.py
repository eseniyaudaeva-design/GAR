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
    if df_rel.empty:
        return

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
    COLOR_MAIN, COLOR_WIDTH, COLOR_DEPTH, COLOR_TREND = '#4F46E5', '#0EA5E9', '#E11D48', '#15803d'
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
        margin=dict(l=10, r=10, t=50, b=40),
        hovermode="x unified",
        height=380
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
                    if morph: sets[set_key].add(morph.parse(w_clean)[0].normal_form.replace('ё', 'е'))
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
        if morph: lemma = morph.parse(word_lower)[0].normal_form

        if word_lower in SPECS_SET or lemma in SPECS_SET: categories['dimensions'].add(word_lower); continue
        if dim_pattern.search(word_lower) or grade_pattern.match(word_lower) or word_lower.isdigit(): categories['dimensions'].add(word_lower); continue
        if word_lower in PRODUCTS_SET or lemma in PRODUCTS_SET: categories['products'].add(word_lower); continue
        
        is_product_root = False
        for prod in PRODUCTS_SET:
            check_root = prod[:-1] if len(prod) > 4 else prod
            if len(check_root) > 3 and check_root in word_lower:
                categories['products'].add(word_lower)
                is_product_root = True; break
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

GARBAGE_LATIN_STOPLIST = {'whatsapp', 'viber', 'telegram', 'skype', 'vk', 'instagram', 'facebook', 'youtube', 'twitter', 'cookie', 'cookies', 'policy', 'privacy', 'agreement', 'terms', 'click', 'submit', 'send', 'zakaz', 'basket', 'cart', 'order', 'call', 'back', 'callback', 'login', 'logout', 'sign', 'register', 'auth', 'account', 'profile', 'search', 'menu', 'nav', 'navigation', 'footer', 'header', 'sidebar', 'img', 'jpg', 'png', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'svg', 'ok', 'error', 'undefined', 'null', 'true', 'false', 'var', 'let', 'const', 'function', 'return', 'ru', 'en', 'com', 'net', 'org', 'biz', 'shop', 'store', 'phone', 'email', 'tel', 'fax', 'mob', 'address', 'copyright', 'all', 'rights', 'reserved', 'div', 'span', 'class', 'id', 'style', 'script', 'body', 'html', 'head', 'meta', 'link'}
SENSITIVE_STOPLIST_RAW = {"украина", "ukraine", "ua", "всу", "зсу", "ато", "киев", "львов", "харьков", "одесса", "днепр", "мариуполь", "донецк", "луганск", "днр", "лнр", "донбасс", "мелитополь", "бердянск", "бахмут", "запорожье", "херсон", "крым", "севастополь", "симферополь"}
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
    except: ARSENKIN_TOKEN = None

if "yandex_dict_key" in st.session_state: YANDEX_DICT_KEY = st.session_state.yandex_dict_key
else:
    try: YANDEX_DICT_KEY = st.secrets["api"]["yandex_dict_key"]
    except: YANDEX_DICT_KEY = None

REGION_MAP = {"Москва": {"ya": 213, "go": 1011969}, "Санкт-Петербург": {"ya": 2, "go": 1011966}, "Екатеринбург": {"ya": 54, "go": 1011868}, "Новосибирск": {"ya": 65, "go": 1011928}, "Казань": {"ya": 43, "go": 1011904}, "Нижний Новгород": {"ya": 47, "go": 1011918}, "Самара": {"ya": 51, "go": 1011956}, "Челябинск": {"ya": 56, "go": 1011882}, "Омск": {"ya": 66, "go": 1011931}, "Краснодар": {"ya": 35, "go": 1011894}, "Киев (UA)": {"ya": 143, "go": 1012852}, "Минск (BY)": {"ya": 157, "go": 1001493}, "Алматы (KZ)": {"ya": 162, "go": 1014601}}
DEFAULT_EXCLUDE_DOMAINS = {"yandex.ru", "avito.ru", "beru.ru", "tiu.ru", "aliexpress.com", "aliexpress.ru", "ebay.com", "auto.ru", "2gis.ru", "sravni.ru", "toshop.ru", "price.ru", "pandao.ru", "instagram.com", "wikipedia.org", "rambler.ru", "hh.ru", "banki.ru", "regmarkets.ru", "zoon.ru", "pulscen.ru", "prodoctorov.ru", "blizko.ru", "domclick.ru", "satom.ru", "quto.ru", "edadeal.ru", "cataloxy.ru", "irr.ru", "onliner.by", "shop.by", "deal.by", "yell.ru", "profi.ru", "irecommend.ru", "otzovik.com", "ozon.ru", "ozon.by", "market.yandex.ru", "youtube.com", "www.youtube.com", "gosuslugi.ru", "www.gosuslugi.ru", "dzen.ru", "2gis.by", "wildberries.ru", "rutube.ru", "vk.com", "facebook.com", "chipdip.ru"}
DEFAULT_EXCLUDE = "\n".join(DEFAULT_EXCLUDE_DOMAINS)
DEFAULT_STOPS = "рублей\nруб\nстр\nул\nшт\nсм\nмм\nмл\nкг\nкв\nм²\nсм²\nм2\nсм2"
PRIMARY_COLOR, PRIMARY_DARK, TEXT_COLOR, LIGHT_BG_MAIN, BORDER_COLOR, HEADER_BG, ROW_BORDER_COLOR = "#277EFF", "#1E63C4", "#3D4858", "#F1F5F9", "#E2E8F0", "#F0F7FF", "#DBEAFE"

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
        .chart-link {{ color: #277EFF !important; font-weight: 600 !important; text-decoration: none !important; border-bottom: 4px solid #CBD5E1 !important; display: inline-block !important; transition: border-color 0.2s ease !important; }}
        .chart-link:hover {{ border-bottom-color: #277EFF !important; cursor: pointer !important; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# STATIC DATA & GEN FUNCTIONS
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
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.encoding = 'utf-8'
    except: return None, None, None, "Ошибка"
    if response.status_code != 200: return None, None, None, f"Ошибка: {response.status_code}"
    soup = BeautifulSoup(response.text, 'html.parser')
    description_div = soup.find('div', class_='description-container')
    target_h2 = description_div.find('h2') if description_div else soup.find('h2')
    page_header = target_h2.get_text(strip=True) if target_h2 else "Описание товара"
    base_text = description_div.get_text(separator="\n", strip=True) if description_div else soup.body.get_text(separator="\n", strip=True)[:5000]
    tags_container = soup.find(class_='popular-tags-inner')
    tags_data = []
    if tags_container:
        for link in tags_container.find_all('a'):
            tag_url = urljoin(url, link.get('href')) if link.get('href') else None
            if tag_url: tags_data.append({'name': link.get_text(strip=True), 'url': tag_url})
    return base_text, tags_data, page_header, None

def generate_ai_content_blocks(client, base_text, tag_name, forced_header, num_blocks=5, seo_words=None):
    if not base_text: return ["Error: No base text"] * 5
    seo_words = seo_words or []
    buckets = [[] for _ in range(5)]
    if seo_words and num_blocks > 0:
        for i, word in enumerate(seo_words): buckets[i % num_blocks].append(word)
    vocab_strs = [", ".join(b) if b else "None" for b in buckets]
    system_instruction = "You are a Senior Editor for a B2B industrial marketplace. Your goal is to write natural, professional, and idiomatic Russian text. Output raw HTML only."
    h_tag_instruction = f"1. Header: <h2>{forced_header}</h2> (Use this EXACT text)."
    if num_blocks > 1: h_tag_instruction += " For blocks 2-N use <h3> tags with relevant technical themes."
    user_prompt = f"""INPUT DATA: Product: "{tag_name}"\nSource Info: \"\"\"{base_text[:3000]}\"\"\"\nTASK: Generate {num_blocks} HTML sections. Separator: |||BLOCK_SEP|||\nSECTION STRUCTURE:\n{h_tag_instruction}\n2. <p> (3-5 sentences). Meaningful commercial/technical text.\n3. Short intro line.\n4. <ul><li>...</li></ul> (Specs/Benefits).\n5. <p> Summary.\nMANDATORY VOCABULARY TO INSERT:\nSection 1: {vocab_strs[0]}\nSection 2: {vocab_strs[1]}\nSection 3: {vocab_strs[2]}\nSection 4: {vocab_strs[3]}\nSection 5: {vocab_strs[4]}\nCRITICAL RULES FOR VOCABULARY:\n1. IDIOMATIC USE ONLY. 2. CHANGE PARTS OF SPEECH if needed. 3. NO "AI" FILLERS. 4. HIGHLIGHT: Wrap the inserted keyword (in its changed form) in <b> tags. 5. QUANTITY: Exactly 1 time per section.\nCONSTRAINTS: Language: Russian (Native Business). Max length: 800 chars/section. NO citations ([1]). NO Markdown."""
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(model="sonar-pro", messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}], temperature=0.75)
            content = response.choices[0].message.content
            content = re.sub(r'\[\d+\]', '', content)
            content = content.replace("```html", "").replace("```", "").strip()
            blocks = [b.strip() for b in content.split("|||BLOCK_SEP|||") if b.strip()]
            while len(blocks) < 5: blocks.append("")
            return blocks[:5]
        except: time.sleep(2)
    return ["API Error"] * 5

# ==========================================
# 7. UI TABS RESTRUCTURED
# ==========================================
tab_seo_main, tab_wholesale_main = st.tabs(["📊 SEO Анализ", "🏭 Оптовый генератор"])

with tab_seo_main:
    col_main, col_sidebar = st.columns([65, 35])
    with col_main:
        st.title("SEO Анализатор")
        if st.button("🧹 Обновить словари (Кэш)", key="clear_cache_btn"):
            st.cache_data.clear(); st.rerun()

        my_input_type = st.radio("Тип страницы", ["Релевантная страница на вашем сайте", "Исходный код страницы или текст", "Без страницы"], horizontal=True, label_visibility="collapsed", key="my_page_source_radio")
        if my_input_type == "Релевантная страница на вашем сайте": st.text_input("URL страницы", placeholder="https://site.ru/catalog/tovar", label_visibility="collapsed", key="my_url_input")
        elif my_input_type == "Исходный код страницы или текст": st.text_area("Исходный код или текст", height=200, label_visibility="collapsed", placeholder="Вставьте HTML", key="my_content_input")

        st.markdown("### Поисковой запрос")
        st.text_input("Основной запрос", placeholder="Например: купить пластиковые окна", label_visibility="collapsed", key="query_input")
        
        st.markdown("### Поиск конкурентов")
        if st.session_state.get('force_radio_switch'):
            st.session_state["competitor_source_radio"] = "Список url-адресов ваших конкурентов"
            st.session_state['force_radio_switch'] = False

        source_type_new = st.radio("Источник", ["Поиск через API Arsenkin (TOP-30)", "Список url-адресов ваших конкурентов"], horizontal=True, label_visibility="collapsed", key="competitor_source_radio")
        filter_help = """✅ ЕСЛИ ГАЛОЧКА СТОИТ: Скрипт берет 30 сайтов, находит среди них слабые/нерелевантные и ВЫКИДЫВАЕТ их. В анализ попадают только сильные.\n⬜ ЕСЛИ ГАЛОЧКИ НЕТ: Скрипт берет просто Топ-10 (или 20) сайтов по порядку из выдачи."""
        use_smart_filter = st.checkbox("⚡ Исключать нерелевантных конкурентов (Умный фильтр)", value=True, help=filter_help, key="cb_smart_filter")
        source_type = "API" if "API" in source_type_new else "Ручной список"
        
        if source_type == "Ручной список":
            if st.session_state.get('analysis_done'):
                col_reset, _ = st.columns([1, 4])
                with col_reset:
                    if st.button("🔄 Новый поиск (Сброс)", type="secondary", help="Сбросить все результаты и ввести новый список"):
                        for k in ['analysis_done', 'analysis_results', 'excluded_urls_auto', 'detected_anomalies', 'serp_trend_info', 'persistent_urls', 'naming_table_df', 'ideal_h1_result']:
                            if k in st.session_state: del st.session_state[k]
                        st.rerun()

            has_exclusions = st.session_state.get('excluded_urls_auto') and len(st.session_state.get('excluded_urls_auto')) > 5
            if has_exclusions:
                c_url_1, c_url_2 = st.columns(2)
                with c_url_1:
                    manual_val = st.text_area("✅ Активные конкуренты (Для анализа)", height=200, key="manual_urls_widget", value=st.session_state.get('persistent_urls', ""))
                    st.session_state['persistent_urls'] = manual_val
                with c_url_2:
                    st.text_area("🚫 Авто-исключенные (Вы можете вернуть их влево)", height=200, key="excluded_urls_widget_display", value=st.session_state.get('excluded_urls_auto', ""))
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
        d_score = results['my_score']['depth']; w_score = results['my_score']['width']
        w_color = "#2E7D32" if w_score >= 80 else ("#E65100" if w_score >= 50 else "#D32F2F")
        if 75 <= d_score <= 88: d_color = "#2E7D32"; d_status = "ИДЕАЛ (Топ)"
        elif 88 < d_score <= 100: d_color = "#D32F2F"; d_status = "ПЕРЕСПАМ (Риск)"
        elif 55 <= d_score < 75: d_color = "#F9A825"; d_status = "Средняя"
        else: d_color = "#D32F2F"; d_status = "Низкая"

        st.success("Анализ готов!")
        st.markdown("""<style>details > summary { list-style: none; } details > summary::-webkit-details-marker { display: none; } .details-card { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; margin-bottom: 10px; overflow: hidden; transition: all 0.2s ease; } .details-card:hover { box-shadow: 0 2px 5px rgba(0,0,0,0.05); border-color: #d1d5db; } .card-summary { padding: 12px 15px; cursor: pointer; font-weight: 700; font-size: 15px; color: #111827; display: flex; justify-content: space-between; align-items: center; background-color: #ffffff; } .card-summary:hover { background-color: #f3f4f6; } .card-content { padding: 15px; border-top: 1px solid #e9ecef; font-size: 14px; color: #374151; line-height: 1.6; background-color: #fcfcfc; } .count-tag { background: #e5e7eb; color: #374151; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; min-width: 25px; text-align: center; } .arrow-icon { font-size: 10px; margin-right: 8px; color: #9ca3af; transition: transform 0.2s; } details[open] .arrow-icon { transform: rotate(90deg); color: #277EFF; }</style>""", unsafe_allow_html=True)
        st.markdown(f"""<div style='display: flex; gap: 20px; flex-wrap: wrap;'><div style='flex: 1; background:{LIGHT_BG_MAIN}; padding:15px; border-radius:8px; border-left: 5px solid {w_color};'><div style='font-size: 12px; color: #666;'>ШИРИНА (Охват тем)</div><div style='font-size: 24px; font-weight: bold; color: {w_color};'>{w_score}/100</div></div><div style='flex: 1; background:{LIGHT_BG_MAIN}; padding:15px; border-radius:8px; border-left: 5px solid {d_color};'><div style='font-size: 12px; color: #666;'>ГЛУБИНА (Цель: ~80)</div><div style='font-size: 24px; font-weight: bold; color: {d_color};'>{d_score}/100 <span style='font-size:14px; font-weight:normal;'>({d_status})</span></div></div></div><br>""", unsafe_allow_html=True)

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
                    st.markdown(f"**⛔ Стоп-слова**"); st.markdown(f"Исключено: **{count_excluded}**"); st.caption("Эти слова автоматически удалены.")
                with cs2:
                    st.text_area("hidden_label", height=100, key="sensitive_words_input_final_widget", label_visibility="collapsed", placeholder="Слова для исключения...", value=current_text_value)
                    if st.button("🔄 Обновить фильтр", type="primary", use_container_width=True):
                        raw_input = st.session_state.get("sensitive_words_input_final_widget", "") # FIXED KEY
                        st.session_state['sensitive_words_input_final'] = raw_input
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
                        st.toast("Фильтр обновлен!", icon="✅"); time.sleep(0.5); st.rerun()

        high = results.get('missing_semantics_high', []); low = results.get('missing_semantics_low', [])
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
                    example_name = res_ideal[0]; report_list = res_ideal[1]
                    formula_str = "Формула не определена"
                    for line in report_list:
                        if "структура" in line or "Схема" in line:
                            formula_str = line.replace("**Самая частая структура:**", "").replace("**Схема:**", "").strip(); break
                    with st.container(border=True):
                        st.markdown("#### 🧪 Идеальная формула названия"); st.info(f"**{formula_str}**", icon="🧩"); st.markdown(f"**Пример генерации:** _{example_name}_")
            
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
            
            should_filter = st.session_state.get("cb_smart_filter", True)
            
            if should_filter:
                bad_urls_set = set(item['url'] for item in bad_urls_dicts)
                clean_data_pool = [d for d in data_for_graph if d['url'] not in bad_urls_set]
                if "API" in current_source_val: final_clean_data = clean_data_pool[:user_target_top_n]
                else: final_clean_data = clean_data_pool
                if bad_urls_dicts:
                    st.session_state['detected_anomalies'] = bad_urls_dicts
                    excluded_list = [item['url'] for item in bad_urls_dicts]
                    st.session_state['excluded_urls_auto'] = "\n".join(excluded_list)
                    st.session_state['persistent_urls'] = "\n".join([d['url'] for d in final_clean_data])
                    st.toast(f"🧹 Авто-фильтр: Исключено {len(bad_urls_dicts)} слабых сайтов.", icon="🗑️")
                else:
                    st.session_state['persistent_urls'] = "\n".join([d['url'] for d in final_clean_data])
                    if 'excluded_urls_auto' in st.session_state: del st.session_state['excluded_urls_auto']
            else:
                clean_data_pool = data_for_graph
                if "API" in current_source_val: final_clean_data = clean_data_pool[:user_target_top_n]
                else: final_clean_data = clean_data_pool
                st.session_state['persistent_urls'] = "\n".join([d['url'] for d in final_clean_data])
                if 'excluded_urls_auto' in st.session_state: del st.session_state['excluded_urls_auto']
                if 'detected_anomalies' in st.session_state: del st.session_state['detected_anomalies']
                st.toast(f"🛡️ Фильтр выключен. Взяты топ-{len(final_clean_data)} сайтов.", icon="ℹ️")

            final_clean_targets = [{'url': d['url'], 'pos': d['pos']} for d in final_clean_data]
            results_final = calculate_metrics(final_clean_data, my_data, settings, my_serp_pos, final_clean_targets)
            st.session_state.analysis_results = results_final
            naming_df = calculate_naming_metrics(final_clean_data, my_data, settings)
            st.session_state.naming_table_df = naming_df 
            st.session_state.ideal_h1_result = analyze_ideal_name(final_clean_data)
            st.session_state.analysis_done = True
            
            if "API" in current_source_val and 'full_graph_data' in st.session_state: df_rel_check = st.session_state['full_graph_data']
            else: df_rel_check = st.session_state.analysis_results['relevance_top']
            
            words_to_check = [x['word'] for x in results_final.get('missing_semantics_high', [])]
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
            st.rerun()

with tab_wholesale_main:
    st.header("🏭 Единый генератор контента")
    cat_products = st.session_state.get('categorized_products', [])
    cat_services = st.session_state.get('categorized_services', [])
    structure_keywords = cat_products + cat_services
    count_struct = len(structure_keywords)

    if 'auto_tags_words' in st.session_state and st.session_state.auto_tags_words:
         tags_list_source = st.session_state.auto_tags_words; promo_list_source = st.session_state.auto_promo_words
    else:
         if count_struct > 0:
            if count_struct < 10: tags_list_source = structure_keywords; promo_list_source = []
            elif count_struct < 30: mid = math.ceil(count_struct / 2); tags_list_source = structure_keywords[:mid]; promo_list_source = structure_keywords[mid:]
            else: part = math.ceil(count_struct / 3); tags_list_source = structure_keywords[:part]; promo_list_source = structure_keywords[part:part*2]
         else: tags_list_source = []; promo_list_source = []
    
    sidebar_default_text = ""
    if count_struct >= 30 and 'auto_tags_words' not in st.session_state: part = math.ceil(count_struct / 3); sidebar_default_text = "\n".join(structure_keywords[part*2:])
    tags_default_text = ", ".join(tags_list_source); promo_default_text = ", ".join(promo_list_source)
    cat_dimensions = st.session_state.get('categorized_dimensions', []); tech_context_default = ", ".join(cat_dimensions) if cat_dimensions else ""
    cat_commercial = st.session_state.get('categorized_commercial', []); cat_general = st.session_state.get('categorized_general', [])
    cat_geo = st.session_state.get('categorized_geo', []); text_context_list_raw = cat_commercial + cat_general
    text_context_default = ", ".join(text_context_list_raw); geo_context_default = ", ".join(cat_geo)
    auto_check_text = bool(text_context_list_raw); auto_check_tags = bool(tags_list_source); auto_check_tables = bool(cat_dimensions)
    auto_check_promo = bool(promo_list_source); auto_check_sidebar = bool(sidebar_default_text.strip()); auto_check_geo = bool(cat_geo)

    with st.container(border=True):
        st.subheader("1. Источник и Доступы")
        col_source, col_key = st.columns([3, 1])
        use_manual_html = st.checkbox("📝 Вставить HTML код страницы", key="cb_manual_html_mode", value=False)
        with col_source:
            if use_manual_html: manual_html_source = st.text_area("Исходный код страницы (HTML)", height=200, placeholder="<html>...</html>", help="Скопируйте сюда исходный код страницы."); main_category_url = None
            else: main_category_url = st.text_input("URL Категории", placeholder="https://site.ru/catalog/...", help="Скрипт соберет товары с этой страницы"); manual_html_source = None
        with col_key:
            default_key = st.session_state.get('pplx_key_cache', "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")
            pplx_api_key = st.text_input("AI API Key", value=default_key, type="password").strip()
            if pplx_api_key: st.session_state.pplx_key_cache = pplx_api_key

    st.subheader("2. Какие блоки генерируем?")
    col_ch1, col_ch2, col_ch3, col_ch4, col_ch5, col_ch6 = st.columns(6)
    with col_ch1: use_text = st.checkbox("🤖 AI Тексты", value=auto_check_text)
    with col_ch2: use_tags = st.checkbox("🏷️ Теги", value=auto_check_tags)
    with col_ch3: use_tables = st.checkbox("🧩 Таблицы", value=auto_check_tables)
    with col_ch4: use_promo = st.checkbox("🔥 Промо", value=auto_check_promo)
    with col_ch5: use_sidebar = st.checkbox("📑 Сайдбар", value=auto_check_sidebar)
    with col_ch6: use_geo = st.checkbox("🌍 Гео-блок", value=auto_check_geo)

    global_tags_list = []; global_promo_list = []; global_sidebar_list = []; global_geo_list = []
    tags_file_content = ""; table_prompts = []; df_db_promo = None; promo_title = "Рекомендуем"; sidebar_content = ""
    text_context_final_list = []; tech_context_final_str = ""; num_text_blocks_val = 5 

    if any([use_text, use_tags, use_tables, use_promo, use_sidebar, use_geo]):
        st.subheader("3. Настройки модулей")
        if use_text:
            with st.container(border=True):
                st.markdown("#### 🤖 1. AI Тексты")
                col_txt1, col_txt2 = st.columns([1, 4])
                with col_txt1: num_text_blocks_val = st.selectbox("Кол-во блоков", [1, 2, 3, 4, 5], index=4, key="sb_num_blocks")
                with col_txt2: ai_words_input = st.text_area("Слова для внедрения (Коммерция + Общие)", value=text_context_default, height=100, key="ai_text_context_editable"); text_context_final_list = [x.strip() for x in re.split(r'[,\n]+', ai_words_input) if x.strip()]

        if use_tags:
            with st.container(border=True):
                st.markdown("#### 🏷️ 2. Теги")
                kws_input_tags = st.text_area("Список (Товары + Услуги) - через запятую", value=tags_default_text, height=100, key="kws_tags_auto")
                global_tags_list = [x.strip() for x in re.split(r'[,\n]+', kws_input_tags) if x.strip()]
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
            dims_str = " ".join(dimensions_list).lower(); gen_str = " ".join(general_list).lower()
            full_context = f"{dims_str} {gen_str} {query_lower}"
            has_sizes_signal = (len(dimensions_list) > 0 or bool(re.search(r'\d+[xх*]\d+', full_context)) or any(x in full_context for x in ['размер', 'габарит', 'толщин', 'диаметр', 'раскрой', 'вес', 'масс']))
            has_gost_signal = any(x in full_context for x in ['гост', 'din', 'aisi', 'astm', 'ту ', 'стандарт'])
            has_grade_signal = any(x in full_context for x in ['марк', 'сплав', 'сталь', 'ст.', 'материал', 'химич', 'состав'])
            priority_stack = []
            if has_grade_signal: priority_stack.append("Марки и сплавы")
            if has_sizes_signal: priority_stack.append("Таблица размеров")
            if has_gost_signal: priority_stack.append("ГОСТы и стандарты")
            if "хим" in full_context and "состав" in full_context:
                 if "Марки и сплавы" in priority_stack: idx = priority_stack.index("Марки и сплавы"); priority_stack.insert(idx+1, "Химический состав")
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
                raw_query = st.session_state.get('query_input', ''); found_dims = st.session_state.get('categorized_dimensions', []); found_general = st.session_state.get('categorized_general', [])
                col_ctx, col_cnt = st.columns([3, 1]) 
                with col_ctx: tech_context_final_str = st.text_area("Контекст для таблиц (Марки, ГОСТ, Размеры)", value=tech_context_default, height=68, key="table_context_editable")
                with col_cnt: cnt = st.selectbox("Кол-во таблиц", [1, 2, 3, 4, 5], index=1, key="num_tbl_vert_select")
                smart_headers_list = generate_context_aware_headers(cnt, raw_query, found_dims, found_general)
                table_presets = ["Технические характеристики", "Свойства", "Параметры изделия", "Общее описание", "Таблица размеров", "Сортамент", "Химический состав", "Физические свойства", "Механические свойства", "Марки и сплавы", "Состав материала", "ГОСТы и стандарты", "Техническая документация", "Требования ГОСТ", "Назначение", "Сферы использования", "Условия эксплуатации", "Где используется", "Классификация", "Модификации", "Аналоги", "Сравнение моделей", "Разновидности"]
                cols = st.columns(cnt)
                for i, col in enumerate(cols):
                    with col:
                        st.caption(f"**Таблица {i+1}**")
                        suggested_topic = smart_headers_list[i]
                        try: default_idx = table_presets.index(suggested_topic)
                        except: default_idx = 0
                        if st.checkbox("Свой заголовок", key=f"cb_tbl_manual_{i}"):
                            selected_topic = st.text_input(f"Название табл. {i+1}", value="", key=f"tbl_topic_custom_{i}", label_visibility="collapsed")
                            if not selected_topic.strip(): selected_topic = "Характеристики" 
                        else: selected_topic = st.selectbox(f"Тема табл. {i+1}", table_presets, index=default_idx, key=f"tbl_topic_select_{i}", label_visibility="collapsed")
                        table_prompts.append(selected_topic)

        if use_promo:
            with st.container(border=True):
                st.markdown("#### 🔥 4. Промо-блок")
                kws_input_promo = st.text_area("Список (Товары + Услуги) - через запятую", value=promo_default_text, height=100, key="kws_promo_auto")
                global_promo_list = [x.strip() for x in re.split(r'[,\n]+', kws_input_promo) if x.strip()]
                col_p1, col_p2 = st.columns([1, 2])
                with col_p1:
                    promo_presets = ["Смотрите также", "Похожие товары", "Вас может заинтересовать", "Рекомендуем", "Другие предложения", "Вам может пригодиться", "Также в этом разделе", "С этим товаром покупают", "Часто покупают вместе", "Сопутствующие товары", "Хиты продаж", "Выбор покупателей", "Лидеры спроса", "Популярное сейчас", "Топ товаров категории", "Лучшая цена", "Спецпредложения", "Успейте заказать", "Не забудьте добавить", "Вы недавно смотрели"]
                    raw_query = st.session_state.get('query_input', '').lower(); comm_words = st.session_state.get('categorized_commercial', [])
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
                    if st.checkbox("Ввести свой заголовок", key="cb_custom_header"): promo_title = st.text_input("Ваш заголовок", placeholder="Смотрите также", key="pr_tit_vert")
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
        st.session_state.gen_result_df = None; st.session_state.unified_excel_data = None
        status_box = st.status("🛠️ Подготовка данных...", expanded=True)
        final_data = [] 
        tags_map = {}; all_tags_links = []
        if use_tags:
            if tags_file_content: s_io = io.StringIO(tags_file_content); all_tags_links = [l.strip() for l in s_io.readlines() if l.strip()]
            elif os.path.exists("data/links_base.txt"):
                with open("data/links_base.txt", "r", encoding="utf-8") as f: all_tags_links = [l.strip() for l in f.readlines() if l.strip()]
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
            if sidebar_content: s_io = io.StringIO(sidebar_content); all_menu_urls = [l.strip() for l in s_io.readlines() if l.strip()]
            elif os.path.exists("data/menu_structure.txt"):
                with open("data/menu_structure.txt", "r", encoding="utf-8") as f: all_menu_urls = [l.strip() for l in f.readlines() if l.strip()]

        missing_words_log = set()
        if use_tags:
            for kw in global_tags_list:
                if kw not in tags_map: missing_words_log.add(kw)
        if use_promo:
            for kw in global_promo_list:
                tr = transliterate_text(kw).replace(' ', '-').replace('_', '-')
                roots = [tr]; 
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
                if any(char.isdigit() for char in w) or any(x in w.lower() for x in ['гост', 'тип', 'форма', 'мм', 'кг']): tech_additions.append(w)
            if tech_additions: tech_context_final_str += "\n" + ", ".join(tech_additions)
            status_box.markdown(f"""<div style="background-color: #FFF4E5; border-left: 5px solid #FF9800; padding: 15px; border-radius: 4px; margin-bottom: 15px; color: #663C00;"><strong>⚠️ Внимание: Часть ссылок не найдена</strong><br><span style="font-size: 0.9em;">Мы не нашли точного совпадения в структуре для: <b>{', '.join(missing_list)}</b>.<br>✅ <u>Они были перенесены в ТЗ для Нейросети (будут в тексте/таблицах).</u></span></div>""", unsafe_allow_html=True); time.sleep(2)

        target_pages = []; soup = None; current_base_url = main_category_url if main_category_url else "http://localhost"
        try:
            if use_manual_html: status_box.write("📂 Обрабатываем HTML код..."); soup = BeautifulSoup(manual_html_source, 'html.parser')
            else:
                status_box.write(f"🕵️ Сканируем категорию: {main_category_url}"); session = requests.Session(); retry = Retry(connect=3, read=3, redirect=3, backoff_factor=0.5); adapter = HTTPAdapter(max_retries=retry); session.mount('http://', adapter); session.mount('https://', adapter); headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                r = session.get(main_category_url, headers=headers, timeout=30, verify=False)
                if r.status_code == 200: soup = BeautifulSoup(r.text, 'html.parser')
                else: status_box.error(f"Ошибка доступа: {r.status_code}"); st.stop()
            if soup:
                tags_container = soup.find(class_='popular-tags-inner')
                if tags_container:
                    for link in tags_container.find_all('a'):
                        href = link.get('href'); name = link.get_text(strip=True)
                        if href and name: full_url = urljoin(current_base_url, href); target_pages.append({'url': full_url, 'name': name})
                if not target_pages:
                    status_box.warning("Теги товаров не найдены (класс .popular-tags-inner). Генерируем для одной страницы."); h1 = soup.find('h1'); name = h1.get_text(strip=True) if h1 else "Товар"; target_pages.append({'url': current_base_url, 'name': name})
        except Exception as e: status_box.error(f"Критическая ошибка: {e}"); st.stop()
            
        urls_to_fetch_names = set(); promo_items_pool = []
        if use_tags:
            for kw, matches in tags_map.items(): urls_to_fetch_names.update(matches)
        if use_promo:
            used_urls = set()
            for kw in global_promo_list:
                if kw in missing_words_log: continue
                tr = transliterate_text(kw).replace(' ', '-').replace('_', '-')
                roots = [tr]; 
                if len(tr) > 5: roots.extend([tr[:-1], tr[:-2]])
                matches = []
                for u in p_img_map.keys():
                    if any(r in u for r in roots): matches.append(u)
                for m in matches:
                    if m not in used_urls: urls_to_fetch_names.add(m); promo_items_pool.append({'url': m, 'img': p_img_map[m]}); used_urls.add(m)
        sidebar_matched_urls = []
        if use_sidebar:
            if global_sidebar_list:
                for kw in global_sidebar_list:
                    if kw in missing_words_log: continue
                    tr = transliterate_text(kw).replace(' ', '-').replace('_', '-'); roots = [tr]; 
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
                done_cnt = 0; prog_fetch = status_box.progress(0)
                for future in concurrent.futures.as_completed(future_to_url):
                    u_res, name_res = future.result(); norm_key = u_res.rstrip('/')
                    if name_res: url_name_cache[norm_key] = name_res
                    else: slug = norm_key.split('/')[-1]; url_name_cache[norm_key] = force_cyrillic_name_global(slug)
                    done_cnt += 1; prog_fetch.progress(done_cnt / len(urls_to_fetch_names))
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
                        curr[part]['__url__'] = url; cache_key = url.rstrip('/'); curr[part]['__name__'] = url_name_cache.get(cache_key, force_cyrillic_name_global(part))
                    curr = curr[part]
            def render_tree_internal(node, level=1):
                html = ""; keys = sorted([k for k in node.keys() if not k.startswith('__')])
                for key in keys:
                    child = node[key]; name = child.get('__name__', force_cyrillic_name_global(key)); url = child.get('__url__'); has_children = any(k for k in child.keys() if not k.startswith('__'))
                    if level == 1:
                        html += '<li class="level-1-header">\n'
                        if has_children: html += f'    <span class="dropdown-toggle">{name}</span>\n    <ul class="collapse-menu list-unstyled">\n' + render_tree_internal(child, level=2) + '    </ul>\n'
                        else: target = url if url else "#"; html += f'    <a href="{target}">{name}</a>\n'
                        html += '</li>\n'
                    elif level == 2:
                        if has_children: html += '<li class="level-2-header">\n' + f'    <span class="dropdown-toggle">{name}</span>\n    <ul class="collapse-menu list-unstyled">\n' + render_tree_internal(child, level=3) + '    </ul>\n'
                        else: target = url if url else "#"; html += f'<li class="level-2-link-special"><a href="{target}">{name}</a></li>\n'
                    elif level >= 3: target = url if url else "#"; html += f'<li class="level-3-link"><a href="{target}">{name}</a></li>\n'
                return html
            inner_html = render_tree_internal(tree, level=1); full_sidebar_code = f"""<div class="page-content-with-sidebar"><button id="mobile-menu-toggle" class="menu-toggle-button">☰</button><div class="sidebar-wrapper"><nav id="sidebar-menu"><ul class="list-unstyled components">{inner_html}</ul></nav></div></div>"""

        client = None
        if openai and (use_text or use_tables or use_geo): client = openai.OpenAI(api_key=pplx_api_key, base_url="https://api.perplexity.ai")

        progress_bar = status_box.progress(0); total_steps = len(target_pages)
        for idx, page in enumerate(target_pages):
            base_text_raw, tags_on_page, real_header_h2, err = get_page_data_for_gen(page['url'])
            header_for_ai = real_header_h2 if real_header_h2 else page['name']
            row_data = {'Page URL': page['url'], 'Product Name': header_for_ai}
            for k, v in STATIC_DATA_GEN.items(): row_data[k] = v
            current_page_seo_words = list(text_context_final_list)
            
            tags_html_parts = []
            if use_tags:
                html_collector = []
                for kw in global_tags_list:
                    if kw not in tags_map: continue 
                    urls = tags_map[kw]; valid_urls = [u for u in urls if u.rstrip('/') != page['url'].rstrip('/')]
                    if valid_urls:
                        selected_url = random.choice(valid_urls); cache_key = selected_url.rstrip('/'); nm = url_name_cache.get(cache_key, kw); html_collector.append(f'<a href="{selected_url}" class="tag-link">{nm}</a>')
                    else:
                        if kw not in current_page_seo_words: current_page_seo_words.append(kw)
                if html_collector: tags_html_parts = ['<div class="popular-tags">'] + html_collector + ['</div>']; row_data['Tags HTML'] = "\n".join(tags_html_parts)
                else: row_data['Tags HTML'] = ""

            if use_promo:
                candidates = [p for p in promo_items_pool if p['url'].rstrip('/') != page['url'].rstrip('/')]
                random.shuffle(candidates); selected_promo = candidates
                if selected_promo:
                    promo_html = f'<div class="promo-section"><h3>{promo_title}</h3><div class="promo-grid" style="display: flex; flex-wrap: nowrap; gap: 15px; overflow-x: auto; padding-bottom: 15px; scrollbar-width: thin;">'
                    for item in selected_promo:
                        p_url = item['url']; p_img = item['img']; cache_key = p_url.rstrip('/'); p_name = url_name_cache.get(cache_key, "Товар")
                        promo_html += f'<div class="promo-card" style="min-width: 220px; width: 220px; flex-shrink: 0; border: 1px solid #eee; padding: 10px; border-radius: 5px; text-align: center;"><a href="{p_url}" style="text-decoration: none; color: #333;"><div style="height: 150px; overflow: hidden; display: flex; align-items: center; justify-content: center; margin-bottom: 10px;"><img src="{p_img}" alt="{p_name}" style="max-height: 100%; max-width: 100%; object-fit: contain;"></div><div style="font-size: 13px; font-weight: bold; line-height: 1.3;">{p_name}</div></a></div>'
                    promo_html += '</div></div>'; row_data['Promo HTML'] = promo_html
                else: row_data['Promo HTML'] = ""

            if use_text and client:
                try:
                    blocks = generate_ai_content_blocks(client, base_text=base_text_raw if base_text_raw else "", tag_name=page['name'], forced_header=header_for_ai, num_blocks=num_text_blocks_val, seo_words=current_page_seo_words)
                    row_data['Text_Block_1'] = blocks[0]; row_data['Text_Block_2'] = blocks[1]; row_data['Text_Block_3'] = blocks[2]; row_data['Text_Block_4'] = blocks[3]; row_data['Text_Block_5'] = blocks[4]
                except Exception as e: row_data['Text_Error'] = str(e)

            if use_tables and client:
                for t_i, t_topic in enumerate(table_prompts):
                    sys_p_table = "You are an expert metallurgist and data analyst. Output ONLY raw HTML <table>. No markdown."; context_hint = ""
                    if tech_context_final_str: context_hint = f"Используй технические данные (марки, ГОСТы): {tech_context_final_str}."
                    usr_p_table = f"""Задача: Составь подробную техническую таблицу для товара "{header_for_ai}".\nТема таблицы: {t_topic}.\n{context_hint}\nТРЕБОВАНИЯ:\n1. Только реальные технические данные.\n2. HTML <table>...</table>.\n3. Без Markdown."""
                    try:
                        resp = client.chat.completions.create(model="sonar-pro", messages=[{"role": "system", "content": sys_p_table}, {"role": "user", "content": usr_p_table}], temperature=0.4)
                        raw_html = resp.choices[0].message.content; clean_html = raw_html.replace("```html", "").replace("```", "").strip(); clean_html = re.sub(r'\[\d+\]', '', clean_html)
                        soup_table = BeautifulSoup(clean_html, 'html.parser'); table_tag = soup_table.find('table')
                        if table_tag:
                            table_tag['style'] = "border-collapse: collapse; width: 100%; border: 2px solid black;"
                            for cell in table_tag.find_all(['th', 'td']): cell['style'] = "border: 2px solid black; padding: 5px;"
                            final_table_html = str(table_tag)
                        else: final_table_html = clean_html
                        row_data[f'Table_{t_i+1}_HTML'] = final_table_html
                    except Exception as e: row_data[f'Table_{t_i+1}_HTML'] = f"Error: {e}"

            if use_sidebar: row_data['Sidebar HTML'] = full_sidebar_code

            if use_geo and client and global_geo_list:
                selected_cities = global_geo_list
                if len(selected_cities) > 20: selected_cities = random.sample(global_geo_list, 20)
                cities_str = ", ".join(selected_cities)
                geo_prompt = f"""Task: Write a short paragraph <p> about delivery options for "{header_for_ai}" to {cities_str}. Output HTML <p> only."""
                try:
                    resp_geo = client.chat.completions.create(model="sonar-pro", messages=[{"role": "system", "content": "You are a logistic summary generator."}, {"role": "user", "content": geo_prompt}], temperature=0.4)
                    clean_geo = resp_geo.choices[0].message.content.replace("```html", "").replace("```", "").strip()
                    row_data['IP_PROP4819'] = clean_geo
                except Exception as e: row_data['IP_PROP4819'] = f"Error: {e}"

            final_data.append(row_data); progress_bar.progress((idx + 1) / total_steps)

        df_result = pd.DataFrame(final_data); st.session_state.gen_result_df = df_result 
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer: df_result.to_excel(writer, index=False)
        st.session_state.unified_excel_data = buffer.getvalue()
        status_box.update(label="✅ Конвейер завершен! Данные готовы.", state="complete", expanded=False)

    if st.session_state.get('unified_excel_data') is not None:
        st.success("Файл успешно сгенерирован!")
        st.download_button(label="📥 СКАЧАТЬ ЕДИНЫЙ EXCEL", data=st.session_state.unified_excel_data, file_name="unified_content_gen.xlsx", mime="application/vnd.ms-excel", key="btn_dl_unified")

    if 'gen_result_df' in st.session_state and st.session_state.gen_result_df is not None:
        st.markdown("---"); st.header("👀 Предпросмотр результата"); df = st.session_state.gen_result_df
        page_options = df['Product Name'].tolist(); selected_page_name = st.selectbox("Выберите страницу для просмотра:", page_options, key="preview_selector")
        row = df[df['Product Name'] == selected_page_name].iloc[0]
        has_text = any((f'Text_Block_{i}' in row and pd.notna(row[f'Text_Block_{i}']) and str(row[f'Text_Block_{i}']).strip()) for i in range(1, 6))
        table_cols = [c for c in df.columns if 'Table_' in c and '_HTML' in c and pd.notna(row[c]) and str(row[c]).strip()]; has_tables = len(table_cols) > 0
        has_tags = 'Tags HTML' in row and pd.notna(row['Tags HTML']) and str(row['Tags HTML']).strip()
        has_sidebar = 'Sidebar HTML' in row and pd.notna(row['Sidebar HTML']) and str(row['Sidebar HTML']).strip()
        has_geo = 'IP_PROP4819' in row and pd.notna(row['IP_PROP4819']) and str(row['IP_PROP4819']).strip()
        has_promo = 'Promo HTML' in row and pd.notna(row['Promo HTML']) and str(row['Promo HTML']).strip()
        has_visual = has_tags or has_sidebar or has_geo or has_promo
        active_tabs = []
        if has_text: active_tabs.append("📝 Текст")
        if has_tables: active_tabs.append("🧩 Таблицы")
        if has_visual: active_tabs.append("🎨 Визуал")
        st.markdown("""<style>.preview-box { border: 1px solid #e0e0e0; padding: 20px; border-radius: 8px; background: #fff; margin-bottom: 20px; } .preview-label { font-size: 12px; font-weight: bold; color: #888; text-transform: uppercase; margin-bottom: 5px; } .popular-tags { display: flex; flex-wrap: wrap; gap: 8px; } .tag-link { background: #f0f2f5; color: #333; padding: 5px 10px; border-radius: 4px; text-decoration: none; font-size: 13px; } table { width: 100%; border-collapse: collapse; font-size: 14px; } th, td { border: 1px solid #ddd; padding: 8px; text-align: left; } th { background-color: #f2f2f2; font-weight: bold; } .sidebar-wrapper ul { list-style-type: none; padding-left: 10px; } .level-1-header { font-weight: bold; margin-top: 10px; color: #277EFF; } .promo-grid { display: flex !important; flex-wrap: wrap; gap: 10px; } .promo-card { width: 23%; box-sizing: border-box; } .promo-card img { max-width: 100%; height: auto; }</style>""", unsafe_allow_html=True)
        if not active_tabs: st.warning("⚠️ Контент пуст.")
        else:
            tabs_objects = st.tabs(active_tabs); tabs_map = dict(zip(active_tabs, tabs_objects))
            if "📝 Текст" in tabs_map:
                with tabs_map["📝 Текст"]:
                    st.subheader(row['Product Name'])
                    for i in range(1, 6):
                        col_key = f'Text_Block_{i}'
                        if col_key in row and pd.notna(row[col_key]):
                            content = str(row[col_key]).strip()
                            if content:
                                with st.container(): st.caption(f"Блок {i}"); st.markdown(f"<div class='preview-box'>{content}</div>", unsafe_allow_html=True)
            if "🧩 Таблицы" in tabs_map:
                with tabs_map["🧩 Таблицы"]:
                    for t_col in table_cols:
                        content = row[t_col]; clean_title = t_col.replace('_HTML', '').replace('_', ' '); st.caption(clean_title); st.markdown(content, unsafe_allow_html=True)
            if "🎨 Визуал" in tabs_map:
                with tabs_map["🎨 Визуал"]:
                    if has_promo: st.markdown('<div class="preview-label">Промо-блок (Рекомендации)</div>', unsafe_allow_html=True); st.markdown(f"<div class='preview-box'>{row['Promo HTML']}</div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        if has_tags: st.markdown('<div class="preview-label">Теги</div>', unsafe_allow_html=True); st.markdown(f"<div class='preview-box'>{row['Tags HTML']}</div>", unsafe_allow_html=True)
                        if has_geo: st.markdown('<div class="preview-label">Гео-блок</div>', unsafe_allow_html=True); st.markdown(f"<div class='preview-box'>{row['IP_PROP4819']}</div>", unsafe_allow_html=True)
                    with c2:
                        if has_sidebar: st.markdown('<div class="preview-label">Сайдбар</div>', unsafe_allow_html=True); st.markdown(f"<div class='preview-box' style='max-height: 400px; overflow-y: auto;'>{row['Sidebar HTML']}</div>", unsafe_allow_html=True)

            # --- ВИЗУАЛ ---
            if "🎨 Визуал" in tabs_map:
                with tabs_map["🎨 Визуал"]:
                    # Вывод Промо
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



