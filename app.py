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
from google import genai
import os
import requests
proxy_url = "http://QYnojH:Uekp4k@196.18.3.35:8000" 

os.environ["http_proxy"] = proxy_url
os.environ["https_proxy"] = proxy_url

try:
    my_ip = requests.get("https://api.ipify.org", timeout=5).text
    st.info(f"🕵️ ВАШ IP ДЛЯ СКРИПТА: {my_ip}")
except Exception as e:
    st.error(f"❌ Прокси не работает: {e}")
    
import random
import streamlit.components.v1 as components
import copy
import plotly.graph_objects as go
import pickle
import datetime

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

try:
    from google import genai
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
    if df_rel.empty:
        return

    # 1. ЖЕСТКАЯ ФИЛЬТРАЦИЯ: Оставляем только то, что > 0
    # Ваш сайт (позиция 0) удаляется из данных для графика
    df = df_rel[df_rel['Позиция'] > 0].copy()
    
    # Если после удаления вашего сайта таблица пуста - выходим
    if df.empty:
        return

    df = df.sort_values(by='Позиция')
    x_indices = np.arange(len(df))
    
    tick_links = []
    
    for _, row in df.iterrows():
        # Чистим имя домена
        raw_name = row['Домен'].replace(' (Вы)', '').strip()
        clean_domain = raw_name.replace('www.', '').split('/')[0]
        
        # Формат: "1. site.ru" (без #)
        label_text = f"{row['Позиция']}. {clean_domain}"
        
        # Обрезаем слишком длинные, но оставляем запас, так как шрифт теперь крупнее
        if len(label_text) > 25: label_text = label_text[:23] + ".."
        
        url_target = row.get('URL', f"https://{raw_name}")
        
        # Используем CSS-класс .chart-link вместо style="..." для работы hover
        link_html = f"<a href='{url_target}' target='_blank' class='chart-link'>{label_text}</a>"
        tick_links.append(link_html)

    # Метрики
    df['Total_Rel'] = (df['Ширина (балл)'] + df['Глубина (балл)']) / 2
    
    # Тренд
    z = np.polyfit(x_indices, df['Total_Rel'], 1)
    p = np.poly1d(z)
    df['Trend'] = p(x_indices)

    # 2. Создаем график
    fig = go.Figure()

    # --- ПАЛИТРА (Premium) ---
    COLOR_MAIN = '#4F46E5'  # Индиго
    COLOR_WIDTH = '#0EA5E9' # Голубой
    COLOR_DEPTH = '#E11D48' # Малиновый
    COLOR_TREND = '#15803d' # Зеленый (Forest Green)

    COMMON_CONFIG = dict(
        mode='lines+markers',
        line=dict(width=3, shape='spline'), 
        marker=dict(size=8, line=dict(width=2, color='white'), symbol='circle')
    )

    # 1. ОБЩАЯ
    fig.add_trace(go.Scatter(
        x=x_indices, y=df['Total_Rel'],
        name='Общая',
        line=dict(color=COLOR_MAIN, **COMMON_CONFIG['line']),
        marker=dict(color=COLOR_MAIN, **COMMON_CONFIG['marker']),
        mode='lines+markers'
    ))

    # 2. ШИРИНА
    fig.add_trace(go.Scatter(
        x=x_indices, y=df['Ширина (балл)'],
        name='Ширина',
        line=dict(color=COLOR_WIDTH, **COMMON_CONFIG['line']),
        marker=dict(color=COLOR_WIDTH, **COMMON_CONFIG['marker']),
        mode='lines+markers'
    ))

    # 3. ГЛУБИНА
    fig.add_trace(go.Scatter(
        x=x_indices, y=df['Глубина (балл)'],
        name='Глубина',
        line=dict(color=COLOR_DEPTH, **COMMON_CONFIG['line']),
        marker=dict(color=COLOR_DEPTH, **COMMON_CONFIG['marker']),
        mode='lines+markers'
    ))

    # 4. ТРЕНД
    fig.add_trace(go.Scatter(
        x=x_indices, y=df['Trend'],
        name='Тренд',
        line=dict(color=COLOR_TREND, **COMMON_CONFIG['line']),
        marker=dict(color=COLOR_TREND, **COMMON_CONFIG['marker']),
        mode='lines+markers',
        opacity=0.8
    ))

# 3. Настройка Layout (КОМПАКТНАЯ ВЕРСИЯ)
    fig.update_layout(
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02, # Легенда прямо над графиком
            xanchor="center", x=0.5,
            font=dict(size=12, color="#111827", family="Inter, sans-serif")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#F3F4F6',
            linecolor='#E5E7EB',
            tickmode='array',
            tickvals=x_indices,
            ticktext=tick_links, 
            
            tickfont=dict(size=11), # Чуть меньше шрифт подписей
            tickangle=-45, 
            
            fixedrange=True,
            dtick=1, 
            range=[-0.5, len(df) - 0.5], 
            automargin=False 
        ),
        yaxis=dict(
            range=[0, 115], 
            showgrid=True, 
            gridcolor='#F3F4F6', 
            gridwidth=1,
            zeroline=False,
            fixedrange=True
        ),
        # === ВОТ ТУТ МЕНЯЕМ РАЗМЕРЫ ===
        # l/r - бока, t - верх, b - низ (под подписи)
        margin=dict(l=10, r=10, t=30, b=110),
        
        hovermode="x unified",
        
        # Общая высота графика (было 550)
        height=400 
    )
    
    # use_container_width=True растягивает график на всю ширину страницы
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"rel_chart_{unique_key}")

def analyze_serp_anomalies(df_rel):
    """
    Анализирует таблицу релевантности (Версия v5 - Robust).
    Порог: 75% от лидера. Принудительная типизация.
    """
    if df_rel.empty:
        return [], [], {"type": "none", "msg": ""}

    # Исключаем "Ваш сайт" из расчетов эталона
    df = df_rel[~df_rel['Домен'].str.contains("\(Вы\)", na=False)].copy()
    
    if df.empty:
        return [], [], {"type": "none", "msg": ""}

    # Принудительно делаем числами (защита от сбоев)
    df['Ширина (балл)'] = pd.to_numeric(df['Ширина (балл)'], errors='coerce').fillna(0)
    df['Глубина (балл)'] = pd.to_numeric(df['Глубина (балл)'], errors='coerce').fillna(0)

    # Считаем средний балл
    df['Total'] = (df['Ширина (балл)'] + df['Глубина (балл)']) / 2
    
    # 1. ИЩЕМ ЛИДЕРА
    max_score = df['Total'].max()
    if max_score < 1: max_score = 1 # Защита от деления на 0
    
    # 2. ЖЕСТКИЙ ПОРОГ: 75% от лидера.
    # Если Лидер=100, порог=75. Все что < 75 - удаляем.
    threshold = max(max_score * 0.75, 40) 
    
    anomalies = []
    normal_urls = []
    
    debug_counts = 0
    
    for _, row in df.iterrows():
        # Достаем ссылку. Защита от пробелов.
        current_url = str(row.get('URL', '')).strip()
        if not current_url or current_url.lower() == 'nan':
             current_url = f"https://{row['Домен']}" 

        score = row['Total']
        
        # АНАЛИЗ
        if score < threshold:
            reason = f"Скор {int(score)} < {int(threshold)} (Лидер {int(max_score)})"
            anomalies.append({'url': current_url, 'reason': reason, 'score': score})
            debug_counts += 1
        else:
            normal_urls.append(current_url)

    # Уведомление с деталями
    if anomalies:
        st.toast(f"🗑️ Фильтр (Лидер {int(max_score)} / Порог {int(threshold)}). Исключено: {len(anomalies)}", icon="⚠️")
    else:
        # Если никого не исключили, пишем почему
        st.toast(f"✅ Все конкуренты ок. (Лидер {int(max_score)} / Порог {int(threshold)}). Мин. балл: {int(df['Total'].min())}", icon="ℹ️")
    
    # Тренд
    x = np.arange(len(df)); y = df['Total'].values
    slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
    trend_msg = "📉 Нормальный топ" if slope < -1 else ("📈 Перевернутый топ" if slope > 1 else "➡️ Ровный топ")

    return normal_urls, anomalies, {"type": "info", "msg": trend_msg}

@st.cache_data
def load_lemmatized_dictionaries():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "data")
    
    # Создаем множества
    sets = {
        "products": set(),
        "commercial": set(),
        "specs": set(),
        "geo": set(),
        "services": set(),
        "sensitive": set()
    }

    # Карта файлов
    files_map = {
        "metal_products.json": "products",
        "commercial_triggers.json": "commercial",
        "geo_locations.json": "geo",
        "services_triggers.json": "services",
        "tech_specs.json": "specs",
        "SENSITIVE_STOPLIST.json": "sensitive"
    }

    for filename, set_key in files_map.items():
        full_path = os.path.join(base_path, filename)
        if not os.path.exists(full_path):
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f) 
                
                words_bucket = []
                if isinstance(data, dict):
                    for cat_list in data.values():
                        words_bucket.extend(cat_list)
                elif isinstance(data, list):
                    words_bucket = data
                
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
                            if morph: 
                                sets[set_key].add(morph.parse(p)[0].normal_form.replace('ё', 'е'))
        except: pass

    # Возвращаем 6 наборов
    return sets["products"], sets["commercial"], sets["specs"], sets["geo"], sets["services"], sets["sensitive"]

def classify_semantics_with_api(words_list, yandex_key):
    # Распаковываем 6 словарей, которые вернула функция загрузки
    PRODUCTS_SET, COMM_SET, SPECS_SET, GEO_SET, SERVICES_SET, SENS_SET = load_lemmatized_dictionaries()
    
    # Объединяем стоп-слова из файла и из глобального списка в коде
    FULL_SENSITIVE = SENS_SET.union(SENSITIVE_STOPLIST)

    if 'debug_geo_count' not in st.session_state:
        st.session_state.debug_geo_count = len(GEO_SET)
    
    st.sidebar.info(f"Словари (из файлов):\n📦 Товары: {len(PRODUCTS_SET)}\n💰 Коммерция: {len(COMM_SET)}\n🛠️ Услуги: {len(SERVICES_SET)}\n🌍 Города: {len(GEO_SET)}")

    dim_pattern = re.compile(r'\d+(?:[\.\,]\d+)?\s?[хx\*×]\s?\d+', re.IGNORECASE)
    grade_pattern = re.compile(r'^([а-яa-z]{1,4}\-?\d+[а-яa-z0-9]*)$', re.IGNORECASE)
    
    categories = {'products': set(), 'services': set(), 'commercial': set(), 
                  'dimensions': set(), 'geo': set(), 'general': set(), 'sensitive': set()}
    
    for word in words_list:
        word_lower = word.lower()
        
        # 1. СТОП-СЛОВА
        is_sensitive = False
        if word_lower in FULL_SENSITIVE: is_sensitive = True
        else:
            for stop_w in FULL_SENSITIVE:
                if len(stop_w) > 3 and stop_w in word_lower: is_sensitive = True; break
        if is_sensitive: categories['sensitive'].add(word_lower); continue
        
        # Лемматизация
        lemma = word_lower
        if morph:
            p = morph.parse(word_lower)[0]
            lemma = p.normal_form

        # 2. РАЗМЕРЫ / ГОСТ
        if word_lower in SPECS_SET or lemma in SPECS_SET:
            categories['dimensions'].add(word_lower); continue
        if dim_pattern.search(word_lower) or grade_pattern.match(word_lower) or word_lower.isdigit():
            categories['dimensions'].add(word_lower); continue

        # 3. ТОВАРЫ (Улучшенная логика)
        if word_lower in PRODUCTS_SET or lemma in PRODUCTS_SET:
            categories['products'].add(word_lower); continue
        
        is_product_root = False
        for prod in PRODUCTS_SET:
            check_root = prod[:-1] if len(prod) > 4 else prod
            if len(check_root) > 3 and check_root in word_lower:
                categories['products'].add(word_lower)
                is_product_root = True
                break
        if is_product_root: continue

        # 4. ГЕО
        if lemma in GEO_SET or word_lower in GEO_SET:
            categories['geo'].add(word_lower); continue
        
        # 5. УСЛУГИ
        if lemma in SERVICES_SET or word_lower in SERVICES_SET:
             categories['services'].add(word_lower); continue
        if lemma.endswith('обработка') or lemma.endswith('изготовление') or lemma == "резка":
            categories['services'].add(word_lower); continue

        # 6. КОММЕРЦИЯ
        if lemma in COMM_SET or word_lower in COMM_SET:
            categories['commercial'].add(word_lower); continue
            
        # 7. ОБЩИЕ
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

# Current lists
if 'categorized_products' not in st.session_state: st.session_state.categorized_products = []
if 'categorized_services' not in st.session_state: st.session_state.categorized_services = []
if 'categorized_commercial' not in st.session_state: st.session_state.categorized_commercial = []
if 'categorized_dimensions' not in st.session_state: st.session_state.categorized_dimensions = []
if 'categorized_geo' not in st.session_state: st.session_state.categorized_geo = []
if 'categorized_general' not in st.session_state: st.session_state.categorized_general = []
if 'categorized_sensitive' not in st.session_state: st.session_state.categorized_sensitive = []

# Original lists (for restoration)
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
    if st.session_state.get("authenticated"):
        return True
    st.markdown("""<style>.main { display: flex; flex-direction: column; justify-content: center; align-items: center; } .auth-logo-box { text-align: center; margin-bottom: 1rem; padding-top: 0; }</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-logo-box"><h3>Вход в систему</h3></div>', unsafe_allow_html=True)
        password = st.text_input("Пароль", type="password", key="password_input", label_visibility="collapsed")
        if st.button("ВОЙТИ", type="primary", use_container_width=True):
            if password == "ZVC01w4_pIquj0bMiaAu":
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
    "Ростов-на-Дону": {"ya": 39, "go": 1012028},
    "Уфа": {"ya": 172, "go": 1012091},
    "Красноярск": {"ya": 62, "go": 1012001},
    "Воронеж": {"ya": 193, "go": 1012134},
    "Пермь": {"ya": 50, "go": 1012015},
    "Волгоград": {"ya": 38, "go": 1012131},
    "Краснодар": {"ya": 35, "go": 1011894},
    "Саратов": {"ya": 194, "go": 1012046},
    "Тюмень": {"ya": 283, "go": 1012089},
    "Тольятти": {"ya": 240, "go": 1012080},
    "Ижевск": {"ya": 44, "go": 1011979},
    "Барнаул": {"ya": 197, "go": 1011855},
    "Иркутск": {"ya": 63, "go": 1011977},
    "Ульяновск": {"ya": 195, "go": 1012092},
    "Хабаровск": {"ya": 76, "go": 1011973},
    "Владивосток": {"ya": 75, "go": 1012129},
    "Ярославль": {"ya": 16, "go": 1012140},
    "Махачкала": {"ya": 28, "go": 1011993},
    "Томск": {"ya": 67, "go": 1012082},
    "Оренбург": {"ya": 48, "go": 1012009},
    "Кемерово": {"ya": 64, "go": 1011985},
    "Новокузнецк": {"ya": 237, "go": 1011987},
    "Рязань": {"ya": 11, "go": 1012033},
    "Набережные Челны": {"ya": 234, "go": 1011905},
    "Пенза": {"ya": 49, "go": 1012013},
    "Липецк": {"ya": 9, "go": 1011991},
    "Тула": {"ya": 15, "go": 1012085},
    "Киров": {"ya": 46, "go": 1011989},
    "Чебоксары": {"ya": 45, "go": 1011880},
    "Калининград": {"ya": 22, "go": 1011981},
    "Курск": {"ya": 8, "go": 1011988},
    "Улан-Удэ": {"ya": 68, "go": 1012090},
    "Ставрополь": {"ya": 36, "go": 1012070},
    "Севастополь": {"ya": 959, "go": 1012048},
    "Сочи": {"ya": 239, "go": 1012053},
    "Россия": {"ya": 225, "go": 2643},
    "Минск (BY)": {"ya": 157, "go": 1001493},
    "Алматы (KZ)": {"ya": 162, "go": 1014601},
    "Астана (KZ)": {"ya": 163, "go": 1014620}
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

# ==========================================
# PARSING & METRICS
# ==========================================
# ... (Остальной код функций без изменений)

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
    # Timeout increased to 10 minutes (120 * 5s)
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
    import streamlit as st
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
        # 1. Создаем объект Soup (Полная страница)
        soup = BeautifulSoup(content, 'html.parser', from_encoding=encoding)
        
        # === НОВОЕ: Собираем Title и Description отдельно ===
        page_title = soup.title.string.strip() if soup.title and soup.title.string else ""
        
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        page_desc = meta_desc_tag['content'].strip() if meta_desc_tag and meta_desc_tag.get('content') else ""
        # ====================================================

        # === ЛОГИКА ТАБЛИЦЫ 2 (Поиск товаров по URL/Ссылке) ===
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
        # ========================================================
        
        h1_tag = soup.find('h1')
        h1_text = h1_tag.get_text(strip=True) if h1_tag else ""

        # 2. Создаем копию для Таблицы 2 (Удаляем блок товаров)
        soup_no_grid = BeautifulSoup(content, 'html.parser', from_encoding=encoding)
        grid_div = soup_no_grid.find('div', class_='an-container-fluid an-container-xl')
        if grid_div: grid_div.decompose()
        
        # === [ВАЖНО] ФИЛЬТРАЦИЯ КОНТЕНТА ПО ГАЛОЧКАМ ===
        tags_to_remove = []
        if settings['noindex']: tags_to_remove.append('noindex')
        
        for s in [soup, soup_no_grid]:
            for c in s.find_all(string=lambda text: isinstance(text, Comment)): c.extract()
            if tags_to_remove:
                for t in s.find_all(tags_to_remove): t.decompose()
            for script in s(["script", "style", "svg", "path", "noscript"]): script.decompose()

        # Текст ссылок (анкоры)
        anchors_list = [a.get_text(strip=True) for a in soup.find_all('a') if a.get_text(strip=True)]
        anchor_text = " ".join(anchors_list)
        
        # Сбор ДОПОЛНИТЕЛЬНОГО текста (Description, Alt, Title)
        extra_text = []
        # Description добавляем в общий текст анализа тоже
        if page_desc: extra_text.append(page_desc)

        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): extra_text.append(img['alt'])
            for t in soup.find_all(title=True): extra_text.append(t['title'])

        # Собираем итоговый текст
        body_text_raw = soup.get_text(separator=' ') + " " + " ".join(extra_text)
        body_text = re.sub(r'\s+', ' ', body_text_raw).strip()

        body_text_no_grid_raw = soup_no_grid.get_text(separator=' ') + " " + " ".join(extra_text)
        body_text_no_grid = re.sub(r'\s+', ' ', body_text_no_grid_raw).strip()

        if not body_text: return None
            
        return {
            'url': url, 
            'domain': urlparse(url).netloc, 
            'body_text': body_text, 
            'body_text_no_grid': body_text_no_grid,
            'anchor_text': anchor_text,
            'h1': h1_text,
            'product_titles': product_titles,
            # !!! НОВЫЕ ПОЛЯ ДЛЯ DASHBOARD !!!
            'meta_title': page_title,
            'meta_desc': page_desc
        }
    except Exception:
        return None

def analyze_meta_gaps(comp_data_full, my_data, settings):
    """
    УМНЫЙ АНАЛИЗАТОР META-ТЕГОВ v2.1
    1. Учитывает вес позиции (слова топов важнее).
    2. Порог вхождения: СТРОГО 50% (слово должно быть у половины конкурентов).
    3. Фильтрует предлоги и союзы.
    """
    if not comp_data_full: return None
    
    # === 1. НАСТРОЙКИ АЛГОРИТМА ===
    TOTAL_COMPS = len(comp_data_full)
    
    # !!! ИСПРАВЛЕНО: СТРОГО 50% !!!
    MIN_OCCURRENCE_PCT = 0.4 
    
    # Минимум 2 сайта, даже если конкурентов всего 3
    MIN_COUNT = max(2, int(TOTAL_COMPS * MIN_OCCURRENCE_PCT))

    # Вспомогательная функция токенизации (Чистка мусора)
    def fast_tokenize(text):
        if not text: return set()
        
        # Список стоп-слов
        stop_garbage = {
            'в', 'на', 'и', 'с', 'со', 'по', 'для', 'от', 'до', 'из', 'к', 'у', 
            'о', 'об', 'за', 'над', 'под', 'при', 'про', 'без', 'через', 'между',
            'а', 'но', 'или', 'да', 'как', 'что', 'чтобы', 'если', 'то', 'ли', 'бы', 'же', 
            'г', 'обл', 'р', 'руб', 'мм', 'см', 'м', 'кг', 'т', 'шт', 'дн',
            'весь', 'все', 'всё', 'свой', 'ваш', 'наш', 'мы', 'вы', 'он', 'она', 'они',
            'купить', 'цена', 'заказать', 'стоимость', 'продажа', 'недорого', 
            'москва', 'спб' 
        }
        # Убираем коммерческие штампы из стоп-листа, чтобы они попадали в рекомендации
        if 'купить' in stop_garbage: stop_garbage.remove('купить') 
        if 'цена' in stop_garbage: stop_garbage.remove('цена')
        
        if settings.get('custom_stops'):
            stop_garbage.update(set(settings['custom_stops']))

        lemmas = set()
        words = re.findall(r'[а-яА-Яa-zA-Z0-9]+', text.lower())
        
        for w in words:
            if len(w) < 2: continue 
            if w in stop_garbage: continue
            
            # NLP Фильтр
            if morph:
                try:
                    p = morph.parse(w)[0]
                    # Исключаем Предлоги, Союзы, Частицы, Местоимения, Междометия
                    if p.tag.POS in {'PREP', 'CONJ', 'PRCL', 'NPRO', 'INTJ'}:
                        continue
                    if p.normal_form in stop_garbage:
                        continue
                    lemmas.add(p.normal_form)
                except: 
                    lemmas.add(w)
            else:
                lemmas.add(w)
        return lemmas

    # === 2. СБОР ДАННЫХ С ВЕСАМИ ===
    # Структура: word -> {'count': 0, 'score': 0.0}
    stats_map = {
        'title': defaultdict(lambda: {'count': 0, 'score': 0.0}),
        'desc': defaultdict(lambda: {'count': 0, 'score': 0.0}),
        'h1': defaultdict(lambda: {'count': 0, 'score': 0.0})
    }
    
    detailed_rows = []

    for i, item in enumerate(comp_data_full):
        # Вес позиции: 1-е место = весомее, чем 10-е
        rank_weight = 1.0 + ( (TOTAL_COMPS - i) / TOTAL_COMPS ) * 1.5
        
        t_tok = fast_tokenize(item.get('meta_title', ''))
        d_tok = fast_tokenize(item.get('meta_desc', ''))
        h_tok = fast_tokenize(item.get('h1', ''))
        
        for w in t_tok:
            stats_map['title'][w]['count'] += 1
            stats_map['title'][w]['score'] += rank_weight
            
        for w in d_tok:
            stats_map['desc'][w]['count'] += 1
            stats_map['desc'][w]['score'] += rank_weight
            
        for w in h_tok:
            stats_map['h1'][w]['count'] += 1
            stats_map['h1'][w]['score'] += rank_weight

        detailed_rows.append({
            'URL': item['url'],
            'Title': item.get('meta_title', ''),
            'Description': item.get('meta_desc', ''),
            'H1': item.get('h1', '')
        })

    # === 3. АНАЛИЗ РАЗРЫВОВ (GAPS) ===
    
    my_tokens = {
        'title': fast_tokenize(my_data.get('meta_title', '')),
        'desc': fast_tokenize(my_data.get('meta_desc', '')),
        'h1': fast_tokenize(my_data.get('h1', ''))
    }

    def process_category(cat_key):
        data = stats_map[cat_key]
        important_words = []
        
        for word, metrics in data.items():
            # 1. Отсекаем слова, которые встречаются реже, чем у 50% конкурентов
            if metrics['count'] < MIN_COUNT:
                continue
            
            # Сохраняем слово и его "важность" (Score)
            important_words.append((word, metrics['score']))
        
        # Сортируем по важности (Score)
        important_words.sort(key=lambda x: x[1], reverse=True)
        
        # Оставляем только ядро (Топ-15 слов, прошедших фильтр 50%)
        core_semantics = [x[0] for x in important_words[:15]]
        
        if not core_semantics:
            return 100, [] 
            
        matches = 0
        missing = []
        
        for w in core_semantics:
            if w in my_tokens[cat_key]:
                matches += 1
            else:
                missing.append(w)
        
        if len(core_semantics) > 0:
            score = int((matches / len(core_semantics)) * 100)
        else:
            score = 100
            
        return score, missing

    s_t, m_t = process_category('title')
    s_d, m_d = process_category('desc')
    s_h, m_h = process_category('h1')

    return {
        'scores': {'title': s_t, 'desc': s_d, 'h1': s_h},
        'missing': {'title': m_t, 'desc': m_d, 'h1': m_h},
        'detailed': detailed_rows,
        'my_data': {
            'Title': my_data.get('meta_title', 'Не определен'),
            'Description': my_data.get('meta_desc', 'Не определен'),
            'H1': my_data.get('h1', 'Не определен')
        }
    }
        
def calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, original_results):
    # Внутренняя функция округления
    def math_round(number):
        return int(number + 0.5)

    all_forms_map = defaultdict(set)
    global_forms_counter = defaultdict(Counter) 
    
    # 1. ОБРАБОТКА МОЕГО САЙТА (для расчета личных баллов)
    if not my_data or not my_data.get('body_text'): 
        my_lemmas, my_forms, my_anchors, my_len = [], {}, [], 0
        my_clean_domain = "local"
    else:
        my_lemmas, my_forms = process_text_detailed(my_data['body_text'], settings)
        my_anchors, _ = process_text_detailed(my_data['anchor_text'], settings)
        my_len = len(my_lemmas)
        for k, v in my_forms.items(): all_forms_map[k].update(v)
        my_clean_domain = my_data['domain'].lower().replace('www.', '').split(':')[0]

    # 2. ОБРАБОТКА КОНКУРЕНТОВ (Формирование базы для Медианы)
    comp_docs = []
    for p in comp_data_full:
        if not p.get('body_text'): continue
        
        # --- ИЗМЕНЕНИЕ: УБРАНА ПРОВЕРКА "ЕСЛИ ЭТО МОЙ САЙТ - ПРОПУСТИТЬ" ---
        # Теперь, если ваш сайт есть в выдаче (Топ-10/20), он попадет в comp_docs 
        # и будет влиять на медиану, как вы и просили.
        
        body, c_forms = process_text_detailed(p['body_text'], settings)
        
        # Сбор реальных форм для отображения
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

    # Метрики корпуса
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

    # IDF
    word_idf_map = {}
    for lemma in vocab:
        df = doc_freqs[lemma]
        if df == 0: continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        word_idf_map[lemma] = max(idf, 0.01)

    # =========================================================================
    # 3. РАСЧЕТ МЕДИАН
    # =========================================================================
    
    table_depth = []
    table_hybrid = []
    missing_semantics_high = []
    missing_semantics_low = []
    
    words_with_median_gt_0 = set() 
    my_found_words_from_median = set() 
    
    for lemma in vocab:
        if lemma in GARBAGE_LATIN_STOPLIST: continue
        
        # 1. СОБИРАЕМ СЫРЫЕ ДАННЫЕ
        raw_counts = [word_counts_per_doc[i][lemma] for i in range(N)]
        sorted_raw = sorted(raw_counts)
        
        if sorted_raw:
            rec_median_absolute = math_round(np.median(sorted_raw))
            obs_min = math_round(sorted_raw[0])
            obs_max = math_round(sorted_raw[-1])
        else:
            rec_median_absolute = 0; obs_min = 0; obs_max = 0

        # 2. СЧИТАЕМ ЦЕЛЕВУЮ МЕДИАНУ
        if settings['norm'] and my_len > 0:
            norm_counts = []
            for i in range(N):
                raw_cnt = raw_counts[i]
                comp_len = c_lens[i]
                if comp_len > 0:
                    normalized_val = raw_cnt * (my_len / comp_len)
                    norm_counts.append(normalized_val)
                else:
                    norm_counts.append(0)
            
            if norm_counts:
                rec_median_target = math_round(np.median(sorted(norm_counts)))
            else:
                rec_median_target = 0
        else:
            rec_median_target = rec_median_absolute

        # --- МОЯ СТАТИСТИКА ---
        my_tf_count = my_lemmas.count(lemma)
        
        if obs_max == 0 and my_tf_count == 0: continue

        # --- ШИРИНА ---
        if rec_median_target >= 1:
            words_with_median_gt_0.add(lemma)
            if my_tf_count > 0:
                my_found_words_from_median.add(lemma)

        # --- УПУЩЕННАЯ СЕМАНТИКА ---
        display_word = lemma
        if global_forms_counter[lemma]:
            display_word = global_forms_counter[lemma].most_common(1)[0][0]

        if my_tf_count == 0:
            weight = word_idf_map.get(lemma, 0) * (rec_median_target if rec_median_target > 0 else 1)
            item = {'word': display_word, 'weight': weight}
            if rec_median_target >= 1: missing_semantics_high.append(item)
            else: missing_semantics_low.append(item)

        # --- РЕКОМЕНДАЦИИ ---
        diff = rec_median_target - my_tf_count
        
        if diff == 0: status = "Норма"; action_text = "✅"; sort_val = 0
        elif diff > 0: status = "Недоспам"; action_text = f"+{diff}"; sort_val = diff
        else: status = "Переспам"; action_text = f"{diff}"; sort_val = abs(diff)

        forms_str = ", ".join(sorted(list(all_forms_map.get(lemma, set())))) if all_forms_map.get(lemma) else lemma

        table_depth.append({
            "Слово": display_word,
            "Словоформы": forms_str,
            "Вхождений у вас": my_tf_count,
            "Медиана": rec_median_absolute, 
            "Минимум (конкур.)": obs_min,
            "Максимум (конкур.)": obs_max,
            "Статус": status,
            "Рекомендация": action_text,
            "is_missing": (my_tf_count == 0),
            "sort_val": sort_val
        })
        
        table_hybrid.append({
            "Слово": display_word,
            "TF-IDF ТОП": round(word_idf_map.get(lemma, 0) * (rec_median_absolute / avg_dl if avg_dl > 0 else 0), 4),
            "TF-IDF у вас": round(word_idf_map.get(lemma, 0) * (my_tf_count / my_len if my_len > 0 else 0), 4),
            "Сайтов": doc_freqs[lemma],
            "Переспам": obs_max
        })

    # =========================================================================
    # 4. ФИНАЛИЗАЦИЯ (БАЛЛЫ)
    # =========================================================================
    
    total_needed = len(words_with_median_gt_0)
    total_found = len(my_found_words_from_median)
    
    if total_needed > 0:
        ratio = total_found / total_needed
        my_width_score_final = int(min(100, ratio * 120))
    else:
        my_width_score_final = 0

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
        if total_needed > 0:
            c_width_val = int(min(100, (c_found / total_needed) * 120))
        else:
            c_width_val = 0
            
        raw_val = calculate_raw_power(doc['body'], c_lens[i])
        comp_raw_scores.append(raw_val)
        competitor_scores_map[doc['url']] = {'width_final': c_width_val, 'raw_depth': raw_val}

    if comp_raw_scores:
        median_raw = np.median(comp_raw_scores)
        ref_val = median_raw if median_raw > 0.1 else 1.0
    else:
        ref_val = 1.0
    
    k_norm = 80.0 / ref_val
    my_raw_bm25 = calculate_raw_power(my_lemmas, my_len)
    my_depth_score_final = int(round(min(100, my_raw_bm25 * k_norm)))

    # Досчитываем глубину всем конкурентам
    for url, data in competitor_scores_map.items():
        data['depth_final'] = int(round(min(100, data['raw_depth'] * k_norm)))

    # Сборка таблицы релевантности
    missing_semantics_high.sort(key=lambda x: x['weight'], reverse=True)
    missing_semantics_low.sort(key=lambda x: x['weight'], reverse=True)
    missing_semantics_low = missing_semantics_low[:500]

    table_rel = []
    my_site_found_in_selection = False
    
    # 1. Проходимся по ИСХОДНОМУ списку из выдачи (Top N)
    for item in original_results:
        url = item['url']
        if url not in competitor_scores_map: continue
        
        row_domain = urlparse(url).netloc.lower().replace('www.', '')
        is_my_site = False
        
        # Проверяем, не наш ли это сайт
        if my_clean_domain and my_clean_domain != "local" and my_clean_domain in row_domain:
            is_my_site = True
            my_site_found_in_selection = True
            display_name = f"{urlparse(url).netloc} (Вы)"
        else:
            display_name = urlparse(url).netloc
            
        scores = competitor_scores_map[url]
        table_rel.append({ 
            "Домен": display_name, 
            "URL": url,
            "Позиция": item['pos'], 
            "Ширина (балл)": scores['width_final'], 
            "Глубина (балл)": scores['depth_final'] 
        })
        
    # 2. Если нашего сайта НЕ было в Топ N, добавляем его вручную в конец
    if not my_site_found_in_selection:
        pos_to_show = my_serp_pos if my_serp_pos > 0 else 0
        my_label = f"{my_data['domain']} (Вы)" if (my_data and my_data.get('domain')) else "Ваш сайт"
        my_full_url = my_data['url'] if (my_data and 'url' in my_data) else "#"
        table_rel.append({ 
            "Домен": my_label, 
            "URL": my_full_url, 
            "Позиция": pos_to_show, 
            "Ширина (балл)": my_width_score_final, 
            "Глубина (балл)": my_depth_score_final 
        })

    # Анализ аномалий
    df_rel_for_analysis = pd.DataFrame(table_rel)
    good_urls, bad_urls_dicts, trend_info = analyze_serp_anomalies(df_rel_for_analysis)
    
    st.session_state['detected_anomalies'] = bad_urls_dicts
    st.session_state['serp_trend_info'] = trend_info
    
    # Сохраняем "чистые" урлы (для следующего запуска)

    return { 
        "depth": pd.DataFrame(table_depth), 
        "hybrid": pd.DataFrame(table_hybrid), 
        "relevance_top": pd.DataFrame(table_rel).sort_values(by='Позиция', ascending=True).reset_index(drop=True), 
        "my_score": {"width": my_width_score_final, "depth": my_depth_score_final}, 
        "missing_semantics_high": missing_semantics_high, 
        "missing_semantics_low": missing_semantics_low 
    }
    
def get_hybrid_word_type(word, main_marker_root, specs_dict=None):
    """
    Классификатор 3.1 (Фикс диапазонов).
    """
    w = word.lower()
    specs_dict = specs_dict or set()
    
    # 1. МАРКЕР
    if w == main_marker_root: return "1. 💎 Маркер (Товар)"
    if morph:
        norm = morph.parse(w)[0].normal_form
        if norm == main_marker_root: return "1. 💎 Маркер (Товар)"

    # 2. СТАНДАРТЫ
    if re.search(r'(gost|din|iso|en|tu|astm|aisi|гост|ост|ту|дин)', w):
        return "6. 📜 Стандарт"

    # 3. РАЗМЕРЫ / ТЕХ. ПАРАМЕТРЫ
    # А. Голые цифры (10, 50.5)
    if re.fullmatch(r'\d+([.,]\d+)?', w): return "5. 🔢 Размеры/Прочее"
    # Б. Размеры с разделителями (10х20, 10*20, 10-20, 10/20) <--- ДОБАВИЛ ТИРЕ И СЛЕШ
    if re.search(r'^\d+[xх*\-/]\d+', w): return "5. 🔢 Размеры/Прочее"
    # В. Единицы (мм, кг)
    if re.search(r'\d+(мм|mm|м|m|kg|кг|bar|бар|атм)$', w): return "5. 🔢 Размеры/Прочее"
    # Г. Префиксы (Ду, Ру, SDR)
    if re.match(r'^(d|dn|pn|sn|sdr|ду|ру|ø)\d+', w): return "5. 🔢 Размеры/Прочее"

    # 4. МАРКИ / СПЛАВЫ
    if w in specs_dict: return "3. 🏗️ Марка/Сплав"
    # Паттерны марок (Буквы+Цифры)
    if re.search(r'\d', w): return "3. 🏗️ Марка/Сплав"

    # 5. ЛАТИНИЦА (Бренды)
    if re.search(r'^[a-z\-]+$', w): return "7. 🔠 Латиница/Бренд"

    # 6. ТЕКСТ
    if morph:
        p = morph.parse(w)[0]
        tag = p.tag
        if {'PREP'} in tag or {'CONJ'} in tag: return "SKIP"
        if {'ADJF'} in tag or {'PRTF'} in tag or {'ADJS'} in tag: return "2. 🎨 Свойства"
        if {'NOUN'} in tag: return "4. 🔗 Дополнения"

    if w.endswith(('ий', 'ый', 'ая', 'ое', 'ые', 'ая')): return "2. 🎨 Свойства"
    return "4. 🔗 Дополнения"
    
def calculate_naming_metrics(comp_data_full, my_data, settings):
    """
    Таблица 2. Без "обрезания" технических слов.
    """
    # Подгрузка словаря
    SPECS_SET = st.session_state.get('categorized_dimensions', set())
    if not SPECS_SET: _, _, SPECS_SET, _, _, _ = load_lemmatized_dictionaries()

    # 1. Мой сайт
    my_tokens = []
    if my_data and my_data.get('body_text_no_grid'):
        # Своя токенизация, чтобы сохранить Ду50
        raw_w = re.findall(r'[а-яА-Яa-zA-Z0-9\-]+', my_data['body_text_no_grid'].lower())
        for w in raw_w:
            # Лемматизируем только чисто текстовые слова
            if not re.search(r'\d', w) and morph:
                my_tokens.append(morph.parse(w)[0].normal_form)
            else:
                my_tokens.append(w)

    # 2. Конкуренты
    all_words_flat = []
    site_vocab_map = []
    
    for p in comp_data_full:
        titles = p.get('product_titles', [])
        valid_titles = [t for t in titles if 5 < len(t) < 150]
        
        if not valid_titles:
            site_vocab_map.append(set())
            continue
            
        curr_site_tokens = set()
        for t in valid_titles:
            words = re.findall(r'[а-яА-Яa-zA-Z0-9\-]+', t.lower())
            for w in words:
                if len(w) < 2: continue
                
                # ЛОГИКА СОХРАНЕНИЯ ФОРМЫ:
                # Если есть цифра -> сохраняем как есть (d50 -> d50)
                if re.search(r'\d', w):
                    token = w
                elif re.search(r'^[a-z]+$', w): # Латиница -> как есть
                    token = w
                elif morph: # Русские слова -> лемматизируем (стальная -> стальной)
                    token = morph.parse(w)[0].normal_form
                else:
                    token = w
                
                all_words_flat.append(token)
                curr_site_tokens.add(token)
                
        site_vocab_map.append(curr_site_tokens)

    if not all_words_flat: return pd.DataFrame()
    N_sites = len(site_vocab_map)

    # 3. Маркер (Самое частое текстовое слово)
    counts = Counter([w for w in all_words_flat if not re.search(r'\d', w)])
    main_marker_root = ""
    # Ищем существительное
    for w, c in counts.most_common(10):
        if morph and 'NOUN' in morph.parse(w)[0].tag: main_marker_root = w; break
    if not main_marker_root and counts: main_marker_root = counts.most_common(1)[0][0]

    # 4. Сбор таблицы
    vocab = sorted(list(set(all_words_flat)))
    table_rows = []
    
    for token in vocab:
        if token in GARBAGE_LATIN_STOPLIST: continue
        
        # Частотность
        sites_with_word = sum(1 for s_set in site_vocab_map if token in s_set)
        freq_percent = int((sites_with_word / N_sites) * 100)
        
        # КЛАССИФИКАЦИЯ
        cat = get_hybrid_word_type(token, main_marker_root, SPECS_SET)
        
        if cat == "SKIP": continue
        
        # Фильтры отображения
        # Марки и Стандарты показываем от 5%
        is_spec = "Марка" in cat or "Стандарт" in cat
        if is_spec and freq_percent < 5: continue
        
        # Обычные слова от 15%
        if not is_spec and "Размеры" not in cat and freq_percent < 15: continue
        
        # Размеры показываем только если они реально частые (например, ходовой диаметр)
        # Иначе таблица будет забита цифрами 10, 11, 12...
        if "Размеры" in cat and freq_percent < 15: continue

        rec_median = 1 if freq_percent > 30 else 0
        my_tf = my_tokens.count(token)
        diff = rec_median - my_tf
        action_text = f"+{diff}" if diff > 0 else ("✅" if diff == 0 else f"{diff}")
        
        table_rows.append({
            "Тип хар-ки": cat[3:],
            "Слово": token, # Выводим токен как есть (с цифрами и буквами)
            "Частотность (%)": f"{freq_percent}%",
            "У Вас": my_tf,
            "Медиана": rec_median,
            "Добавить": action_text,
            "raw_freq": freq_percent,
            "cat_sort": int(cat[0])
        })
        
    df = pd.DataFrame(table_rows)
    if not df.empty:
        df = df.sort_values(by=["cat_sort", "raw_freq"], ascending=[True, False])
        
    return df

def analyze_ideal_name(comp_data_full):
    """
    Строит структуру с учетом Марок и ГОСТов.
    """
    # Подгружаем словарь
    SPECS_SET = st.session_state.get('categorized_dimensions', set())
    if not SPECS_SET: _, _, SPECS_SET, _, _, _ = load_lemmatized_dictionaries()

    titles = []
    for d in comp_data_full:
        ts = d.get('product_titles', [])
        titles.extend([t for t in ts if 5 < len(t) < 150])
    
    if not titles: return "Нет данных", []

    # Маркер
    all_w = []
    for t in titles: all_w.extend(re.findall(r'[а-яА-Яa-zA-Z0-9\-]+', t.lower()))
    c = Counter(all_w)
    main_marker_root = ""
    for w, _ in c.most_common(5):
        if not re.search(r'\d', w):
             if morph and 'NOUN' in morph.parse(w)[0].tag: main_marker_root = w; break
             elif not morph: main_marker_root = w; break
    if not main_marker_root and c: main_marker_root = c.most_common(1)[0][0]

    # Анализ паттернов
    structure_counter = Counter()
    vocab_by_type = defaultdict(Counter)
    
    sample = titles[:500]
    
    for t in sample:
        words = re.findall(r'[а-яА-Яa-zA-Z0-9\-]+', t.lower())
        pattern = []
        
        for w in words:
            if len(w) < 2: continue
            
            # Классификация с учетом словаря
            cat_full = get_hybrid_word_type(w, main_marker_root, SPECS_SET)
            if cat_full == "SKIP": continue
            
            # Упрощенное имя типа ("Свойства", "Марка/Сплав", "Стандарт")
            # "3. 🏗️ Марка/Сплав" -> "Марка/Сплав"
            try:
                cat_short = cat_full.split('.', 1)[1].strip().split(' ', 1)[1] # Берем текст после иконки
            except:
                cat_short = cat_full # Fallback
            
            vocab_by_type[cat_short][w] += 1
            
            if not pattern or pattern[-1] != cat_short:
                pattern.append(cat_short)
        
        if pattern:
            structure_str = " + ".join(pattern)
            structure_counter[structure_str] += 1
            
    # Сборка
    if not structure_counter: return "Структура не найдена", []
    
    best_struct_str, _ = structure_counter.most_common(1)[0]
    best_struct_list = best_struct_str.split(" + ")
    
    final_parts = []
    used_words = set()
    
    for block in best_struct_list:
        # Для переменных параметров ставим заглушку
        if "Размеры" in block or "Стандарт" in block or "Марка" in block:
            # Пытаемся найти самый частый пример, если он очень популярен
            top_cand = vocab_by_type[block].most_common(1)
            if top_cand and top_cand[0][1] > (len(sample) * 0.3): # Если встречается у 30%
                 final_parts.append(top_cand[0][0])
            else:
                 final_parts.append(f"[{block.upper()}]")
            continue
            
        # Для слов (Маркер, Свойства) берем ТОП-1
        candidates = vocab_by_type[block].most_common(3)
        for w, cnt in candidates:
            if w not in used_words:
                if "Маркер" in block: w = w.capitalize()
                final_parts.append(w)
                used_words.add(w)
                break
                
    ideal_name = " ".join(final_parts)
    
    # Отчет
    report = []
    report.append(f"**Схема:** {best_struct_str}")
    report.append("")
    report.append("**Популярные значения:**")
    for block in best_struct_list:
        if "Размеры" in block: continue
        top = [f"{w}" for w, c in vocab_by_type[block].most_common(3)]
        report.append(f"- **{block}**: {', '.join(top)}")
            
    return ideal_name, report

def render_paginated_table(df, title_text, key_prefix, default_sort_col=None, use_abs_sort_default=False, default_sort_order="Убывание", show_controls=True):
    if df.empty: st.info(f"{title_text}: Нет данных."); return
    col_t1, col_t2 = st.columns([7, 3])
    with col_t1: st.markdown(f"### {title_text}")
    
    # Инициализация дефолтов в Session State
    if f'{key_prefix}_sort_col' not in st.session_state: 
        st.session_state[f'{key_prefix}_sort_col'] = default_sort_col if (default_sort_col and default_sort_col in df.columns) else df.columns[0]
    
    if f'{key_prefix}_sort_order' not in st.session_state: 
        st.session_state[f'{key_prefix}_sort_order'] = default_sort_order

    search_query = st.text_input(f"🔍 Поиск ({title_text})", key=f"{key_prefix}_search")
    if search_query:
        mask = df.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
        df_filtered = df[mask].copy()
    else: df_filtered = df.copy()

    if df_filtered.empty: st.warning("Ничего не найдено."); return

    # === ЛОГИКА ОТОБРАЖЕНИЯ КОНТРОЛОВ ===
    if show_controls:
        with st.container():
            st.markdown("<div class='sort-container'>", unsafe_allow_html=True)
            col_s1, col_s2, col_sp = st.columns([2, 2, 4])
            with col_s1:
                current_sort = st.session_state[f'{key_prefix}_sort_col']
                if current_sort not in df_filtered.columns: current_sort = df_filtered.columns[0]
                sort_col = st.selectbox("🗂 Сортировать по:", df_filtered.columns, key=f"{key_prefix}_sort_box", index=list(df_filtered.columns).index(current_sort))
                st.session_state[f'{key_prefix}_sort_col'] = sort_col
            with col_s2:
                def_index = 0 if st.session_state[f'{key_prefix}_sort_order'] == "Убывание" else 1
                sort_order = st.radio("Порядок:", ["Убывание", "Возрастание"], horizontal=True, key=f"{key_prefix}_order_box", index=def_index)
                st.session_state[f'{key_prefix}_sort_order'] = sort_order
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Если контролы скрыты, берем значения из session_state (дефолтные)
        sort_col = st.session_state[f'{key_prefix}_sort_col']
        sort_order = st.session_state[f'{key_prefix}_sort_order']

    ascending = (sort_order == "Возрастание")
    
    # Применение сортировки
    if use_abs_sort_default and sort_col == "Рекомендация" and "sort_val" in df_filtered.columns: 
        df_filtered = df_filtered.sort_values(by="sort_val", ascending=ascending)
    elif ("Добавить" in sort_col or "+/-" in sort_col) and df_filtered[sort_col].dtype == object:
        try:
            df_filtered['_temp_sort'] = df_filtered[sort_col].astype(str).str.replace(r'[^\d]', '', regex=True)
            df_filtered['_temp_sort'] = pd.to_numeric(df_filtered['_temp_sort'], errors='coerce').fillna(0)
            df_filtered = df_filtered.sort_values(by='_temp_sort', ascending=ascending).drop(columns=['_temp_sort'])
        except: df_filtered = df_filtered.sort_values(by=sort_col, ascending=ascending)
    else: 
        # Проверка наличия колонки (на случай смены данных)
        if sort_col in df_filtered.columns:
            df_filtered = df_filtered.sort_values(by=sort_col, ascending=ascending)

    df_filtered = df_filtered.reset_index(drop=True); df_filtered.index = df_filtered.index + 1
    
    # Экспорт
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        export_df = df_filtered.copy()
        if "is_missing" in export_df.columns: del export_df["is_missing"]
        if "sort_val" in export_df.columns: del export_df["sort_val"]
        export_df.to_excel(writer, index=False, sheet_name='Data')
    excel_data = buffer.getvalue()
    with col_t2: st.download_button(label="📥 Скачать Excel", data=excel_data, file_name=f"{key_prefix}_export.xlsx", mime="application/vnd.ms-excel", key=f"{key_prefix}_down")

    # Пагинация
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
        if st.button("⬅️", key=f"{key_prefix}_next", disabled=(current_page >= total_pages), use_container_width=True):
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
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.encoding = 'utf-8'
    except Exception as e: return None, None, None, f"Ошибка соединения: {e}"
    
    if response.status_code != 200: return None, None, None, f"Ошибка статуса: {response.status_code}"
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 1. ЗАГОЛОВОК: Ищем строго H2 (как вы просили)
    # Сначала ищем в контентной части (часто бывает h2 в меню, который нам не нужен)
    # Если есть класс description-container, ищем внутри него, если нет - первый H2 на странице
    description_div = soup.find('div', class_='description-container')
    
    target_h2 = None
    if description_div:
        target_h2 = description_div.find('h2')
    
    if not target_h2:
        target_h2 = soup.find('h2')
        
    page_header = target_h2.get_text(strip=True) if target_h2 else "Описание товара" # Дефолт, если H2 нет совсем

    # 2. Фактура (текст)
    base_text = description_div.get_text(separator="\n", strip=True) if description_div else soup.body.get_text(separator="\n", strip=True)[:5000]
    
    # 3. Теги
    tags_container = soup.find(class_='popular-tags-inner')
    tags_data = []
    if tags_container:
        links = tags_container.find_all('a')
        for link in links:
            tag_url = urljoin(url, link.get('href')) if link.get('href') else None
            if tag_url: tags_data.append({'name': link.get_text(strip=True), 'url': tag_url})
            
    return base_text, tags_data, page_header, None

def generate_ai_content_blocks(api_key, base_text, tag_name, forced_header, num_blocks=5, seo_words=None):
    if not base_text: return ["Error: No base text"] * num_blocks
    
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://litellm.tokengate.ru/v1")
    
    seo_words = seo_words or []
    seo_instruction_block = ""
    
    if seo_words:
        seo_list_str = ", ".join(seo_words)
        seo_instruction_block = f"""
--- ВАЖНАЯ ИНСТРУКЦИЯ ПО SEO-СЛОВАМ ---
Тебе нужно внедрить в текст следующие слова в любой подходящей под контекст лемме: {{{seo_list_str}}}

ПРАВИЛА ВНЕДРЕНИЯ И ВЫДЕЛЕНИЯ:
1. РАСПРЕДЕЛЕНИЕ: Раскидай слова по всем {num_blocks} блокам.
2. СТРОГИЙ ЗАПРЕТ: Используй тег <b> ТОЛЬКО для этих SEO-слов. Не выделяй жирным ничего другого.
3. ЕСТЕСТВЕННОСТЬ: Меняй словоформы под контекст. Текст должен быть естественным и логичным, не пиши чушь.
-------------------------------------------
"""

    # 2. ПРОМТЫ (ТЕКСТ ОСТАВЛЕН БЕЗ ИЗМЕНЕНИЙ)
    system_instruction = (
        "Ты — профессиональный технический копирайтер и верстальщик. "
        "Твоя цель — писать глубокий, технически полезный текст для профессионалов, насыщенный фактами и цифрами. "
        "Ты выдаешь ТОЛЬКО HTML-код. "
        "Стиль: Деловой, экспертный, но \"человечный\" и понятный. Избегай канцеляризмов и пространных рассуждений. "
        "Факты и конкретика: Все суждения подкрепляй измеримыми фактами, цифрами, ссылками на ГОСТы, марки стали и другие нормативы. "
        "Используй поисковые инструменты для проверки и обогащения текста актуальной информацией. "
        "Коммерческая направленность: Текст должен продавать. Говори от лица компании-производителя/поставщика. "
        "Вместо \"проверенный поставщик\" используй формулировки, подчеркивающие собственное производство и экспертизу. "
        "Формула Главреда для B2B: В тексте должны быть ответы на вопросы: что это? какую проблему решает? кому подойдет? "
        "какие есть разновидности? Дополнительно раскрой информацию о стандартах производства, складских запасах и возможности изготовления под заказ. "
        "СТРОГИЕ ЗАПРЕТЫ: "
        "1. Не используй упоминания Украины, украинских городов (Киев, Львов и др.), политические темы, валюту гривну. Контент строго для РФ. "
        "2. НИКОГДА не используй ссылки на источники ни в тексте, ни в списках. Чисти текст от них полностью. "
        "3. Именна собственные, названия городов пиши с заглавной буквы. Марки пиши в соответствии с марочниками. ГОСТ всегда заглавными."
    )

    user_prompt = f"""
    ИСХОДНЫЕ ДАННЫЕ:
    Название товара: "{tag_name}"
    Базовый текст (фактура): \"\"\"{base_text[:3500]}\"\"\"
    
    {seo_instruction_block}
    
    ЗАДАЧА:
    Напиши {num_blocks} HTML-блоков, разделенных строго разделителем: |||BLOCK_SEP|||
    
    ОБЩИЕ ТРЕБОВАНИЯ:
    1. ОБЪЕМ: Каждый блок должен содержать максимум 800 символов. Раскрывай тему подробно.
    2. ЧИСТОТА: Исключи любые ссылки на источники.
    3. ПОЛЬЗА: Текст должен быть технически грамотным и полезным для специалиста по закупкам. Избегай "воды".
    
    ТРЕБОВАНИЯ К СТРУКТУРЕ КАЖДОГО БЛОКА:
    Каждый из {num_blocks} блоков должен строго соблюдать следующий порядок элементов:
    1. Заголовок (<h2> только для 1-го блока, <h3> для блоков 2-5).
    2. Первый абзац текста (<p>) - развернутый, информативный.
    3. Вводное предложение, подводящее к списку (например: "Основные характеристики:", "Сферы применения:").
    4. Маркированный список (<ul> c <li>).
    5. Второй (завершающий) абзац текста (<p>) - развернутый.
    
    ТЕМЫ БЛОКОВ:
    --- БЛОК 1 (Вводный) ---
    - Заголовок: <h2>{forced_header}</h2>
    - Описание товара, назначение, ключевые особенности.
    
    --- БЛОКИ 2, 3, 4, 5 (Технические детали) ---
    - Заголовки: <h3> (Характеристики, Применение, Производство, Особенности, Сортамент и т.д.).
    - Используй фактуру из "Базового текста".
    
    ФИНАЛЬНЫЕ УСЛОВИЯ:
    - Никаких вводных слов типа "Вот ваш код".
    - Никакого Markdown (```).
    - Только чистый HTML, разбитый через |||BLOCK_SEP|||.
    """
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7 
        )
        content = response.choices[0].message.content
        
        # === ЖЕСТКАЯ ЧИСТКА ОТ МУСОРА И ЖИРНОГО ===
        content = re.sub(r'^```[a-zA-Z]*\s*', '', content.strip())
        content = re.sub(r'\s*```$', '', content.strip())
        content = content.strip().lstrip('`.').strip()
        # Удаляем жирное выделение, если оно все же просочилось
        content = re.sub(r'</?(b|strong)>', '', content, flags=re.IGNORECASE)
        
        blocks = [b.strip() for b in content.split("|||BLOCK_SEP|||") if b.strip()]
        
        cleaned_blocks = []
        for b in blocks:
            cb = re.sub(r'^```[a-zA-Z]*', '', b).strip().lstrip('`.').strip()
            if cb: cleaned_blocks.append(cb)
            
        while len(cleaned_blocks) < num_blocks: cleaned_blocks.append("")
        return cleaned_blocks[:num_blocks]
    except Exception as e:
        return [f"API Error: {str(e)}"] * num_blocks

# ==========================================
# 7. UI TABS RESTRUCTURED
# ==========================================
tab_seo_main, tab_wholesale_main, tab_projects = st.tabs(["📊 SEO Анализ", "🏭 Оптовый генератор", "📁 Проекты"])

# ------------------------------------------
# TAB 1: SEO ANALYSIS (KEPT AS IS)
# ------------------------------------------
with tab_seo_main:
    col_main, col_sidebar = st.columns([65, 35])
    
    # === ЛЕВАЯ КОЛОНКА (ОСНОВНАЯ) ===
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
        
        # --- Обработка авто-переключения ---
        if st.session_state.get('force_radio_switch'):
            st.session_state["competitor_source_radio"] = "Список url-адресов ваших конкурентов"
            st.session_state['force_radio_switch'] = False
        # -----------------------------------------------

        source_type_new = st.radio("Источник", ["Поиск через API Arsenkin (TOP-30)", "Список url-адресов ваших конкурентов"], horizontal=True, label_visibility="collapsed", key="competitor_source_radio")
        source_type = "API" if "API" in source_type_new else "Ручной список"
        
        if source_type == "Ручной список":
            # --- ВСТАВИТЬ ЭТОТ БЛОК ТУТ ---
            # Проверяем, есть ли отложенное обновление от фильтра
            if 'temp_update_urls' in st.session_state:
                st.session_state['persistent_urls'] = st.session_state['temp_update_urls']
                del st.session_state['temp_update_urls']

            # Кнопка сброса
            if st.session_state.get('analysis_done'):
                col_reset, _ = st.columns([1, 4])
                with col_reset:
                    if st.button("🔄 Новый поиск (Сброс)", type="secondary"):
                        keys_to_clear = ['analysis_done', 'analysis_results', 'persistent_urls', 'excluded_urls_auto', 'detected_anomalies']
                        for k in keys_to_clear:
                            if k in st.session_state: del st.session_state[k]
                        st.rerun()

            # Инициализация переменной (если нет)
            if 'persistent_urls' not in st.session_state:
                st.session_state['persistent_urls'] = ""

            has_exclusions = st.session_state.get('excluded_urls_auto') and len(st.session_state.get('excluded_urls_auto')) > 5
            
            if has_exclusions:
                c_url_1, c_url_2 = st.columns(2)
                with c_url_1:
                    # ПРОСТО ВИДЖЕТ. Без value=..., так как мы используем key.
                    # Значение само подтянется из st.session_state['persistent_urls']
                    st.text_area(
                        "✅ Активные конкуренты (Для анализа)", 
                        height=200, 
                        key="persistent_urls" 
                    )
                with c_url_2:
                    st.text_area(
                        "🚫 Авто-исключенные", 
                        height=200, 
                        value=st.session_state.get('excluded_urls_auto', ""),
                        disabled=True # Сделал неактивным, чтобы не путать
                    )
            else:
                st.text_area(
                    "Список ссылок (каждая с новой строки)", 
                    height=200, 
                    key="persistent_urls"
                )

        # ГРАФИК
        if st.session_state.get('analysis_done') and st.session_state.get('analysis_results'):
            results = st.session_state.analysis_results
            if 'relevance_top' in results and not results['relevance_top'].empty:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 График релевантности (Нажмите, чтобы раскрыть)", expanded=False):
                    st.info(
                        """
                        **Логика расширенного анализа**  
                        График строится по максимальной выборке из парсинга для **калибровки линии тренда** и точного выявления аномалий. Это позволяет системе отличить качественные ресурсы от случайных "выбросов" в выдаче.
                        
                        *Важно: Все рекомендации ниже (семантическое ядро, таблицы, структура) формируются строго по выбранному вами лимиту (Топ-10/20) на основе уже очищенных данных, если вы выбрали "Авто-фильтр слабых сайтов. В случае если "Авто-фильтр" не активирован, на результат влияют все ссылки добавленные врчную.*
                        """, 
                        icon="ℹ️"
                    )
                    graph_data = st.session_state.get('full_graph_data', results['relevance_top'])
                    render_relevance_chart(graph_data, unique_key="main")
                st.markdown("<br>", unsafe_allow_html=True)

        # --- КНОПКА ЗАПУСКА ---
        def run_analysis_callback():
            saved_filter_state = st.session_state.get('settings_auto_filter', True)
            keys_to_clear = [
                'analysis_results', 'analysis_done', 'naming_table_df',
                'ideal_h1_result', 'gen_result_df', 'unified_excel_data',
                'detected_anomalies', 'serp_trend_info',
                'excluded_urls_auto'
            ]
            for k in keys_to_clear:
                if k in st.session_state: del st.session_state[k]
            st.session_state.settings_auto_filter = saved_filter_state
            for k in list(st.session_state.keys()):
                if k.endswith('_page'): st.session_state[k] = 1
            st.session_state.start_analysis_flag = True

        st.markdown("<br>", unsafe_allow_html=True) # Отступ перед кнопкой
        st.button(
            "ЗАПУСТИТЬ АНАЛИЗ", 
            type="primary", 
            use_container_width=True, 
            key="start_analysis_btn",
            on_click=run_analysis_callback 
        )

    # === ПРАВАЯ КОЛОНКА (САЙДБАР) ===
    with col_sidebar:
        if not ARSENKIN_TOKEN:
             new_arsenkin = st.text_input("Arsenkin Token", type="password", key="input_arsenkin")
             if new_arsenkin: st.session_state.arsenkin_token = new_arsenkin; ARSENKIN_TOKEN = new_arsenkin 
        if not YANDEX_DICT_KEY:
             new_yandex = st.text_input("Yandex Dict Key", type="password", key="input_yandex")
             if new_yandex: st.session_state.yandex_dict_key = new_yandex; YANDEX_DICT_KEY = new_yandex
        
        st.markdown("⚙️ Настройки поиска")
        st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0"], key="settings_ua")
        st.selectbox("Поисковая система", ["Яндекс", "Google", "Яндекс + Google"], key="settings_search_engine")
        st.selectbox("Регион поиска", list(REGION_MAP.keys()), key="settings_region")
        st.selectbox("Кол-во конкурентов для анализа", [10, 20], index=0, key="settings_top_n")
        
        # Инициализация чекбоксов
        if "settings_noindex" not in st.session_state: st.session_state.settings_noindex = True
        if "settings_alt" not in st.session_state: st.session_state.settings_alt = False
        if "settings_numbers" not in st.session_state: st.session_state.settings_numbers = False
        if "settings_norm" not in st.session_state: st.session_state.settings_norm = True
        if "settings_auto_filter" not in st.session_state: st.session_state.settings_auto_filter = True

        st.checkbox("Исключать <noindex>", key="settings_noindex")
        st.checkbox("Учитывать Alt/Title", key="settings_alt")
        st.checkbox("Учитывать числа", key="settings_numbers")
        st.checkbox("Нормировать по длине", key="settings_norm")
        st.checkbox("Авто-фильтр слабых сайтов", key="settings_auto_filter", help="Сайты с низкой релевантностью будут автоматически перенесены в список исключенных.")
        
        # === [ИЗМЕНЕНИЕ] СПИСКИ ПЕРЕНЕСЕНЫ СЮДА ===
        st.markdown("---")
        st.markdown("🛑 **Исключения**")
        
        if "settings_excludes" not in st.session_state: st.session_state.settings_excludes = DEFAULT_EXCLUDE
        if "settings_stops" not in st.session_state: st.session_state.settings_stops = DEFAULT_STOPS

        st.text_area("Не учитывать домены", height=100, key="settings_excludes", help="Домены, которые парсер пропустит сразу.")
        st.text_area("Стоп-слова", height=100, key="settings_stops", help="Слова, которые не попадут в анализ.")
# ==========================================
    # БЛОК 1: ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
    # ==========================================
    if st.session_state.analysis_done and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        d_score = results['my_score']['depth']
        w_score = results['my_score']['width']
        
        # Цвета баллов
        w_color = "#2E7D32" if w_score >= 80 else ("#E65100" if w_score >= 50 else "#D32F2F")
        
        if 75 <= d_score <= 88:
            d_color = "#2E7D32"; d_status = "ИДЕАЛ (Топ)"
        elif 88 < d_score <= 100:
            d_color = "#D32F2F"; d_status = "ПЕРЕСПАМ (Риск)"
        elif 55 <= d_score < 75:
            d_color = "#F9A825"; d_status = "Средняя"
        else:
            d_color = "#D32F2F"; d_status = "Низкая"

        st.success("Анализ готов!")
        
        # Стили
        st.markdown("""
        <style>
            details > summary { list-style: none; }
            details > summary::-webkit-details-marker { display: none; }
            .details-card { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; margin-bottom: 10px; }
            .card-summary { padding: 12px 15px; cursor: pointer; font-weight: 700; display: flex; justify-content: space-between; }
            .count-tag { background: #e5e7eb; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
            .flat-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; height: 340px; display: flex; flex-direction: column; }
            .flat-header { height: 50px; padding: 0 20px; font-weight: 700; border-bottom: 1px solid #f3f4f6; display: flex; align-items: center; justify-content: space-between; }
            .flat-content { flex-grow: 1; padding: 15px 20px; overflow-y: auto; font-size: 13px; line-height: 1.4; }
            .flat-footer { height: 150px; padding: 12px 20px; border-top: 1px solid #f3f4f6; background: #fafafa; }
            .flat-len-badge { padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 10px; }
            .flat-miss-tag { border: 1px solid #fecaca; color: #991b1b; padding: 2px 6px; font-size: 11px; border-radius: 4px; margin: 2px; display: inline-block; }
        </style>
        """, unsafe_allow_html=True)

        # Вывод баллов
        st.markdown(f"""
        <div style='display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px;'>
            <div style='flex: 1; background:{LIGHT_BG_MAIN}; padding:15px; border-radius:8px; border-left: 5px solid {w_color};'>
                <div style='font-size: 12px; color: #666;'>ШИРИНА (Охват тем)</div>
                <div style='font-size: 24px; font-weight: bold; color: {w_color};'>{w_score}/100</div>
            </div>
            <div style='flex: 1; background:{LIGHT_BG_MAIN}; padding:15px; border-radius:8px; border-left: 5px solid {d_color};'>
                <div style='font-size: 12px; color: #666;'>ГЛУБИНА (Цель: ~80)</div>
                <div style='font-size: 24px; font-weight: bold; color: {d_color};'>{d_score}/100 <span style='font-size:14px; font-weight:normal;'>({d_status})</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- РАСЧЕТ META (Чтобы показать их первыми) ---
        my_data_saved = st.session_state.get('saved_my_data')
        meta_res = None
        
        if 'raw_comp_data' in st.session_state and my_data_saved:
            # Настройки для анализатора
            s_meta = {
                'noindex': True, 'alt_title': False, 'numbers': False, 'norm': True, 
                'ua': "Mozilla/5.0", 'custom_stops': st.session_state.get('settings_stops', "").split()
            }
            meta_res = analyze_meta_gaps(st.session_state['raw_comp_data'], my_data_saved, s_meta)

        # --- ВЫВОД META DASHBOARD (КАРТОЧКИ) ---
        if meta_res:
            st.markdown("### 🧬 Рекомендации Title, Description и H1")
            
            # Хелперы для отрисовки
            def check_len_status(text, type_key):
                length = len(text) if text else 0
                limits = {'Title': (30, 70), 'Description': (150, 250), 'H1': (20, 60)}
                mn, mx = limits.get(type_key, (0,0))
                if mn <= length <= mx: return length, "ХОРОШО", "#059669", "#ECFDF5"
                return length, "ПЛОХО", "#DC2626", "#FEF2F2"

            def render_flat_card(col, label, type_key, icon, txt, score, missing):
                length, status, col_txt, col_bg = check_len_status(txt, type_key)
                rel_col = "#10B981" if score >= 90 else ("#F59E0B" if score >= 50 else "#EF4444")
                
                miss_html = ""
                if missing:
                    tags = "".join([f'<span class="flat-miss-tag">{w}</span>' for w in missing[:10]])
                    miss_html = f"<div style='margin-top:5px;'>{tags}</div>"
                else:
                    miss_html = "<div style='color:#059669; font-weight:bold; margin-top:10px;'>✔ Всё отлично</div>"

                html = f"""
                <div class="flat-card">
                    <div class="flat-header">
                        <div>{icon} {label}</div>
                        <span class="flat-len-badge" style="background:{col_bg}; color:{col_txt}">{length} зн.</span>
                    </div>
                    <div class="flat-content">{txt if txt else '<span style="color:#ccc">Нет данных</span>'}</div>
                    <div class="flat-footer">
                        <div style="display:flex; justify-content:space-between; font-weight:bold; font-size:11px; color:#9ca3af;">
                            <span>РЕЛЕВАНТНОСТЬ</span> 
                            <span style="color:{rel_col}">{score}%</span>
                        </div>
                        <div style="width:100%; height:6px; background:#e5e7eb; border-radius:3px; margin-top:5px; overflow:hidden;">
                            <div style="width:{score}%; height:100%; background:{rel_col};"></div>
                        </div>
                        {miss_html}
                    </div>
                </div>
                """
                col.markdown(html, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            m_s = meta_res['scores']; m_m = meta_res['missing']; m_d = meta_res['my_data']
            
            render_flat_card(c1, "Title", "Title", "📑", m_d['Title'], m_s['title'], m_m['title'])
            render_flat_card(c2, "Description", "Description", "📝", m_d['Description'], m_s['desc'], m_m['desc'])
            render_flat_card(c3, "H1 Заголовок", "H1", "#️⃣", m_d['H1'], m_s['h1'], m_m['h1'])
            
            st.markdown("<br>", unsafe_allow_html=True)

# 1. СЕМАНТИЧЕСКОЕ ЯДРО
        with st.expander("🛒 Семантическое ядро", expanded=True):
            if not st.session_state.get('orig_products') and not st.session_state.get('categorized_general'):
                st.info("⚠️ Данные отсутствуют. Запустите анализ.")
            else:
                # --- ФУНКЦИЯ ПЕРЕСЧЕТА (CALLBACK) ---
                def sync_semantics_with_stoplist():
                    # 1. Считываем, что пользователь оставил/написал в поле стоп-слов
                    raw_input = st.session_state.get('sensitive_words_input_final', "")
                    # Создаем сет (множество) для быстрого поиска, переводим в нижний регистр
                    current_stop_set = set(w.strip().lower() for w in raw_input.split('\n') if w.strip())

                    # 2. Пересобираем отображаемые списки из Мастер-списков (orig_...)
                    # Проверяем: если слова нет в стоп-листе — оно идет в работу
                    st.session_state.categorized_products = [w for w in st.session_state.orig_products if w.lower() not in current_stop_set]
                    st.session_state.categorized_services = [w for w in st.session_state.orig_services if w.lower() not in current_stop_set]
                    st.session_state.categorized_commercial = [w for w in st.session_state.orig_commercial if w.lower() not in current_stop_set]
                    st.session_state.categorized_geo = [w for w in st.session_state.orig_geo if w.lower() not in current_stop_set]
                    st.session_state.categorized_dimensions = [w for w in st.session_state.orig_dimensions if w.lower() not in current_stop_set]
                    st.session_state.categorized_general = [w for w in st.session_state.orig_general if w.lower() not in current_stop_set]

                    # 3. Синхронизируем с генератором (чтобы мусор не попал в теги)
                    all_active_products = st.session_state.categorized_products
                    if len(all_active_products) < 20:
                        st.session_state.auto_tags_words = all_active_products
                        st.session_state.auto_promo_words = []
                    else:
                        mid = math.ceil(len(all_active_products) / 2)
                        st.session_state.auto_tags_words = all_active_products[:mid]
                        st.session_state.auto_promo_words = all_active_products[mid:]
                    
                    st.toast("Списки обновлены!", icon="✅")

                # --- ОТОБРАЖЕНИЕ КАРТОЧЕК ---
                c1, c2, c3 = st.columns(3)
                with c1: render_clean_block("Товары", "🧱", st.session_state.categorized_products)
                with c2: render_clean_block("Гео", "🌍", st.session_state.categorized_geo)
                with c3: render_clean_block("Коммерция", "💰", st.session_state.categorized_commercial)
                
                c4, c5, c6 = st.columns(3)
                with c4: render_clean_block("Услуги", "🛠️", st.session_state.categorized_services)
                with c5: render_clean_block("Размеры/ГОСТ", "📏", st.session_state.categorized_dimensions)
                with c6: render_clean_block("Общие", "📂", st.session_state.categorized_general)

                # --- БЛОК СТОП-СЛОВ (РЕДАКТИРУЕМЫЙ) ---
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🛑 Стоп-лист")
                st.caption("Сюда автоматически попали слова из внутреннего списка стоп-слов, они не будут влиять на расчеты при анализе. Добавьте свои или удалите лишние.")

                col_text, col_btn = st.columns([4, 1])
                
                with col_text:
                    # Используем key, чтобы значение сохранялось в session_state
                    st.text_area(
                        "Список исключений",
                        height=150,
                        key="sensitive_words_input_final", 
                        label_visibility="collapsed"
                    )
                
                with col_btn:
                    st.write("") # Отступ
                    st.button(
                        "🔄 Применить и пересчитать", 
                        type="primary", 
                        use_container_width=True,
                        on_click=sync_semantics_with_stoplist
                    )
                    st.info("Удалите слово из списка слева, чтобы вернуть его в группы выше.")

        # 2. ТАБЛИЦА РЕЛЕВАНТНОСТИ
        with st.expander("🏆 4. Релевантность конкурентов (Таблица)", expanded=True):
            render_paginated_table(results['relevance_top'], "4. Релевантность", "tbl_rel", 
                                   default_sort_col="Позиция", default_sort_order="Возрастание", show_controls=False)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("👇 Дополнительные данные")

        # 3. НАЙМИНГ
        with st.expander("🏷️ Рекомендации по названию товаров", expanded=False):
            if 'naming_table_df' in st.session_state and not st.session_state.naming_table_df.empty:
                st.dataframe(st.session_state.naming_table_df, use_container_width=True, hide_index=True)
            else:
                st.info("Нет данных.")

        # 4. ДЕТАЛИ META (ТАБЛИЦА) - ВОТ ТУТ БЫЛА ОШИБКА
        with st.expander("🕵️ Мета-данные конкурентов", expanded=False):
            # Вставляем защиту: если meta_res нет, не строим таблицу
            if meta_res and 'detailed' in meta_res:
                df_meta_table = pd.DataFrame(meta_res['detailed'])
                # Добавляем строку "Ваш сайт"
                my_row = pd.DataFrame([{
                    'URL': 'ВАШ САЙТ', 
                    'Title': meta_res['my_data']['Title'], 
                    'Description': meta_res['my_data']['Description'], 
                    'H1': meta_res['my_data']['H1']
                }])
                df_meta_table = pd.concat([my_row, df_meta_table], ignore_index=True)
                
                st.dataframe(
                    df_meta_table, 
                    use_container_width=True, 
                    column_config={
                        "URL": st.column_config.LinkColumn("Ссылка"),
                        "Title": st.column_config.TextColumn("Title", width="medium"),
                        "Description": st.column_config.TextColumn("Description", width="large"),
                        "H1": st.column_config.TextColumn("H1", width="small"),
                    }
                )
            else:
                st.warning("Данные по мета-тегам недоступны (возможно, ошибка при анализе).")

        # 5. УПУЩЕННАЯ СЕМАНТИКА
        high = results.get('missing_semantics_high', [])
        low = results.get('missing_semantics_low', [])
        
        if high or low:
            # Считаем общую сумму
            total_missing = len(high) + len(low)
            
            with st.expander(f"🧩 Упущенная семантика ({total_missing})", expanded=False):
                # 1. ВАЖНЫЕ (Медиана >= 1) - Синяя плашка
                if high: 
                    words_high = ", ".join([x['word'] for x in high])
                    st.markdown(f"""
                    <div style='background:#EBF5FF; padding:12px; border-radius:8px; border:1px solid #BFDBFE; color:#1E40AF; margin-bottom:10px;'>
                        <div style='font-weight:bold; margin-bottom:4px;'>🔥 Важные (Есть у большинства конкурентов):</div>
                        <div style='font-size:14px; line-height:1.5;'>{words_high}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 2. ДОПОЛНИТЕЛЬНЫЕ (Медиана < 1) - Серая плашка
                if low: 
                    words_low = ", ".join([x['word'] for x in low])
                    st.markdown(f"""
                    <div style='background:#F8FAFC; padding:12px; border-radius:8px; border:1px solid #E2E8F0; color:#475569;'>
                        <div style='font-weight:bold; margin-bottom:4px;'>🔸 Дополнительные (Встречаются реже):</div>
                        <div style='font-size:13px; line-height:1.5;'>{words_low}</div>
                    </div>
                    """, unsafe_allow_html=True)

# 6. ГЛУБИНА (ЗАКРЫТО)
        with st.expander("📉 1. Глубина (Детальная таблица)", expanded=False):
            render_paginated_table(
                results['depth'], 
                "Глубина", 
                "tbl_depth_1", 
                default_sort_col="Рекомендация", 
                use_abs_sort_default=True
            )

        # 7. TF-IDF (ЗАКРЫТО)
        with st.expander("🧮 3. TF-IDF Анализ", expanded=False):
            render_paginated_table(
                results['hybrid'], 
                "3. TF-IDF", 
                "tbl_hybrid", 
                default_sort_col="TF-IDF ТОП", 
                show_controls=False 
            )
# ==========================================
    # БЛОК 2: СКАНИРОВАНИЕ И РАСЧЕТ
    # ==========================================
    if st.session_state.get('start_analysis_flag'):
        st.session_state.start_analysis_flag = False
        
        # Настройки парсинга
        settings = {
            'noindex': st.session_state.settings_noindex, 
            'alt_title': st.session_state.settings_alt, 
            'numbers': st.session_state.settings_numbers, 
            'norm': st.session_state.settings_norm, 
            'ua': st.session_state.settings_ua, 
            'custom_stops': st.session_state.settings_stops.split()
        }
        
        my_data, my_domain, my_serp_pos = None, "", 0
        current_input_type = st.session_state.get("my_page_source_radio")
        
        # 1. Обработка ВАШЕЙ страницы
        if current_input_type == "Релевантная страница на вашем сайте":
            with st.spinner("Скачивание вашей страницы..."):
                my_data = parse_page(st.session_state.my_url_input, settings, st.session_state.query_input)
                if not my_data: st.error("Ошибка скачивания вашей страницы."); st.stop()
                my_domain = urlparse(st.session_state.my_url_input).netloc
        elif current_input_type == "Исходный код страницы или текст":
            my_data = {'url': 'Local', 'domain': 'local', 'body_text': st.session_state.my_content_input, 'anchor_text': ''}

        st.session_state['saved_my_data'] = my_data 
            
        # 2. Сбор КАНДИДАТОВ
        candidates_pool = []
        current_source_val = st.session_state.get("competitor_source_radio")
        
        # ИСПРАВЛЕНИЕ: Берем настройку пользователя (10 или 20) для ФИНАЛА
        user_target_top_n = st.session_state.settings_top_n
        # А скачиваем всегда МАКСИМУМ (30), чтобы было из чего выбирать
        download_limit = 30 
        
        if "API" in current_source_val:
            if not ARSENKIN_TOKEN: st.error("Отсутствует API токен Arsenkin."); st.stop()
            with st.spinner(f"API Arsenkin (Запрос Топ-30)..."):
                raw_top = get_arsenkin_urls(st.session_state.query_input, st.session_state.settings_search_engine, st.session_state.settings_region, ARSENKIN_TOKEN, depth_val=30)
                
                if not raw_top: st.stop()
                
                excl = [d.strip() for d in st.session_state.settings_excludes.split('\n') if d.strip()]
                agg_list = [
                    "avito", "ozon", "wildberries", "market.yandex", "tiu", "youtube", "vk.com", "yandex",
                    "leroymerlin", "petrovich", "satom", "pulscen", "blizko", "deal.by", "satu.kz", "prom.ua",
                    "wikipedia", "dzen", "rutube", "kino", "otzovik", "irecommend", "profi.ru", "zoon", "2gis",
                    "megamarket.ru", "lamoda.ru", "utkonos.ru", "vprok.ru", "allbiz.ru", "all-companies.ru",
                    "orgpage.ru", "list-org.com", "rusprofile.ru", "e-katalog.ru", "kufar.by", "wildberries.kz",
                    "ozon.kz", "kaspi.kz", "pulscen.kz", "allbiz.kz", "wildberries.uz", "olx.uz", "pulscen.uz",
                    "allbiz.uz", "wildberries.kg", "pulscen.kg", "allbiz.kg", "all.biz", "b2b-center.ru"
                ]
                excl.extend(agg_list)
                for res in raw_top:
                    dom = urlparse(res['url']).netloc.lower()
                    if my_domain and (my_domain in dom or dom in my_domain):
                        if my_serp_pos == 0 or res['pos'] < my_serp_pos: 
                            my_serp_pos = res['pos']
                    is_garbage = False
                    for x in excl:
                        if x.lower() in dom:
                            is_garbage = True
                            break
                    if is_garbage: continue
                    candidates_pool.append(res)
        else:
            raw_input_urls = st.session_state.get("persistent_urls", "")
            candidates_pool = [{'url': u.strip(), 'pos': i+1} for i, u in enumerate(raw_input_urls.split('\n')) if u.strip()]

        if not candidates_pool: st.error("После фильтрации не осталось кандидатов."); st.stop()
        
        # 3. СКАЧИВАНИЕ (Всех 30)
        comp_data_valid = []
        with st.status(f"🕵️ Сканирование (Всего кандидатов: {len(candidates_pool)})...", expanded=True) as status:
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                futures = {
                    executor.submit(parse_page, item['url'], settings, st.session_state.query_input): item 
                    for item in candidates_pool
                }
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
            # Сначала берем ВСЕХ, кто скачался (до 30), для графика
            data_for_graph = comp_data_valid[:download_limit]
            targets_for_graph = [{'url': d['url'], 'pos': d['pos']} for d in data_for_graph]

        # 5. РАСЧЕТ МЕТРИК (ДВОЙНОЙ ПРОГОН)
        with st.spinner("Анализ и фильтрация..."):
            
            # --- ЭТАП 1: Черновой прогон (по всем 30 сайтам) ---
            # Это нужно, чтобы построить график и найти аномалии
            results_full = calculate_metrics(data_for_graph, my_data, settings, my_serp_pos, targets_for_graph)
            
            # Сохраняем ПОЛНЫЕ данные для графика (чтобы на нем были все)
            st.session_state['full_graph_data'] = results_full['relevance_top']
            
            # Анализ аномалий по полному списку
            df_rel_check = results_full['relevance_top']
            good_urls, bad_urls_dicts, trend = analyze_serp_anomalies(df_rel_check)
            st.session_state['serp_trend_info'] = trend
            
            # --- ЭТАП 2: Отбор чистовых (Топ-10/20 без мусора) ---
            
# 1. Берем данные тех сайтов, которые НЕ в списке плохих
            bad_urls_set = set(item['url'] for item in bad_urls_dicts)
            
            # === ИСПРАВЛЕННАЯ ЛОГИКА ФИЛЬТРАЦИИ ===
            # Если это API - мы фильтруем и режем топ.
            # Если это РУЧНОЙ режим - мы НЕ фильтруем (доверяем пользователю).
            if "API" in current_source_val:
                clean_data_pool = [d for d in data_for_graph if d['url'] not in bad_urls_set]
                final_clean_data = clean_data_pool[:user_target_top_n]
            else:
                # В ручном режиме используем ВСЕХ скачанных, не фильтруем "слабых"
                final_clean_data = data_for_graph 
            
            # <--- ВАЖНО: Строка сохранения идет СТРОГО ПОСЛЕ блока if/else --->
            st.session_state['raw_comp_data'] = final_clean_data
            # ------------------------------------------------------------------

            final_clean_targets = [{'url': d['url'], 'pos': d['pos']} for d in final_clean_data]
            
            # 3. ФИНАЛЬНЫЙ РАСЧЕТ (Только по элите)
            results_final = calculate_metrics(final_clean_data, my_data, settings, my_serp_pos, final_clean_targets)
            
# 3. ФИНАЛЬНЫЙ РАСЧЕТ (Только по элите)
            results_final = calculate_metrics(final_clean_data, my_data, settings, my_serp_pos, final_clean_targets)
            st.session_state.analysis_results = results_final
            
            # --- Остальная логика (нейминг, семантика) ---
            naming_df = calculate_naming_metrics(final_clean_data, my_data, settings)
            st.session_state.naming_table_df = naming_df 
            st.session_state.ideal_h1_result = analyze_ideal_name(final_clean_data)
            st.session_state.analysis_done = True
            
            # ==========================================
            # 🔥 БЛОК: КЛАССИФИКАЦИЯ СЕМАНТИКИ (СТРОГО ЗДЕСЬ)
            # ==========================================
            words_to_check = [x['word'] for x in results_final.get('missing_semantics_high', [])]
            
            # Если "важных" слов мало, берем и дополнительные, чтобы заполнить фильтры
            if len(words_to_check) < 5:
                words_to_check.extend([x['word'] for x in results_final.get('missing_semantics_low', [])[:20]])

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

                # Сохраняем оригиналы
                st.session_state.orig_products = categorized['products'] + categorized['sensitive']
                st.session_state.orig_services = categorized['services'] + categorized['sensitive']
                st.session_state.orig_commercial = categorized['commercial'] + categorized['sensitive']
                st.session_state.orig_geo = categorized['geo'] + categorized['sensitive']
                st.session_state.orig_dimensions = categorized['dimensions'] + categorized['sensitive']
                st.session_state.orig_general = categorized['general'] + categorized['sensitive']
                
                st.session_state['sensitive_words_input_final'] = "\n".join(categorized['sensitive'])

            # Обновление списков для генератора
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
            # ==========================================
            # КОНЕЦ БЛОКА КЛАССИФИКАЦИИ
            # ==========================================
            
            
            # === УМНАЯ ФИЛЬТРАЦИЯ (Smart Filter Logic) ===
            
            # 1. Берем данные для проверки аномалий
            if "API" in current_source_val and 'full_graph_data' in st.session_state:
                df_rel_check = st.session_state['full_graph_data']
            else:
                df_rel_check = st.session_state.analysis_results['relevance_top']
            
            # 2. Анализ аномалий
            good_urls, bad_urls_dicts, trend = analyze_serp_anomalies(df_rel_check)
            st.session_state['serp_trend_info'] = trend
            
            # Настройка фильтра
            is_filter_enabled = st.session_state.get("settings_auto_filter", True)
            
            def get_strict_key(u):
                if not u: return ""
                return str(u).lower().strip().replace("https://", "").replace("http://", "").replace("www.", "").rstrip('/')

            final_clean_text = ""
            
            # --- ЛОГИКА РАСПРЕДЕЛЕНИЯ ---
            if is_filter_enabled and bad_urls_dicts:
                # 1. Сохраняем плохих
                st.session_state['detected_anomalies'] = bad_urls_dicts
                
                blacklist_keys = set()
                excluded_display_list = []
                for item in bad_urls_dicts:
                    raw_u = item.get('url', '')
                    if raw_u:
                        blacklist_keys.add(get_strict_key(raw_u))
                        excluded_display_list.append(str(raw_u).strip())
                
                st.session_state['excluded_urls_auto'] = "\n".join(excluded_display_list)
                
                # 2. Собираем хороших
                clean_active_list = []
                seen_keys = set()
                for u in good_urls:
                    key = get_strict_key(u)
                    if key and key not in blacklist_keys and key not in seen_keys:
                        clean_active_list.append(str(u).strip())
                        seen_keys.add(key)
                
                final_clean_text = "\n".join(clean_active_list)
                st.toast(f"Фильтр сработал. Исключено: {len(blacklist_keys)}", icon="✂️")
            
            else:
                # Фильтр выключен или плохих нет - берем всё
                clean_all = []
                seen_all = set()
                combined_pool = good_urls + [x['url'] for x in (bad_urls_dicts or [])]
                for u in combined_pool:
                    key = get_strict_key(u)
                    if key and key not in seen_all:
                        clean_all.append(str(u).strip())
                        seen_all.add(key)
                
                final_clean_text = "\n".join(clean_all)
                # Чистим старые ошибки
                st.session_state.pop('excluded_urls_auto', None)
                st.session_state.pop('detected_anomalies', None)

            # === ФИНАЛЬНАЯ ЗАПИСЬ И ПЕРЕЗАГРУЗКА ===
            # Сохраняем во ВРЕМЕННУЮ переменную
            st.session_state['temp_update_urls'] = final_clean_text
            
            # Ставим флаг переключения радио-кнопки
            st.session_state['force_radio_switch'] = True
            
            # Перезагружаем страницу, чтобы применить изменения СВЕРХУ
            st.rerun()

# ------------------------------------------
# TAB 2: WHOLESALE GENERATOR (COMBINED)
# ------------------------------------------
with tab_wholesale_main:
    st.header("🏭 Единый генератор контента")
    
    # ==========================================
    # 0. ПОДГОТОВКА ДАННЫХ (ИЗ ТЕКУЩЕГО СОСТОЯНИЯ)
    # ==========================================
    cat_products = st.session_state.get('categorized_products', [])
    cat_services = st.session_state.get('categorized_services', [])
    
    # 1. Для Тегов и Промо (Сайдбар исключен)
    structure_keywords = cat_products + cat_services
    count_struct = len(structure_keywords)

    if 'auto_tags_words' in st.session_state and st.session_state.auto_tags_words:
         tags_list_source = st.session_state.auto_tags_words
         promo_list_source = st.session_state.auto_promo_words
    else:
         if count_struct > 0:
            if count_struct < 10:
                tags_list_source = structure_keywords
                promo_list_source = []
            else:
                # Делим всегда пополам (Теги / Промо), Сайдбар игнорируем
                mid = math.ceil(count_struct / 2)
                tags_list_source = structure_keywords[:mid]
                promo_list_source = structure_keywords[mid:]
         else:
             tags_list_source = []
             promo_list_source = []
    
    # Сайдбар всегда пустой
    sidebar_default_text = ""

    tags_default_text = ", ".join(tags_list_source)
    promo_default_text = ", ".join(promo_list_source)

    # 2. Для Таблиц (Размеры/ГОСТ)
    cat_dimensions = st.session_state.get('categorized_dimensions', [])
    tech_context_default = ", ".join(cat_dimensions) if cat_dimensions else ""

    # 3. Разделение Коммерции/Общих и ГЕО
    cat_commercial = st.session_state.get('categorized_commercial', [])
    cat_general = st.session_state.get('categorized_general', [])
    cat_geo = st.session_state.get('categorized_geo', [])
    
    # ИСКЛЮЧАЕМ ГЕО из текстового контекста
    text_context_list_raw = cat_commercial + cat_general
    text_context_default = ", ".join(text_context_list_raw)
    
    # Формируем дефолт для ГЕО блока
    geo_context_default = ", ".join(cat_geo)

    # --- АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ АКТИВНОСТИ МОДУЛЕЙ ---
    auto_check_text = bool(text_context_list_raw)
    auto_check_tags = bool(tags_list_source)
    auto_check_tables = bool(cat_dimensions)
    auto_check_promo = bool(promo_list_source)
    auto_check_geo = bool(cat_geo)

    # ==========================================
    # 1. ВВОДНЫЕ ДАННЫЕ
    # ==========================================
    with st.container(border=True):
        st.subheader("1. Источник и Доступы")
        
        col_source, col_key = st.columns([3, 1])
        
        use_manual_html = st.checkbox("📝 Вставить HTML код страницы", key="cb_manual_html_mode", value=False)
        
        with col_source:
            if use_manual_html:
                manual_html_source = st.text_area(
                    "Исходный код страницы (HTML)", 
                    height=200, 
                    placeholder="<html>...</html>", 
                    help="Скопируйте сюда исходный код страницы."
                )
                main_category_url = None
            else:
                main_category_url = st.text_input(
                    "URL Категории", 
                    placeholder="https://site.ru/catalog/...", 
                    help="Скрипт соберет товары с этой страницы"
                )
                manual_html_source = None

        with col_key:
            try:
                key_from_secrets = st.secrets["GEMINI_KEY"]
            except (FileNotFoundError, KeyError):
                key_from_secrets = ""

            default_key = st.session_state.get('gemini_key_cache', key_from_secrets)
            gemini_api_key = st.text_input("Google Gemini API Key", value=default_key, type="password")

    # ==========================================
    # 2. ВЫБОР МОДУЛЕЙ
    # ==========================================
    st.subheader("2. Какие блоки генерируем?")
    st.info("ℹ️ **Авто-настройка:** Галочки активированы автоматически там, где после анализа нашлись подходящие слова.")
    col_ch1, col_ch2, col_ch3, col_ch4, col_ch5, col_ch6 = st.columns(6)
    
    with col_ch1: use_text = st.checkbox("🤖 AI Тексты", value=auto_check_text)
    with col_ch2: use_tags = st.checkbox("🏷️ Теги", value=auto_check_tags)
    with col_ch3: use_tables = st.checkbox("🧩 Таблицы", value=auto_check_tables)
    with col_ch4: use_promo = st.checkbox("🔥 Промо", value=auto_check_promo)
    
    # ОТКЛЮЧАЕМ САЙДБАР ЗДЕСЬ
    with col_ch5: use_sidebar = st.checkbox("📑 Сайдбар (Откл)", value=False, disabled=True, key="sidebar_disabled_ui")
    
    with col_ch6: use_geo = st.checkbox("🌍 Гео-блок", value=auto_check_geo)

    # ==========================================
    # 3. НАСТРОЙКИ МОДУЛЕЙ
    # ==========================================
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
    
    # Переменная для количества блоков текста (по дефолту 5)
    num_text_blocks_val = 5 

    if any([use_text, use_tags, use_tables, use_promo, use_sidebar, use_geo]):
        st.subheader("3. Настройки модулей")

        # --- AI TEXT ---
        if use_text:
            with st.container(border=True):
                st.markdown("#### 🤖 1. AI Тексты")
                
                # Добавляем выбор количества блоков
                col_txt1, col_txt2 = st.columns([1, 4])
                with col_txt1:
                    num_text_blocks_val = st.selectbox("Кол-во блоков", [1, 2, 3, 4, 5], index=4, key="sb_num_blocks")
                
                with col_txt2:
                    ai_words_input = st.text_area(
                        "Слова для внедрения (Коммерция + Общие)", 
                        value=text_context_default, 
                        height=100, 
                        key="ai_text_context_editable",
                        help="Эти слова нейросеть постарается внедрить в текст."
                    )
                
                text_context_final_list = [x.strip() for x in re.split(r'[,\n]+', ai_words_input) if x.strip()]

        # --- TAGS ---
        if use_tags:
            with st.container(border=True):
                st.markdown("#### 🏷️ 2. Теги")
                kws_input_tags = st.text_area(
                    "Список (Товары + Услуги) - через запятую", 
                    value=tags_default_text, 
                    height=100, 
                    key="kws_tags_auto"
                )
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

# --- ФУНКЦИЯ ГЛУБОКОГО АНАЛИЗА КОНТЕКСТА ДЛЯ ТАБЛИЦ ---
        def generate_context_aware_headers(count, query, dimensions_list, general_list):
            """
            Анализирует запрос И найденные слова (размеры, общие), 
            чтобы понять, какие типы таблиц нужны.
            """
            query_lower = query.lower()
            
            # Превращаем списки слов в одну строку для быстрого поиска
            dims_str = " ".join(dimensions_list).lower()
            gen_str = " ".join(general_list).lower()
            full_context = f"{dims_str} {gen_str} {query_lower}"
            
            # --- 1. ДЕТЕКТОРЫ СИГНАЛОВ (Ищем признаки в семантике) ---
            
            # Признак размеров: есть цифры с 'х' (10х20), слова мм, кг, тонна, размер
            has_sizes_signal = (
                len(dimensions_list) > 0 or 
                bool(re.search(r'\d+[xх*]\d+', full_context)) or 
                any(x in full_context for x in ['размер', 'габарит', 'толщин', 'диаметр', 'раскрой', 'вес', 'масс'])
            )
            
            # Признак стандартов: ГОСТ, ОСТ, ТУ, DIN, AISI
            has_gost_signal = any(x in full_context for x in ['гост', 'din', 'aisi', 'astm', 'ту ', 'стандарт'])
            
            # Признак марок/материала: сталь, сплав, марка, ст.3, 09г2с
            has_grade_signal = any(x in full_context for x in ['марк', 'сплав', 'сталь', 'ст.', 'материал', 'химич', 'состав'])
            
            # Признак применения: для чего, сфера, использование
            has_usage_signal = any(x in full_context for x in ['применен', 'сфер', 'назначен', 'использ'])

            # --- 2. СБОРКА ОЧЕРЕДИ (PRIORITY QUEUE) ---
            # Мы расставляем таблицы в порядке логической важности для пользователя
            
            priority_stack = []
            
            # Если это металлопрокат (есть марки/сплавы), обычно сначала ставят Характеристики или Марки
            if has_grade_signal:
                priority_stack.append("Марки и сплавы")
                
            # Если найдены конкретные размеры - это супер важно, ставим в начало
            if has_sizes_signal:
                # Если уже есть Марки, то Размеры вторыми. Если нет - первыми.
                priority_stack.append("Таблица размеров")
                
            # Если есть упоминание ГОСТов
            if has_gost_signal:
                priority_stack.append("ГОСТы и стандарты")
                
            # Если запрос про хим состав (специфично)
            if "хим" in full_context and "состав" in full_context:
                 # Вставляем "Химический состав" вместо "Марки и сплавы" или рядом
                 if "Марки и сплавы" in priority_stack:
                     idx = priority_stack.index("Марки и сплавы")
                     priority_stack.insert(idx+1, "Химический состав")
                 else:
                     priority_stack.append("Химический состав")

            # --- 3. ЗАПОЛНЕНИЕ ПУСТОТ (DEFAULTS) ---
            # Если мы выбрали 5 таблиц, а сигналов нашли только на 2, нужно добить остальное
            defaults = [
                "Технические характеристики", # Универсальная заглушка
                "Свойства",
                "Сферы использования",
                "Параметры изделия",
                "Аналоги"
            ]
            
            final_headers = []
            # Сначала добавляем то, что нашли умным поиском
            for p in priority_stack:
                if p not in final_headers: final_headers.append(p)
            
            # Добиваем стандартными
            for d in defaults:
                if d not in final_headers: final_headers.append(d)
                
            # Если вдруг всё равно мало (редкий случай)
            while len(final_headers) < count:
                final_headers.append("Характеристики")
                
            return final_headers[:count]

        # --- БЛОК ИНТЕРФЕЙСА TABLES ---
        if use_tables:
            with st.container(border=True):
                st.markdown("#### 🧩 3. Таблицы")
                
                # ДАННЫЕ ДЛЯ АНАЛИЗА (Берем из session_state, куда сохранили результаты анализа)
                raw_query = st.session_state.get('query_input', '')
                found_dims = st.session_state.get('categorized_dimensions', []) # Словарь размеров
                found_general = st.session_state.get('categorized_general', []) # Словарь общих слов
                
                col_ctx, col_cnt = st.columns([3, 1]) 
                
                with col_ctx:
                    tech_context_final_str = st.text_area(
                        "Контекст для таблиц (Марки, ГОСТ, Размеры)", 
                        value=tech_context_default, # Здесь лежат найденные размеры
                        height=68, 
                        key="table_context_editable",
                        help="Эти данные помогут AI составить правильную таблицу."
                    )
                
                with col_cnt:
                    cnt_options = [1, 2, 3, 4, 5]
                    cnt = st.selectbox("Кол-во таблиц", cnt_options, index=1, key="num_tbl_vert_select")

                # --- ЗАПУСК АНАЛИЗАТОРА ---
                # Формируем список заголовков на основе ТОГО, ЧТО НАШЛИ В СЕМАНТИКЕ
                smart_headers_list = generate_context_aware_headers(cnt, raw_query, found_dims, found_general)

                table_presets = [
                    "Технические характеристики", "Свойства", "Параметры изделия",
                    "Общее описание", "Таблица размеров", "Сортамент",
                    "Химический состав", "Физические свойства", "Механические свойства",
                    "Марки и сплавы", "Состав материала", "ГОСТы и стандарты",
                    "Техническая документация", "Требования ГОСТ", "Назначение",
                    "Сферы использования", "Условия эксплуатации", "Где используется",
                    "Классификация", "Модификации", "Аналоги",
                    "Сравнение моделей", "Разновидности"
                ]
                
                table_prompts = []
                st.write("") 
                
                cols = st.columns(cnt)
                
                for i, col in enumerate(cols):
                    with col:
                        st.caption(f"**Таблица {i+1}**")
                        
                        # Авто-выбор на основе анализатора
                        suggested_topic = smart_headers_list[i]
                        
                        try: default_idx = table_presets.index(suggested_topic)
                        except: default_idx = 0
                        
                        is_manual = st.checkbox("Свой заголовок", key=f"cb_tbl_manual_{i}")
                        
                        if is_manual:
                            selected_topic = st.text_input(
                                f"Название табл. {i+1}", value="", 
                                key=f"tbl_topic_custom_{i}", label_visibility="collapsed"
                            )
                            if not selected_topic.strip(): selected_topic = "Характеристики" 
                        else:
                            selected_topic = st.selectbox(
                                f"Тема табл. {i+1}", 
                                table_presets, 
                                index=default_idx, # <--- УМНЫЙ ИНДЕКС
                                key=f"tbl_topic_select_{i}",
                                label_visibility="collapsed"
                            )
                        
                        table_prompts.append(selected_topic)

# --- PROMO (С АНАЛИЗОМ КОММЕРЧЕСКИХ ФАКТОРОВ) ---
        if use_promo:
            with st.container(border=True):
                st.markdown("#### 🔥 4. Промо-блок")
                
                kws_input_promo = st.text_area(
                    "Список (Товары + Услуги) - через запятую", 
                    value=promo_default_text, 
                    height=100, 
                    key="kws_promo_auto"
                )
                global_promo_list = [x.strip() for x in re.split(r'[,\n]+', kws_input_promo) if x.strip()]
                if not global_promo_list: st.warning("⚠️ Список пуст!")
                
                st.markdown("---")
                
                col_p1, col_p2 = st.columns([1, 2])
                with col_p1:
                    promo_presets = [
                        "Смотрите также", "Похожие товары", "Вас может заинтересовать",
                        "Рекомендуем", "Другие предложения", "Вам может пригодиться",
                        "Также в этом разделе", "С этим товаром покупают", "Часто покупают вместе",
                        "Сопутствующие товары", "Хиты продаж", "Выбор покупателей",
                        "Лидеры спроса", "Популярное сейчас", "Топ товаров категории",
                        "Лучшая цена", "Спецпредложения", "Успейте заказать",
                        "Не забудьте добавить", "Вы недавно смотрели"
                    ]

                    # --- ЛОГИКА АНАЛИЗА КОММЕРЦИИ ---
                    # Берем запрос + список коммерческих слов из анализа (Tab 1)
                    raw_query = st.session_state.get('query_input', '').lower()
                    comm_words = st.session_state.get('categorized_commercial', [])
                    
                    # Объединяем в одну строку для проверки
                    comm_context = f"{raw_query} {' '.join(comm_words)}".lower()
                    
                    target_header = "Смотрите также" # Дефолт (информационный)

                    # 1. Явная коммерция (есть слова 'цена', 'купить' в семантике или запросе)
                    is_commercial = any(x in comm_context for x in ["купить", "цена", "заказ", "стоимость", "прайс", "магазин", "корзина"])
                    
                    # 2. Акционные слова
                    is_promo = any(x in comm_context for x in ["акция", "скидк", "распродаж", "выгодн"])
                    
                    # 3. Рейтинговые слова
                    is_top = any(x in comm_context for x in ["топ", "лучш", "рейтинг", "популярн"])

                    if is_promo:
                        target_header = "Спецпредложения"
                    elif is_top:
                        target_header = "Лидеры спроса"
                    elif is_commercial:
                        # Если это явная коммерция, лучше "Покупают вместе" или "Рекомендуем"
                        target_header = "С этим товаром покупают"
                    
                    try: promo_smart_idx = promo_presets.index(target_header)
                    except: promo_smart_idx = 0

                    use_custom_header = st.checkbox("Ввести свой заголовок", key="cb_custom_header")
                    
                    if use_custom_header:
                        promo_title = st.text_input("Ваш заголовок", placeholder="Смотрите также", key="pr_tit_vert")
                    else:
                        promo_title = st.selectbox(
                            "Варианты заголовка", 
                            promo_presets, 
                            index=promo_smart_idx, # <--- УМНЫЙ ВЫБОР
                            key="promo_header_select"
                        )

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

        # --- SIDEBAR ---
        if use_sidebar:
            with st.container(border=True):
                st.markdown("#### 📑 5. Сайдбар")
                kws_input_sidebar = st.text_area(
                    "Список (Товары + Услуги) - с новой строки", 
                    value=sidebar_default_text, 
                    height=100, 
                    key="kws_sidebar_auto"
                )
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

        # --- GEO BLOCK ---
        if use_geo:
            with st.container(border=True):
                st.markdown("#### 🌍 6. Гео-блок")
                kws_input_geo = st.text_area(
                    "Список городов/регионов (из вкладки Анализ) - через запятую", 
                    value=geo_context_default, 
                    height=100, 
                    key="kws_geo_auto"
                )
                global_geo_list = [x.strip() for x in re.split(r'[,\n]+', kws_input_geo) if x.strip()]
                
                if not global_geo_list:
                    st.warning("⚠️ Список городов пуст!")
                else:
                    st.info(f"Будет сгенерирован текст доставки для поля IP_PROP4819 с упоминанием этих городов.")

    st.markdown("---")
    
# ==========================================
    # 4. ЗАПУСК ГЕНЕРАЦИИ (ИСПРАВЛЕННАЯ ЛОГИКА)
    # ==========================================
    
    ready_to_go = True
    if use_manual_html:
        if not manual_html_source: ready_to_go = False
    else:
        if not main_category_url: ready_to_go = False

    if (use_text or use_tables or use_geo) and not gemini_api_key: ready_to_go = False
    if use_promo and df_db_promo is None: ready_to_go = False

# ==========================================
    # 🆘 БЛОК ДИАГНОСТИКИ (Gemini 2.0 Flash)
    # ==========================================
    st.markdown("---")
    with st.expander("🛠️ ДИАГНОСТИКА API (Если есть ошибки)", expanded=True):
        if st.button("📡 ПРОВЕРИТЬ GEMINI 2.0"):
            if not gemini_api_key:
                st.error("❌ Ключ API не введен!")
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=gemini_api_key, base_url="https://litellm.tokengate.ru/v1")
                    response = client.chat.completions.create(
                        model="google/gemini-2.5-pro",
                        messages=[{"role": "user", "content": "Say OK"}]
                    )
                    st.success(f"✅ УСПЕХ! Ответ: {response.choices[0].message.content}")
                except Exception as e:
                    st.error(f"❌ ОШИБКА: {str(e)}")
                    if "404" in str(e):
                        st.info("Попробуйте получить список доступных моделей (возможно, нужна другая):")
                        try:
                            models = [m.name for m in genai.list_models()]
                            st.code("\n".join(models))
                        except: pass

    st.markdown("---")

# ==========================================
    # 4. ЗАПУСК ГЕНЕРАЦИИ (ПАКЕТНАЯ ОБРАБОТКА + LOGS)
    # ==========================================

    # Повторная проверка готовности (на случай если она была только выше)
    if use_manual_html:
        if not manual_html_source: ready_to_go = False
    else:
        if not main_category_url: ready_to_go = False
    if (use_text or use_tables or use_geo) and not gemini_api_key: ready_to_go = False
    if use_promo and df_db_promo is None: ready_to_go = False

# ==========================================
    # 4. УМНЫЙ ЗАПУСК (АВТО-ЦЕПОЧКА) - ВЕРСИЯ С ЗАЩИТОЙ ОТ ДУБЛЕЙ
    # ==========================================
    st.markdown("### 🚀 Управление запуском (Авто-цепочка)")

    # Инициализация переменных
    if 'auto_run_active' not in st.session_state: st.session_state.auto_run_active = False
    if 'auto_current_index' not in st.session_state: st.session_state.auto_current_index = 0

    # КОЛОНКИ УПРАВЛЕНИЯ
    col_batch1, col_batch2, col_batch3 = st.columns([1, 1, 2])
    
    with col_batch1:
        # Если авто-режим активен - поле заблокировано и показывает текущий прогресс
        if st.session_state.auto_run_active:
            start_val = st.session_state.auto_current_index
            st.text_input("Текущий старт:", value=str(start_val), disabled=True)
            start_index = start_val
        else:
            # Если стоим - можно менять руками
            start_index = st.number_input("Начать с товара № (с 0)", min_value=0, value=0, step=1)

    with col_batch2:
        safe_batch_size = st.number_input("Размер пачки (шт)", min_value=1, value=5, help="Лучше 3-5 шт.")
        
    with col_batch3:
        st.write("")
        st.write("")
        enable_auto_chain = st.checkbox("🔄 Включить АВТО-ЦЕПОЧКУ", value=True, help="Скрипт будет сам перезагружаться и продолжать.")

    # === КНОПКИ СТАРТ / СТОП ===
    c_start, c_stop = st.columns([2, 1])
    with c_start:
        # Кнопка СТАРТ
        # Если авто-режим уже идет, кнопка выглядит нажатой (но неактивна для клика, чтобы не двоилось)
        btn_label = "🚀 ЗАПУСТИТЬ ПРОЦЕСС"
        start_clicked = st.button(btn_label, type="primary", disabled=(not ready_to_go or st.session_state.auto_run_active), use_container_width=True)
    
    with c_stop:
        # Кнопка СТОП
        if st.button("🛑 СТОП", type="secondary", use_container_width=True, help="Нажмите, чтобы остановить цепочку после текущей пачки."):
            st.session_state.auto_run_active = False
            st.warning("⛔ Команда остановки принята. Скрипт остановится после текущей пачки.")
            # Мы не делаем st.stop() тут, чтобы дать скрипту возможность отрисовать интерфейс, 
            # просто снимаем флаг активности.

    # ЛОГИКА ЗАПУСКА
    # Мы запускаемся, если нажали Старт ИЛИ если флаг активности уже True (после перезагрузки)
    should_run = start_clicked or st.session_state.auto_run_active

    if should_run:
        # Включаем флаг, если это первый клик
        if not st.session_state.auto_run_active:
             st.session_state.auto_run_active = True
             st.session_state.auto_current_index = start_index

        # Инициализация таблицы (если нет)
        if 'gen_result_df' not in st.session_state or st.session_state.gen_result_df is None:
             st.session_state.gen_result_df = pd.DataFrame(columns=[
                'Page URL', 'Product Name', 'IP_PROP4839', 'IP_PROP4817', 'IP_PROP4818', 
                'IP_PROP4819', 'IP_PROP4820', 'IP_PROP4821', 'IP_PROP4822', 'IP_PROP4823', 
                'IP_PROP4824', 'IP_PROP4816', 'IP_PROP4825', 'IP_PROP4826', 'IP_PROP4834', 
                'IP_PROP4835', 'IP_PROP4836', 'IP_PROP4837', 'IP_PROP4838', 'IP_PROP4829', 'IP_PROP4831'
            ])

        EXCEL_COLUMN_ORDER = st.session_state.gen_result_df.columns.tolist()
        TEXT_CONTAINERS = ['IP_PROP4839', 'IP_PROP4816', 'IP_PROP4838', 'IP_PROP4829', 'IP_PROP4831']

        # Подготовка контекста
        raw_txt_val = st.session_state.get("ai_text_context_editable", "")
        if not raw_txt_val: raw_txt_val = text_context_default
        actual_text_list = [x.strip() for x in re.split(r'[,\n]+', raw_txt_val) if x.strip()]

        raw_geo_val = st.session_state.get("kws_geo_auto", "")
        if not raw_geo_val: raw_geo_val = geo_context_default
        actual_geo_list = [x.strip() for x in re.split(r'[,\n]+', raw_geo_val) if x.strip()]

        user_num_blocks = st.session_state.get("sb_num_blocks", 5)

        # Плейсхолдеры
        live_download_placeholder = st.empty()
        live_table_placeholder = st.empty()
        log_container = st.status(f"🚀 Работаем... (Начали с {start_index})", expanded=True)

        # API Client
        client = None
        if (use_text or use_tables or use_geo) and gemini_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=gemini_api_key, base_url="https://litellm.tokengate.ru/v1")
            except Exception as e:
                log_container.error(f"Ошибка API: {e}")
                st.session_state.auto_run_active = False
                st.stop()
        
        # --- ПОДГОТОВКА ТЕГОВ ---
        all_tags_links = []
        if use_tags:
            if tags_file_content: all_tags_links = [l.strip() for l in io.StringIO(tags_file_content).readlines() if l.strip()]
            elif os.path.exists("data/links_base.txt"):
                with open("data/links_base.txt", "r", encoding="utf-8") as f: all_tags_links = [l.strip() for l in f.readlines() if l.strip()]

        tags_data_prepared = [] 
        if use_tags:
            for kw in global_tags_list:
                tr = transliterate_text(kw).replace(' ', '-').replace('_', '-')
                matches = [u for u in all_tags_links if tr in u.lower()]
                if matches: tags_data_prepared.append((kw, matches)) 
                else:
                    if kw not in actual_text_list: actual_text_list.append(kw)

        p_img_map = {}
        if use_promo and df_db_promo is not None:
            for _, row in df_db_promo.iterrows():
                u = str(row.iloc[0]).strip(); img = str(row.iloc[1]).strip()
                if u and u != 'nan' and img and img != 'nan': p_img_map[u.rstrip('/')] = img
        # -----------------------------------------------

        # Функция парсинга
        def resolve_real_names(urls_list, status_msg=""):
            if not urls_list: return {}
            results_map = {}
            if status_msg: log_container.write(status_msg)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(get_breadcrumb_only, u, st.session_state.settings_ua): u for u in urls_list}
                for future in concurrent.futures.as_completed(future_to_url):
                    url_key = future_to_url[future]
                    try:
                        extracted_name = future.result()
                        if extracted_name: results_map[url_key] = extracted_name
                    except: pass
            return results_map

        # СБОР СТРАНИЦ
        log_container.write("📥 Проверка списка страниц...")
        target_pages = []
        try:
            if use_manual_html:
                soup_main = BeautifulSoup(manual_html_source, 'html.parser')
            else:
                session = requests.Session()
                r = session.get(main_category_url, timeout=30, verify=False)
                if r.status_code == 200: soup_main = BeautifulSoup(r.text, 'html.parser')
                else: st.stop()
            
            if soup_main:
                tags_container = soup_main.find(class_='popular-tags-inner')
                if tags_container:
                    for link in tags_container.find_all('a'):
                        target_pages.append({'url': urljoin(main_category_url or "http://localhost", link.get('href')), 'name': link.get_text(strip=True)})
                if not target_pages:
                    h1_found = soup_main.find('h1')
                    target_pages.append({'url': main_category_url or "local", 'name': h1_found.get_text(strip=True) if h1_found else "Товар"})
        except Exception as e:
            log_container.error(f"Ошибка сбора страниц: {e}")
            st.session_state.auto_run_active = False
            st.stop()

        # === РАСЧЕТ ПАЧКИ ===
        total_found = len(target_pages)
        if start_index >= total_found:
             st.session_state.auto_run_active = False
             st.success("🎉 Все товары обработаны!")
             st.stop()

        end_index = min(start_index + safe_batch_size, total_found)
        target_pages_batch = target_pages[start_index:end_index]
        
        log_container.write(f"📊 ПАЧКА: {start_index+1} — {end_index} из {total_found}")

        # === ЦИКЛ ГЕНЕРАЦИИ ===
        for i, page in enumerate(target_pages_batch):
            
            # [ВАЖНО] ЗАЩИТА ОТ ДУБЛЕЙ
            # Проверяем, есть ли уже этот URL в нашей таблице результатов
            current_urls_in_df = st.session_state.gen_result_df['Page URL'].values
            if page['url'] in current_urls_in_df:
                log_container.warning(f"⚠️ Пропуск дубля: {page['name']} (Уже есть в таблице)")
                continue  # Пропускаем итерацию, переходим к следующему

            current_num = start_index + i + 1
            log_container.write(f"▶️ **[{current_num}/{total_found}] {page['name']}**")
            
            try:
                base_text_raw, _, real_header_h2, _ = get_page_data_for_gen(page['url'])
                header_for_ai = real_header_h2 if real_header_h2 else page['name']
                row_data = {col: "" for col in EXCEL_COLUMN_ORDER}
                row_data['Page URL'] = page['url']; row_data['Product Name'] = header_for_ai
                for k, v in STATIC_DATA_GEN.items():
                    if k in row_data: row_data[k] = v
                
                injections = []

                # ТЕГИ
                if use_tags and tags_data_prepared:
                    tags_pool = tags_data_prepared
                    if len(tags_pool) > 15: tags_pool = random.sample(tags_pool, 15)
                    selected_urls_map = {} 
                    for kw, links in tags_pool:
                        valid = [u for u in links if u.rstrip('/') != page['url'].rstrip('/')]
                        if valid:
                             sel = random.choice(valid); selected_urls_map[sel] = kw
                    urls_to_fetch = list(selected_urls_map.keys())
                    real_names_map = resolve_real_names(urls_to_fetch)
                    html_t = []
                    for u in urls_to_fetch:
                        display_name = real_names_map.get(u, selected_urls_map[u])
                        if display_name == selected_urls_map[u]: display_name = display_name.capitalize()
                        html_t.append(f'<a href="{u}" class="tag-item">{display_name}</a>')
                    if html_t:
                        tags_block = f'''<div class="popular-tags-text"><div class="popular-tags-inner-text"><div class="tag-items">{"\n".join(html_t)}</div></div></div>'''
                        injections.append(tags_block)

                # ТАБЛИЦЫ
                if use_tables and client:
                    for t_topic in table_prompts:
                        ctx = f"Данные: {tech_context_final_str}"
                        prompt_tbl = f"Create HTML <table> for '{header_for_ai}'. Topic: {t_topic}. Context: {ctx}. No markdown."
                        try:
                            resp = client.chat.completions.create(model="google/gemini-2.5-pro", messages=[{"role": "user", "content": prompt_tbl}], temperature=0)
                            raw_table = resp.choices[0].message.content.replace("```html", "").replace("```", "").strip()
                            injections.append(raw_table)
                        except: pass
                
                # ПРОМО
                if use_promo and p_img_map:
                    p_cands = [u for u in p_img_map.keys() if u.rstrip('/') != page['url'].rstrip('/')]
                    if p_cands:
                        sel_p = random.sample(p_cands, min(8, max(3, len(p_cands))))
                        promo_names_map = resolve_real_names(sel_p)
                        gallery_items = []
                        for u in sel_p:
                            nm = promo_names_map.get(u, force_cyrillic_name_global(u.split("/")[-1]))
                            img_src = p_img_map[u]
                            item_html = f'''<div class="gallery-item"><h3><a href="{u}" target="_blank">{nm}</a></h3><figure><a href="{u}" target="_blank"><picture><img src="{img_src}" loading="lazy"></picture></a></figure></div>'''
                            gallery_items.append(item_html)

                        p_html = f'''
<style>.outer-full-width-section {{ padding: 25px 0; width: 100%; }}.gallery-content-wrapper {{ max-width: 1400px; margin: 0 auto; padding: 25px 15px; box-sizing: border-box; border-radius: 10px; overflow: hidden; background-color: #F6F7FC; }}h3.gallery-title {{ color: #3D4858; font-size: 1.8em; font-weight: normal; padding: 0; margin-top: 0; margin-bottom: 15px; text-align: left; }}.five-col-gallery {{ display: flex; justify-content: flex-start; align-items: flex-start; gap: 20px; margin-bottom: 0; padding: 0; list-style: none; flex-wrap: nowrap !important; overflow-x: auto !important; padding-bottom: 15px; }}.gallery-item {{ flex: 0 0 260px !important; box-sizing: border-box; text-align: center; scroll-snap-align: start; }}.gallery-item h3 {{ font-size: 1.1em; margin-bottom: 8px; font-weight: normal; text-align: center; line-height: 1.1em; display: block; min-height: 40px; }}.gallery-item h3 a {{ text-decoration: none; color: #333; display: block; height: 100%; display: flex; align-items: center; justify-content: center; transition: color 0.2s ease; }}.gallery-item h3 a:hover {{ color: #007bff; }}.gallery-item figure {{ width: 100%; margin: 0; float: none !important; height: 260px; overflow: hidden; margin-bottom: 5px; border-radius: 8px; }}.gallery-item figure a {{ display: block; height: 100%; text-decoration: none; }}.gallery-item img {{ width: 100%; height: 100%; display: block; margin: 0 auto; object-fit: cover; transition: transform 0.3s ease; border-radius: 8px; }}.gallery-item figure a:hover img {{ transform: scale(1.05); }}</style>
<div class="outer-full-width-section"><div class="gallery-content-wrapper"><h3 class="gallery-title">{promo_title}</h3><div class="five-col-gallery">{"".join(gallery_items)}</div></div></div>'''
                        injections.append(p_html)

                # ТЕКСТ
                blocks = [""] * 5
                if use_text and client:
                    log_container.write(f"   ↳ 🤖 Пишем текст...")
                    blocks_raw = generate_ai_content_blocks(gemini_api_key, base_text_raw or "", page['name'], header_for_ai, user_num_blocks, actual_text_list)
                    cleaned_blocks = [b.replace("```html", "").replace("```", "").strip() for b in blocks_raw]
                    for i_b in range(len(cleaned_blocks)):
                        if i_b < 5: blocks[i_b] = cleaned_blocks[i_b]

                # СЛИЯНИЕ
                effective_blocks_count = max(1, user_num_blocks)
                for i_inj, inj in enumerate(injections):
                    target_idx = i_inj % effective_blocks_count
                    blocks[target_idx] = blocks[target_idx] + "\n\n" + inj

                # ГЕО
                if use_geo and client:
                    log_container.write(f"   ↳ 🌍 Пишем доставку...")
                    try:
                         cities = ", ".join(random.sample(actual_geo_list, min(15, len(actual_geo_list))))
                         prompt_geo = f"Write ONE HTML paragraph about delivery to {cities}. No markdown."
                         resp = client.chat.completions.create(model="google/gemini-2.5-pro", messages=[{"role": "user", "content": prompt_geo}], temperature=0.1)
                         row_data['IP_PROP4819'] = resp.choices[0].message.content.replace("```html", "").replace("```", "").strip()
                    except: pass

                for i_c, c_name in enumerate(TEXT_CONTAINERS):
                    row_data[c_name] = blocks[i_c]

                # === СОХРАНЕНИЕ ===
                new_row_df = pd.DataFrame([row_data])
                st.session_state.gen_result_df = pd.concat([st.session_state.gen_result_df, new_row_df], ignore_index=True)
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    st.session_state.gen_result_df.to_excel(writer, index=False)
                st.session_state.unified_excel_data = buffer.getvalue()
                
                try: st.session_state.gen_result_df.to_excel("backup_auto.xlsx", index=False)
                except: pass

                live_table_placeholder.dataframe(st.session_state.gen_result_df.tail(3), use_container_width=True)
                
                with live_download_placeholder.container():
                    st.download_button(
                        label=f"💾 СКАЧАТЬ ТЕКУЩИЙ РЕЗУЛЬТАТ ({len(st.session_state.gen_result_df)} стр.)",
                        data=st.session_state.unified_excel_data,
                        file_name=f"FULL_RESULT_{int(time.time())}.xlsx",
                        mime="application/vnd.ms-excel",
                        key=f"dl_live_{int(time.time())}_{i}"
                    )

            except Exception as e:
                log_container.error(f"Сбой: {e}")

        log_container.update(label=f"✅ Пачка {start_index}-{end_index} готова!", state="complete", expanded=False)
        
        # === ЛОГИКА АВТО-ПЕРЕЗАПУСКА ===
        if enable_auto_chain:
            # Проверяем, не нажали ли СТОП в процессе
            if not st.session_state.auto_run_active:
                st.warning("⛔ Цепочка была остановлена вручную.")
            else:
                next_start = end_index
                if next_start < total_found:
                    st.session_state.auto_current_index = next_start
                    st.info(f"⏳ Перезагрузка через 1 сек... Следующая пачка с {next_start}.")
                    time.sleep(1)
                    st.rerun() 
                else:
                    st.session_state.auto_run_active = False
                    st.balloons()
                    st.success("🏁 ГЕНЕРАЦИЯ ПОЛНОСТЬЮ ЗАВЕРШЕНА!")

    # КНОПКА СКАЧИВАНИЯ (ПОЯВЛЯЕТСЯ СРАЗУ ПОСЛЕ ПЕРВОЙ СТРОКИ)
    if st.session_state.get('unified_excel_data') is not None:
        count = len(st.session_state.gen_result_df)
        st.success(f"Готово! Обработано строк: {count}")
        st.download_button(
            label=f"📥 СКАЧАТЬ РЕЗУЛЬТАТ ({count} стр.)",
            data=st.session_state.unified_excel_data,
            file_name=f"wholesale_result_{int(time.time())}.xlsx",
            mime="application/vnd.ms-excel",
            key="btn_dl_fixed"
        )

    # ==========================================
    # 5. БЛОК ПРЕДПРОСМОТРА
    # ==========================================
    if 'gen_result_df' in st.session_state and st.session_state.gen_result_df is not None and not st.session_state.gen_result_df.empty:
        st.markdown("---")
        st.header("👀 Предпросмотр по колонкам")
        
        st.markdown("""
        <style>
            .preview-box {
                border: 1px solid #e2e8f0;
                background-color: #ffffff;
                padding: 20px;
                border-radius: 8px;
                max-height: 600px;
                overflow-y: auto;
                box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
            }
        </style>
        """, unsafe_allow_html=True)

        df_p = st.session_state.gen_result_df
        
        if 'Product Name' in df_p.columns:
            sel_p = st.selectbox("Страница:", df_p['Product Name'].tolist(), key="ws_prev_sel")
            row_p = df_p[df_p['Product Name'] == sel_p].iloc[0]
            
            # Фильтр табов
            relevant_cols = []
            if use_text or use_sidebar or use_tags or use_tables or use_promo:
                relevant_cols.extend(['IP_PROP4839', 'IP_PROP4816', 'IP_PROP4838', 'IP_PROP4829', 'IP_PROP4831'])
            if use_geo:
                relevant_cols.append('IP_PROP4819')

            active_tabs = [c for c in relevant_cols if str(row_p.get(c, "")).strip() != ""]
            
            if active_tabs:
                tabs = st.tabs(active_tabs)
                for i, col in enumerate(active_tabs):
                    with tabs[i]:
                        content_to_show = str(row_p[col])
                        st.markdown(f"<div class='preview-box'>{content_to_show}</div>", unsafe_allow_html=True)
            else:
                st.info("Нет данных для отображения.")
# ==========================================
# TAB 3: PROJECT MANAGER (SAVE/LOAD)
# ==========================================
with tab_projects:
    st.header("📁 Управление проектами")
    st.markdown("Здесь вы можете сохранить текущее состояние анализа в файл или загрузить ранее сохраненный проект.")

    col_save, col_load = st.columns(2)

    # --- ФУНКЦИЯ ВОССТАНОВЛЕНИЯ (CALLBACK) ---
    def restore_state_callback(data_to_restore):
        """
        Эта функция запускается ДО перерисовки интерфейса.
        Поэтому здесь можно безопасно обновлять session_state.
        """
        try:
            state_dict = data_to_restore["state"]
            restored_count = 0
            
            # 1. Обновляем session_state
            for k, v in state_dict.items():
                st.session_state[k] = v
                restored_count += 1
            
            # 2. Принудительные флаги
            st.session_state['analysis_done'] = True
            
            # 3. Уведомление (появится после перезагрузки)
            st.toast(f"✅ Успешно восстановлено {restored_count} параметров!", icon="🎉")
            
        except Exception as e:
            st.error(f"Ошибка внутри callback: {e}")

    # --- БЛОК СОХРАНЕНИЯ ---
    with col_save:
        with st.container(border=True):
            st.subheader("💾 Сохранить проект")
            
            if not st.session_state.get('analysis_done'):
                st.warning("⚠️ Сначала проведите анализ (Вкладка SEO), чтобы было что сохранять.")
            else:
                st.info("Будут сохранены: все таблицы, списки семантики, настройки, ссылки конкурентов и результаты генерации.")
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                query_slug = transliterate_text(st.session_state.get('query_input', 'project'))[:20]
                default_filename = f"GAR_PRO_{query_slug}_{timestamp}.pkl"
                
                project_snapshot = {
                    "meta": {
                        "version": "2.6",
                        "date": str(datetime.datetime.now())
                    },
                    "state": {}
                }
                
                # Ключи для сохранения
                keys_to_save = [
                    'analysis_results', 'analysis_done', 'naming_table_df', 'ideal_h1_result',
                    'detected_anomalies', 'serp_trend_info', 'full_graph_data',
                    'categorized_products', 'categorized_services', 'categorized_commercial',
                    'categorized_dimensions', 'categorized_geo', 'categorized_general', 'categorized_sensitive',
                    'orig_products', 'orig_services', 'orig_commercial', 
                    'orig_dimensions', 'orig_geo', 'orig_general',
                    'sensitive_words_input_final', 'auto_tags_words', 'auto_promo_words',
                    'my_url_input', 'query_input', 'my_content_input', 'my_page_source_radio',
                    'competitor_source_radio', 'persistent_urls', 'excluded_urls_auto',
                    'settings_excludes', 'settings_stops', 'arsenkin_token', 'yandex_dict_key',
                    'settings_ua', 'settings_search_engine', 'settings_region', 'settings_top_n',
                    'settings_noindex', 'settings_alt', 'settings_numbers', 'settings_norm',
                    'gen_result_df', 'unified_excel_data'
                ]
                
                for k in keys_to_save:
                    if k in st.session_state:
                        project_snapshot["state"][k] = st.session_state[k]

                try:
                    pickle_data = pickle.dumps(project_snapshot)
                    st.download_button(
                        label="📥 Скачать файл проекта (.pkl)",
                        data=pickle_data,
                        file_name=default_filename,
                        mime="application/octet-stream",
                        type="primary",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Ошибка при упаковке данных: {e}")

    # --- БЛОК ЗАГРУЗКИ ---
    with col_load:
        with st.container(border=True):
            st.subheader("📂 Загрузить проект")
            
            uploaded_file = st.file_uploader("Выберите файл .pkl", type=["pkl"], key="project_loader")
            
            if uploaded_file is not None:
                try:
                    loaded_data = pickle.load(uploaded_file)
                    
                    if isinstance(loaded_data, dict) and "state" in loaded_data:
                        date_str = loaded_data['meta'].get('date', 'Неизвестно')
                        st.success(f"Проект распознан! (Дата: {date_str})")
                        
                        # ИСПОЛЬЗУЕМ ON_CLICK И ARGS
                        # Это главное исправление: функция restore_state_callback вызовется ДО того,
                        # как Streamlit начнет отрисовывать виджеты заново.
                        st.button(
                            "🚀 ВОССТАНОВИТЬ СОСТОЯНИЕ", 
                            type="primary", 
                            use_container_width=True,
                            on_click=restore_state_callback,
                            args=(loaded_data,)
                        )
                    else:
                        st.error("❌ Неверный формат файла проекта.")
                except Exception as e:
                    st.error(f"❌ Ошибка чтения файла: {e}")


























