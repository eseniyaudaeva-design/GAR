import streamlit as st
import pymorphy3 as pymorphy2
import pandas as pd
import numpy as np
import requests
import gzip
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
import csv
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
# ЯДРО БАЗЫ ДАННЫХ (КЭШИРОВАНИЕ SEO-АНАЛИЗА НА 90 ДНЕЙ)
# ==========================================
import sqlite3
import json
import datetime

def init_seo_db():
    conn = sqlite3.connect('seo_cache.db', timeout=10)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS seo_analysis (
            query TEXT PRIMARY KEY,
            timestamp TEXT,
            parsed_data TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_seo_db()

def get_cached_analysis(query, region="Москва"):
    if not query: return None
    try:
        conn = sqlite3.connect('seo_cache.db', timeout=10)
        c = conn.cursor()
        
        # Авто-чистка старья (90 дней)
        expiry_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute('DELETE FROM seo_analysis WHERE timestamp < ?', (expiry_date,))
        conn.commit()
        
        # Составляем уникальный ключ: запрос + регион
        db_key = f"{query.lower().strip()}_{region.lower().strip()}"
        
        c.execute('SELECT timestamp, parsed_data FROM seo_analysis WHERE query = ?', (db_key,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[1])
    except sqlite3.OperationalError:
        return None 
    return None

def save_cached_analysis(query, region, data_for_graph):
    try:
        conn = sqlite3.connect('seo_cache.db', timeout=10)
        c = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Составляем уникальный ключ: запрос + регион
        db_key = f"{query.lower().strip()}_{region.lower().strip()}"
        
        c.execute('''
            INSERT OR REPLACE INTO seo_analysis (query, timestamp, parsed_data)
            VALUES (?, ?, ?)
        ''', (db_key, timestamp, json.dumps(data_for_graph)))
        conn.commit()
        conn.close()
    except: pass

# ==========================================
# ДВИЖОК ГЕНЕРАЦИИ ОТЗЫВОВ (БЕЗ ИИ)
# ==========================================
import pymorphy3
import random
import re
import pandas as pd
import datetime

@st.cache_resource
def init_morph():
    return pymorphy3.MorphAnalyzer()

morph = init_morph()

LSI_BRIDGES = [
    {"template": "Отдельно хочу отметить **{}**.", "case": "accs"},
    {"template": "Также порадовало наличие **{}**.", "case": "gent"},
    {"template": "Обратили внимание на **{}** – всё отлично.", "case": "accs"},
    {"template": "Кстати, с **{}** тоже никаких проблем не возникло.", "case": "ablt"},
    {"template": "К слову, **{}** тут на высшем уровне.", "case": "nomn"}
]

def inflect_lsi_phrase(phrase, target_case):
    words = str(phrase).split()
    inflected_words = []
    for word in words:
        parsed_word = morph.parse(word)[0]
        try:
            inf_word = parsed_word.inflect({target_case})
            if inf_word:
                inflected_words.append(inf_word.word)
            else:
                inflected_words.append(word)
        except AttributeError:
            inflected_words.append(word)
    return " ".join(inflected_words)

def generate_random_date():
    start_date = datetime.datetime(2026, 1, 1)
    end_date = datetime.datetime(2026, 2, 10)
    random_days = random.randrange((end_date - start_date).days + 1)
    return (start_date + datetime.timedelta(days=random_days)).strftime("%d.%m.%Y")

def build_review_from_repo(template, variables_dict, repo_fio, lsi_words):
    def replace_var(match):
        var_name = match.group(1).strip()
        if var_name == "дата":
            return generate_random_date()
        if var_name in variables_dict:
            return str(random.choice(variables_dict[var_name])).strip()
        return match.group(0)

    draft = re.sub(r'\{([^}]+)\}', replace_var, str(template))
    
    forbidden_roots = [
        "украин", "ukrain", "ua", "всу", "зсу", "ато", "сво", "войн",
        "киев", "львов", "харьков", "одесс", "днепр", "мариуполь",
        "донец", "луганс", "днр", "лнр", "донбасс", "мелитополь",
        "бердянск", "бахмут", "запорожь", "херсон", "крым",
        "политик", "спецоперац"
    ]
    clean_lsi = [w for w in lsi_words if not any(root in str(w).lower() for root in forbidden_roots) and len(str(w)) > 2]
    
    used_lsi = []
    if clean_lsi:
        lsi_word = random.choice(clean_lsi)
        bridge = random.choice(LSI_BRIDGES)
        inflected_lsi = inflect_lsi_phrase(lsi_word, bridge["case"])
        lsi_sentence = bridge["template"].format(inflected_lsi)
        
        sentences = [s.strip() for s in draft.split('.') if s.strip()]
        insert_pos = random.randint(1, max(1, len(sentences)))
        sentences.insert(insert_pos, lsi_sentence)
        draft = ". ".join(sentences) + "."
        used_lsi.append(inflected_lsi)

    draft = re.sub(r'\s+', ' ', draft)
    draft = draft.replace(' .', '.').replace(' ,', ',').replace(' - ', ' – ')
    sentences = draft.split('. ')
    draft = '. '.join([s.capitalize() for s in sentences]).strip()
    
    # Сборка ФИО
    random_name = "Аноним"
    available_genders = [g for g in ['MALE', 'FEMALE'] if repo_fio[g]['names'] and repo_fio[g]['surnames']]
    if available_genders:
        chosen_gender = random.choice(available_genders)
        rand_name = random.choice(repo_fio[chosen_gender]['names'])
        rand_surname = random.choice(repo_fio[chosen_gender]['surnames'])
        
        if repo_fio[chosen_gender]['patronymics'] and random.random() > 0.5:
            rand_patronymic = random.choice(repo_fio[chosen_gender]['patronymics'])
            random_name = f"{rand_name} {rand_patronymic} {rand_surname}"
        else:
            random_name = f"{rand_name} {rand_surname}"

    return random_name, draft, used_lsi
# ==========================================

# ==========================================
# FIX FOR PYTHON 3.11+
# ==========================================
if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

try:
    # Используем pymorphy3, но импортируем как pymorphy2, 
    # чтобы не переписывать весь остальной код
    import pymorphy3 as pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    USE_NLP = True
except ImportError:
    # Резервный вариант, если вдруг стоит старая версия
    try:
        import pymorphy2
        morph = pymorphy2.MorphAnalyzer()
        USE_NLP = True
    except Exception as e:
        st.error(f"❌ ОШИБКА: Не удалось загрузить pymorphy. Детали: {e}")
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


# ... (тут идут импорты) ...
import datetime

# === ВСТАВИТЬ СЮДА (СТРОКА ~40) ===
if 'SUPER_GLOBAL_KEY' not in st.session_state:
    st.session_state.SUPER_GLOBAL_KEY = ""
    # Пробуем подтянуть из secrets сразу при старте
    try: st.session_state.SUPER_GLOBAL_KEY = st.secrets["GEMINI_KEY"]
    except: pass
# ==================================

@st.cache_data
def load_names_db():
    """Загрузка миллионника из папки data"""
    if os.path.exists("data/users_db.csv.gz"):
        return pd.read_csv("data/users_db.csv.gz", compression='gzip')
    return pd.DataFrame(columns=['template_type', 'username', 'gender'])

def get_diverse_authors(n):
    """Выбор авторов: ровно 1 аноним, остальные равномерно распределены"""
    df = load_names_db()
    if df.empty: return [{"name": "Имя скрыто", "type": "anonymous", "gender": "Н"}] * n
    
    df_anon = df[df['template_type'] == 'anonymous']
    df_others = df[df['template_type'] != 'anonymous']
    
    res = []
    if not df_anon.empty:
        res.append(df_anon.sample(1).iloc[0].to_dict())
    
    needed = n - len(res)
    other_types = df_others['template_type'].unique()
    if len(other_types) > 0:
        per_type = needed // len(other_types)
        rem = needed % len(other_types)
        for t in other_types:
            count = per_type + (1 if rem > 0 else 0)
            if rem > 0: rem -= 1
            slice_dt = df_others[df_others['template_type'] == t]
            if not slice_dt.empty:
                res.extend(slice_dt.sample(min(len(slice_dt), count)).to_dict('records'))
    
    random.shuffle(res)
    return [{"name": r['username'], "type": r['template_type'], "gender": r['gender']} for r in res[:n]]

def get_balanced_ratings(n):
    """Генерирует оценки: одна 3.5 обязательна, средний балл плавающий (4.7 - 4.95)"""
    target_avg = random.uniform(4.7, 4.95) # У каждой категории будет свой средний балл
    ratings = [3.5] # Гарантированная 3.5
    for _ in range(n - 1): ratings.append(5.0) # Забиваем остальное пятерками
    
    # Постепенно снижаем некоторые пятерки до 4.5 или 4.0, пока не приблизимся к таргету
    indices = list(range(1, n))
    random.shuffle(indices)
    for idx in indices:
        current_avg = sum(ratings) / n
        if current_avg <= target_avg: break
        
        new_val = random.choice([4.0, 4.5])
        if (sum(ratings) - ratings[idx] + new_val) / n >= 4.7:
            ratings[idx] = new_val
    
    random.shuffle(ratings)
    return ratings
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

def get_h1_from_url(url):
    """Парсит H1 со страницы для использования в качестве поискового запроса."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        # Используем прокси, если они настроены глобально в вашем скрипте
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            h1 = soup.find('h1')
            if h1:
                return h1.get_text(strip=True)
    except Exception as e:
        print(f"Ошибка парсинга H1: {e}")
    return ""

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
    # Добавляем проверку на None, чтобы график не падал при переключении категорий
    if df_rel is None or (isinstance(df_rel, pd.DataFrame) and df_rel.empty):
        return

    # Защита от пустых колонок
    if 'Позиция' not in df_rel.columns:
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

# @st.cache_data
def load_lemmatized_dictionaries():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "data")
    
    sets = {"products": set(), "commercial": set(), "specs": set(), "geo": set(), "services": set(), "sensitive": set()}
    files_map = {
        "metal_products.json": "products", "commercial_triggers.json": "commercial",
        "geo_locations.json": "geo", "services_triggers.json": "services",
        "tech_specs.json": "specs", "SENSITIVE_STOPLIST.json": "sensitive"
    }

    for filename, set_key in files_map.items():
        # 1. Ищем файл и в папке data, и в корне (чтобы точно не промахнуться)
        path_data = os.path.join(base_path, filename)
        path_root = os.path.join(script_dir, filename)
        full_path = path_data if os.path.exists(path_data) else (path_root if os.path.exists(path_root) else None)
        
        if not full_path: continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f) 
                
                words_bucket = []
                # 2. Безопасное извлечение (чтобы не разбивал строки на отдельные буквы)
                if isinstance(data, dict):
                    for val in data.values():
                        if isinstance(val, list): words_bucket.extend(val)
                        elif isinstance(val, str): words_bucket.append(val)
                elif isinstance(data, list):
                    words_bucket = data
                
                # 3. Нормализация
                for phrase in words_bucket:
                    w_clean = str(phrase).lower().strip().replace('ё', 'е')
                    # Прибиваем все виды тире к одному стандартному дефису
                    w_clean = re.sub(r'[\—\–\−]', '-', w_clean)
                    
                    if len(w_clean) < 2: continue
                    sets[set_key].add(w_clean)
                    if morph:
                        sets[set_key].add(morph.parse(w_clean)[0].normal_form.replace('ё', 'е'))
        except Exception as e:
            # Теперь если в JSON будет опечатка, скрипт покажет тебе красную ошибку, а не промолчит!
            st.error(f"❌ Ошибка загрузки файла {filename}: {e}")

    return sets["products"], sets["commercial"], sets["specs"], sets["geo"], sets["services"], sets["sensitive"]


def classify_semantics_with_api(words_list, yandex_key):
    PRODUCTS_SET, COMM_SET, SPECS_SET, GEO_SET, SERVICES_SET, SENS_SET = load_lemmatized_dictionaries()
    FULL_SENSITIVE = SENS_SET.union(SENSITIVE_STOPLIST)

    # === КУВАЛДА ДЛЯ ГЕО ===
    # Очищаем все города в словаре от дефисов для железобетонного сравнения
    GEO_SET_CLEAN = {str(g).replace('-', '').replace(' ', '') for g in GEO_SET}
    # =======================

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
        
        lemma = word_lower
        if morph: lemma = morph.parse(word_lower)[0].normal_form

        # === 2. РАЗМЕРЫ / ГОСТ ===
        if word_lower in SPECS_SET or lemma in SPECS_SET or dim_pattern.search(word_lower) or grade_pattern.match(word_lower) or word_lower.isdigit():
            categories['dimensions'].add(word_lower); continue
            
        # === 3. ГЕО (ЖЕЛЕЗОБЕТОННАЯ ПРОВЕРКА) ===
        word_clean = word_lower.replace('-', '').replace(' ', '')
        lemma_clean = lemma.replace('-', '').replace(' ', '')
        
        # --- ЛОВУШКА ДЛЯ ПЕТЕРБУРГА ---
        if "петербург" in word_lower:
            st.error(f"🛑 ОТЛАДКА: Из текста пришло: '{word_clean}'. В словаре нашлось: {[g for g in GEO_SET_CLEAN if 'петербург' in g]}")
        # ------------------------------
        
        if word_clean in GEO_SET_CLEAN or lemma_clean in GEO_SET_CLEAN:
            categories['geo'].add(word_lower); continue
        
        if word_clean in GEO_SET_CLEAN or lemma_clean in GEO_SET_CLEAN:
            categories['geo'].add(word_lower); continue

        # === 4. УСЛУГИ ===
        if lemma in SERVICES_SET or word_lower in SERVICES_SET or lemma.endswith(('обработка', 'изготовление')) or lemma == "резка":
            categories['services'].add(word_lower); continue

        # === 5. КОММЕРЦИЯ ===
        if lemma in COMM_SET or word_lower in COMM_SET:
            categories['commercial'].add(word_lower); continue

        # === 6. ТОВАРЫ ===
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
            
        # === 7. ОБЩИЕ ===
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
if 'reviews_results' not in st.session_state: st.session_state.reviews_results = []
if 'reviews_queue' not in st.session_state: st.session_state.reviews_queue = []
if 'reviews_automode_active' not in st.session_state: st.session_state.reviews_automode_active = False
if 'reviews_current_index' not in st.session_state: st.session_state.reviews_current_index = 0
if 'reviews_per_query' not in st.session_state: st.session_state.reviews_per_query = 3
if 'pending_widget_updates' not in st.session_state: st.session_state.pending_widget_updates = {}

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
STOP_POS = {'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO'}
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
        r = requests.post(url_set, headers=headers, json=payload, timeout=60)
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
        r_final = requests.post(url_get, headers=headers, json={"task_id": task_id}, timeout=200)
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

    # Список частей речи, которые мы ВЫКИДЫВАЕМ (союзы, предлоги, местоимения и т.д.)
    # Добавь INTJ (междометия), если нужно еще чище
    BAD_POS = {'PREP', 'CONJ', 'PRCL', 'NPRO', 'INTJ'}

    for w in words:
        # 1. Фильтр по длине: убираем всё, что короче 3 символов (было < 2)
        if len(w) < 3: 
            continue
            
        # 2. Фильтр цифр и кастомных стоп-слов
        if not settings['numbers'] and w.isdigit(): continue
        if w in stops: continue
        
        lemma = w
        if USE_NLP and n_gram == 1:
            p = morph.parse(w)[0]
            # Если часть речи в списке мусора (STOP_POS) — пропускаем слово
            if p.tag.POS in STOP_POS: 
                continue
            lemma = p.normal_form.replace('ё', 'е')
            # Дополнительная проверка: если лемма стала слишком короткой после очистки
            if len(lemma) < 3:
                continue

        lemmas.append(lemma)
        forms_map[lemma].add(w)
        
    return lemmas, forms_map # Не забудь про return, если он был в конце

def check_positions_NO_ALT(query, target_url, region_name, api_token):
    """
    Абсолютно новая функция.
    Гарантированно не отправляет alt_urls.
    """
    url_set = "https://arsenkin.ru/api/tools/set"
    url_check = "https://arsenkin.ru/api/tools/check"
    url_get = "https://arsenkin.ru/api/tools/get"
    headers = {"Authorization": f"Bearer {api_token}", "Content-type": "application/json"}
    
    # Регион
    reg_ids = REGION_MAP.get(region_name, {"ya": 213})
    region_id_int = int(reg_ids['ya'])
    
    # === JSON СТРОГО БЕЗ ALT_URLS ===
    payload = {
        "tools_name": "positions",
        "data": {
            "queries": [str(query)],
            "url": str(target_url).strip(),
            # СТРОКА alt_urls ПОЛНОСТЬЮ УДАЛЕНА ОТСЮДА
            "subdomain": True,
            "se": [{"type": 2, "region": region_id_int}],
            "format": 0
        }
    }

    try:
        # 1. ЗАПУСК
        r = requests.post(url_set, headers=headers, json=payload, timeout=20)
        
        # Если сервер вернул 500 или 400
        if r.status_code != 200:
            return 0, {"error": f"HTTP {r.status_code}", "text": r.text}
            
        resp = r.json()
        if "error" in resp: return 0, resp
        
        task_id = resp.get("task_id")
        if not task_id: return 0, {"error": "No Task ID", "resp": resp}
        
        # 2. ОЖИДАНИЕ
        for i in range(40):
            time.sleep(2)
            r_c = requests.post(url_check, headers=headers, json={"task_id": task_id})
            if r_c.json().get("status") == "finish":
                break
        else:
            return 0, {"error": "Timeout"}

        # 3. РЕЗУЛЬТАТ
        r_g = requests.post(url_get, headers=headers, json={"task_id": task_id})
        data = r_g.json()
        
        res_list = data.get("result", [])
        if not res_list: return 0, data
            
        item = res_list[0]
        pos = item.get('position')
        if pos is None: pos = item.get('pos')
        
        if str(pos) in ['0', '-', '', 'None']:
            return 0, item 
            
        return int(pos), None

    except Exception as e:
        return 0, {"error": f"Crash: {str(e)}"}

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
    
    # 1. Твой расширенный список + единицы измерения
    stop_garbage = {
        'в', 'на', 'и', 'с', 'со', 'по', 'для', 'от', 'до', 'из', 'к', 'у', 
        'о', 'об', 'за', 'над', 'под', 'при', 'про', 'без', 'через', 'между',
        'а', 'но', 'или', 'да', 'как', 'что', 'чтобы', 'если', 'то', 'ли', 'бы', 'же', 
        'г', 'обл', 'р', 'руб', 'мм', 'см', 'м', 'кг', 'т', 'шт', 'дн',
        'весь', 'все', 'всё', 'свой', 'ваш', 'наш', 'мы', 'вы', 'он', 'она', 'они',
        'купить', 'цена', 'заказать', 'стоимость', 'продажа', 'недорого', 
        'москва', 'спб',
        # Добавляем точно по твоему списку:
        'рублей', 'стр', 'ул', 'кв', 'м²', 'см²', 'м2', 'см2'
    }

    # Убираем коммерцию, если нужно (как в твоем исходнике)
    if 'купить' in stop_garbage: stop_garbage.remove('купить') 
    if 'цена' in stop_garbage: stop_garbage.remove('цена')
    
    if settings.get('custom_stops'):
        stop_garbage.update(set(settings['custom_stops']))

    lemmas = set()
    # ИСПРАВЛЕННАЯ РЕГУЛЯРКА: добавлена поддержка цифр и спецсимволов площадей ²
    words = re.findall(r'[а-яА-Яa-zA-Z0-9²]+', text.lower())
    
    for w in words:
        # Фильтр длины
        if len(w) < 2: continue 
        
        # Проверка ДО лемматизации (на случай "руб", "м2")
        if w in stop_garbage: continue
        
        if morph:
            try:
                p = morph.parse(w)[0]
                # Исключаем служебные части речи
                if p.tag.POS in {'PREP', 'CONJ', 'PRCL', 'NPRO', 'INTJ'}:
                    continue
                
                normal_form = p.normal_form
                # Проверка ПОСЛЕ лемматизации (на случай "рублей" -> "рубль")
                # Чтобы "рубль" тоже отсекался, если в списке есть "руб"
                if normal_form in stop_garbage:
                    continue
                
                # Дополнительная проверка на сокращения (р., руб. и т.д.)
                if any(normal_form.startswith(s) for s in ['рубл', 'метр', 'сантим', 'килогр']):
                    if w in stop_garbage or normal_form in stop_garbage:
                        continue

                lemmas.add(normal_form)
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
    import math
    import pandas as pd
    import numpy as np
    from collections import Counter, defaultdict
    import re
    from urllib.parse import urlparse

    if morph is None:
        st.error("CRITICAL: Лемматизация не работает!")
        return { "depth": pd.DataFrame(), "hybrid": pd.DataFrame(), "relevance_top": pd.DataFrame(), "my_score": {"width": 0, "depth": 0}, "missing_semantics_high": [], "missing_semantics_low": [] }

    # Карта частей речи
    POS_MAP = {
        'NOUN': 'Сущ', 'ADJF': 'Прил', 'ADJS': 'Прил',
        'VERB': 'Гл', 'INFN': 'Гл', 'PRTF': 'Прич', 'PRTS': 'Прич',
        'GRND': 'Деепр', 'NUMR': 'Числ', 'ADVB': 'Нареч',
        'NPRO': 'Местоим', 'PREP': 'Предлог', 'CONJ': 'Союз', 'PRCL': 'Частица', 'INTJ': 'Междом'
    }

    # === 1. АНАЛИЗАТОР (С ЧИСТКОЙ МУСОРА) ===
    def analyze_text_structure(text):
        if not text: return [], {}, 0
        
        # ЧЕРНЫЙ СПИСОК (Технический мусор)
        trash_stop_list = {
            'руб', 'рублей', 'кг', 'ул', 'наш', 'ваш', 'ru', 'com', 'net', 'org', 
            'стр', 'шт', 'см', 'мм', 'мл', 'кв', 'тел', 'факс', 'пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс',
            'description', 'keywords', 'content', 'viewport', 'charset', 'utf-8', 'html', 'body', 
            'div', 'span', 'class', 'style', 'script', 'function', 'return', 'var', 'let', 'const',
            'цена', 'купить', 'заказать', 'корзина', 'каталог', 'г', 'обл', 'д', 'pro', 'max', 'min',
            'width', 'height', 'px', 'em', 'rem', 'color', 'background', 'border', 'padding', 'margin',
            'true', 'false', 'null', 'undefined', 'nan', 'id', 'src', 'href', 'link', 'rel', 'type',
            'mil', 'armox', 'target', 'blank', 'self', 'parent', 'top'
        }

        words = re.findall(r'[а-яА-ЯёЁa-zA-Z0-9\-]+', text.lower())
        
        lemma_pos_list = []
        forms_map = defaultdict(set)
        valid_word_count = 0

        for w in words:
            if len(w) < 3 or w in trash_stop_list: continue
            if not settings['numbers'] and w.isdigit(): continue
            
            p = morph.parse(w)[0]
            if p.tag.POS in {'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO'}: continue
            
            lemma = p.normal_form.replace('ё', 'е')
            if lemma in trash_stop_list or len(lemma) < 3: continue

            pos_tag = p.tag.POS
            pos_ru = POS_MAP.get(pos_tag, 'Прочее')
            
            key = (lemma, pos_ru)
            lemma_pos_list.append(key)
            forms_map[key].add(w)
            valid_word_count += 1
            
        return lemma_pos_list, forms_map, valid_word_count

    # === 2. СБОР СЫРЫХ ДАННЫХ ===
    # Сначала собираем раздельно: (броневой, Сущ) и (броневой, Прил)
    global_stats_raw = defaultdict(lambda: {
        'sum_tf': 0.0, 
        'forms': set(), 
        'counts_list': [] # Список вхождений по документам [0, 5, 2, ...]
    })

    all_text_blocks = [] 
    N_sites = len(comp_data_full) if len(comp_data_full) > 0 else 1
    PASSAGE_SIZE = 20 

    for p in comp_data_full:
        if not p.get('body_text'): 
            # Если текста нет, добавляем пустые данные, чтобы длина counts_list была корректной
            continue
            
        doc_tokens, doc_forms, doc_len = analyze_text_structure(p['body_text'])
        
        if doc_len > 0:
            doc_counter = Counter(doc_tokens) # Ключи здесь (lemma, pos)
            
            # ВАЖНО: Нам нужно заполнить counts_list для всех известных ключей
            # Но мы пока не знаем всех ключей. Поэтому собираем локально, а потом сольем.
            # Упростим: global_stats_raw собирает данные. counts_list будем наращивать.
            # Но чтобы позиции совпадали с N_sites, нужно инициализировать нулями.
            pass

    # ПЕРЕПИСАННЫЙ СБОР ДАННЫХ ДЛЯ КОРРЕКТНОГО СЛИЯНИЯ
    # 1. Сначала найдем все уникальные ключи (lemma, pos) во всех документах
    # 2. Потом пройдемся по документам и заполним матрицу
    
    # Структура: { (lemma, pos): [count_doc1, count_doc2, ...] }
    matrix_counts = defaultdict(lambda: [0] * N_sites)
    matrix_forms = defaultdict(set)
    matrix_sum_tf = defaultdict(float)
    
    # Для IDF (пассажи)
    # Здесь упростим: считаем IDF по леммам сразу, чтобы не мучиться с объединением пассажей
    lemma_docs_count = Counter()

    # === ДОБАВЛЯЕМ МАССИВ ДЛИН ===
    doc_lengths = [0] * N_sites

    for idx, p in enumerate(comp_data_full):
        if not p.get('body_text'): continue
        doc_tokens, doc_forms, doc_len = analyze_text_structure(p['body_text'])
        
        # === СОХРАНЯЕМ ДЛИНУ ===
        doc_lengths[idx] = doc_len

    for idx, p in enumerate(comp_data_full):
        if not p.get('body_text'): continue
        doc_tokens, doc_forms, doc_len = analyze_text_structure(p['body_text'])
        
        if doc_len > 0:
            # Уникальные леммы в документе для IDF
            unique_lemmas_here = set(t[0] for t in doc_tokens)
            lemma_docs_count.update(unique_lemmas_here)
            
            # Подсчет для матрицы
            doc_counter = Counter(doc_tokens)
            for (lemma, pos), count in doc_counter.items():
                matrix_counts[(lemma, pos)][idx] = count
                matrix_sum_tf[(lemma, pos)] += (count / doc_len)
                matrix_forms[(lemma, pos)].update(doc_forms[(lemma, pos)])

    # ВАШ САЙТ (Сырые данные)
    my_counts_map_raw = Counter() # Ключ: (lemma, pos)
    my_clean_domain = "local"
    if my_data and my_data.get('body_text'):
        my_tokens, my_forms, my_len = analyze_text_structure(my_data['body_text'])
        my_counts_map_raw = Counter(my_tokens)
        if my_data.get('domain'):
            my_clean_domain = my_data.get('domain').lower().replace('www.', '').split(':')[0]

    # === 🔥 ЭТАП СЛИЯНИЯ (MERGE) 🔥 ===
    # Группируем ключи по лемме
    grouped_keys = defaultdict(list)
    for (lemma, pos) in matrix_counts.keys():
        grouped_keys[lemma].append(pos)

    final_stats = {}

    for lemma, pos_list in grouped_keys.items():
        if not pos_list: continue
        
        # 1. Определяем победителя по суммарной частоте (sum_tf)
        # Сортируем части речи: у кого sum_tf больше
        sorted_pos = sorted(pos_list, key=lambda p: matrix_sum_tf[(lemma, p)], reverse=True)
        winner_pos = sorted_pos[0]
        
        # 2. Инициализируем данные победителя
        merged_counts = list(matrix_counts[(lemma, winner_pos)]) # Копия списка
        merged_sum_tf = matrix_sum_tf[(lemma, winner_pos)]
        merged_forms = matrix_forms[(lemma, winner_pos)].copy()
        
        # 3. ПРИПЛЮСОВЫВАЕМ ПРОИГРАВШИХ
        for loser_pos in sorted_pos[1:]:
            # Складываем TF
            merged_sum_tf += matrix_sum_tf[(lemma, loser_pos)]
            
            # Объединяем формы
            merged_forms.update(matrix_forms[(lemma, loser_pos)])
            
            # Складываем вхождения по каждому документу (векторное сложение)
            loser_counts = matrix_counts[(lemma, loser_pos)]
            for i in range(N_sites):
                merged_counts[i] += loser_counts[i]

        # 4. Считаем docs_containing для ОБЪЕДИНЕННОГО слова
        # (сколько документов имеют count > 0 после слияния)
        merged_docs_containing = sum(1 for c in merged_counts if c > 0)
        
        # Формируем красивый список частей речи для отображения
        # Например: "Прил, Сущ"
        display_pos = ", ".join(sorted_pos)
        
        final_stats[lemma] = {
            'pos': display_pos,
            'sum_tf': merged_sum_tf,
            'forms': merged_forms,
            'counts_list': merged_counts,
            'docs_containing': merged_docs_containing
        }

    # === 3. РАСЧЕТ ТАБЛИЦ ===
    table_depth = []
    table_hybrid = []
    missing_semantics_high = []
    missing_semantics_low = []
    words_with_median_gt_0 = set()
    my_found_words = set()

    # Сортируем по алфавиту
    sorted_lemmas = sorted(final_stats.keys())

    # Пассажи считаем упрощенно через lemma_docs_count (это IDF)
    # Но лучше использовать merged_docs_containing для точности в рамках TF-IDF
    # N_passages в этой версии заменим на N_sites для классического IDF документа
    
    for lemma in sorted_lemmas:
        data = final_stats[lemma]
        df_docs = data['docs_containing']
        if df_docs == 0: continue
        
        # IDF (по документам, не по пассажам, так надежнее при слиянии)
        idf = math.log(N_sites / (1 + df_docs)) + 1
        avg_tf = data['sum_tf'] / N_sites
        tf_idf_value = avg_tf * idf
        
        # СЧИТАЕМ ВХОЖДЕНИЯ У ВАС (ТОЖЕ СУММИРУЕМ ВСЕ ВАРИАНТЫ)
        my_total_count = 0
        for (m_lemma, m_pos), cnt in my_counts_map_raw.items():
            if m_lemma == lemma:
                my_total_count += cnt

        # 3.1. TF-IDF
        table_hybrid.append({
            "Слово": lemma,
            "Часть речи": data['pos'], # Выведет "Прил, Сущ"
            "TF-IDF ТОП": tf_idf_value, 
            "IDF": idf, 
            "Кол-во сайтов": df_docs,
            "Вхождений у вас": my_total_count
        })

        # 3.2. ГЛУБИНА
        raw_counts = data['counts_list']
        # raw_counts уже имеет длину N_sites
        rec_median = int(np.median(raw_counts) + 0.5)
        obs_max = max(raw_counts) if raw_counts else 0
        
        if not (obs_max == 0 and my_total_count == 0):
            if rec_median >= 1:
                words_with_median_gt_0.add(lemma)
                if my_total_count > 0: my_found_words.add(lemma)

            if my_total_count == 0:
                weight = tf_idf_value * (rec_median if rec_median > 0 else 0.5)
                item = {'word': lemma, 'weight': weight}
                if rec_median >= 1: missing_semantics_high.append(item)
                else: missing_semantics_low.append(item)

            forms_str = ", ".join(sorted(list(data['forms'])))[:100]
            diff = rec_median - my_total_count
            status = "Норма" if diff == 0 else ("Недоспам" if diff > 0 else "Переспам")
            action_text = "✅" if diff == 0 else (f"+{diff}" if diff > 0 else f"{diff}")

            table_depth.append({
                "Слово": lemma, "Словоформы": forms_str, "Вхождений у вас": my_total_count,
                "Медиана": rec_median, "Максимум (конкур.)": obs_max,
                "Статус": status, "Рекомендация": action_text,
                "is_missing": (my_total_count == 0), "sort_val": abs(diff)
            })

    # --- 4. ФИНАЛ ---
    df_hybrid = pd.DataFrame(table_hybrid)
    if not df_hybrid.empty:
        df_hybrid = df_hybrid.sort_values(by="TF-IDF ТОП", ascending=False).head(1000)
        df_hybrid["TF-IDF ТОП"] = df_hybrid["TF-IDF ТОП"].apply(lambda x: float(f"{x:.6f}"))
        df_hybrid["IDF"] = df_hybrid["IDF"].round(2)

    total_needed = len(words_with_median_gt_0)
    total_found = len(my_found_words)
    my_width_score = int(min(100, (total_found / total_needed) * 105)) if total_needed > 0 else 0

    # ==== НОВЫЙ БЛОК: РАСЧЕТ ГЛУБИНЫ (С УЧЕТОМ НОРМИРОВАНИЯ) ====
    use_norm = settings.get('norm', True)
    
    # Функция для пересчета вхождений: сырые штуки ИЛИ плотность на 1000 слов
    def get_weighted_count(raw_c, d_len):
        if not use_norm: return raw_c
        if d_len == 0: return 0
        return (raw_c / d_len) * 1000

    total_median_sum = 0
    word_medians = {}
    
    # 1. Считаем эталонный объем Топа (справедливый)
    for lemma in words_with_median_gt_0:
        data = final_stats.get(lemma)
        if data:
            weighted_counts = [
                get_weighted_count(data['counts_list'][i], doc_lengths[i]) 
                for i in range(N_sites)
            ]
            rec_median = np.median(weighted_counts)
            word_medians[lemma] = rec_median
            total_median_sum += rec_median

    # 2. Считаем глубину для Вашего сайта
    my_depth_sum = 0
    my_len_val = my_len if 'my_len' in locals() and my_len > 0 else 1000 
    
    for lemma in words_with_median_gt_0:
        my_c = sum(cnt for (m_lemma, m_pos), cnt in my_counts_map_raw.items() if m_lemma == lemma)
        my_w_count = get_weighted_count(my_c, my_len_val)
        
        rec_median = word_medians.get(lemma, 0)
        # Штраф за переспам (режем по медиане) или недоспам (берем что есть)
        my_depth_sum += min(my_w_count, max(0.01, rec_median)) 

    my_depth_score = int(min(100, (my_depth_sum / total_median_sum) * 105)) if total_median_sum > 0 else 0
    # ==============================================================

    table_rel = []
    my_site_found = False
    for item in original_results:
        url = item['url']
        try:
            idx = next(i for i, x in enumerate(comp_data_full) if x['url'] == url)
        except StopIteration:
            continue

        doc_data = comp_data_full[idx]
        width_val = 0
        depth_val = 0
        
        if doc_data and doc_data.get('body_text'):
             # Считаем Ширину
             toks, _, _ = analyze_text_structure(doc_data['body_text'])
             lemmas_only = set(t[0] for t in toks)
             inter = lemmas_only.intersection(words_with_median_gt_0)
             width_val = int(min(100, (len(inter) / total_needed) * 105)) if total_needed > 0 else 0

             # Считаем Глубину конкурента
             doc_depth_sum = 0
             d_len = doc_lengths[idx]
             for lemma in words_with_median_gt_0:
                 data = final_stats.get(lemma)
                 if data:
                     w_count = get_weighted_count(data['counts_list'][idx], d_len)
                     rec_median = word_medians.get(lemma, 0)
                     doc_depth_sum += min(w_count, max(0.01, rec_median))
             
             depth_val = int(min(100, (doc_depth_sum / total_median_sum) * 105)) if total_median_sum > 0 else 0
        
        d_name = urlparse(url).netloc
        if my_clean_domain != "local" and my_clean_domain in d_name:
            d_name += " (Вы)"
            my_site_found = True
            
        table_rel.append({ 
            "Домен": d_name, "URL": url, "Позиция": item['pos'], 
            "Ширина (балл)": width_val, "Глубина (балл)": depth_val 
        })

    if not my_site_found:
        my_u_val = my_data.get('url', '#') if my_data else '#'
        table_rel.append({ 
            "Домен": "Ваш сайт", "URL": my_u_val, "Позиция": my_serp_pos, 
            "Ширина (балл)": my_width_score, "Глубина (балл)": my_depth_score 
        })

    missing_semantics_high.sort(key=lambda x: x['weight'], reverse=True)
    missing_semantics_low.sort(key=lambda x: x['weight'], reverse=True)
    
    good_urls, bad_urls_dicts, trend_info = analyze_serp_anomalies(pd.DataFrame(table_rel))

    return { 
        "depth": pd.DataFrame(table_depth), 
        "hybrid": df_hybrid, 
        "relevance_top": pd.DataFrame(table_rel).sort_values(by='Позиция'), 
        "my_score": {"width": my_width_score, "depth": my_depth_score}, 
        "missing_semantics_high": missing_semantics_high, 
        "missing_semantics_low": missing_semantics_low[:500],
        "debug_width": {"found": total_found, "needed": total_needed},
        "bad_urls": bad_urls_dicts
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

def run_seo_analysis_background(query, api_token):
    """
    Фоновый запуск SEO-анализа.
    ПОЛНАЯ СИМУЛЯЦИЯ РАБОТЫ ВКЛАДКИ 1 (ОБНОВЛЯЕТ UI).
    """
    # 1. Используем ТЕ ЖЕ настройки, что и на вкладке 1 по умолчанию
    settings = {
        'noindex': True, 
        'alt_title': False, 
        'numbers': False, 
        'norm': True, 
        'ua': "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", 
        'custom_stops': []
    }
    
    # Режим "Без страницы"
    my_data = {'url': 'Local', 'domain': 'local', 'body_text': '', 'anchor_text': ''}
    
    if not api_token: return []
    
    try:
        # === 1. ИМИТАЦИЯ НАЖАТИЯ "Поиск через API" ===
        raw_top = get_arsenkin_urls(query, "Яндекс", "Москва", api_token, depth_val=10)
        if not raw_top: return []
        
        candidates = [item for item in raw_top if not any(x in item['url'] for x in ["avito", "ozon", "wildberries", "market", "tiu"])]
        candidates = candidates[:10]
        if not candidates: return []

        comp_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(parse_page, item['url'], settings, query): item for item in candidates}
            for f in concurrent.futures.as_completed(futures):
                try:
                    res = f.result()
                    if res:
                        res['pos'] = futures[f]['pos']
                        comp_data.append(res)
                except: pass
        
        if not comp_data: return []

        # === 2. РАСЧЕТ МЕТРИК (с очисткой от дублей, как мы настроили выше) ===
        targets = [{'url': d['url'], 'pos': d['pos']} for d in comp_data]
        results = calculate_metrics(comp_data, my_data, settings, 0, targets)
        
        # =========================================================
        # 🔥 ОБНОВЛЕНИЕ STATE (Чтобы вкладка 1 увидела результат)
        # =========================================================
        
        st.session_state['analysis_done'] = True
        st.session_state['analysis_results'] = results
        st.session_state['raw_comp_data'] = comp_data 
        
        # 1. Вписываем запрос в поле ввода на 1 вкладке
        st.session_state['query_input'] = query
        
        # 2. Переключаем режим "Ваша страница" -> "Без страницы"
        if 'my_page_source_radio' not in st.session_state:
            st.session_state['my_page_source_radio'] = "Без страницы"
        
        # 3. Переключаем источник конкурентов -> "API"
        # Это самое важное, чтобы UI не искал список ссылок
        st.session_state['competitor_source_radio'] = "Поиск через API Arsenkin (TOP-30)"
        
        # 4. Обновляем вспомогательные таблицы
        st.session_state['naming_table_df'] = calculate_naming_metrics(comp_data, my_data, settings)
        st.session_state['ideal_h1_result'] = analyze_ideal_name(comp_data)
        st.session_state['full_graph_data'] = results['relevance_top']
        
        # 5. Тренды
        _, _, trend = analyze_serp_anomalies(results['relevance_top'])
        st.session_state['serp_trend_info'] = trend

        # =========================================================

        # Возвращаем 15 лучших слов (TF-IDF) для LSI генератора
        df_hybrid = results.get('hybrid')
        if df_hybrid is not None and not df_hybrid.empty:
            return df_hybrid.head(15)['Слово'].tolist()
            
    except Exception as e:
        print(f"Background SEO Error: {e}")
        return []
    
    return []

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
    'IP_PROP4824': "Контакты для связи",
    'IP_PROP4825': "Персональный менеджер",
    'IP_PROP4826': "Современный практический подход",
    'IP_PROP4834': "Персональный менеджер",
    'IP_PROP4835': "Точно в срок",
    'IP_PROP4836': "Гибкие условия расчета",
    'IP_PROP4837': "Порядок в ГОСТах"
}

def get_page_data_for_gen(url):
    # Возвращаем стандартный requests, который работал нормально
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        # verify=False помогает от простых SSL ошибок, но не ломает кодировку
        response = requests.get(url, headers=headers, timeout=20, verify=False)
        # Принудительная кодировка часто помогает на ру-сайтах
        if response.encoding != 'utf-8':
            response.encoding = response.apparent_encoding
    except Exception as e: 
        return None, None, None, f"Ошибка соединения: {e}"
    
    if response.status_code != 200: 
        return None, None, None, f"Ошибка статуса: {response.status_code}"
    
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
    except:
        return None, None, None, "Ошибка парсинга"
    
    # 1. ЗАГОЛОВОК
    description_div = soup.find('div', class_='description-container')
    target_h2 = None
    if description_div:
        target_h2 = description_div.find('h2')
    
    if not target_h2:
        target_h2 = soup.find('h2')
    
    # ИЗМЕНЕНИЕ: Если H2 нет, возвращаем None, чтобы скрипт взял имя товара из ссылки
    page_header = target_h2.get_text(strip=True) if target_h2 else None

    # 2. Фактура (текст)
    if description_div:
        base_text = description_div.get_text(separator="\n", strip=True)
    else:
        # Чистим скрипты, чтобы в текст не попал мусор
        for s in soup(['script', 'style']): s.decompose()
        base_text = soup.body.get_text(separator="\n", strip=True)[:6000]
    
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
    import re
    
    client = OpenAI(api_key=api_key, base_url="https://litellm.tokengate.ru/v1")
    
    seo_words = seo_words or []
    seo_instruction_block = ""
    
    # === 1. ОБЪЯВЛЯЕМ ВСЕ ПЕРЕМЕННЫЕ СТРОГО ДО ПРОМПТА ===
    h1_marker = tag_name
    h2_topic = forced_header if forced_header else tag_name
    lsi_string = ", ".join(seo_words) if seo_words else "Нет дополнительных слов"
    
    stop_words_list = (
        "является, представляет собой, ключевой компонент, широко применяется, "
        "обладают, характеризуются, отличается, разнообразие, широкий спектр, "
        "оптимальный, уникальный, данный, этот, изделия, материалы, "
        "высокое качество, доступная цена, индивидуальный подход, "
        "доставка, оплата, условия поставки, звоните, менеджер"
    )

    contact_html_block = (
        'Предлагаем консультацию с менеджером по номеру '
        '<nobr><a href="tel:#PHONE#" class="a_404 ct_phone">#PHONE#</a></nobr>, '
        'либо пишите на почту <a href="mailto:#EMAIL#" class="a_404">#EMAIL#</a>.'
    )
    
    # === 2. ИНСТРУКЦИЯ ПО SEO ===
    if seo_words:
        seo_list_str = ", ".join(seo_words)
        seo_instruction_block = f"""
--- СТРОГАЯ ИНСТРУКЦИЯ ПО КЛЮЧЕВЫМ СЛОВАМ ---
Тебе нужно внедрить в текст следующие LSI-слова: {seo_list_str}
ПРАВИЛА РАБОТЫ СО СЛОВАМИ:
1. ОБЯЗАТЕЛЬНО используй каждое слово из списка ровно ОДИН раз. Не повторяй их!
2. Вписывай слова максимально естественно, меняя падежи, числа и формы.
3. ОБЯЗАТЕЛЬНО выделяй все вставленные SEO-слова жирным шрифтом (оборачивай в **слово** или <strong>слово</strong>), чтобы их было видно.
4. Качество: Если ключ выглядит как мусор или название конкурента — игнорируй его.
5. КАТЕГОРИЧЕСКИЙ ЗАПРЕТ НА ЛАТИНИЦУ: Никаких английских слов, транслита (типа 'truba', 'stal').
ИСКЛЮЧЕНИЕ: Только международные марки стали (AISI 304, S355), стандарты (DIN, ISO, EN) и ГОСТ. 
Всё остальное пиши ТОЛЬКО кириллицей.
Если видишь в ключевых словах латиницу (не марку) — ИГНОРИРУЙ ЕЁ.
-------------------------------------------
"""

    # === 3. СИСТЕМНАЯ РОЛЬ ===
    system_instruction = (
        "Ты — технический редактор. Ты пишешь фактами, связным русским языком. "
        "Ты соблюдаешь HTML-структуру (списки <ul>, таблицы). "
        "Ты умеешь грамотно вписывать ключевые слова, меняя их форму и порядок слов. "
        "Если нужно вписать много ключевых слов — ты увеличиваешь объем текста, чтобы они смотрелись естественно."
    )
    
    # === 4. ПОЛЬЗОВАТЕЛЬСКИЙ ПРОМПТ (Твоя структура сохранена) ===
    user_prompt = f"""
    ИСТОЧНИК ДАННЫХ ДЛЯ РАБОТЫ (ФАКТУРА):
    \"\"\"{base_text[:3500]}\"\"\"

    КРИТИЧЕСКОЕ ПРАВИЛО: Напиши ровно {num_blocks} HTML-блоков. 
    Каждый блок должен строго отделяться друг от друга разделителем: |||BLOCK_SEP|||

    {seo_instruction_block}

    ЗАДАЧА: Напиши техническую статью.
    [I] ГЛАВНЫЕ ПРАВИЛА РАБОТЫ С КЛЮЧОМ ("{h1_marker}"):
    
    1. В ЗАГОЛОВКАХ H3 (СТРОГО ЦЕЛИКОМ):
       - Здесь фраза "{h1_marker}" должна стоять ЦЕЛИКОМ (рядом).
       - МОЖНО: Склонять (Монтаж трубы стальной).
       - НЕЛЬЗЯ: Разрывать слова или заменять синонимами.
    2. В ТЕКСТЕ И АБЗАЦАХ (МЯГКОЕ ВХОЖДЕНИЕ):
       - Общая плотность слов из ключа — 1.5%.
       - ВАЖНО: В тексте ТЫ ОБЯЗАН РАЗБИВАТЬ фразу, менять порядок слов.
       - ПЛОХО: "Купить трубу стальную можно..." (спам).
       - ХОРОШО: "Стальная поверхность трубы обеспечивает..." (разбил слова).
       - ХОРОШО: "Для этой трубы характерна стальная структура..." (поменял местами).
    [II] ЛОГИКА HTML (СТРОГО):
    
    1. СПИСКИ:
       - <ol>: ТОЛЬКО для пошаговых процессов.
       - <ul>: ДЛЯ ХАРАКТЕРИСТИК, СФЕР, СВОЙСТВ (Списки №1, №2, №3 — СТРОГО <ul>).
       - ВАЖНО: Не используй двоеточие (:) ВНУТРИ пунктов списка.
       
    2. ТАБЛИЦА:
       - Класс: "brand-accent-table".
       - Шапка через <thead> и <th>.

    [III] СТРУКТУРА ТЕКСТА:
    
    1.1. Заголовок: <h2>{h2_topic}</h2>.
    
    1.2.
    БЭНГЕР: 3-4 связных предложения. Опиши товар "{h1_marker}" нормальным языком (что это, ГОСТ, материал).
    
    1.3.
    Абзац 1 + Контакты: 
    {contact_html_block}
    
    1.4.
    Подводка к списку 1 (:).
    
    1.5. Список №1 (6 пунктов): ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ.
    (Формат: <ul>). "И" меняй на запятую. Цифры значащие.
    1.6. Абзац 2. Описание производства.
    
    1.7. ТАБЛИЦА ХАРАКТЕРИСТИК (СПРАВОЧНАЯ):
    4-5 строк. Без дублей списка №1.
    ИСПОЛЬЗУЙ ЭТОТ КОД:
    <table class="brand-accent-table">
        <thead>
            <tr>
                <th>Параметр</th>
                <th>Значение</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>ГОСТ / ТУ</td><td>[Данные]</td></tr>
            <tr><td>Марка сплава</td><td>[Данные]</td></tr>
            <tr><td>[Параметр 3]</td><td>[Данные]</td></tr>
            <tr><td>[Параметр 4]</td><td>[Данные]</td></tr>
        </tbody>
    </table>
    
    1.8.
    Подзаголовок H3 (ШАБЛОН): 
    "Классификация {h1_marker} (род. падеж, с маленькой буквы)"
    (Тут ключ целиком!).
    1.9. Абзац 3. Виды, типы. (Тут разбивай ключ).
    
    1.10. Подводка к списку 2 (:).
    
    1.11.
    Список №2 (6 пунктов): СФЕРЫ ПРИМЕНЕНИЯ.
    (Формат: <ul>).
       
    1.12. Абзац 4. Условия эксплуатации.
                          
    1.13.
    Подзаголовок H3 (ШАБЛОН):
    "Монтаж {h1_marker} (род. падеж)" ИЛИ "Обработка {h1_marker} (род. падеж)".
    (Тут ключ целиком!).
    
    1.14.
    Абзац 5. Технология работы.
    
    1.15. Подводка к списку 3 (:).
    
    1.16. Список №3 (6 пунктов): ЭКСПЛУАТАЦИОННЫЕ СВОЙСТВА.
    (Без союзов "и").
    Формат: <ul>.
       
    1.17. Абзац 6. Резюме и отгрузка.

    [IV] ДОПОЛНИТЕЛЬНО (LSI ЯДРО):
    Список слов: {lsi_string}
    
    ПРАВИЛА LSI:
    1. ИСПОЛЬЗУЙ ВЕСЬ СПИСОК (Общие + Семантика).
    2. Каждое слово — РОВНО 1 РАЗ (не повторяй).
    3. Выделяй каждое вставленное слово тегом <b>жирный</b>.
    4. Если слов много — УВЕЛИЧИВАЙ ОБЪЕМ ТЕКСТА. Пиши дополнительные предложения, чтобы слова вписывались плавно, а не "обрубками".
    Смысл и связность важнее краткости.

    [V] СТОП-СЛОВА: ({stop_words_list}).
    
    ВЫВОД: ТОЛЬКО HTML КОД.
    СТРОГИЕ ПРАВИЛА ОФОРМЛЕНИЯ И ФИЛЬТРАЦИИ (КРИТИЧЕСКИ ВАЖНО):
    1. Оформление списков: Каждый пункт любого маркированного списка должен строго заканчиваться точкой с запятой (;), а самый последний пункт списка — точкой (.).
    Без исключений.
    2. Написание диапазонов: Числовые диапазоны (длина, вес, размер) пиши через тире и с сокращением единиц измерения.
    Пример: "4-9 м", "10-20 мм". КАТЕГОРИЧЕСКИ ЗАПРЕЩАЕТСЯ писать "от 4 до 9 метров".
    - Исключение: для температурных диапазонов с минусовыми значениями используй слова (например, "от -10 до +50 °C").
    3. Игнорирование конкурентов: Если в списке переданных ключевых слов тебе попадется странный мусор на латинице или названия чужих магазинов/компаний, полностью ИГНОРИРУЙ ИХ.
    Из латиницы разрешается писать только марки сталей и стандарты (AISI 304, DIN и т.д.).
    4. Характеристики в списках: Если в пункте списка перечисляется свойство (характеристика) и его значения, ОБЯЗАТЕЛЬНО ставь тире (–) между названием свойства и списком значений.
    5. Максимально сократи использование союза "и". В 90% случаев заменяй его запятой при перечислении или просто перестраивай предложение.
    Текст должен быть динамичным и лаконичным, без лишнего "нагромождения" связок.
    """

    try:
        # === ВЫЗОВ API ===
        response = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=8192 # <--- Лимит, чтобы текст не обрывался
        )
        
        raw_content = response.choices[0].message.content
        
        if not raw_content:
            return ["Error: API вернул пустой ответ (возможно, сработал фильтр безопасности)"] * num_blocks

        # === БРОНЕБОЙНАЯ ОБРАБОТКА (ЧТОБЫ НЕ ТЕРЯТЬ ДЕНЬГИ) ===
        try:
            # Очистка Markdown
            content = re.sub(r'^```[a-zA-Z]*\s*', '', raw_content.strip())
            content = re.sub(r'\s*```$', '', content.strip())
            
            # Разбивка
            blocks = [b.strip() for b in content.split("|||BLOCK_SEP|||") if b.strip()]
            
            # Если нейросеть проигнорировала разделитель, отдаем весь текст в первом блоке
            if not blocks:
                blocks = [content]
                
            cleaned_blocks = []
            for b in blocks:
                cb = b.strip()
                cb = cb.replace("**", "")
                if cb: cleaned_blocks.append(cb)
                
            # Гарантируем нужное количество блоков, чтобы скрипт не упал
            while len(cleaned_blocks) < num_blocks: 
                cleaned_blocks.append("")
                
            # Обработка ТОЛЬКО первого блока (без риска стереть весь текст)
            first_block = cleaned_blocks[0]
            first_block = re.sub(r'^<h[23][^>]*>.*?</h[23]>', '', first_block, flags=re.IGNORECASE).strip()
            final_h2_text = forced_header if forced_header else tag_name
            cleaned_blocks[0] = f"<h2>{final_h2_text}</h2>\n{first_block}"

            return cleaned_blocks[:num_blocks]

        except Exception as parse_error:
            # ЕСЛИ СКРИПТ СЛОМАЛСЯ ПРИ НАРЕЗКЕ - МЫ ВОЗВРАЩАЕМ СЫРОЙ ТЕКСТ (ДЕНЬГИ СПАСЕНЫ)
            safe_blocks = [f"<h2>{forced_header if forced_header else tag_name}</h2>\n{raw_content}"]
            while len(safe_blocks) < num_blocks:
                safe_blocks.append("")
            return safe_blocks[:num_blocks]

    except Exception as e:
        return [f"Error: Ошибка соединения с API - {str(e)}"] * num_blocks

# ==========================================
# НОВЫЕ ФУНКЦИИ ДЛЯ LSI ГЕНЕРАТОРА (ВСТАВИТЬ СЮДА)
# ==========================================

def inflect_lsi_phrase(phrase, target_case):
    morph = pymorphy2.MorphAnalyzer()
    words = str(phrase).split()
    inflected_words = []
    for word in words:
        parsed_word = morph.parse(word)[0]
        try:
            inf_word = parsed_word.inflect({target_case})
            inflected_words.append(inf_word.word if inf_word else word)
        except: inflected_words.append(word)
    return " ".join(inflected_words)

def generate_random_date():
    start = datetime.datetime(2026, 1, 1)
    end = datetime.datetime(2026, 2, 10)
    delta = end - start
    return (start + datetime.timedelta(days=random.randrange(delta.days + 1))).strftime("%d.%m.%Y")

def build_review_from_repo(template, variables_dict, repo_fio, lsi_words):
    # Выбираем случайное LSI слово
    lsi_word = random.choice(lsi_words) if lsi_words else ""
    
    # Пытаемся вставить LSI слово ВНУТРИ шаблона вместо подходящей переменной
    if lsi_word:
        parsed = morph.parse(lsi_word)[0]
        lsi_gender = parsed.tag.gender
        lsi_number = parsed.tag.number
        
        # Ищем подходящие плейсхолдеры (например, {товар_сущ_муж})
        placeholders = re.findall(r'\{([^}]+)\}', template)
        found_slot = None
        for p in placeholders:
            is_product = any(x in p for x in ['товар', 'сущ', 'вид_проката'])
            # Проверяем род и число, чтобы не было "купили труба (муж.род слот)"
            gender_ok = True
            if '_муж' in p and lsi_gender != 'masc': gender_ok = False
            if '_жен' in p and lsi_gender != 'femn': gender_ok = False
            if '_мнч' in p and lsi_number != 'plur': gender_ok = False
            
            if is_product and gender_ok:
                found_slot = p
                break
        
        if found_slot:
            # Склоняем LSI под падеж слота
            target_case = 'nomn'
            case_map = {'_вин': 'accs', '_ВП': 'accs', '_род': 'gent', '_творит': 'ablt', '_им': 'nomn'}
            for sfx, c in case_map.items():
                if sfx in found_slot:
                    target_case = c
                    break
            inflected = inflect_lsi_phrase(lsi_word, target_case)
            template = template.replace(f"{{{found_slot}}}", f"**{inflected}**", 1)
        else:
            # Если слота нет, делаем рандомную вставку через твои же переменные
            v_intro = random.choice(variables_dict.get('вводное_слово', ['Кстати']))
            v_eval = random.choice(variables_dict.get('оценка1_хар_товар_ед_им', ['на высоте']))
            template += f" {v_intro}, **{lsi_word}** {v_eval}."

    # Заполняем остальные переменные
    def replace_var(match):
        v = match.group(1).strip()
        if v == "дата":
            # Твои даты 2026 года
            start = datetime.date(2026, 1, 1)
            return (start + datetime.timedelta(days=random.randint(0, 40))).strftime("%d.%m.%Y")
        if v in variables_dict:
            return str(random.choice(variables_dict[v])).strip()
        return match.group(0)

    final_draft = re.sub(r'\{([^}]+)\}', replace_var, template)
    
    # ФИО из твоего файла
    fio_row = repo_fio.sample(1).iloc[0]
    author = f"{fio_row['Имя']} {fio_row['Фамилия']}"
    
    return author, final_draft
def generate_faq_gemini(api_key, h1, lsi_words, target_count=5):
    import json
    import re
    from openai import OpenAI
    
    try:
        client = OpenAI(api_key=api_key, base_url="https://litellm.tokengate.ru/v1")
    except Exception as e:
        return [{"Тип": "Ошибка", "Вопрос": "Ошибка инициализации API", "Ответ": str(e)}]
    
    forbidden_roots = [
        "украин", "ukrain", "ua", "всу", "зсу", "ато", "сво", "войн",
        "киев", "львов", "харьков", "одесс", "днепр", "мариуполь",
        "донец", "луганс", "днр", "лнр", "донбасс", "мелитополь",
        "бердянск", "бахмут", "запорожь", "херсон", "крым",
        "политик", "спецоперац", "msk", "spb", "мск", "спб", "тк ", "ооо ", "ип "
    ]
    
    clean_lsi = []
    for w in lsi_words:
        w_lower = str(w).lower()
        if not any(root in w_lower for root in forbidden_roots):
            if re.match(r'^[a-z]{2,4}$', w_lower) and w_lower not in ['aisi', 'din', 'iso', 'en']:
                continue
            clean_lsi.append(w)
            
    lsi_text = ", ".join(clean_lsi)
    
    prompt_1 = f"""
    Ты технический эксперт в металлопрокате и B2B продажах.
    Составь FAQ для страницы "{h1}".
    
    "КРИТИЧЕСКОЕ ПРАВИЛО КОЛИЧЕСТВА: Ты ОБЯЗАН сгенерировать ровно {target_count} пар Вопрос-Ответ. Ни больше, ни меньше. "
    "Если информации в исходном тексте не хватает на {target_count} вопросов, ТЫ ДОЛЖЕН самостоятельно придумать реалистичные технические детали "
    "(о монтаже, ГОСТах, доставке, аналогах, сроках службы), чтобы добить количество строго до {target_count}. "
    "За выдачу меньшего количества последует системная ошибка."

    СТРУКТУРА ВОПРОСОВ (ОБЯЗАТЕЛЬНО):
    1. 50% — КОММЕРЧЕСКИЕ: доставка по РФ, способы оплаты, наличие, отгрузка, резка в размер.
    2. 50% — ИНФОРМАЦИОННЫЕ: технические характеристики, ГОСТы, марки стали и их отличия.

    ТЕХНИЧЕСКАЯ ГРАМОТНОСТЬ:
    - ГОСТ всегда пиши заглавными буквами (напр. ГОСТ 10704-91).
    - Исправляй названия марок стали (AISI 304, Ст3сп), даже если в LSI они написаны с ошибками.
    - Диапазоны параметров пиши через тире без пробелов (напр. 10-20 мм).
    
    СТРОЖАЙШИЙ ЗАПРЕТ:
    Категорически запрещено использовать любые упоминания Украины, политики, войны, конкурентов или сокращений типа 'msk', 'спб'.
    
    УСЛОВИЯ:
    1. Список LSI-слов: {lsi_text}.
    2. ВЫДЕЛИ ЖИРНЫМ ШРИФТОМ (**слово**) все использованные LSI-слова!
    3. Напиши ровно {target_count} вопросов и ответов.
    4. ВЕРНИ СТРОГО В JSON, добавив поле "Тип": [{{"Тип": "Коммерческий" или "Информационный", "Вопрос": "...", "Ответ": "..."}}]
    """
    
    try:
        res_1 = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[{"role": "user", "content": prompt_1}],
            temperature=0.3,
            max_tokens=8192
        )
        draft_text = res_1.choices[0].message.content.strip()
        
        if draft_text.startswith("```json"): draft_text = draft_text[7:]
        if draft_text.startswith("```"): draft_text = draft_text[3:]
        if draft_text.endswith("```"): draft_text = draft_text[:-3]
        draft_text = draft_text.strip()
        
        prompt_2 = f"""
        Я сгенерировал черновик FAQ для страницы "{h1}". Вот он (JSON):
        {draft_text}

        Выступи в роли строгого коммерческого редактора.
        ПРАВИЛА:
        1. СТРОГИЙ ЗАПРЕТ: Вычисти любые следы политики, Украины или мусорных сокращений (msk, спб).
        2. ФАКТЧЕКИНГ: Убедись, что все ГОСТы написаны заглавными, а марки стали корректно.
        3. Удали фразы ИИ ("Важно отметить", "Конечно").
        4. Ответы должны быть короткими и полезными.
        5. Сохрани заданное количество вопросов ({target_count}).
        6. ОБЯЗАТЕЛЬНО СОХРАНИ выделение жирным шрифтом (**слово**) для LSI!
        7. ВЕРНИ ТОЛЬКО ГОЛЫЙ JSON-МАССИВ строго в таком формате:
        [{{"Тип": "Коммерческий" или "Информационный", "Вопрос": "...", "Ответ": "..."}}]
        """
        
        res_2 = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[{"role": "user", "content": prompt_2}],
            temperature=0.3,
            max_tokens=8192
        )
        final_text = res_2.choices[0].message.content.strip()
        
        if final_text.startswith("```json"): final_text = final_text[7:]
        if final_text.startswith("```"): final_text = final_text[3:]
        if final_text.endswith("```"): final_text = final_text[:-3]
        final_text = final_text.strip()
        
        parsed_data = json.loads(final_text)
        
        for item in parsed_data:
            if "Вопрос" in item:
                item["Вопрос"] = item["Вопрос"].replace('—', '&ndash;').replace('–', '&ndash;').replace('&mdash;', '&ndash;')
            if "Ответ" in item:
                item["Ответ"] = item["Ответ"].replace('—', '&ndash;').replace('–', '&ndash;').replace('&mdash;', '&ndash;')
                
        return parsed_data
        
    except Exception as e:
        return [{"Тип": "Ошибка", "Вопрос": "Ошибка генерации", "Ответ": str(e)}]

def generate_reviews_deepseek(api_key, h2_header, lsi_words, target_count, chosen_authors):
    import json
    import random
    import re
    from openai import OpenAI
    from datetime import date, timedelta
    
    try:
        import pymorphy3 as pymorphy2
    except ImportError:
        import pymorphy2
    morph = pymorphy2.MorphAnalyzer()

    # --- 1. ФИЛЬТРАЦИЯ LSI СЛОВ ---
    clean_lsi = []
    for word in lsi_words:
        try:
            parsed = morph.parse(word)[0]
            if any(tag in parsed.tag for tag in ['Geox', 'VERB', 'INFN', 'PREP', 'CONJ', 'PRCL', 'INTJ']) or len(word) < 3:
                continue
            clean_lsi.append(word)
        except:
            if len(word) >= 3: clean_lsi.append(word)
            
    final_lsi = clean_lsi[:15]

    # --- 2. МАТЕМАТИКА ОЦЕНОК ---
    def get_balanced_ratings(n):
        if n <= 1: return [5.0] * n
        ratings = [3.5]
        for _ in range(n - 1): ratings.append(5.0)
        indices = list(range(1, n))
        random.shuffle(indices)
        for idx in indices:
            if (sum(ratings) / n) <= 4.75: break
            new_val = random.choice([4.0, 4.5])
            if (sum(ratings) - ratings[idx] + new_val) / n >= 4.71:
                ratings[idx] = new_val
        random.shuffle(ratings)
        return ratings

    ratings = get_balanced_ratings(target_count)

    # --- 3. РАНДОМНЫЕ ДАТЫ ---
    start_dt = date(2020, 1, 1)
    end_dt = date(2026, 2, 10)
    delta_days = (end_dt - start_dt).days
    raw_dates = [(start_dt + timedelta(days=random.randint(0, delta_days))).strftime("%d.%m.%Y") for _ in range(target_count)]

    # --- 4. РАСПРЕДЕЛЕНИЕ СТИЛЕЙ ---
    lower_count = int(target_count * 0.5)
    upper_count = target_count - lower_count
    case_pool = ['lower'] * lower_count + ['upper'] * upper_count
    random.shuffle(case_pool)

    # --- 5. СПИСКИ ПЕРЕМЕННЫХ ИЗ ТВОЕЙ СХЕМЫ ---
    male_personas = [
        "Мужчина, делает забор или навес на даче своими руками",
        "Прораб небольшой бригады, строит частный дом или баню",
        "Частный сварщик (ИП/самозанятый), варит лестницы, мангалы, фермы на заказ",
        "Владелец участка, закупает материалы для заливки фундамента",
        "Мелкий подрядчик, занимается монтажом заборов и кровли"
    ]
    
    focus_points = [
        "Менеджер по телефону толково проконсультировал и помог рассчитать нужное количество без лишних обрезков",
        "На складе погрузили нормально, металл чистый (без сильной ржавчины), менеджер быстро оформил доки",
        "Удобно, что менеджер подсказал взять всю мелочевку (краску, крепеж, расходники) сразу в одном месте",
        "Заказывал резку в размер, порезали ровно, геометрия норм",
        "Оформление документов и выставление счета заняло минимум времени, цены актуальные",
        "Адекватный водитель на доставке, смог аккуратно заехать в узкие ворота СНТ/участка"
    ]
    focus_weights = [25, 25, 15, 15, 10, 10]

    styles_and_lengths = [
        "Короткий (1-2 предложения). Максимально сухо и по делу. Пишет с телефона.",
        "Разговорный (3-4 предложения). Простой слог, как сообщение в мессенджере. Пару раз пропущены запятые перед 'что' или 'а'.",
        "Ультракороткий (до 5 слов). Формальная отписка занятого человека.",
        "Разговорный (2-3 предложения). Грамотный, но без сложных деепричастных оборотов."
    ]

    authors_listing = []

    # ФУНКЦИЯ ДЛЯ ОПРЕДЕЛЕНИЯ ПОЛА ПО ИМЕНИ
    def guess_gender(author_name):
        name_lower = str(author_name).lower()
        parts = name_lower.split()
        last_word = parts[-1] if parts else ""
        exceptions_male = ['илья', 'никита', 'данила', 'саша', 'женя', 'миша', 'коля', 'николай', 'кузьма']
        if last_word in exceptions_male: return 'Мужской'
        if last_word.endswith(('а', 'я', 'ва', 'на', 'ова', 'ева', 'ина')) and not re.search(r'[a-z0-9]', last_word):
            return 'Женский'
        return 'Мужской'

    # Устанавливаем индексы для уникальных фишек
    indices = list(range(target_count))
    emoji_index = random.choice(indices) if indices else 0

    for i in range(target_count):
        author_data = chosen_authors[i] if i < len(chosen_authors) else {}
        name = author_data.get('name', f'Автор_{i}')
        
        passed_gender = author_data.get('gender')
        if passed_gender in ['Ж', 'Женский', 'F', 'female']: assigned_gender = 'Женский'
        elif passed_gender in ['М', 'Мужской', 'M', 'male']: assigned_gender = 'Мужской'
        else: assigned_gender = guess_gender(name)
        
        if assigned_gender == 'Женский':
            persona = "Женщина, заказывает стройматериалы на участок по списку от мужа или строителей"
        else:
            persona = random.choice(male_personas)

        if ratings[i] == 5.0:
            sentiment = random.choice([
                "5 звезд: все прошло гладко, обычная рабочая покупка, без лишних восторгов",
                "5 звезд: очень помог человеческий фактор (менеджер вошел в положение, подсказал по размерам или остаткам)"
            ])
        elif ratings[i] in [4.0, 4.5]:
            sentiment = random.choice([
                f"{ratings[i]} звезды: товар и цена отличные, но пришлось подождать на погрузке минут 30. Менеджер извинился за задержку.",
                f"{ratings[i]} звезды: привезли все, но забыли мелкую позицию. Менеджер был на связи и быстро решил вопрос довозом."
            ])
        else:
            sentiment = "3.5 звезды: была реальная задержка по срокам доставки на пару дней, НО менеджер постоянно был на связи, вырулил казус и сделал хорошую скидку (или бонус), поэтому оценка не двойка."

        focus = random.choices(focus_points, weights=focus_weights, k=1)[0]
        style = random.choice(styles_and_lengths)

        case_rule = "(ВНИМАНИЕ: пиши весь текст с маленькой буквы)" if case_pool[i] == 'lower' else ""
        
        # Строгая раздача смайликов
        emoji_rule = "СТРОГО ОБЯЗАТЕЛЬНО добавь в конец текста 1 смайлик (👍, 👌 или 🔥)." if i == emoji_index else "СМАЙЛИКИ КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНЫ."

        authors_listing.append(
            f"--- ОТЗЫВ #{i+1} ---\n"
            f"Имя: {name}\n"
            f"Кто пишет (Персонаж): {persona}\n"
            f"Что купили (Товары): товар категории '{h2_header}' и немного сопутствующей мелочевки.\n"
            f"Суть отзыва (Фокус): {focus}\n"
            f"Оценка (Эмоция): {sentiment}\n"
            f"Стиль и объем текста: {style} {case_rule}\n"
            f"Смайлики: {emoji_rule}\n"
        )

    nl = chr(10)
    
    try:
        client = OpenAI(api_key=api_key, base_url="https://litellm.tokengate.ru/v1")
        
        prompt = f"""Твоя задача — сгенерировать пачку гиперреалистичных отзывов покупателей строительной базы. Тексты должны выглядеть так, будто их оставили реальные люди на картах или форуме.

СТРОГИЕ ПРАВИЛА ГЕНЕРАЦИИ (НЕ НАРУШАТЬ):
1. ЗАПРЕТ НА КОМПАНИИ: Категорически запрещено использовать слова "компания", "фирма", "конкуренты", "поставщик", а также любые названия магазинов. Пиши "на базе", "у них", "здесь".
2. РЕАЛИСТИЧНАЯ ДОСТАВКА: Жесткий запрет писать, что доставили "в тот же день", "сегодня", "за пару часов". Доставка занимает минимум день-два. Пиши "привезли в назначенный срок", "ждал пару дней".
3. СТОП-СЛОВА (Маркеры ИИ): Никогда не используй слова: безупречный, высококачественный, настоятельно рекомендую, порадовало, в целом, подводя итог, оптимальный, профессионализм, клиентоориентированность, превзошел ожидания, на высшем уровне, данного.
4. ЗАПРЕТ НА ШАБЛОНЫ И СТРУКТУРУ: Не используй приветствия и прощания. Начинай текст сразу с сути. Не делай обобщающих выводов в конце.
5. САМОПРОВЕРКА НА ФАЛЬШЬ (ВНУТРЕННИЙ РЕДАКТОР): Перед тем как выдать текст, проверь его. Люди НЕ говорят "избранный товар", "огромное преимущество", "отличный элемент для моих задач" или "со своей задачей справляется". Если фраза звучит как рекламный буклет — перепиши ее простым бытовым языком (например, "взял для навеса, пойдет"). Не коверкай слова до абсурда (не пиши "адно" вместо "одно").
6. РАБОТА С LSI И КЛЮЧАМИ (КРИТИЧЕСКИ ВАЖНО):
СПИСОК LSI: {", ".join(final_lsi)} 
ОБЯЗАТЕЛЬНО распредели эти слова так, чтобы каждое встретилось ровно 1 раз на всю пачку. Оборачивай в <b>...</b>. 
СТРОГО СКЛОНЯЙ ключи по падежам и числам по правилам русского языка! Запрещено вставлять их криво. (Например, нельзя писать "взял на каркас для <b>лестница</b>" — нужно писать "на каркас для <b>лестницы</b>").

ВВОДНЫЕ ДАННЫЕ ДЛЯ КАЖДОГО ОТЗЫВА:
{nl.join(authors_listing)}

ВЕРНИ СТРОГО JSON МАССИВ:
[{{ "Имя": "...", "Оценка": ..., "Текст": "..." }}]
"""
        
        resp = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85
        )
        content = re.sub(r'```json\s*|```', '', resp.choices[0].message.content).strip()
        reviews = json.loads(content)
        
        # === ПОСТ-ПРОЦЕССИНГ (Регистр и точки) ===
        for i in range(len(reviews)):
            if i < len(raw_dates):
                reviews[i]["Дата"] = raw_dates[i]
            else:
                reviews[i]["Дата"] = date.today().strftime("%d.%m.%Y")
                
            text = reviews[i].get("Текст", "")
            if text:
                if case_pool[i] == 'lower':
                    text = text[0].lower() + text[1:]
                elif case_pool[i] == 'upper':
                    text = text[0].upper() + text[1:]
                
                # Удаляем точки на конце в 60% случаев. Смайлики при этом не трогаются, так как они не в списке знаков препинания.
                if random.random() <= 0.60:
                    while text and text[-1] in ['.', '!', '?', ',', ';']:
                        text = text[:-1]
                        
                reviews[i]["Текст"] = text
                    
        return reviews
    except Exception as e:
        return [{"Имя": "Ошибка", "Текст": str(e), "Оценка": 5.0, "Дата": date.today().strftime("%d.%m.%Y")}]
def generate_full_article_v2(api_key, h1_marker, h2_topic, lsi_list):
    if not api_key: return "Error: No API Key"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://litellm.tokengate.ru/v1")
    except ImportError: return "Error: Library 'openai' not installed"
    
    lsi_string = ", ".join(lsi_list)
    
    stop_words_list = (
        "является, представляет собой, ключевой компонент, широко применяется, "
        "обладают, характеризуются, отличается, разнообразие, широкий спектр, "
        "оптимальный, уникальный, данный, этот, изделия, материалы, "
        "высокое качество, доступная цена, индивидуальный подход, "
        "доставка, оплата, условия поставки, звоните, менеджер"
    )

    contact_html_block = (
        'Предлагаем консультацию с менеджером по номеру '
        '<nobr><a href="tel:#PHONE#" onclick="ym(document.querySelector(\'#ya_counter\').getAttribute(\'data-counter\'),\'reachGoal\',\'tel\');gtag(\'event\', \'Click po nomeru telefona\', {{\'event_category\' : \'Click\', \'event_label\' : \'po nomeru telefona\'}});gtag(\'event\', \'Lead_Goal\', {{\'event_category\' : \'Click\', \'event_label\' : \'Leads Goal\'}});" class="a_404 ct_phone">#PHONE#</a></nobr>, '
        'либо пишите на почту <a href="mailto:#EMAIL#" onclick="ym(document.querySelector(\'#ya_counter\').getAttribute(\'data-counter\'),\'reachGoal\',\'email\');gtag(\'event\', \'Click napisat nam\', {{\'event_category\' : \'Click\', \'event_label\' : \'napisat nam\'}});gtag(\'event\', \'Lead_Goal\', {{\'event_category\' : \'Click\', \'event_label\' : \'Leads Goal\'}});" class="a_404">#EMAIL#</a>.'
    )

    system_instruction = (
        "Ты — технический редактор. Ты пишешь фактами, связным русским языком. "
        "Ты соблюдаешь HTML-структуру (списки <ul>, таблицы). "
        "Ты умеешь грамотно вписывать ключевые слова, меняя их форму и порядок слов. "
        "Если нужно вписать много ключевых слов — ты увеличиваешь объем текста, чтобы они смотрелись естественно."
    )
    
    # В ВАШЕМ ПРОМТЕ:
    # {exact_h2} в старом коде — это то, про что писать статью. 
    # В новой логике: 
    # - {h1_marker} — это предмет статьи (ключевое слово для H3 и текста).
    # - {h2_topic} — это точный заголовок H2.
    
    user_prompt = f"""
    ЗАДАЧА: Напиши техническую статью.
    
    [I] ГЛАВНЫЕ ПРАВИЛА РАБОТЫ С КЛЮЧОМ ("{h1_marker}"):
    
    1. В ЗАГОЛОВКАХ H3 (СТРОГО ЦЕЛИКОМ):
       - Здесь фраза "{h1_marker}" должна стоять ЦЕЛИКОМ (рядом).
       - МОЖНО: Склонять (Монтаж трубы стальной).
       - НЕЛЬЗЯ: Разрывать слова или заменять синонимами.
       
    2. В ТЕКСТЕ И АБЗАЦАХ (МЯГКОЕ ВХОЖДЕНИЕ):
       - Общая плотность слов из ключа — 1.5%.
       - ВАЖНО: В тексте ТЫ ОБЯЗАН РАЗБИВАТЬ фразу, менять порядок слов.
       - ПЛОХО: "Купить трубу стальную можно..." (спам).
       - ХОРОШО: "Стальная поверхность трубы обеспечивает..." (разбил слова).
       - ХОРОШО: "Для этой трубы характерна стальная структура..." (поменял местами).
       
    [II] ЛОГИКА HTML (СТРОГО):
    
    1. СПИСКИ:
       - <ol>: ТОЛЬКО для пошаговых процессов.
       - <ul>: ДЛЯ ХАРАКТЕРИСТИК, СФЕР, СВОЙСТВ (Списки №1, №2, №3 — СТРОГО <ul>).
       - ВАЖНО: Не используй двоеточие (:) ВНУТРИ пунктов списка.
       
    2. ТАБЛИЦА:
       - Класс: "brand-accent-table".
       - Шапка через <thead> и <th>.

    [III] СТРУКТУРА ТЕКСТА:
    
    1.1. Заголовок: <h2>{h2_topic}</h2>.
    
    1.2. БЭНГЕР: 3-4 связных предложения. Опиши товар "{h1_marker}" нормальным языком (что это, ГОСТ, материал).
    
    1.3. Абзац 1 + Контакты: 
    {contact_html_block}
    
    1.4. Подводка к списку 1 (:).
    
    1.5. Список №1 (6 пунктов): ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ.
    (Формат: <ul>). "И" меняй на запятую. Цифры значащие.
       
    1.6. Абзац 2. Описание производства.
    
    1.7. ТАБЛИЦА ХАРАКТЕРИСТИК (СПРАВОЧНАЯ):
    4-5 строк. Без дублей списка №1.
    ИСПОЛЬЗУЙ ЭТОТ КОД:
    <table class="brand-accent-table">
        <thead>
            <tr>
                <th>Параметр</th>
                <th>Значение</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>ГОСТ / ТУ</td><td>[Данные]</td></tr>
            <tr><td>Марка сплава</td><td>[Данные]</td></tr>
            <tr><td>[Параметр 3]</td><td>[Данные]</td></tr>
            <tr><td>[Параметр 4]</td><td>[Данные]</td></tr>
        </tbody>
    </table>
    
    1.8. Подзаголовок H3 (ШАБЛОН): 
    "Классификация {h1_marker} (род. падеж, с маленькой буквы)"
    (Тут ключ целиком!).
    
    1.9. Абзац 3. Виды, типы. (Тут разбивай ключ).
    
    1.10. Подводка к списку 2 (:).
    
    1.11. Список №2 (6 пунктов): СФЕРЫ ПРИМЕНЕНИЯ.
    (Формат: <ul>).
       
    1.12. Абзац 4. Условия эксплуатации.
                          
    1.13. Подзаголовок H3 (ШАБЛОН):
    "Монтаж {h1_marker} (род. падеж)" ИЛИ "Обработка {h1_marker} (род. падеж)".
    (Тут ключ целиком!).
    
    1.14. Абзац 5. Технология работы.
    
    1.15. Подводка к списку 3 (:).
    
    1.16. Список №3 (6 пунктов): ЭКСПЛУАТАЦИОННЫЕ СВОЙСТВА.
    (Без союзов "и"). Формат: <ul>.
       
    1.17. Абзац 6. Резюме и отгрузка.

    [IV] ДОПОЛНИТЕЛЬНО (LSI ЯДРО):
    Список слов: {{{lsi_string}}}
    
    ПРАВИЛА LSI:
    1. ИСПОЛЬЗУЙ ВЕСЬ СПИСОК (Общие + Семантика).
    2. Каждое слово — РОВНО 1 РАЗ (не повторяй).
    3. Выделяй каждое вставленное слово тегом <b>жирный</b>.
    4. Если слов много — УВЕЛИЧИВАЙ ОБЪЕМ ТЕКСТА. Пиши дополнительные предложения, чтобы слова вписывались плавно, а не "обрубками". Смысл и связность важнее краткости.

    [V] СТОП-СЛОВА: ({stop_words_list}).
    
    ВЫВОД: ТОЛЬКО HTML КОД.
    СТРОГИЕ ПРАВИЛА ОФОРМЛЕНИЯ И ФИЛЬТРАЦИИ (КРИТИЧЕСКИ ВАЖНО):
1. Оформление списков: Каждый пункт любого маркированного списка должен строго заканчиваться точкой с запятой (;), а самый последний пункт списка — точкой (.). Без исключений.
2. Написание диапазонов: Числовые диапазоны (длина, вес, размер) пиши через тире и с сокращением единиц измерения. Пример: "4-9 м", "10-20 мм". КАТЕГОРИЧЕСКИ ЗАПРЕЩАЕТСЯ писать "от 4 до 9 метров". 
   - Исключение: для температурных диапазонов с минусовыми значениями используй слова (например, "от -10 до +50 °C").
3. Игнорирование конкурентов: Если в списке переданных ключевых слов тебе попадется странный мусор на латинице или названия чужих магазинов/компаний, полностью ИГНОРИРУЙ ИХ. Из латиницы разрешается писать только марки сталей и стандарты (AISI 304, DIN и т.д.).
4. Характеристики в списках: Если в пункте списка перечисляется свойство (характеристика) и его значения, ОБЯЗАТЕЛЬНО ставь тире (–) между названием свойства и списком значений.
5. Максимально сократи использование союза "и". В 90% случаев заменяй его запятой при перечислении или просто перестраивай предложение. Текст должен быть динамичным и лаконичным, без лишнего "нагромождения" связок.
    """
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.25
        )
        content = response.choices[0].message.content
        content = re.sub(r'^```html', '', content.strip())
        content = re.sub(r'^```', '', content.strip())
        content = re.sub(r'```$', '', content.strip())
        
        # --- ОЧИСТКА ---
        # Теги <b> НЕ удаляем!
        content = content.replace('—', '&ndash;').replace('–', '&ndash;').replace('&mdash;', '&ndash;')
        content = content.replace('**', '').replace('__', '')
        
        return content
    except Exception as e:
        return f"API Error: {str(e)}"

def scrape_h1_h2_from_url(url):
    """
    Заходит на страницу, забирает H1 (как маркер) и первый релевантный H2.
    """
    # 1. Попытка через curl_cffi (чтобы обойти 403)
    try:
        from curl_cffi import requests as cffi_requests
        r = cffi_requests.get(
            url, 
            impersonate="chrome110", 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'},
            timeout=20
        )
        content = r.content
        encoding = r.encoding if r.encoding else 'utf-8'
    except:
        # 2. Fallback на requests
        try:
            import requests
            import urllib3
            urllib3.disable_warnings()
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20, verify=False)
            content = r.content
            encoding = r.apparent_encoding
        except Exception as e:
            return None, None, f"Ошибка соединения: {e}"

    if r.status_code != 200:
        return None, None, f"HTTP Error {r.status_code}"

    try:
        soup = BeautifulSoup(content, 'html.parser', from_encoding=encoding)
        
        # --- ИЩЕМ H1 ---
        h1_tag = soup.find('h1')
        h1_text = h1_tag.get_text(strip=True) if h1_tag else ""
        
        # --- ИЩЕМ H2 ---
        # Сначала пробуем найти H2 внутри контентного блока (частая практика)
        h2_text = ""
        content_div = soup.find('div', class_=re.compile(r'(desc|content|text|article)'))
        if content_div:
            h2_tag = content_div.find('h2')
            if h2_tag: h2_text = h2_tag.get_text(strip=True)
        
        # Если не нашли в контенте, берем просто первый попавшийся H2 на странице
        if not h2_text:
            h2_tag = soup.find('h2')
            if h2_tag: h2_text = h2_tag.get_text(strip=True)
            
        # Если H2 вообще нет, используем H1 как тему (fallback)
        if not h2_text:
            h2_text = h1_text

        if not h1_text:
            return None, None, "H1 не найден"

        return h1_text, h2_text, "OK"

    except Exception as e:
        return None, None, f"Ошибка парсинга: {e}"

# === ФИКС ОШИБКИ ВИДЖЕТОВ (StreamlitAPIException) ===
if 'pending_widget_updates' in st.session_state:
    for k, v in st.session_state['pending_widget_updates'].items():
        st.session_state[k] = v
    del st.session_state['pending_widget_updates']

# ==========================================
# ДОПОЛНИТЕЛЬНЫЕ ПРОВЕРКИ (ТУРГЕНЕВ И TEXT.RU)
# ==========================================
def check_turgenev_sync(text, key): # <-- добавили key
    url = 'https://turgenev.ashmanov.com/'
    params = {
        'api': 'risk',
        'key': key, # <-- теперь ключ берется из переменной
        'text': text,
        'more': '1'
    }
    try:
        r = requests.post(url, data=params, timeout=20)
        res = r.json()
        if 'error' in res:
            return f"Ошибка: {res['error']}"
        return f"{res.get('risk', '-')} ({res.get('level', '-')})"
    except Exception as e:
        return "Ошибка соединения"

def send_textru_sync(text, key):
    url = 'https://api.text.ru/post'
    payload = {
        'userkey': key,
        'text': text
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        res = r.json()
        return res.get('text_uid')
    except:
        return None

def check_textru_status_sync(uid, key):
    url = 'https://api.text.ru/post'
    payload = {
        'userkey': key,
        'uid': uid,
        'jsonvisible': 'detail'
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        res = r.json()
        if 'text_unique' in res:
            return f"{res['text_unique']}%"
        elif res.get('error_code') == 181:
            return "processing"
        else:
            return f"Ошибка: {res.get('error_desc', res.get('error_code', 'Unknown'))}"
    except:
        return "error"

def validate_topic_deepseek(api_key, h1, h2, text):
    """Проверяет соответствие текста теме через DeepSeek-v3.2"""
    if not api_key or not text: return True
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://litellm.tokengate.ru/v1")
        prompt = f"""
        Ты - строгий SEO-асессор.
        Тебе передан сгенерированный текст.
        Проверь, действительно ли этот текст написан на заданную тему:
        H1: {h1}
        H2: {h2}

        Бывает, что нейросеть ошибается и пишет про смежные категории или вообще о другом товаре/услуге.
        Если текст точно раскрывает заявленную тему, ответь строго одним словом: YES
        Если текст написан про другое или тема не раскрыта, ответь строго одним словом: NO
        """
        # Обрезаем текст для экономии токенов (суть ясна по первым 3000 символам)
        user_msg = f"Текст для проверки:\n{text[:3000]}"

        response = client.chat.completions.create(
            model="deepseek/deepseek-v3.2",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=10
        )
        ans = response.choices[0].message.content.strip().upper()
        if "NO" in ans: return False
        return True
    except Exception as e:
        return True # При ошибке API пропускаем, чтобы не блочить весь процесс
# ==========================================
# 7. UI TABS RESTRUCTURED
# ==========================================
def global_stop_callback():
    st.session_state.lsi_automode_active = False
    st.session_state.faq_automode_active = False
    st.session_state.reviews_automode_active = False # Добавлено
    st.session_state.auto_run_active = False
    st.session_state.start_analysis_flag = False

tab_seo_main, tab_wholesale_main, tab_projects, tab_monitoring, tab_lsi_gen, tab_faq_gen, tab_reviews_gen = st.tabs(["📊 SEO Анализ", "🏭 Оптовый генератор", "📁 Проекты", "📉 Мониторинг позиций", "📝 LSI Тексты", "❓ FAQ Генератор", "💬 Отзывы"])

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

        # Узнаем, запущена ли генерация
        is_running = st.session_state.get('start_analysis_flag', False)

        my_input_type = st.radio("Тип страницы", ["Релевантная страница на вашем сайте", "Исходный код страницы или текст", "Без страницы"], horizontal=True, label_visibility="collapsed", key="my_page_source_radio", disabled=is_running)
        if my_input_type == "Релевантная страница на вашем сайте":
            st.text_input("URL страницы", placeholder="https://site.ru/catalog/tovar", label_visibility="collapsed", key="my_url_input", disabled=is_running)
        elif my_input_type == "Исходный код страницы или текст":
            st.text_area("Исходный код или текст", height=200, label_visibility="collapsed", placeholder="Вставьте HTML", key="my_content_input", disabled=is_running)

        st.markdown("### Поисковой запрос")
        st.text_input("Основной запрос", placeholder="Например: купить пластиковые окна", label_visibility="collapsed", key="query_input", disabled=is_running)
        
        st.markdown("### Поиск конкурентов")
        
        # --- Обработка авто-переключения ---
        if st.session_state.get('force_radio_switch'):
            st.session_state["competitor_source_radio"] = "Список url-адресов ваших конкурентов"
        st.session_state['force_radio_switch'] = False
        # -----------------------------------------------

        source_type_new = st.radio("Источник", ["Поиск через API Arsenkin (TOP-30)", "Список url-адресов ваших конкурентов"], horizontal=True, label_visibility="collapsed", key="competitor_source_radio", disabled=is_running)
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
        if st.session_state.get('analysis_done'):
            results = st.session_state.analysis_results
            
# ==========================================
        # БЛОК ЭКСПОРТА РЕЗУЛЬТАТОВ (УМНЫЙ EXCEL)
        # ==========================================
            st.markdown("---")
            st.subheader("💾 Экспорт результатов")
    
            export_format = st.radio(
                "Выберите формат файла:", 
                ["📊 Excel (С графиками и группами)", "⚙️ JSON (Для разработчиков)"], 
                horizontal=True
            )
    
            import datetime
            import io
            from collections import defaultdict
    
            if "Excel" in export_format:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    workbook = writer.book
                    
                    # --- ЛИСТ 1: РЕЛЕВАНТНОСТЬ И ГРАФИК ---
                    if 'relevance_top' in results:
                        df_rel = results['relevance_top']
                        df_rel.to_excel(writer, sheet_name='Релевантность', index=False)
                        worksheet = writer.sheets['Релевантность']
                        
                        max_row = len(df_rel)
                        
                        chart = workbook.add_chart({'type': 'line'})
                        chart.add_series({
                            'name':       ['Релевантность', 0, 3],
                            'categories': ['Релевантность', 1, 0, max_row, 0],
                            'values':     ['Релевантность', 1, 3, max_row, 3],
                            'line':       {'color': '#1f77b4', 'width': 2.5}
                        })
                        chart.add_series({
                            'name':       ['Релевантность', 0, 4],
                            'categories': ['Релевантность', 1, 0, max_row, 0],
                            'values':     ['Релевантность', 1, 4, max_row, 4],
                            'line':       {'color': '#d62728', 'width': 2.5}
                        })
                        chart.set_title ({'name': 'Анализ конкурентов ТОПа'})
                        chart.set_size({'width': 750, 'height': 400})
                        
                        worksheet.insert_chart('G2', chart)
    
                        if 'bad_urls' in results and results['bad_urls']:
                            start_row_bad = max_row + 3
                            format_header = workbook.add_format({'bold': True, 'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                            worksheet.write(start_row_bad, 0, "ОТСЕЯННЫЕ КОНКУРЕНТЫ (АНОМАЛИИ)", format_header)
                            df_bad = pd.DataFrame(results['bad_urls'])
                            df_bad.to_excel(writer, sheet_name='Релевантность', startrow=start_row_bad+1, index=False)
    
                        worksheet.set_column('A:B', 30)
    
                    # --- ЛИСТ 2: УПУЩЕННАЯ СЕМАНТИКА (ЕДИНАЯ СВОДКА) ---
                    def get_category_for_word(w):
                        w_low = w.lower()
                        if w_low in [x.lower() for x in st.session_state.get('categorized_products', [])]: return "📦 Товары"
                        if w_low in [x.lower() for x in st.session_state.get('categorized_commercial', [])]: return "💰 Коммерция"
                        if w_low in [x.lower() for x in st.session_state.get('categorized_services', [])]: return "🛠️ Услуги"
                        if w_low in [x.lower() for x in st.session_state.get('categorized_geo', [])]: return "🌍 Гео"
                        if w_low in [x.lower() for x in st.session_state.get('categorized_dimensions', [])]: return "📏 Размеры/ГОСТ"
                        if w_low in [x.lower() for x in st.session_state.get('categorized_general', [])]: return "📂 Общие"
                        return "❓ Остальные"
    
                    # БЕЗОПАСНОЕ ИЗВЛЕЧЕНИЕ СЛОВ (ИСПРАВЛЕННАЯ ОШИБКА)
                    high_words = []
                    for item in results.get('missing_semantics_high', []):
                        if isinstance(item, dict) and 'word' in item:
                            high_words.append(item['word'])
                        elif isinstance(item, dict) and 'lemma' in item:
                            high_words.append(item['lemma'])
                        elif isinstance(item, str):
                            high_words.append(item)
    
                    low_words = []
                    for item in results.get('missing_semantics_low', []):
                        if isinstance(item, dict) and 'word' in item:
                            low_words.append(item['word'])
                        elif isinstance(item, dict) and 'lemma' in item:
                            low_words.append(item['lemma'])
                        elif isinstance(item, str):
                            low_words.append(item)
    
                    cat_order = ["📦 Товары", "💰 Коммерция", "🛠️ Услуги", "🌍 Гео", "📏 Размеры/ГОСТ", "📂 Общие", "❓ Остальные"]
                    grouped_sem = {k: {'high': [], 'low': []} for k in cat_order}
    
                    for w in high_words: grouped_sem[get_category_for_word(w)]['high'].append(w)
                    for w in low_words: grouped_sem[get_category_for_word(w)]['low'].append(w)
    
                    sem_rows = []
                    for cat in cat_order:
                        if grouped_sem[cat]['high'] or grouped_sem[cat]['low']:
                            sem_rows.append({
                                "Группа": cat,
                                "Основные (Важные)": ", ".join(grouped_sem[cat]['high']),
                                "Дополнительные": ", ".join(grouped_sem[cat]['low'])
                            })
    
                    if sem_rows:
                        df_sem = pd.DataFrame(sem_rows)
                        df_sem.to_excel(writer, sheet_name='Упущенная_Семантика', index=False)
                        worksheet_sem = writer.sheets['Упущенная_Семантика']
                        worksheet_sem.set_column('A:A', 20)
                        worksheet_sem.set_column('B:C', 80)
    
                    # --- ЛИСТ 3: ГЛУБИНА ---
                    if 'depth' in results:
                        results['depth'].to_excel(writer, sheet_name='Матрица_Глубины', index=False)
    
                    # --- ЛИСТ 4: TF-IDF ---
                    if 'hybrid' in results:
                        results['hybrid'].to_excel(writer, sheet_name='TF-IDF', index=False)
    
                st.download_button(
                    label="📥 Скачать готовый отчет",
                    data=excel_buffer.getvalue(),
                    file_name=f"SEO_Отчет_{datetime.datetime.now().strftime('%d_%m_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                def custom_serializer(obj):
                    if isinstance(obj, pd.DataFrame): return obj.to_dict(orient='records')
                    import numpy as np
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return str(obj)
    
                import json
                json_data = json.dumps(results, default=custom_serializer, ensure_ascii=False, indent=4)
                st.download_button(
                    label="📥 Скачать JSON",
                    data=json_data,
                    file_name=f"SEO_Raw_{datetime.datetime.now().strftime('%d_%m_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            st.markdown("---")
        # ==========================================
        # НИЖЕ ИДУТ ТВОИ ГРАФИКИ И ТАБЛИЦЫ
        # ==========================================
            if 'relevance_top' in results and not results['relevance_top'].empty:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 График релевантности (Нажмите, чтобы раскрыть)", expanded=False):                    
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

        is_running = st.session_state.get('start_analysis_flag', False)
        st.checkbox("Исключать <noindex>", key="settings_noindex", disabled=is_running)
        st.checkbox("Учитывать Alt/Title", key="settings_alt", disabled=is_running)
        st.checkbox("Учитывать числа", key="settings_numbers", disabled=is_running)
        st.checkbox("Нормировать по длине", key="settings_norm", disabled=is_running)
        st.checkbox("Авто-фильтр слабых сайтов", key="settings_auto_filter", help="Сайты с низкой релевантностью будут автоматически перенесены в список исключенных.", disabled=is_running)
        
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

# Вывод ОТЛАДКИ для Ширины (чтобы понять, почему 95)
        if 'debug_width' in results:
            found = results['debug_width']['found']
            needed = results['debug_width']['needed']
            pct = int((found / needed * 100)) if needed > 0 else 0
            st.caption(f"🔍 Диагностика Ширины: Найдено **{found}** из **{needed}** обязательных слов ({pct}%).")
        
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

# ==================================================================
            # 🔥 HOOK ДЛЯ LSI ГЕНЕРАТОРА (ВКЛАДКА 5) - ИСПРАВЛЕННАЯ ВЕРСИЯ v2
            # ==================================================================
            if st.session_state.get('lsi_automode_active'):
                
                # 1. Достаем данные текущей задачи
                current_idx = st.session_state.get('lsi_processing_task_id')
                
                # Защита: проверяем существование очереди и индекса
                if 'bg_tasks_queue' not in st.session_state or current_idx is None or current_idx >= len(st.session_state.bg_tasks_queue):
                    st.session_state.lsi_automode_active = False
                    st.success("Все задачи выполнены (или очередь пуста)!")
                    st.stop()

                task = st.session_state.bg_tasks_queue[current_idx]
                
                # 2. Достаем LSI (TF-IDF) из результатов анализа
                lsi_words = []
                
                # --- ИСПРАВЛЕНИЕ: БЕРЕМ ИЗ SESSION_STATE, А НЕ ИЗ ЛОКАЛЬНОЙ ПЕРЕМЕННОЙ ---
                results_data = st.session_state.get('analysis_results')
                
                if results_data and results_data.get('hybrid') is not None and not results_data['hybrid'].empty:
                    # Берем топ-15 слов
                    lsi_words = results_data['hybrid'].head(15)['Слово'].tolist()
                # --------------------------------------------------------------------------
                
                # 3. Добавляем общие слова из настроек
                common_lsi = ["гарантия", "доставка", "цена", "купить", "оптом", "в наличии"] 
                combined_lsi = list(set(common_lsi + lsi_words))
                
# 4. ГЕНЕРИРУЕМ СТАТЬЮ
                # --- ИСПРАВЛЕНИЕ: Ищем ключ везде ---
                api_key_gen = st.session_state.get('gemini_key_persistent')
                if not api_key_gen:
                    api_key_gen = st.session_state.get('bulk_api_key_v3')
                if not api_key_gen:
                    try: api_key_gen = st.secrets["GEMINI_KEY"]
                    except: pass
                
                html_out = ""
                status_code = "Error"
                
                if not api_key_gen:
                    html_out = "Ошибка: Нет API ключа Gemini (введите на вкладке 5). Нажмите Enter после ввода ключа!"
                    status_code = "Key Error"
                else:
                    try:
                        html_out = generate_full_article_v2(api_key_gen, task['h1'], task['h2'], combined_lsi)
                        status_code = "OK"
                    except Exception as e:
                        html_out = f"Error generating: {e}"
                        status_code = "Gen Error"

                # 5. СОХРАНЯЕМ РЕЗУЛЬТАТ В СПИСОК ВКЛАДКИ 5
                if 'bg_results' not in st.session_state:
                    st.session_state.bg_results = []
                    
                found_existing = False
                for existing_res in st.session_state.bg_results:
                    if existing_res['h1'] == task['h1'] and existing_res['h2'] == task['h2']:
                        existing_res['content'] = html_out
                        existing_res['lsi_added'] = lsi_words
                        existing_res['status'] = status_code
                        found_existing = True
                        break
                
                if not found_existing:
                    st.session_state.bg_results.append({
                        "h1": task['h1'],
                        "h2": task['h2'],
                        "source_url": task.get('source_url', '-'),
                        "lsi_added": lsi_words,
                        "content": html_out,
                        "status": status_code
                    })

                # 6. ПЛАНИРУЕМ СЛЕДУЮЩУЮ ЗАДАЧУ
                next_task_idx = current_idx + 1
                
                if next_task_idx < len(st.session_state.bg_tasks_queue):
                    next_task = st.session_state.bg_tasks_queue[next_task_idx]
                    
                    st.toast(f"✅ Готово: {task['h1']}. Переход к: {next_task['h1']}...")
                    
                    # === ПРИНУДИТЕЛЬНОЕ ВОССТАНОВЛЕНИЕ ===
                    # Если ключ вдруг удалился, но есть в secrets или другой переменной - восстанавливаем
                    if 'bulk_api_key_v3' not in st.session_state:
                         # Пытаемся найти в persist или secrets
                         recovered = st.session_state.get('gemini_key_persistent') or st.secrets.get("GEMINI_KEY", "")
                         if recovered:
                             st.session_state.bulk_api_key_v3 = recovered

# === ТОЧЕЧНАЯ ОЧИСТКА СТАРЫХ РЕЗУЛЬТАТОВ ===
                    keys_to_clear = [
                        'analysis_results', 'analysis_done', 'naming_table_df', 
                        'ideal_h1_result', 'raw_comp_data', 'full_graph_data',
                        'detected_anomalies', 'serp_trend_info', 'excluded_urls_auto'
                    ]
                    for k in keys_to_clear:
                        st.session_state.pop(k, None)
                        
                    # УСТАНОВКА ПАРАМЕТРОВ ДЛЯ СЛЕДУЮЩЕГО
                    st.session_state['pending_widget_updates'] = {
                        'query_input': next_task['h1'],
                        'competitor_source_radio': "Поиск через API Arsenkin (TOP-30)",
                        'my_page_source_radio': "Без страницы",
                        'my_url_input': "",
                        'settings_region': st.session_state.get('lsi_settings_region', 'Москва') # <--- ПЕРЕДАЕМ РЕГИОН
                    }
                    st.session_state['lsi_processing_task_id'] = next_task_idx
                    st.session_state['start_analysis_flag'] = True 
                    st.session_state['analysis_done'] = False
                    
                    time.sleep(0.5)
                    st.rerun()
                    
                else:
                    st.session_state.lsi_automode_active = False
                    st.balloons()
                    st.success("🏁 ВСЕ ЗАДАЧИ В ОЧЕРЕДИ ОБРАБОТАНЫ! Результаты на вкладке 5.")
            
            # ==================================================================

# 5. УПУЩЕННАЯ СЕМАНТИКА + МОТОР АВТОМАТИЗАЦИИ
        high = results.get('missing_semantics_high', [])
        low = results.get('missing_semantics_low', [])
        
        if high or low:
            total_missing = len(high) + len(low)
            with st.expander(f"🧩 Упущенная семантика ({total_missing})", expanded=False):
                if high: 
                    words_high = ", ".join([x['word'] for x in high])
                    st.markdown(f"<div style='background:#EBF5FF; padding:12px; border-radius:8px; border:1px solid #BFDBFE; color:#1E40AF; margin-bottom:10px;'><b>🔥 Важные:</b><br>{words_high}</div>", unsafe_allow_html=True)
                if low: 
                    words_low = ", ".join([x['word'] for x in low])
                    st.markdown(f"<div style='background:#F8FAFC; padding:12px; border-radius:8px; border:1px solid #E2E8F0; color:#475569;'><b>🔸 Дополнительные:</b><br>{words_low}</div>", unsafe_allow_html=True)

            # --- ВОТ ЭТОТ КУСОК ОЖИВЛЯЕТ СКРИПТ ---
            if st.session_state.get('lsi_automode_active'):
                with st.status("🛠️ Работает автоматика: перенос данных...", expanded=True) as status:
                    
                    # 1. Забираем слова
                    st.write("📝 Считываю таблицу TF-IDF...")
                    current_lsi_words = [x['word'] for x in high] if high else []
                    
                    # 2. Ищем текущую задачу
                    t_id = st.session_state.get('lsi_processing_task_id', 0)
                    if t_id < len(st.session_state.bg_tasks_queue):
                        task = st.session_state.bg_tasks_queue[t_id]
                        
                        # 3. Записываем во вкладку 5
                        st.write(f"📂 Сохраняю LSI для: **{task['h1']}**")
                        
                        new_rec = {
                            "h1": task['h1'],
                            "h2": task['h2'],
                            "lsi_added": current_lsi_words,
                            "content": "",
                            "status": "Ready",
                            "date": "10.02.2026"
                        }
                        
                        if 'bg_results' not in st.session_state:
                            st.session_state.bg_results = []
                        
                        # Проверка на дубли
                        is_exist = False
                        for r in st.session_state.bg_results:
                            if r['h1'] == task['h1']:
                                r['lsi_added'] = current_lsi_words
                                is_exist = True
                                break
                        if not is_exist:
                            st.session_state.bg_results.append(new_rec)
                        
                        # 4. ГЕНЕРАЦИЯ (Сразу здесь!)
                        st.write("🧠 Отправляю LSI и ТЗ в нейросеть Gemini...")
                        try:
                            # Убедись, что функция называется именно так
                            text = generate_article_with_gemini(task['h1'], current_lsi_words)
                            
                            # Находим запись и сохраняем текст
                            for r in st.session_state.bg_results:
                                if r['h1'] == task['h1']:
                                    r['content'] = text
                                    r['status'] = "Done"
                            st.write("✅ Текст сгенерирован успешно!")
                        except Exception as e:
                            st.error(f"❌ Ошибка нейросети: {e}")

                        # 5. ПЕРЕХОД К СЛЕДУЮЩЕМУ
                        next_id = t_id + 1
                        if next_id < len(st.session_state.bg_tasks_queue):
                            st.write(f"⏭️ Переключаюсь на следующий ключ: **{st.session_state.bg_tasks_queue[next_id]['h1']}**")
                            
                            st.session_state.lsi_processing_task_id = next_id
                            st.session_state.query_input = st.session_state.bg_tasks_queue[next_id]['h1']
                            st.session_state.start_analysis_flag = True
                            st.session_state.analysis_done = False # Сбрасываем, чтобы пошел новый поиск
                            
                            time.sleep(2)
                            status.update(label="🔄 Запускаю новый цикл анализа...", state="running")
                            st.rerun()
                        else:
                            st.session_state.lsi_automode_active = False
                            status.update(label="🏁 Все задачи выполнены!", state="complete")
                            st.success("Готово! Проверь вкладку 5.")
                            st.balloons()

# =========================================================
        # 🔥 БЛОК АВТОМАТИЗАЦИИ: ПЕРЕНОС В ТАБ 5 И ГЕНЕРАЦИЯ
        # =========================================================
        if st.session_state.get('lsi_automode_active'):
            with st.status("🚀 Автоматизация: обработка LSI и генерация...", expanded=True) as status:
                
                # 1. Извлекаем слова из результатов анализа
                st.write("📥 Собираю LSI-слова из таблицы TF-IDF...")
                current_lsi = [x['word'] for x in high] if high else []
                
                # 2. Находим текущую задачу в очереди
                task_id = st.session_state.get('lsi_processing_task_id', 0)
                task = st.session_state.bg_tasks_queue[task_id]
                
                # 3. Формируем запись для Вкладки №5
                st.write(f"💾 Переношу данные для ключа: **{task['h1']}**")
                
                new_entry = {
                    "h1": task['h1'],
                    "h2": task['h2'],
                    "lsi_added": current_lsi,
                    "content": "",  # Сюда запишем текст ниже
                    "status": "Generating",
                    "date": "05.02.2026" # Установлено согласно вашим правилам
                }
                
                if 'bg_results' not in st.session_state:
                    st.session_state.bg_results = []
                
                # Добавляем в результаты (или обновляем, если уже есть)
                # Чтобы не было дублей при случайном реране
                existing_idx = next((i for i, r in enumerate(st.session_state.bg_results) if r['h1'] == task['h1']), None)
                if existing_idx is not None:
                    st.session_state.bg_results[existing_idx] = new_entry
                    res_idx = existing_idx
                else:
                    st.session_state.bg_results.append(new_entry)
                    res_idx = len(st.session_state.bg_results) - 1

                # 4. ГЕНЕРАЦИЯ ТЕКСТА (чтобы не заходить на 5 вкладку вручную)
                st.write("🤖 Gemini генерирует текст статьи... Подождите.")
                try:
                    # Вызываем вашу функцию генерации. 
                    # Убедитесь, что название функции совпадает (обычно generate_article_with_gemini)
                    generated_text = generate_article_with_gemini(task['h1'], current_lsi)
                    st.session_state.bg_results[res_idx]['content'] = generated_text
                    st.session_state.bg_results[res_idx]['status'] = "Done"
                    st.write("✅ Текст успешно сгенерирован и сохранен!")
                except Exception as e:
                    st.error(f"❌ Ошибка Gemini: {e}")
                    st.session_state.bg_results[res_idx]['status'] = "Error"

                # 5. ПЕРЕХОД К СЛЕДУЮЩЕМУ КЛЮЧУ
                next_task_idx = task_id + 1
                if next_task_idx < len(st.session_state.bg_tasks_queue):
                    st.write(f"⏭️ Следующий ключ в очереди: **{st.session_state.bg_tasks_queue[next_task_idx]['h1']}**")
                    
                    # Обновляем состояние для следующей итерации
                    st.session_state.lsi_processing_task_id = next_task_idx
                    st.session_state.query_input = st.session_state.bg_tasks_queue[next_task_idx]['h1']
                    st.session_state.start_analysis_flag = True 
                    st.session_state.analysis_done = False # Сбрасываем флаг, чтобы запустить новый поиск
                    
                    status.update(label="🔄 Перехожу к анализу следующего ключа...", state="running")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.session_state.lsi_automode_active = False
                    status.update(label="🏁 ВСЕ ЗАДАЧИ ЗАВЕРШЕНЫ!", state="complete")
                    st.balloons()
                    st.success("Все ключи из списка обработаны. Результаты на вкладке 5.")

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
            
# 2. Сбор КАНДИДАТОВ И ПРОВЕРКА КЭША БД
        current_source_val = st.session_state.get("competitor_source_radio")
        user_target_top_n = st.session_state.settings_top_n
        download_limit = 30 # ВСЕГДА КАЧАЕМ 30 для TF-IDF
        
        # Получаем текущий регион
        current_region = st.session_state.get('settings_region', 'Москва')
        
        cached_data_for_graph = None
        if "API" in current_source_val and current_input_type == "Без страницы":
            # Передаем регион в функцию
            cached_data_for_graph = get_cached_analysis(st.session_state.query_input, current_region)

        if cached_data_for_graph:
            st.toast(f"⚡ Найдено в кэше ({current_region})! Парсинг пропущен", icon="🗄️")
            data_for_graph = cached_data_for_graph
            targets_for_graph = [{'url': d['url'], 'pos': d['pos']} for d in data_for_graph]
        else:
            candidates_pool = []
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
                data_for_graph = comp_data_valid[:download_limit]
                targets_for_graph = [{'url': d['url'], 'pos': d['pos']} for d in data_for_graph]
                
                # +++ СОХРАНЯЕМ В БД ТОЛЬКО ЧТО СКАЧАННОЕ (С УЧЕТОМ РЕГИОНА) +++
                if "API" in current_source_val and current_input_type == "Без страницы":
                    save_cached_analysis(st.session_state.query_input, current_region, data_for_graph)

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
        # 🔥 ПОЛНЫЙ ДВИЖОК ОТЗЫВОВ (ВЕРСИЯ 6: УПУЩЕННАЯ СЕМАНТИКА + ТОВАРЫ)
        # ==========================================
        if st.session_state.get('reviews_automode_active'):
            try:
               # === НОВЫЙ СБОР СЛОВ ДЛЯ ОТЗЫВОВ ===
                # 1. Берем ТОЛЬКО "основные важные" (high) из первой вкладки
                res_seo = st.session_state.get('analysis_results', {})
                missing_high = [x['word'] for x in res_seo.get('missing_semantics_high', [])]
                
                # 2. Берем слова, которые вылетели при генерации на 2-й вкладке (теги, промо, текст)
                unused_from_gen = list(st.session_state.get('global_unused_for_reviews', set()))
                
                # 3. Объединяем их в единый список кандидатов (убираем дубликаты)
                raw_candidates = list(set(missing_high + unused_from_gen))
                
                # 4. (Опционально) Очищаем буфер неиспользованных слов, чтобы они не переносились на следующие проекты
                # st.session_state['global_unused_for_reviews'] = set()
                # ===================================
                
                # Подтягиваем категории (если они были определены на вкладке 1)
                known_products = set(st.session_state.get('categorized_products', []))
                known_services = set(st.session_state.get('categorized_services', [])) # <-- Добавили услуги
                
                # === БЛОКИРОВКА ГЕО И МУСОРА ===
                known_geo = set(st.session_state.get('categorized_geo', []))
                known_geo.update(st.session_state.get('orig_geo', []))
                
                try:
                    _, _, _, dict_geo, _, _ = load_lemmatized_dictionaries()
                    known_geo.update(dict_geo)
                except:
                    pass
                
                lsi_nouns = []
                
                STOP_NOUNS = {
                    'код', 'сайт', 'каталог', 'меню', 'корзина', 'поиск', 'ссылка', 'страница', 
                    'версия', 'ошибка', 'руб', 'грн', 'шт', 'раз', 'два', 'три', 'номер', 
                    'телефон', 'адрес', 'email', 'фильтр', 'сортировка', 'артикул', 'наличие',
                    'акция', 'скидка', 'хит', 'новинка', 'клик', 'вход', 'регистрация', 'главная',
                    'карта', 'новость', 'статья', 'отзыв', 'вакансия', 'оплата', 'доставка',
                    'город', 'регион', 'россия', 'москва', 'спб', 'доставка', 'производство', 'завод'
                }

                if raw_candidates:
                    for w in raw_candidates:
                        w_clean = str(w).lower().strip()
                        
                        if (len(w_clean) > 2 
                            and not re.search(r'[a-zA-Z0-9]', w_clean) 
                            and w_clean not in STOP_NOUNS
                            and w_clean not in known_geo):
                            
                            parsed = morph.parse(w_clean)[0]
                            is_name_or_geo = any(tag in parsed.tag for tag in ['Name', 'Surn', 'Patr', 'Geox'])
                            is_orphan_modifier = any(tag in parsed.tag for tag in ['ADJF', 'ADJS', 'PRTF', 'PRTS', 'ADVB', 'GRND'])
                            
                            is_noun = 'NOUN' in parsed.tag
                            is_known_product = w_clean in known_products
                            is_known_service = w_clean in known_services
                            
                            if is_known_product or (is_noun and not is_name_or_geo and not is_orphan_modifier):
                                priority = 1 if is_known_product else 0
                                
                                # === ОПРЕДЕЛЯЕМ ТИП СЛОВА ДЛЯ ШАБЛОНА ===
                                if is_known_product: w_type = 'product'
                                elif is_known_service: w_type = 'service'
                                else: w_type = 'general'
                                
                                lsi_nouns.append({
                                    'word': w_clean,
                                    'parse': parsed,
                                    'priority': priority,
                                    'type': w_type # <-- Сохраняем тип
                                })
                
                # СОРТИРОВКА: Сначала слова из категории "Товары", потом просто существительные из упущенного
                # Shuffle делаем внутри групп, чтобы сохранять рандом, но соблюдать приоритет
                lsi_nouns.sort(key=lambda x: x['priority'], reverse=True)
                
                # Если слов мало, можно перемешать топ-20, чтобы не шли всегда одни и те же товары
                if len(lsi_nouns) > 5:
                    top_slice = lsi_nouns[:10]
                    random.shuffle(top_slice)
                    lsi_nouns[:10] = top_slice

                curr_idx = st.session_state.get('reviews_current_index', 0)
                queue = st.session_state.get('reviews_queue', [])
                
                if curr_idx < len(queue):
                    task = queue[curr_idx]
                else:
                    st.session_state.reviews_automode_active = False
                    st.rerun()

                # 2. Загрузка справочников
                import os
                if os.path.exists("dicts/fio.csv"):
                    df_fio = pd.read_csv("dicts/fio.csv", sep=";")
                else:
                    df_fio = pd.DataFrame([{"Фамилия": "Покупатель", "Имя": ""}])

                if os.path.exists("dicts/templates.csv"):
                    df_templates = pd.read_csv("dicts/templates.csv", sep=";")
                else:
                    df_templates = pd.DataFrame([
                        {"Шаблон": "Заказывали {товар}. Все хорошо."}, 
                        {"Шаблон": "Качество на уровне, {товар} пришел вовремя."},
                        {"Шаблон": "Нормальная компания. Был нужен {товар}, помогли подобрать."}
                    ])
                
                var_dict = {}
                if os.path.exists("dicts/vars.csv"):
                    df_vars = pd.read_csv("dicts/vars.csv", sep=";")
                    for _, row in df_vars.iterrows():
                        v_name = str(row['Переменная']).strip()
                        if pd.notna(row['Значения']):
                            var_dict[f"{{{v_name}}}"] = [v.strip() for v in str(row['Значения']).split('|')]
                            
                if "{товар}" not in var_dict:
                    var_dict["{товар}"] = ["заказ", "товар", "продукцию"]

                # Умные фразы-конструкторы по категориям
                LSI_SENTENCES_PROD = [
                    {"tpl": "Отдельно отмечу качество {}.", "case": "gent"}, 
                    {"tpl": "Заказывали {} оптом.", "case": "accs"},       
                    {"tpl": "Партия {} пришла без брака.", "case": "gent"},        
                    {"tpl": "Проблем с {} не возникло.", "case": "ablt"},    
                    {"tpl": "Сейчас {} в наличии.", "case": "nomn"}          
                ]
                
                LSI_SENTENCES_SERV = [
                    {"tpl": "Также потребовалась {}.", "case": "nomn"},
                    {"tpl": "Отдельное спасибо за {}.", "case": "accs"},
                    {"tpl": "С {} справились на отлично.", "case": "ablt"},
                    {"tpl": "Кстати, {} здесь на хорошем уровне.", "case": "nomn"}
                ]
                
                LSI_SENTENCES_GEN = [
                    {"tpl": "Обратили внимание на {}.", "case": "accs"},
                    {"tpl": "Порадовало наличие {}.", "case": "gent"},
                    {"tpl": "К слову, с {} проблем не возникло.", "case": "ablt"},
                    {"tpl": "Хорошо продумали {}.", "case": "accs"}
                ]

                with st.spinner(f"📦 Сборка отзывов для: {task.get('q', 'запроса')}..."):
                    for _ in range(st.session_state.get('reviews_per_query', 3)):
                        # ФИО
                        f_row = df_fio.sample(n=1).iloc[0]
                        c_fio = f"{f_row.get('Имя', '')} {f_row.get('Фамилия', '')}".strip()
                        if not c_fio: c_fio = "Клиент"

                        # Шаблон
                        final_text = random.choice(df_templates['Шаблон'].values)
                        used_lsi_word = None
                        
                        # --- ВНЕДРЕНИЕ LSI (С ПРИОРИТЕТОМ ТОВАРОВ) ---
                        if "{товар}" in final_text and lsi_nouns:
                            top_n = min(len(lsi_nouns), 10)
                            lsi_obj = lsi_nouns[random.randint(0, top_n - 1)]
                            replacement = f"**{lsi_obj['word']}**"
                            final_text = final_text.replace("{товар}", replacement, 1)
                            used_lsi_word = True

                        # Заполнение остальных переменных
                        tags = re.findall(r"\{[а-яА-ЯёЁa-zA-Z0-9_]+\}", final_text)
                        for t in tags:
                            if t in var_dict:
                                final_text = final_text.replace(t, random.choice(var_dict[t]), 1)
                            elif t == "{дата}":
                                dt = (datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 60))).strftime("%d.%m.%Y")
                                final_text = final_text.replace("{дата}", dt)
                                
                        # 2. Если {товар} не был заменен, вставляем отдельное предложение
                        if not used_lsi_word and lsi_nouns:
                            top_n = min(len(lsi_nouns), 10)
                            lsi_obj = lsi_nouns[random.randint(0, top_n - 1)]
                            
                            parsed_word = lsi_obj['parse']
                            w_type = lsi_obj.get('type', 'general')
                            
                            # Выбираем правильный набор шаблонов
                            if w_type == 'product':
                                tpl_list = LSI_SENTENCES_PROD
                            elif w_type == 'service':
                                tpl_list = LSI_SENTENCES_SERV
                            else:
                                tpl_list = LSI_SENTENCES_GEN
                                
                            tpl_obj = random.choice(tpl_list)
                            
                            try:
                                inflected = parsed_word.inflect({tpl_obj['case']})
                                w_res = inflected.word if inflected else lsi_obj['word']
                            except:
                                w_res = lsi_obj['word']
                            
                            w_res_bold = f"**{w_res}**"
                            add_sentence = tpl_obj['tpl'].format(w_res_bold)
                            
                            sentences = [s.strip() for s in final_text.split('.') if len(s) > 1]
                            if sentences:
                                idx = random.randint(0, len(sentences))
                                sentences.insert(idx, add_sentence)
                            else:
                                sentences = [final_text, add_sentence]
                                
                            # Исправление регистра (Capitalization)
                            capitalized_sentences = []
                            for s in sentences:
                                clean_s = s.strip()
                                if clean_s:
                                    cap_s = clean_s[0].upper() + clean_s[1:]
                                    capitalized_sentences.append(cap_s)
                            
                            final_text = ". ".join(capitalized_sentences) + "."

                        # Финальная чистка
                        final_text = re.sub(r"\{[^}]+\}", "", final_text)
                        final_text = final_text.replace("..", ".").replace(" .", ".").replace(" ,", ",")
                        
                        # === ОЧИСТКА АРТЕФАКТОВ ПУНКТУАЦИИ ===
                        final_text = re.sub(r'([!?]+)\s*\.', r'\1', final_text) 
                        final_text = re.sub(r'\s+([.,!?])', r'\1', final_text)
                        # =====================================

                        final_text = re.sub(r'\s+', ' ', final_text).strip()
                        
                        if final_text:
                            final_text = final_text[0].upper() + final_text[1:]

                        st.session_state.reviews_results.append({
                            "ФИО": c_fio,
                            "Запрос": task.get('q', '-'),
                            "URL": task.get('url', '-'),
                            "Отзыв": final_text
                        })

                # Переход дальше
                n_idx = curr_idx + 1
                if n_idx < len(queue):
                    st.session_state.reviews_current_index = n_idx
                    nxt = queue[n_idx]
                    
                    keys_to_clear = [
                        'analysis_results', 'analysis_done', 'naming_table_df', 
                        'ideal_h1_result', 'raw_comp_data', 'full_graph_data',
                        'detected_anomalies', 'serp_trend_info', 'excluded_urls_auto'
                    ]
                    for k in keys_to_clear:
                        st.session_state.pop(k, None)
                    
                    st.session_state['pending_widget_updates'] = {
                        'query_input': nxt.get('q'),
                        'my_url_input': nxt.get('url', ''),
                        'my_page_source_radio': "Релевантная страница на вашем сайте" if nxt.get('url') != 'manual' else "Без страницы"
                    }
                    st.session_state.start_analysis_flag = True
                    st.toast(f"🔄 Обработка: {nxt.get('q')}")
                    import time
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.session_state.reviews_automode_active = False
                    st.success("✅ Готово!")
                    st.balloons()
            
            except Exception as e:
                st.error(f"❌ Ошибка генерации: {e}")
                st.session_state.reviews_automode_active = False
        # ==========================================
        # 🔥 БЛОК: КЛАССИФИКАЦИЯ СЕМАНТИКИ (ИСПРАВЛЕННЫЙ)
        # ==========================================
        words_to_check = [x['word'] for x in results_final.get('missing_semantics_high', [])]
        if len(words_to_check) < 5:
            words_to_check.extend([x['word'] for x in results_final.get('missing_semantics_low', [])[:20]])

        if not words_to_check:
            st.session_state.categorized_products = []
            st.session_state.categorized_services = []
            st.session_state.categorized_commercial = []
            st.session_state.categorized_dimensions = []
            st.session_state.categorized_geo = []
            st.session_state.categorized_general = []
            st.session_state.categorized_sensitive = []
        else:
            if 'categorized_products' not in st.session_state or not st.session_state.categorized_products:
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

        # Готовим обновления для виджетов
        if 'pending_widget_updates' not in st.session_state:
            st.session_state['pending_widget_updates'] = {}
        
        updates = st.session_state['pending_widget_updates']

        if words_to_check and 'categorized_sensitive' in locals() or 'categorized' in locals():
            # Если классификация только что прошла, берем из локальной переменной
            sens = categorized['sensitive'] if 'categorized' in locals() else st.session_state.categorized_sensitive
            updates['sensitive_words_input_final'] = "\n".join(sens)
        
        all_found_products = st.session_state.get('categorized_products', [])
        count_prods = len(all_found_products)
        
        if count_prods > 0:
            if count_prods < 20:
                st.session_state.auto_tags_words = all_found_products
                st.session_state.auto_promo_words = []
            else:
                half_count = int(math.ceil(count_prods / 2))
                st.session_state.auto_tags_words = all_found_products[:half_count]
                st.session_state.auto_promo_words = all_found_products[half_count:]
            
            updates['tags_products_edit_final'] = "\n".join(st.session_state.auto_tags_words)
            updates['promo_keywords_area_final'] = "\n".join(st.session_state.auto_promo_words)
        
        st.session_state['pending_widget_updates'] = updates

        # --- ГРАФИКИ ---
        current_source_val = st.session_state.get('competitor_source_radio', '')
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

# ==================================================================
        # 🔥 HOOK ДЛЯ LSI ГЕНЕРАТОРА (ВКЛАДКА 5)
        # Если этот анализ был заказан Вкладкой 5, мы генерируем текст и идем дальше
        # ==================================================================
        if st.session_state.get('lsi_automode_active'):
            
            # 1. Достаем данные текущей задачи
            current_idx = st.session_state.get('lsi_processing_task_id')
            task = st.session_state.bg_tasks_queue[current_idx]
            
            # 2. Достаем LSI (TF-IDF) из результатов анализа Вкладки 1
            lsi_words = []
            if results_final.get('hybrid') is not None and not results_final['hybrid'].empty:
                lsi_words = results_final['hybrid'].head(15)['Слово'].tolist()
            
            # 3. Добавляем общие слова из настроек (нужно сохранить их в session_state во вкладке 5)
            # (Предполагаем, что они есть, или берем дефолт)
# 3. Добавляем общие слова из поля ввода
            raw_common = st.session_state.get('common_lsi_input', "гарантия, звоните, консультация, купить, оплата, оптом, отгрузка, под заказ, поставка, прайс-лист, цены")
            common_lsi = [w.strip() for w in raw_common.split(",") if w.strip()]
            combined_lsi = list(set(common_lsi + lsi_words))
            
# 4. ГЕНЕРИРУЕМ СТАТЬЮ
            # Читаем из SUPER_GLOBAL_KEY (который мы создали в Шаге 1)
            api_key_gen = st.session_state.get('SUPER_GLOBAL_KEY')
            
            # Фолбэк: если вдруг его нет, пробуем старый метод
            if not api_key_gen:
                api_key_gen = st.session_state.get('bulk_api_key_v3')
            
            try:
                html_out = generate_full_article_v2(api_key_gen, task['h1'], task['h2'], combined_lsi)
                status_code = "OK"
            except Exception as e:
                html_out = f"Error: {e}"
                status_code = "Error"

            # 5. СОХРАНЯЕМ РЕЗУЛЬТАТ В СПИСОК ВКЛАДКИ 5
            st.session_state.bg_results.append({
                "h1": task['h1'],
                "h2": task['h2'],
                "source_url": task.get('source_url', '-'),
                "lsi_added": lsi_words,
                "content": html_out,
                "status": status_code
            })

            # 6. ПЛАНИРУЕМ СЛЕДУЮЩУЮ ЗАДАЧУ
            finished_ids = set(f"{r['h1']}|{r['h2']}" for r in st.session_state.bg_results)
            next_task_idx = -1
            
            for i, t in enumerate(st.session_state.bg_tasks_queue):
                unique_id = f"{t['h1']}|{t['h2']}"
                if unique_id not in finished_ids:
                    next_task_idx = i
                    break
            
            st.write(f"DEBUG: Найдена следующая задача под индексом: {next_task_idx}")

            if next_task_idx != -1:
# === ТОЧЕЧНАЯ ОЧИСТКА СТАРЫХ РЕЗУЛЬТАТОВ ===
                keys_to_clear = [
                    'analysis_results', 'analysis_done', 'naming_table_df', 
                    'ideal_h1_result', 'raw_comp_data', 'full_graph_data',
                    'detected_anomalies', 'serp_trend_info', 'excluded_urls_auto'
                ]
                for k in keys_to_clear:
                    st.session_state.pop(k, None)

                # 3. БЕРЕМ НОВУЮ ЗАДАЧУ
                next_task = st.session_state.bg_tasks_queue[next_task_idx]
                
                # Ставим статус "В работе" для таблицы очереди
                st.session_state.bg_tasks_queue[next_task_idx]['status'] = "🔍 Инициализация парсинга..."
                
                # === ТОЧЕЧНАЯ ОЧИСТКА СТАРЫХ РЕЗУЛЬТАТОВ ===
                keys_to_clear = [
                    'analysis_results', 'analysis_done', 'naming_table_df', 
                    'ideal_h1_result', 'raw_comp_data', 'full_graph_data',
                    'detected_anomalies', 'serp_trend_info', 'excluded_urls_auto'
                ]
                for k in keys_to_clear:
                    st.session_state.pop(k, None)
                    
                # УСТАНОВКА ПАРАМЕТРОВ ДЛЯ СЛЕДУЮЩЕГО
                    st.session_state['pending_widget_updates'] = {
                        'query_input': next_task['h1'],
                        'competitor_source_radio': "Поиск через API Arsenkin (TOP-30)",
                        'my_page_source_radio': "Без страницы",
                        'my_url_input': "",
                        'settings_region': st.session_state.get('lsi_settings_region', 'Москва') # <--- ПЕРЕДАЕМ РЕГИОН
                    }
                
                # Включаем "автопилот"
                st.session_state['start_analysis_flag'] = True
                st.session_state['analysis_done'] = False
                st.session_state['lsi_processing_task_id'] = next_task_idx
                
                st.toast(f"🚀 Начинаем работу над: {next_task['h1']}")
                time.sleep(1)
                st.rerun()

        if not st.session_state.get('lsi_automode_active'):
            st.rerun()

# ==========================================
# TAB 2: WHOLESALE GENERATOR (SMART PIPELINE V11 - БРОНЕБОЙНЫЙ ТЕКСТ)
# ==========================================
with tab_wholesale_main:
    # 0. Инициализация расширенного датафрейма
    if 'gen_result_df' not in st.session_state or st.session_state.gen_result_df is None:
         st.session_state.gen_result_df = pd.DataFrame(columns=[
            'Page URL', 'Product Name', 'IP_PROP4839', 'IP_PROP4817', 'IP_PROP4818', 
            'IP_PROP4819', 'IP_PROP4820', 'IP_PROP4821', 'IP_PROP4822', 'IP_PROP4823', 
            'IP_PROP4824', 'IP_PROP4816', 'IP_PROP4825', 'IP_PROP4826', 'IP_PROP4834', 
            'IP_PROP4835', 'IP_PROP4836', 'IP_PROP4837', 'IP_PROP4838', 'IP_PROP4829', 'IP_PROP4831',
            'FAQ Коммерческий вопрос', 'FAQ Коммерческий ответ', 
            'FAQ Информационный вопрос', 'FAQ Информационный ответ',
            'Весь текст целиком', 
            'DeepSeek Контекст', 'DeepSeek Комментарий',
            'Риск Тургенев', 'Тургенев Комментарий',
            'Уникальность', 'Text.ru Комментарий', 'Text.ru UID'
        ])
    else:
        # Безопасное добавление колонок, если сессия уже запущена
        for col in ['FAQ Коммерческий вопрос', 'FAQ Коммерческий ответ', 'FAQ Информационный вопрос', 'FAQ Информационный ответ']:
            if col not in st.session_state.gen_result_df.columns:
                st.session_state.gen_result_df[col] = ""

    st.header("🏭 Умный Оптовый Конвейер (V11 - Бронебойный текст)")
    st.info("Исправлена критическая ошибка генерации текста. Теперь текст пишется всегда, а доп. блоки (таблицы, теги) встраиваются ПОД текстом.")

# --- НЕВИДИМЫЙ ХУК АВТО-КОНВЕЙЕРА ---
    if st.session_state.get('ws_automode_active') and st.session_state.get('ws_waiting_for_analysis') and st.session_state.get('analysis_done'):
        task_idx = st.session_state.auto_current_index
        queue = st.session_state.ws_bg_tasks_queue
        
        if task_idx < len(queue):
            current_task = queue[task_idx]
            h1_marker = current_task.get('h1', current_task['name'])
            h2_header = current_task.get('h2', current_task['name'])
            
            with st.status(f"⚙️ Обработка: {h2_header} (Товар {task_idx + 1} из {len(queue)})", expanded=True) as status_logger:
                step_logger = st.empty() # <--- СОЗДАЕМ ЕДИНУЮ КОРОБКУ ДЛЯ СТАТУСОВ
                step_logger.info("⏳ Этап 1: Семантика получена. Начинаем распределение слов...")
                
                row_data = {col: "" for col in st.session_state.gen_result_df.columns}
                row_data['Page URL'] = current_task['url']
                row_data['Product Name'] = h2_header
                
                try:
                    for k, v in STATIC_DATA_GEN.items():
                        if k in row_data: row_data[k] = v
                except NameError: pass 
                
                # ВОТ ЭТОТ TRY Я СЛУЧАЙНО ПРОПУСТИЛ В ПРОШЛЫЙ РАЗ!
                try: 
                    cat_dimensions = st.session_state.get('categorized_dimensions', [])
                    cat_commercial = st.session_state.get('categorized_commercial', [])
                    cat_general = st.session_state.get('categorized_general', [])
                    cat_geo = st.session_state.get('categorized_geo', [])
                    structure_keywords = st.session_state.get('categorized_products', []) + st.session_state.get('categorized_services', [])
                    faq_cands = st.session_state.get('categorized_info', []) # Запас для FAQ
                    
                    # --- ЧИТАЕМ ИЗ НАДЕЖНОГО СЕЙФА ---
                    global_text = st.session_state.get('safe_ws_global_text', True)
                    global_tables = st.session_state.get('safe_ws_global_tables', True)
                    global_tags = st.session_state.get('safe_ws_global_tags', True)
                    global_promo = st.session_state.get('safe_ws_global_promo', True)
                    global_geo = st.session_state.get('safe_ws_global_geo', True)
                    global_faq = st.session_state.get('safe_ws_global_faq', True)
                    global_reviews = st.session_state.get('safe_ws_global_reviews', True)

                    auto_num_blocks_setting = st.session_state.get('safe_ws_num_blocks_val', 5)
                    use_auto_blocks = st.session_state.get('safe_ws_auto_blocks', True)
                    current_faq_count = st.session_state.get('safe_ws_faq_count', 4)
                    rev_count = st.session_state.get('safe_ws_reviews_count', 3)

                    # ДОСТАЕМ ЗАМОРОЖЕННЫЕ КЛЮЧИ И ГАЛОЧКИ
                    gemini_api_key = st.session_state.get('safe_gemini_key', '')
                    turgenev_api_key = st.session_state.get('safe_turgenev_key', '')
                    textru_api_key = st.session_state.get('safe_textru_key', '')

                    use_turgenev_chk = st.session_state.get('safe_use_turgenev', False)
                    use_textru_chk = st.session_state.get('safe_use_textru', False)
                    use_ds_chk = st.session_state.get('safe_use_ds', True)
                    # ----------------------------------)
                    
                    # =================================================================
                    # УМНЫЙ ПОИСК ПО БАЗАМ И РАСПРЕДЕЛЕНИЕ (ДО ГЕНЕРАЦИИ!)
                    # =================================================================
                    
                    # 1. Загрузка базы для ТЕГОВ (links_base.xlsx)
                    links_data = []
                    if global_promo or global_tags:
                        db_path = "data/links_base.xlsx" if os.path.exists("data/links_base.xlsx") else "links_base.xlsx"
                        if not os.path.exists(db_path) and os.path.exists("data/links_base..xlsx"):
                            db_path = "data/links_base..xlsx"
                            
                        if os.path.exists(db_path):
                            try:
                                df_links = pd.read_excel(db_path)
                                for _, r in df_links.iterrows():
                                    u_link = str(r.iloc[0]).strip().rstrip('/')
                                    if not u_link or u_link == 'nan': continue
                                    name_val = str(r.iloc[1]).strip() if len(df_links.columns) > 1 else ""
                                    links_data.append({'url': u_link, 'name': name_val})
                            except Exception as e:
                                status_logger.error(f"Ошибка чтения {db_path}: {e}")

                    # 2. Загрузка базы для ПРОМО (images_db.xlsx)
                    images_data = []
                    if global_promo:
                        img_db_path = "data/images_db.xlsx" if os.path.exists("data/images_db.xlsx") else "images_db.xlsx"
                        if os.path.exists(img_db_path):
                            try:
                                df_img = pd.read_excel(img_db_path)
                                for _, r in df_img.iterrows():
                                    u_link = str(r.iloc[0]).strip().rstrip('/')
                                    if not u_link or u_link == 'nan': continue
                                    img_val = str(r.iloc[1]).strip() if len(df_img.columns) > 1 else ""
                                    images_data.append({'url': u_link, 'img': img_val})
                            except Exception as e:
                                status_logger.error(f"Ошибка чтения {img_db_path}: {e}")

                    # --- МЯСОРУБКА ДЛЯ СЛОВ (ДРОБИМ СТРОКИ) ---
                    raw_candidates = list(set(structure_keywords))
                    all_candidates = []
                    for raw_kw in raw_candidates:
                        clean_str = re.sub(r'^.*?[–-]\s*', '', str(raw_kw))
                        parts = re.split(r'[,;]', clean_str)
                        for p in parts:
                            p = p.strip().strip('.')
                            if len(p) > 2:
                                all_candidates.append(p)
                                
                    all_candidates = list(set(all_candidates))
                    
                    # Разделяем слова на блоки ДО поиска
                    tags_1_cands = all_candidates[:15]
                    promo_cands = all_candidates[15:20]
                    tags_2_cands = all_candidates[20:]

                    used_urls = set([current_task['url'].rstrip('/')])
                    unmatched_kws = []
                    tags_block_1 = []
                    promo_block = []
                    tags_block_2 = []

                    import random

                    # --- ФУНКЦИЯ ПОИСКА ДЛЯ ТЕГОВ (Ищет по Названию) ---
                    def get_tag_data(kw):
                        kw_clean = kw.lower().strip()
                        kw_stem = kw_clean[:4] if len(kw_clean) > 4 else kw_clean
                        matches = []
                        
                        for item in links_data:
                            if item['url'].rstrip('/') in used_urls: continue
                            db_n_low = item['name'].lower()
                            # Ищем корень слова во втором столбце
                            if kw_stem in db_n_low or kw_clean in db_n_low:
                                matches.append(item)
                                
                        if matches:
                            chosen = random.choice(matches) # БЕРЕМ РАНДОМНОЕ СОВПАДЕНИЕ
                            used_urls.add(chosen['url'].rstrip('/'))
                            return chosen['url'], chosen['name']
                        return None, None

                    # --- ФУНКЦИЯ ПОИСКА И ПАРСИНГА ДЛЯ ПРОМО (Ищет транслит в URL) ---
                    def get_promo_data(kw):
                        kw_clean = kw.lower().strip()
                        kw_stem = kw_clean[:4] if len(kw_clean) > 4 else kw_clean
                    
                        # Делаем транслит корня (со встроенной подстраховкой)
                        try:
                            from transliterate import translit
                            kw_translit = translit(kw_stem, 'ru', reversed=True).replace("'", "")
                        except Exception:
                            t_dict = {'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'e','ж':'zh','з':'z','и':'i','й':'i','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f','х':'h','ц':'c','ч':'ch','ш':'sh','щ':'sch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya'}
                            kw_translit = "".join([t_dict.get(c, c) for c in kw_stem])
                    
                        matches = []
                        for item in images_data:
                            u_clean = item['url'].rstrip('/')
                            if u_clean in used_urls:
                                continue
                            # Ищем транслит в URL (первый столбец)
                            if kw_translit in u_clean.lower():
                                matches.append(item)
                                
                        if matches:
                            chosen = random.choice(matches) # БЕРЕМ РАНДОМНОЕ СОВПАДЕНИЕ
                            u_target = chosen['url']
                            img_target = chosen['img']
                            if str(img_target) == 'nan' or not img_target:
                                img_target = "https://via.placeholder.com/260"
                            
                            # ИДЕМ НА САЙТ И ПАРСИМ ХЛЕБНЫЕ КРОШКИ (ДЛЯ НАЗВАНИЯ ПРОМО)
                            promo_title = kw.capitalize() # Заглушка по умолчанию
                            try:
                                resp = requests.get(u_target, timeout=5)
                                if resp.status_code == 200:
                                    soup = BeautifulSoup(resp.text, 'html.parser')
                                    
                                    # Ищем контейнер с крошками (стандартные классы)
                                    breadcrumbs = soup.find(['ul', 'div', 'nav'], class_=re.compile(r'breadcrumb|breadcrumbs|nav-path', re.I))
                                    
                                    if breadcrumbs:
                                        # Ищем все элементы внутри
                                        crumbs = breadcrumbs.find_all(['li', 'span', 'a'])
                                        # Чистим от разделителей (/, » и т.д.)
                                        clean_crumbs = [c.get_text(strip=True) for c in crumbs if len(c.get_text(strip=True)) > 2 and c.get_text(strip=True) not in ['/', '\\', '>', '»', '•', '-']]
                                        if clean_crumbs:
                                            promo_title = clean_crumbs[-1] # Берем последний пункт
                                    else:
                                        # Резервный вариант, если крошек нет - берем H1
                                        h1_tag = soup.find('h1')
                                        if h1_tag: 
                                            promo_title = h1_tag.get_text(strip=True)
                            except Exception:
                                pass 
                                
                            used_urls.add(u_target.rstrip('/'))
                            return u_target, promo_title, img_target
                            
                        # ВОТ ЭТОЙ СТРОЧКИ НЕ ХВАТАЛО (Возвращаем 3 пустые переменные, если совпадений нет)
                        return None, None, None

                    # --- РАСПРЕДЕЛЯЕМ ПО БЛОКАМ ---
                    
                    # 1. Плитка тегов 1
                    if global_tags:
                        for kw in tags_1_cands:
                            u, n = get_tag_data(kw)
                            if u: tags_block_1.append({"url": u, "name": n, "kw": kw})
                            else: unmatched_kws.append(kw)
                    else: unmatched_kws.extend(tags_1_cands)

                    # 2. Промо-блок (с парсингом)
                    if global_promo:
                        for kw in promo_cands:
                            u, n, img = get_promo_data(kw)
                            if u: promo_block.append({"url": u, "name": n, "img": img, "kw": kw})
                            else: unmatched_kws.append(kw)
                    else: unmatched_kws.extend(promo_cands)

                    # 3. Плитка тегов 2
                    if global_tags:
                        for kw in tags_2_cands:
                            u, n = get_tag_data(kw)
                            if u: tags_block_2.append({"url": u, "name": n, "kw": kw})
                            else: unmatched_kws.append(kw)
                    else: unmatched_kws.extend(tags_2_cands)
                    # =================================================================
                    # 4. УМНОЕ РАСПРЕДЕЛЕНИЕ ОСТАТКОВ (ИЗОЛИРОВАННЫЕ ПОТОКИ)
                    # =================================================================
                    import random
                    
                    # 1. Формируем "Чистый пул" (Только коммерция и остатки товаров/услуг).
                    # Сюда НЕ ПОПАДАЕТ корпоративный SEO-мусор из cat_general!
                    safe_cands = list(set(cat_commercial + unmatched_kws))
                    random.shuffle(safe_cands)
                    
                    faq_cands = st.session_state.get('categorized_info', []) # Запас для FAQ
                    review_cands = []
                    
                    # Отщипываем слова для FAQ (только из чистых!)
                    if global_faq:
                        chunk_size = min(7, len(safe_cands))
                        faq_cands.extend(safe_cands[:chunk_size])
                        safe_cands = safe_cands[chunk_size:]
            
                    # Отщипываем слова для Отзывов (только из чистых!)
                    if st.session_state.get('ws_global_reviews', True):
                        chunk_size = min(7, len(safe_cands))
                        review_cands.extend(safe_cands[:chunk_size])
                        safe_cands = safe_cands[chunk_size:]
            
                    # 2. Формируем пул для Главного текста.
                    # Сюда идут остатки чистых слов + ВЕСЬ cat_general (корпоративная вода).
                    # В SEO-статье нейросеть нормально переварит слова типа "проект", "успешно", "кабинет".
                    final_text_seo_list = list(set(safe_cands + cat_general))
                    random.shuffle(final_text_seo_list)
            
                    # Правильный подсчет вообще всех собранных слов для отчета
                    total_collected = len(cat_commercial) + len(cat_general) + len(structure_keywords) + len(cat_dimensions) + len(cat_geo)
                    with st.expander(f"📊 ОТЧЕТ: Распределение слов (Всего собрано: {total_collected} шт.)", expanded=True):
                        st.write(f"**В Текст ({len(final_text_seo_list)} шт)** (Включая общую семантику)")
                        st.write(f"**В FAQ ({len(faq_cands)} шт)**")
                        st.write(f"**В Отзывы ({len(review_cands)} шт)** (Только коммерция и товары)")
                        st.write(f"**В Плитку тегов ({len(tags_block_1) + len(tags_block_2)} шт)**")
                        st.write(f"**В Промо-блок ({len(promo_block)} шт)**")
                        st.write(f"**В Таблицу (ГОСТ/Размеры) ({len(cat_dimensions)} шт)**")
                        st.write(f"**В Гео-Блок ({len(cat_geo)} шт)**")

                    # =================================================================
                    # ГЕНЕРАЦИЯ ТЕКСТА
                    # =================================================================
                    curr_use_text = global_text
                    curr_use_tables = global_tables and (len(cat_dimensions) > 0)
                    curr_use_geo = global_geo and (len(cat_geo) > 0)
                    
                    base_text_raw = current_task.get('base_text', '')
                    b_text_str = str(base_text_raw).strip() if base_text_raw is not None else ""
                    if not b_text_str or b_text_str == "None":
                        safe_base_text = "Техническая информация о товаре. Основные параметры и характеристики для профессионалов."
                    else:
                        safe_base_text = base_text_raw

                    blocks = [""] * 5
                    generated_full_text = ""
                    gemini_api_key = st.session_state.get('SUPER_GLOBAL_KEY', '')
                    
                    if not gemini_api_key:
                        status_logger.error("❌ ОШИБКА: Отсутствует API-ключ Gemini!")
                    
                    from openai import OpenAI
                    client = OpenAI(api_key=gemini_api_key, base_url="https://litellm.tokengate.ru/v1") if gemini_api_key else None
                    
                    if curr_use_text and client:
                        words_count = len(final_text_seo_list)
                        # 🔥 БЕРЕМ СОХРАНЕННОЕ КОЛИЧЕСТВО БЛОКОВ
                        auto_num_blocks = st.session_state.get('safe_ws_num_blocks_val', 5)
                        
                        # 🔥 ПРОВЕРЯЕМ СОХРАНЕННУЮ ГАЛОЧКУ АВТО-РАСПРЕДЕЛЕНИЯ
                        if st.session_state.get('safe_ws_auto_blocks', True):
                            if words_count <= 15: auto_num_blocks = 3
                            elif words_count <= 25: auto_num_blocks = 4
                            else: auto_num_blocks = 5
                        
                        status_logger.write(f"🤖 Пишем SEO-текст (Слов: {words_count} ➔ Блоков: {auto_num_blocks})...")
                        blocks_raw = generate_ai_content_blocks(gemini_api_key, safe_base_text, h1_marker, h2_header, auto_num_blocks, final_text_seo_list)
                        
                        if not blocks_raw or "Error" in str(blocks_raw[0]):
                            step_logger.error(f"❌ Нейросеть вернула ошибку: {blocks_raw[0]}")
                        else:
                            cleaned_blocks = [b.replace("```html", "").replace("```", "").strip() for b in blocks_raw]
                            for i_b in range(len(cleaned_blocks)):
                                if i_b < 5: blocks[i_b] = cleaned_blocks[i_b]
                                
                            generated_full_text = " ".join(blocks)
                            # Убрали спам об успешной генерации, статус просто переключится на следующий шаг

                    # =================================================================
                    # ГЕНЕРАЦИЯ ТАБЛИЦЫ
                    # =================================================================
                    if curr_use_tables and client:
                        step_logger.warning("🧩 Этап 3: Верстаем таблицу размеров...")
                        dims_str = ", ".join(cat_dimensions)
                        prompt_tbl = f"""ТЫ — СТРОГИЙ ТЕХНОЛОГ. Задача: Сгенерировать HTML-таблицу для "{h2_header}".
ВВОДНЫЕ: Контекст текста: {generated_full_text[:3000]}. Обязательные параметры: [{dims_str}].
ПРАВИЛА: 
1. НЕ дублируй инфу из текста! 
2. Добавь НОВЫЕ технические характеристики.
3. Максимум 5 колонок.
4. Формат: <table class="brand-accent-table"><thead><tr>...</tr></thead><tbody>...</tbody></table>
Выдай только HTML код."""
                        try:
                            resp = client.chat.completions.create(model="google/gemini-2.5-pro", messages=[{"role": "user", "content": prompt_tbl}], temperature=0.25)
                            raw_table = resp.choices[0].message.content.replace("```html", "").replace("```", "").strip()
                            if "<table" in raw_table:
                                cl_tab = raw_table[raw_table.find("<table"):raw_table.find("</table>")+8]
                                if "brand-accent-table" not in cl_tab: cl_tab = cl_tab.replace("<table", "<table class='brand-accent-table'", 1)
                        except Exception as e:
                            status_logger.error(f"Ошибка таблицы: {e}")

                    # =================================================================
                    # ГЕНЕРАЦИЯ FAQ
                    # =================================================================
                    final_faq_html = ""
                    if global_faq and client:
                        # 🔥 БЕРЕМ СОХРАНЕННОЕ КОЛИЧЕСТВО ВОПРОСОВ
                        current_faq_count = st.session_state.get('safe_ws_faq_count', 4)
                        status_logger.write(f"❓ Генерируем FAQ ({current_faq_count} вопросов)...")
                        try:
                            faq_json = generate_faq_gemini(gemini_api_key, h2_header, faq_cands, target_count=current_faq_count)
                            if isinstance(faq_json, list) and len(faq_json) > 0 and "Вопрос" in faq_json[0]:
                                comm_items = [item for item in faq_json if "коммерч" in item.get("Тип", "").lower()]
                                info_items = [item for item in faq_json if "информац" in item.get("Тип", "").lower()]
                                
                                if 'faq_export_data' not in st.session_state:
                                    st.session_state.faq_export_data = []
                                    
                                # Функция для чистки маркдауна (конвертация в HTML)
                                def clean_faq_md(text):
                                    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', str(text))
                                    
                                for item in faq_json:
                                    st.session_state.faq_export_data.append({
                                        'Page URL': current_task['url'],
                                        'Product Name': h2_header,
                                        'Тип вопроса': item.get("Тип", ""),
                                        'Вопрос': clean_faq_md(item.get("Вопрос", "")),
                                        'Ответ': clean_faq_md(item.get("Ответ", ""))
                                    })
                                
                                faq_html_parts = ['<div class="faq-section">', f'<div class="h2"><h2>Частые вопросы по {h2_header}</h2></div>']
                                
                                # --- УМНЫЙ ФИЛЬТР МАРКДАУНА В HTML ---
                                def md_to_html(text):
                                    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', str(text))
                                
                                if comm_items:
                                    faq_html_parts.append('<div class="faq-category"><div class="h3"><h3>Коммерческие вопросы</h3></div>')
                                    for item in comm_items: 
                                        q_clean = md_to_html(item.get("Вопрос", ""))
                                        a_clean = md_to_html(item.get("Ответ", ""))
                                        faq_html_parts.append(f'<div class="faq-item"><div class="h4"><h4>{q_clean}</h4></div><p>{a_clean}</p></div>')
                                    faq_html_parts.append('</div>')
                                    
                                if info_items:
                                    faq_html_parts.append('<div class="faq-category"><div class="h3"><h3>Информационные вопросы</h3></div>')
                                    for item in info_items: 
                                        q_clean = md_to_html(item.get("Вопрос", ""))
                                        a_clean = md_to_html(item.get("Ответ", ""))
                                        faq_html_parts.append(f'<div class="faq-item"><div class="h4"><h4>{q_clean}</h4></div><p>{a_clean}</p></div>')
                                    faq_html_parts.append('</div>')
                                    
                                faq_html_parts.append('</div>')
                                final_faq_html = "\n".join(faq_html_parts)
                        except Exception as e:
                            status_logger.error(f"Ошибка FAQ: {e}")
                    # =================================================================
                    # ОБНОВЛЕННАЯ ГЕНЕРАЦИЯ С ОЦЕНКАМИ И МИЛЛИОННИКОМ
                    # =================================================================
                    global_reviews = st.session_state.get('safe_ws_global_reviews', True)
                    final_reviews_html = ""
                    
                    if global_reviews and gemini_api_key:
                        rev_count = st.session_state.get('safe_ws_reviews_count', 10)
                        status_logger.write(f"💬 Работаем над категорией: {h2_header}...")
                        
                        # 1. Получаем уникальных авторов (1 аноним + остальные)
                        chosen_authors = get_diverse_authors(rev_count)
                        
                        # 2. Генерируем отзывы
                        reviews_json = generate_reviews_deepseek(gemini_api_key, h2_header, review_cands, rev_count, chosen_authors)
                        
                       # --- ВОТ ЭТОТ БЛОК ВСТАВЛЯТЬ СРАЗУ ПОСЛЕ ВЫЗОВА generate_reviews_deepseek ---

                        # 1. СТИЛИ (Они делают отзывы "дорогими" на вид)
                        # --- 1. ПРИНУДИТЕЛЬНЫЕ СТИЛИ (ВСТАВЛЯТЬ СЮДА) ---
                        st.markdown("""
                        <style>
                            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
                            .stReviewContainer { font-family: 'Inter', sans-serif; background: #f1f5f9; padding: 20px; border-radius: 15px; }
                            .review-card {
                                background: #ffffff !important;
                                border: 1px solid #cbd5e1 !important;
                                border-left: 8px solid #3b82f6 !important;
                                border-radius: 12px !important;
                                padding: 25px !important;
                                margin-bottom: 20px !important;
                                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
                            }
                            .author-name { font-weight: 700; color: #0f172a; font-size: 1.15rem; }
                            .star-gold { color: #eab308; font-size: 1.3rem; }
                            .star-gray { color: #d1d5db; font-size: 1.3rem; }
                            .review-date { color: #64748b; font-size: 0.85rem; margin-top: 5px; }
                            .review-body { color: #334155; line-height: 1.6; margin-top: 15px; font-size: 1rem; }
                            .review-body strong { color: #1d4ed8; background: #dbeafe; padding: 0 3px; border-radius: 4px; }
                        </style>
                        """, unsafe_allow_html=True)

                        # --- 2. ЦИКЛ ОТРИСОВКИ ---
                        if isinstance(reviews_json, list) and len(reviews_json) > 0:
                            st.write(f"### 📋 Предпросмотр отзывов ({len(reviews_json)} шт.)")
                            
                            review_sources = ["Яндекс Карты", "Google Карты", "2ГИС", "Flamp", "Blizko"]
                            
                            for i, item in enumerate(reviews_json):
                                r_name = item.get('Имя', 'Клиент')
                                r_date = item.get('Дата', '')
                                r_rating = float(item.get('Оценка', 5.0))
                                r_text_raw = item.get('Текст', '')
                                r_source = random.choice(review_sources)
                                
                                # Превращаем **слово** в <b>слово</b> (перестраховка, если ИИ ошибся)
                                r_text_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', str(r_text_raw))
                                
                                # ГЕНЕРАЦИЯ ЗВЕЗД (HTML)
                                full_stars = int(r_rating)
                                has_half = 1 if r_rating % 1 != 0 else 0
                                stars_html = '<span class="star-gold">' + '★' * full_stars + '</span>'
                                if has_half: stars_html += '<span class="star-gold">½</span>'
                                stars_html += '<span class="star-gray">' + '☆' * (5 - full_stars - has_half) + '</span>'

                                # ВЫВОД КАРТОЧКИ В ПРОЦЕССЕ ГЕНЕРАЦИИ
                                st.markdown(f'''
                                    <div class="review-card">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <div>
                                                <div class="author-name">{r_name}</div>
                                                <div class="review-date">📅 {r_date} | 📍 {r_source}</div>
                                            </div>
                                            <div style="text-align: right;">
                                                <div class="star-gold">{stars_html}</div>
                                                <div style="font-weight: bold; color: #1e293b;">{r_rating}</div>
                                            </div>
                                        </div>
                                        <div class="review-body">{r_text_html}</div>
                                    </div>
                                ''', unsafe_allow_html=True)

                                # СОХРАНЕНИЕ ДЛЯ ЭКСПОРТА В EXCEL
                                if 'ws_reviews_export_data' not in st.session_state:
                                    st.session_state.ws_reviews_export_data = []
                                    
                                st.session_state.ws_reviews_export_data.append({
                                    'Page URL': current_task['url'],
                                    'Product Name': h2_header,
                                    'Имя': r_name,
                                    'Источник': r_source,
                                    'Оценка': r_rating,
                                    'Дата': r_date,
                                    'Отзыв': r_text_html
                                })
                        else:
                            st.error(f"Генерация для '{h2_header}' вернула пустой список или ошибку.")
                    # =================================================================
                    # ГЕО-ДОСТАВКА
                    # =================================================================
                    if curr_use_geo and client:
                        status_logger.write("🌍 Добавляем гео-доставку...")
                        try:
                            cities = ", ".join(cat_geo[:15])
                            prompt_geo = f"Напиши один HTML параграф (<p>) о доставке товара '{h2_header}' в города: {cities}. Выдай только HTML."
                            resp = client.chat.completions.create(model="google/gemini-2.5-pro", messages=[{"role": "user", "content": prompt_geo}], temperature=0.5)
                            row_data['IP_PROP4819'] = resp.choices[0].message.content.replace("```html", "").replace("```", "").strip()
                        except: pass

                    # =================================================================
                    # СБОРКА КОНТЕНТА В БЛОКИ (ЖЕСТКОЕ ПРАВИЛО: 1 ЭЛЕМЕНТ В 1 БЛОК)
                    # =================================================================
                    html_injections = {}
                    
                    if curr_use_tables and 'cl_tab' in locals():
                        html_injections['table'] = f'<div class="table-scroll-wrapper">\n{cl_tab}\n</div>'
                    
                    if global_tags and tags_block_1:
                        html_t1 =[f'<a href="{i["url"]}" class="tag-item">{i["name"]}</a>' for i in tags_block_1]
                        html_injections['tags1'] = f'<div class="popular-tags-text"><div class="popular-tags-inner-text"><div class="tag-items">{"\n".join(html_t1)}</div></div></div>'
                    
                    if global_promo and promo_block:
                        # ИСПРАВЛЕННАЯ HTML-СТРУКТУРА ПРОМО (картинки больше не растягиваются)
                        g_items =[
                            f'<div class="gallery-item">'
                            f'<figure class="gallery-img-wrap"><a href="{i["url"]}" target="_blank"><picture><img src="{i["img"]}" loading="lazy" alt="{i["name"]}"></picture></a></figure>'
                            f'<div class="gallery-title-wrap"><h3><a href="{i["url"]}" target="_blank">{i["name"]}</a></h3></div>'
                            f'</div>' 
                            for i in promo_block
                        ]
                        html_injections['promo'] = f'<div class="outer-full-width-section promo-block-section"><div class="gallery-content-wrapper"><h3 class="gallery-title">Рекомендуем</h3><div class="five-col-gallery">{"".join(g_items)}</div></div></div>'
                    
                    if global_tags and tags_block_2:
                        html_t2 = [f'<a href="{i["url"]}" class="tag-item">{i["name"]}</a>' for i in tags_block_2]
                        html_injections['tags2'] = f'<div class="popular-tags-text"><div class="popular-tags-inner-text"><div class="tag-items">{"\n".join(html_t2)}</div></div></div>'

                    # --- СТРОГАЯ ЛОГИКА РАСПРЕДЕЛЕНИЯ ---
                    # Собираем все доступные элементы в жесткую очередь (порядок вывода)
                    available_injections =[]
                    if 'tags1' in html_injections: available_injections.append(html_injections['tags1'])
                    if 'table' in html_injections: available_injections.append(html_injections['table'])
                    if 'promo' in html_injections: available_injections.append(html_injections['promo'])
                    if 'tags2' in html_injections: available_injections.append(html_injections['tags2'])

                    # Находим реальные текстовые блоки (которые нейросеть вернула не пустыми)
                    active_text_slots =[i for i, b in enumerate(blocks) if b.strip()]
                    
                    # Защита от "сплошного текста" без разделителей (если ИИ выдал все одним куском)
                    if not active_text_slots:
                        active_text_slots = [0]
                        blocks[0] = ""
                        
                    inj_idx = 0
                    for slot_idx in active_text_slots:
                        # 1-й абзац (Вводный) всегда оставляем ЧИСТЫМ для текста (если блоков больше 1)
                        if slot_idx == 0 and len(active_text_slots) > 1:
                            continue
                            
                        # Вшиваем ровно 1 элемент в конец текущего абзаца
                        if inj_idx < len(available_injections):
                            blocks[slot_idx] += "\n\n" + available_injections[inj_idx]
                            inj_idx += 1

                    # Если элементов больше, чем абзацев (например, 4 элемента, но нейросеть дала 2 блока),
                    # то оставшиеся аккуратно складываем списком в самый последний блок.
                    if active_text_slots:
                        last_slot = active_text_slots[-1]
                        while inj_idx < len(available_injections):
                            blocks[last_slot] += "\n\n" + available_injections[inj_idx]
                            inj_idx += 1

                    TEXT_CONTAINERS =['IP_PROP4839', 'IP_PROP4816', 'IP_PROP4838', 'IP_PROP4829', 'IP_PROP4831']
                    for i_c, c_name in enumerate(TEXT_CONTAINERS):
                        if i_c < len(blocks): row_data[c_name] = blocks[i_c]

                    # Собираем итоговую статью (Склеиваем Текст + FAQ + Отзывы)
                    merged_html = "".join(blocks)
                    if final_faq_html: merged_html += f"\n\n{final_faq_html}"
                    if final_reviews_html: merged_html += f"\n\n{final_reviews_html}"
                    
                    row_data['Весь текст целиком'] = merged_html
                    row_data['FAQ HTML'] = final_faq_html
                    
                    # =================================================================
                    # ПРОВЕРКИ АНТИСПАМ И TEXT.RU (С ИСПОЛЬЗОВАНИЕМ ЗАМОРОЖЕННЫХ КЛЮЧЕЙ)
                    # =================================================================
                    pure_text_for_check = BeautifulSoup(generated_full_text, "html.parser").get_text(separator=" ").strip()
                    row_data['DeepSeek Контекст'] = "-"; row_data['DeepSeek Комментарий'] = "-"
                    row_data['Риск Тургенев'] = "-"; row_data['Тургенев Комментарий'] = "-"
                    row_data['Уникальность'] = "-"; row_data['Text.ru Комментарий'] = "-"; row_data['Text.ru UID'] = None
                    
                    step_logger.warning("🔍 Этап 6: Отправляем на проверки (Антиспам и Уникальность)...")
                    
                    if use_ds_chk and gemini_api_key and pure_text_for_check:
                        try:
                            is_valid = validate_topic_deepseek(gemini_api_key, h1_marker, h2_header, pure_text_for_check)
                            row_data['DeepSeek Контекст'] = "YES" if is_valid else "NO"
                            row_data['DeepSeek Комментарий'] = "Ок" if is_valid else "Ошибка: не по теме"
                        except: row_data['DeepSeek Комментарий'] = "Сбой API"
                        
                    if use_turgenev_chk and turgenev_api_key and pure_text_for_check:
                        try:
                            turg_val = check_turgenev_sync(pure_text_for_check, turgenev_api_key)
                            row_data['Риск Тургенев'] = turg_val
                            t_num = float(re.search(r'\d+\.?\d*', str(turg_val)).group())
                            row_data['Тургенев Комментарий'] = "Ок" if t_num <= 5 else "Риск > 5 (Нужно править)"
                        except: row_data['Тургенев Комментарий'] = "Сбой API"
                            
                    if use_textru_chk and textru_api_key and pure_text_for_check:
                        try:
                            uid = send_textru_sync(pure_text_for_check, textru_api_key)
                            if uid:
                                row_data['Text.ru UID'] = uid; row_data['Уникальность'] = "⏳ Проверяется..."; row_data['Text.ru Комментарий'] = "В очереди"
                            else: row_data['Text.ru Комментарий'] = "Ошибка отправки"
                        except: row_data['Text.ru Комментарий'] = "Сбой API"

                    status_logger.update(label=f"✅ {h2_header} успешно сгенерирован!", state="complete", expanded=False)

                # ТЕПЕРЬ EXCEPT СВЯЗАН С TRY И ВСЕ РАБОТАЕТ!
                except Exception as e:
                    row_data['Весь текст целиком'] = f"❌ ОШИБКА ГЕНЕРАЦИИ: {e}"
                    status_logger.update(label=f"❌ Ошибка: {h2_header}", state="error", expanded=True)
                    status_logger.error(f"Сбой: {e}")
                    
                finally:
                    st.session_state.gen_result_df = pd.concat([st.session_state.gen_result_df, pd.DataFrame([row_data])], ignore_index=True)
                    st.session_state.auto_current_index += 1
                    st.session_state.last_stopped_index = st.session_state.auto_current_index
                    st.session_state.ws_waiting_for_analysis = False
                    
                    if st.session_state.auto_current_index < len(queue):
                        next_task = queue[st.session_state.auto_current_index]
                        p_source = "Релевантная страница на вашем сайте" if next_task.get('url') and next_task['url'] != 'manual' else "Без страницы"
                        st.session_state['pending_widget_updates'] = {
                            'query_input': next_task.get('h1', next_task['name']),
                            'my_page_source_radio': p_source,
                            'my_url_input': next_task.get('url', ''),
                            'competitor_source_radio': "Поиск через API Arsenkin (TOP-30)",
                            'settings_region': st.session_state.get('ws_settings_region', 'Москва')
                        }
                        st.session_state.start_analysis_flag = True
                        st.session_state.pop('analysis_done', None)
                        st.session_state.pop('analysis_results', None)
                        st.session_state.ws_waiting_for_analysis = True
                        st.rerun()
                    else:
                        st.session_state.ws_automode_active = False
                        st.rerun()

    # ==========================================
    # ИНТЕРФЕЙС И НАСТРОЙКИ
    # ==========================================
    with st.container(border=True):
        st.subheader("1. Режим работы и Валидация")
        gen_mode = st.radio("Тип страниц:", ["Подфильтровые (ссылки)", "Родительские (URL)", "Родительские (Вручную H1+H2)"], horizontal=True)
        
        default_reg_val = st.session_state.get('settings_region', 'Москва')
        try: def_index_ws = list(REGION_MAP.keys()).index(default_reg_val)
        except: def_index_ws = 0
        ws_region = st.selectbox("Регион для парсера:", list(REGION_MAP.keys()), index=def_index_ws, key="ws_settings_region")
        
        st.markdown("**Анализ контента:**")
        cv1, cv2, cv3 = st.columns(3)
        with cv1:
            if st.checkbox("📚 Риск Тургенева", key="use_turgenev_bulk"): 
                turg_val = st.text_input("🔑 API-ключ", value=st.session_state.get('TURGENEV_GLOBAL_KEY', ''), type="password", key="turg_key_widget")
                if turg_val: st.session_state['TURGENEV_GLOBAL_KEY'] = turg_val
        with cv2:
            if st.checkbox("🚀 Text.ru", key="use_textru_bulk"): 
                txtru_val = st.text_input("🔑 API-ключ", value=st.session_state.get('TEXTRU_GLOBAL_KEY', ''), type="password", key="txtru_key_widget")
                if txtru_val: st.session_state['TEXTRU_GLOBAL_KEY'] = txtru_val
        with cv3:
            st.checkbox("🧠 Валидация DeepSeek", key="use_ds_bulk", value=True)

    with st.container(border=True):
        st.subheader("2. Данные и Глобальные Разрешения")
        try: key_from_secrets = st.secrets["GEMINI_KEY"]
        except: key_from_secrets = ""
        
        # Железобетонное сохранение ключа в сессию
        ws_gem_key_input = st.text_input("🔑 Google Gemini API Key:", value=st.session_state.get('SUPER_GLOBAL_KEY', key_from_secrets), type="password", key="ws_gem_key_fixed")
        if ws_gem_key_input:
            st.session_state['SUPER_GLOBAL_KEY'] = ws_gem_key_input
        
    # Упаковываем всё в один контейнер, чтобы настройки не «разлетались»
        with st.container(border=True):
            col_left, col_right = st.columns([1.2, 1], gap="medium")

            with col_left:
                st.write("📝 **Ввод данных**")
                if "Подфильтровые" in gen_mode or "URL" in gen_mode:
                    raw_urls = st.text_area("Список ссылок:", height=215, placeholder="https://...", key="ws_area_urls")
                else:
                    # H1 и H2 теперь компактно в два столбика
                    h1_c, h2_c = st.columns(2)
                    raw_h1 = h1_c.text_area("H1 (Маркеры):", height=215, key="ws_area_h1")
                    raw_h2 = h2_c.text_area("H2 (Заголовки):", height=215, key="ws_area_h2")

            with col_right:
                st.write("⚙️ **Настройки генерации**")
                
                # Блок текста: всегда умный авто-расчет (3-5 блоков)
                st.checkbox("🤖 AI Тексты (Авто: 3-5 блоков)", value=True, key="ws_global_text")
                st.session_state.safe_ws_auto_blocks = True # Принудительно
                st.session_state.safe_ws_num_blocks_val = 5 # Технический максимум контейнеров
                
                st.write("---") # Разделитель
                
                # Остальные настройки в два столбика (ровная сетка)
                st.write("**Доп. элементы:**")
                grid_1, grid_2 = st.columns(2)
                with grid_1:
                    st.checkbox("🧩 Таблицы", value=True, key="ws_global_tables")
                    st.checkbox("🏷️ Теги", value=True, key="ws_global_tags")
                    st.checkbox("❓ FAQ", value=True, key="ws_global_faq", disabled=is_running)
                    # Увеличили лимит вопросов до 50
                    st.number_input("Количество вопросов FAQ", min_value=2, max_value=50, value=4, step=1, key="ws_faq_count", disabled=is_running)
                    
                with grid_2:
                    st.checkbox("🔥 Промо", value=True, key="ws_global_promo", disabled=is_running)
                    st.checkbox("🌍 Гео-блок", value=True, key="ws_global_geo", disabled=is_running)
                    
                    # Добавлена галочка и счетчик для отзывов
                    st.checkbox("💬 Отзывы", value=True, key="ws_global_reviews", disabled=is_running)
                    st.number_input("Количество отзывов", min_value=1, max_value=50, value=3, step=1, key="ws_reviews_count", disabled=is_running)
                
            
    c_start, c_stop = st.columns([2, 1])
    with c_start:
        is_running = st.session_state.get('ws_automode_active', False)
        if not is_running:
            if st.button("🚀 ЗАПУСТИТЬ АНАЛИЗ И ГЕНЕРАЦИЮ", type="primary", use_container_width=True):
                
                # --- ЗАМОРАЖИВАЕМ ВСЕ НАСТРОЙКИ, КЛЮЧИ И ГАЛОЧКИ (СПАСЕНИЕ ОТ СБРОСА) ---
                st.session_state.safe_ws_global_text = st.session_state.get('ws_global_text', True)
                st.session_state.safe_ws_global_tables = st.session_state.get('ws_global_tables', True)
                st.session_state.safe_ws_global_tags = st.session_state.get('ws_global_tags', True)
                st.session_state.safe_ws_global_promo = st.session_state.get('ws_global_promo', True)
                st.session_state.safe_ws_global_geo = st.session_state.get('ws_global_geo', True)
                st.session_state.safe_ws_global_faq = st.session_state.get('ws_global_faq', True)
                st.session_state.safe_ws_global_reviews = st.session_state.get('ws_global_reviews', True)
                
                st.session_state.safe_ws_faq_count = st.session_state.get('ws_faq_count', 4)
                st.session_state.safe_ws_reviews_count = st.session_state.get('ws_reviews_count', 3)
                st.session_state.safe_ws_num_blocks_val = st.session_state.get('ws_num_blocks_val', 5)
                st.session_state.safe_ws_auto_blocks = st.session_state.get('ws_auto_blocks', True)

                # ЗАМОРАЖИВАЕМ API-КЛЮЧИ И ГАЛОЧКИ
                st.session_state.safe_gemini_key = st.session_state.get('SUPER_GLOBAL_KEY', '')
                st.session_state.safe_turgenev_key = st.session_state.get('TURGENEV_GLOBAL_KEY', '')
                st.session_state.safe_textru_key = st.session_state.get('TEXTRU_GLOBAL_KEY', '')

                st.session_state.safe_use_turgenev = st.session_state.get('use_turgenev_bulk', False)
                st.session_state.safe_use_textru = st.session_state.get('use_textru_bulk', False)
                st.session_state.safe_use_ds = st.session_state.get('use_ds_bulk', True)
                # ----------------------------------------------------------------------
                queue =[]
                if "URL" in gen_mode or "Подфильтровые" in gen_mode:
                    urls = [u.strip() for u in raw_urls.split('\n') if u.strip()]
                    with st.spinner("Сбор данных со ссылок (ожидайте)..."):
                        for u in urls:
                            h1_s, h2_s, _ = scrape_h1_h2_from_url(u) if "URL" in gen_mode else ("", "", "")
                            b_text, _, _, _ = get_page_data_for_gen(u) if u else ("", "", "", "")
                            queue.append({'url': u, 'h1': h1_s or u.split('/')[-1], 'h2': h2_s or u.split('/')[-1], 'base_text': b_text, 'name': h1_s or u})
                else:
                    h1s =[x.strip() for x in raw_h1.split('\n') if x.strip()]
                    h2s =[x.strip() for x in raw_h2.split('\n') if x.strip()]
                    for h1, h2 in zip(h1s, h2s): queue.append({'url': 'manual', 'h1': h1, 'h2': h2, 'base_text': '', 'name': h1})
                
                if queue:
                    # === АКТИВИРУЕМ ЗАПУСК ТОЛЬКО ПОСЛЕ УСПЕШНОГО СБОРА ОЧЕРЕДИ ===
                    st.session_state.ws_bg_tasks_queue = queue
                    st.session_state.auto_current_index = 0
                    st.session_state.ws_automode_active = True
                    st.session_state.ws_waiting_for_analysis = True
                    st.session_state.start_analysis_flag = True
                    
                    first_task = queue[0]
                    f_source = "Релевантная страница на вашем сайте" if first_task.get('url') and first_task['url'] != 'manual' else "Без страницы"
                    st.session_state['pending_widget_updates'] = {
                        'query_input': first_task.get('h1', first_task['name']),
                        'my_page_source_radio': f_source,
                        'my_url_input': first_task.get('url', ''),
                        'competitor_source_radio': "Поиск через API Arsenkin (TOP-30)",
                        'settings_region': st.session_state.get('ws_settings_region', 'Москва')
                    }
                    st.session_state.pop('analysis_done', None)
                    st.session_state.pop('analysis_results', None)
                    st.rerun() # Теперь обновляем страницу, когда все данные подготовлены
                else:
                    st.error("❌ Очередь пуста! Проверьте введенные данные.")
        else:
            q_len = len(st.session_state.get('ws_bg_tasks_queue', []))
            curr = st.session_state.get('auto_current_index', 0)
            st.info(f"⏳ Конвейер в работе: Обработка {curr + 1} из {q_len} ... (Смотри предпросмотр внизу)")

    with c_stop:
        if is_running:
            if st.button("⛔ ОСТАНОВИТЬ КОНВЕЙЕР", type="secondary", use_container_width=True):
                st.session_state.ws_automode_active = False
                st.session_state.ws_waiting_for_analysis = False
                st.rerun()

    # --- ФОНОВЫЙ ОПРОС TEXT.RU И ПРЕДПРОСМОТР ---
    if not st.session_state.gen_result_df.empty:
        has_pending = any("⏳" in str(row.get('Уникальность', '')) for _, row in st.session_state.gen_result_df.iterrows())
        st.markdown("---")
        
        if has_pending and not is_running:
                st.warning("⚠️ Есть тексты в очереди Text.ru. Обновите статусы вручную:")
                if st.button("🔄 ОБНОВИТЬ СТАТУСЫ TEXT.RU", type="primary", use_container_width=True):
                    txtru_key_active = st.session_state.get('TEXTRU_GLOBAL_KEY', '')
                    with st.spinner("Стучимся в Text.ru..."):
                        for idx, row in st.session_state.gen_result_df.iterrows():
                            if "⏳" in str(row.get('Уникальность', '')):
                                uid = row.get('Text.ru UID')
                                if uid:
                                    stts = check_textru_status_sync(uid, txtru_key_active)
                                    if stts not in ["processing", "error"] and "Ошибка" not in stts:
                                        st.session_state.gen_result_df.at[idx, 'Уникальность'] = stts
                                        st.session_state.gen_result_df.at[idx, 'Text.ru UID'] = None
                                        try:
                                            u_num = float(re.search(r'\d+\.?\d*', str(stts)).group())
                                            st.session_state.gen_result_df.at[idx, 'Text.ru Комментарий'] = "Ок" if u_num >= 95 else "Уникальность < 95%"
                                        except:
                                            st.session_state.gen_result_df.at[idx, 'Text.ru Комментарий'] = "Проверено"
                    # st.rerun()  <--- ОТКЛЮЧЕНО
        elif has_pending and is_running:
            txtru_key_active = st.session_state.get('TEXTRU_GLOBAL_KEY', '')
            if txtru_key_active:
                updated_any = False
                for idx, row in st.session_state.gen_result_df.iterrows():
                    if "⏳" in str(row.get('Уникальность', '')):
                        uid = row.get('Text.ru UID')
                        if uid:
                            stts = check_textru_status_sync(uid, txtru_key_active)
                            if stts not in ["processing", "error"] and "Ошибка" not in stts:
                                st.session_state.gen_result_df.at[idx, 'Уникальность'] = stts
                                st.session_state.gen_result_df.at[idx, 'Text.ru UID'] = None
                                try:
                                    u_num = float(re.search(r'\d+\.?\d*', str(stts)).group())
                                    st.session_state.gen_result_df.at[idx, 'Text.ru Комментарий'] = "Ок" if u_num >= 95 else "Уникальность < 95%"
                                except:
                                    st.session_state.gen_result_df.at[idx, 'Text.ru Комментарий'] = "Проверено"
                                updated_any = True
                # if updated_any:   <--- ОТКЛЮЧЕНО
                #     st.rerun()    <--- ОТКЛЮЧЕНО

        elif has_pending and is_running:
            txtru_key_active = st.session_state.get('TEXTRU_GLOBAL_KEY', '') # <--- ИЗМЕНЕНА ЭТА СТРОКА
            if txtru_key_active:
                updated_any = False
                for idx, row in st.session_state.gen_result_df.iterrows():
                    if "⏳" in str(row.get('Уникальность', '')):
                        uid = row.get('Text.ru UID')
                        if uid:
                            stts = check_textru_status_sync(uid, txtru_key_active)
                            if stts not in ["processing", "error"] and "Ошибка" not in stts:
                                st.session_state.gen_result_df.at[idx, 'Уникальность'] = stts
                                st.session_state.gen_result_df.at[idx, 'Text.ru UID'] = None
                                try:
                                    u_num = float(re.search(r'\d+\.?\d*', str(stts)).group())
                                    st.session_state.gen_result_df.at[idx, 'Text.ru Комментарий'] = "Ок" if u_num >= 95 else "Уникальность < 95%"
                                except: st.session_state.gen_result_df.at[idx, 'Text.ru Комментарий'] = "Проверено"
                                updated_any = True
                if updated_any: st.rerun()

        # --- 1. ОПРЕДЕЛЕНИЕ ФУНКЦИИ ПОДСВЕТКИ (ИСПРАВЛЯЕТ NameError) ---
        def highlight_bad_results(row):
            styles = [''] * len(row)
            err_style = 'background-color: #ffe6e6; color: #cc0000; font-weight: bold;'
            col_idx = {name: i for i, name in enumerate(row.index)}
            
            # Подсветка DeepSeek (если БРАК)
            if str(row.get('DeepSeek Контекст')) == "NO" and 'DeepSeek Контекст' in col_idx:
                styles[col_idx['DeepSeek Контекст']] = err_style
            # Подсветка Тургенева
            try:
                t_val = str(row.get('Риск Тургенев', '0'))
                t_num = float(re.search(r'\d+\.?\d*', t_val).group())
                if t_num > 5 and 'Риск Тургенев' in col_idx:
                    styles[col_idx['Риск Тургенев']] = err_style
            except: pass
            # Подсветка Уникальности
            try:
                u_val = str(row.get('Уникальность', '100'))
                u_num = float(re.search(r'\d+\.?\d*', u_val).group())
                if u_num < 95 and 'Уникальность' in col_idx:
                    styles[col_idx['Уникальность']] = err_style
            except: pass
            # Подсветка Комментария (если не Ок)
            if str(row.get('Комментарий')) != "Ок" and 'Комментарий' in col_idx:
                styles[col_idx['Комментарий']] = err_style
            return styles

        # --- 2. ПОДГОТОВКА ТАБЛИЦЫ ДЛЯ ЭКСПОРТА ---
        df_export = st.session_state.gen_result_df.copy()
        
        def build_unified_comment(row):
            errs = []
            if str(row.get('DeepSeek Контекст')) == "NO": 
                errs.append("Текст должен быть строго по теме")
            try:
                t_val = str(row.get('Риск Тургенев', '0'))
                t_num = float(re.search(r'\d+\.?\d*', t_val).group())
                if t_num > 5: errs.append("Риск Тургенева должен быть не более 5")
            except: pass
            try:
                u_val = str(row.get('Уникальность', '100'))
                u_num = float(re.search(r'\d+\.?\d*', u_val).group())
                if u_num < 95: errs.append("Уникальность от 95%")
            except: pass
            return " | ".join(errs) if errs else "Ок"

        df_export['Комментарий'] = df_export.apply(build_unified_comment, axis=1)
        
        # Очистка мусора для Excel (Убираем пустые колонки FAQ, чтобы они не мешались в основной таблице)
        cols_to_drop_excel = [
            'Text.ru UID', 'FAQ HTML', 'DeepSeek Комментарий', 'Тургенев Комментарий', 'Text.ru Комментарий',
            'FAQ Коммерческий вопрос', 'FAQ Коммерческий ответ', 'FAQ Информационный вопрос', 'FAQ Информационный ответ'
        ]
        df_export_clean = df_export.drop(columns=[c for c in cols_to_drop_excel if c in df_export.columns], errors='ignore')

        # --- 3. ПОСТОЯННОЕ РЕШЕНИЕ: АВТО-ОПРОС TEXT.RU (БЕЗ КНОПКИ) ---
        has_pending_uids = False
        if 'Text.ru UID' in st.session_state.gen_result_df.columns:
            has_pending_uids = any("⏳" in str(row.get('Уникальность', '')) for _, row in st.session_state.gen_result_df.iterrows())

        # Если генератор НЕ работает, но есть тексты в очереди - опрашиваем API автоматически
        if has_pending_uids and not st.session_state.get('ws_automode_active', False):
            txtru_key_active = st.session_state.get('TEXTRU_GLOBAL_KEY', '')
            if txtru_key_active:
                st.info("🔄 Автоматическое обновление статусов Text.ru...")
                updated_any = False
                for idx, row in st.session_state.gen_result_df.iterrows():
                    if "⏳" in str(row.get('Уникальность', '')):
                        uid = row.get('Text.ru UID')
                        if uid:
                            stts = check_textru_status_sync(uid, txtru_key_active)
                            if stts not in ["processing", "error"] and "Ошибка" not in stts:
                                st.session_state.gen_result_df.at[idx, 'Уникальность'] = stts
                                st.session_state.gen_result_df.at[idx, 'Text.ru UID'] = None
                                updated_any = True
                if updated_any:
                    st.rerun()

        # --- 4. КНОПКИ СКАЧИВАНИЯ ---
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # ЛИСТ 1: Основные результаты генерации
            df_export_clean.to_excel(writer, sheet_name='Результаты', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Результаты']
            # ... (оставьте тут ваши старые форматы)
            
            # ЛИСТ 2: Выгрузка базы FAQ на отдельный лист
            if 'faq_export_data' in st.session_state and st.session_state.faq_export_data:
                df_faq = pd.DataFrame(st.session_state.faq_export_data)
                df_faq.to_excel(writer, sheet_name='База FAQ', index=False)
                
            # ЛИСТ 3: Выгрузка Отзывов на отдельный лист (НОВОЕ)
            if 'ws_reviews_export_data' in st.session_state and st.session_state.ws_reviews_export_data:
                df_rev_export = pd.DataFrame(st.session_state.ws_reviews_export_data)
                df_rev_export.to_excel(writer, sheet_name='Отзывы', index=False)
                w_rev = writer.sheets['Отзывы']
                w_rev.set_column('A:B', 30)
                w_rev.set_column('C:D', 18)  # Имя и Источник
                w_rev.set_column('E:F', 12)  # Оценка и Дата
                w_rev.set_column('G:G', 80)  # Отзыв

        col_dl, col_cl = st.columns([2, 1])
        with col_dl:
            st.download_button(
                label=f"📥 СКАЧАТЬ EXCEL ({len(df_export_clean)} шт.)",
                data=buffer.getvalue(),
                file_name=f"wholesale_texts_{int(time.time())}.xlsx",
                mime="application/vnd.ms-excel",
                use_container_width=True
            )
        with col_cl:
            if st.button("🗑️ Очистить таблицу", use_container_width=True):
                st.session_state.gen_result_df = st.session_state.gen_result_df.iloc[0:0]
                if 'faq_export_data' in st.session_state:
                    st.session_state.faq_export_data = []
                if 'ws_reviews_export_data' in st.session_state:
                    st.session_state.ws_reviews_export_data = []
                st.rerun()

        # --- 5. ТЕХНИЧЕСКАЯ ТАБЛИЦА (ТЕПЕРЬ БЕЗ ОШИБОК) ---
        with st.expander("👀 Техническая таблица результатов", expanded=False):
            # Используем df_export_clean, из которого уже вырезаны пустые FAQ и лишние системные колонки
            st.dataframe(df_export_clean.style.apply(highlight_bad_results, axis=1), use_container_width=True)

        st.markdown("---")

        # --- 6. ВИЗУАЛЬНЫЙ ПРЕДПРОСМОТР (ВКЛАДКИ) ---
        st.markdown("### 🖥️ Визуальный предпросмотр")
        
        if not df_export.empty and 'Product Name' in df_export.columns:
            all_products = df_export['Product Name'].tolist()
            sel_p = st.selectbox("Выберите товар:", all_products, index=len(all_products)-1, key="ws_preview_final_sel")
            
            if sel_p:
                row_p = df_export[df_export['Product Name'] == sel_p].iloc[0]
                
                # Поля для визуала (БЕЗ ГЕО-блока IP_PROP4819)
                visual_fields = ['IP_PROP4839', 'IP_PROP4816', 'IP_PROP4838', 'IP_PROP4829', 'IP_PROP4831']
                active_visual = [c for c in visual_fields if str(row_p.get(c, "")).strip() != ""]
                
                # Стили CSS (переиспользованы из твоего кода)
                st.markdown("""
                    <style>
                        .preview-box { border: 1px solid #e2e8f0; background-color: #ffffff; padding: 20px; border-radius: 8px; margin-bottom: 25px; box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06); }
                        .block-title { color: #277EFF; margin-top: 30px; margin-bottom: 10px; font-size: 1.2em; font-weight: 600; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; }
                        .table-scroll-wrapper { width: 100%; overflow-x: auto; margin: 20px 0; }
                        .brand-accent-table { width: 100%; border-collapse: collapse; text-align: left; font-family: sans-serif; }
                        .brand-accent-table th { background-color: #277EFF; color: white; padding: 12px; font-weight: 500; border: none; }
                        .brand-accent-table td { padding: 12px; border-bottom: 1px solid #eee; color: #333; }
                        .popular-tags-text { margin: 20px 0; }
                        .tag-item { display: inline-block; padding: 6px 12px; margin: 4px; background: #f0f4f8; border-radius: 4px; text-decoration: none; color: #277EFF; font-size: 14px; }
                        .gallery-content-wrapper { background: #F6F7FC; padding: 20px; border-radius: 10px; margin: 20px 0; }
                        .promo-block-section { margin: 30px 0; }
                        .five-col-gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }
                        .gallery-item { display: flex; flex-direction: column; justify-content: space-between; background: white; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
                        .gallery-img-wrap { height: 160px; display: flex; align-items: center; justify-content: center; overflow: hidden; margin: 0 0 15px 0; }
                        .gallery-img-wrap img { max-width: 100%; max-height: 100%; object-fit: contain; }
                        .gallery-title-wrap h3 { font-size: 14px; margin: 0; line-height: 1.3; font-weight: 500; }
                        .gallery-title-wrap a { text-decoration: none; color: #277EFF; }
                        .gallery-title-wrap a:hover { text-decoration: underline; }
                        .gallery-item h3 { font-size: 14px; margin-top: 10px; font-weight: normal; }
                        .gallery-item a { text-decoration: none; color: #333; }
                        .faq-section { margin: 20px 0; padding: 20px; background: #F6F7FC; border-radius: 8px; border: 1px solid #e2e8f0; }
                    </style>
                """, unsafe_allow_html=True)
                
                tabs_v = st.tabs(["📝 Текст и блоки", "❓ FAQ", "💬 Отзывы"])
                with tabs_v[0]:
                    if active_visual:
                        for col in active_visual:
                            b_name = "ГЛАВНЫЙ ТЕКСТ" if col == "IP_PROP4839" else col.replace("IP_PROP", "БЛОК ")
                            st.markdown(f"<div class='block-title'>{b_name}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='preview-box'>{str(row_p[col])}</div>", unsafe_allow_html=True)
                    else:
                        st.info("Текстовые блоки отсутствуют.")
                
                with tabs_v[1]:
                    # ВЫВОДИМ FAQ HTML
                    f_html = str(row_p.get('FAQ HTML', '')).strip()
                    if f_html and f_html != "None":
                        st.markdown(f"<div class='faq-section'>{f_html}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("FAQ для этого товара не генерировался.")
                with tabs_v[2]:
                    st.markdown("### Сгенерированные отзывы:")
                    if 'ws_reviews_export_data' in st.session_state and st.session_state.ws_reviews_export_data:
                        current_reviews = [r for r in st.session_state.ws_reviews_export_data if r['Product Name'] == sel_p]
                        if current_reviews:
                            for rev in current_reviews:
                                r_rating = rev.get('Оценка', 5.0)
                                full_stars = int(r_rating)
                                has_half = 1 if r_rating % 1 != 0 else 0
                                stars_html = '<span style="color: #eab308; font-size: 1.2rem;">' + '★' * full_stars + '</span>'
                                if has_half: stars_html += '<span style="color: #eab308; font-size: 1.2rem;">½</span>'
                                stars_html += '<span style="color: #d1d5db; font-size: 1.2rem;">' + '☆' * (5 - full_stars - has_half) + '</span>'

                                # Отрисовка Имени, Источника, Звезд и Оценки
                                st.markdown(f"**{rev.get('Имя', 'Аноним')}** 🗓️ *{rev.get('Дата', '')}* | 📍 *{rev.get('Источник', 'Яндекс Карты')}* | {stars_html} **{r_rating}**", unsafe_allow_html=True)
                                st.markdown(f"<div style='padding:10px; border-left: 3px solid #ccc; margin-bottom: 20px;'>{rev.get('Отзыв', '')}</div>", unsafe_allow_html=True)
                        else:
                            st.info("Для этого товара нет отзывов.")
                    else:
                        st.info("База отзывов пуста. Убедитесь, что галочка генерации отзывов была включена.")
# ==========================================

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

# ==========================================
# МОНИТОРИНГ: ЧИСТАЯ ВЕРСИЯ (БЕЗ МУСОРА)
# ==========================================
import os
import pandas as pd
import datetime
import time
import requests
import json
from urllib.parse import urlparse

TRACK_FILE = "monitoring.csv"

def add_to_tracking(url, keyword):
    if not os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "w", encoding="utf-8") as f:
            f.write("URL;Keyword;Date;Position\n")
    try:
        existing = pd.read_csv(TRACK_FILE, sep=";")
        if ((existing['URL'] == url) & (existing['Keyword'] == keyword)).any(): return
    except: pass
    with open(TRACK_FILE, "a", encoding="utf-8") as f:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        f.write(f"{url};{keyword};{today};0\n")

# Функция нормализации (убирает www и http для сравнения)
def normalize_url(u):
    if not u: return ""
    u = str(u).lower().strip()
    u = u.replace("https://", "").replace("http://", "").replace("www.", "")
    if u.endswith("/"): u = u[:-1]
    return u

with tab_monitoring:
    st.header("📉 Трекер позиций (DEBUG MODE)")

    # Выбор региона
    default_reg_val = st.session_state.get('settings_region', 'Москва')
    try: def_index = list(REGION_MAP.keys()).index(default_reg_val)
    except: def_index = 0

    col_reg, col_btn, col_del = st.columns([2, 2, 1])
    
    with col_reg:
        selected_mon_region = st.selectbox("Регион:", list(REGION_MAP.keys()), index=def_index, label_visibility="collapsed")

    # Форма добавления
    with st.expander("➕ Добавить запрос вручную", expanded=False):
        with st.form("add_clean_manual"):
            col_u, col_k = st.columns(2)
            u_in = col_u.text_input("URL страницы/сайта")
            k_in = col_k.text_input("Ключевое слово")
            if st.form_submit_button("Добавить в список"):
                if u_in and k_in:
                    add_to_tracking(u_in, k_in)
                    st.success(f"Добавлено: {k_in}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Заполните оба поля")

    if not os.path.exists(TRACK_FILE):
        st.info("Список пуст.")
    else:
        try: df_mon = pd.read_csv(TRACK_FILE, sep=";")
        except: df_mon = pd.DataFrame()

        if df_mon.empty:
            st.info("Файл базы пуст.")
        else:
            with col_btn:
                if st.button("🚀 ОБНОВИТЬ ПОЗИЦИИ", type="primary", use_container_width=True):
                    if not ARSENKIN_TOKEN:
                        st.error("❌ ОШИБКА: Нет токена!")
                    else:
                        status_container = st.status("🚀 Начинаем...", expanded=True)
                        progress_bar = status_container.progress(0)
                        
                        reg_ids = REGION_MAP.get(selected_mon_region, {"ya": 213})
                        rid_int = int(reg_ids['ya'])
                        
                        total_rows = len(df_mon)

                        for i, row in df_mon.iterrows():
                            kw = str(row['Keyword']).strip()
                            target_url_raw = str(row['URL']).strip()
                            
                            # === 1. ВЫДЕЛЯЕМ ЧИСТЫЙ ДОМЕН ДЛЯ API ===
                            # API в поле "url" хочет "site.ru", а не "site.ru/page"
                            parsed_url = urlparse(target_url_raw)
                            clean_domain = parsed_url.netloc.replace("www.", "")
                            if not clean_domain: clean_domain = target_url_raw.split('/')[0]

                            status_container.write(f"📡 Запрос: **{kw}** (Домен: {clean_domain})...")

                            payload = {
                                "tools_name": "positions",
                                "data": {
                                    "queries": [kw],
                                    "url": clean_domain, # <--- ОТПРАВЛЯЕМ ТОЛЬКО ДОМЕН
                                    "subdomain": True,
                                    "se": [{"type": 2, "region": rid_int}],
                                    "format": 0
                                }
                            }
                            
                            try:
                                # SET
                                r_set = requests.post("https://arsenkin.ru/api/tools/set", headers={"Authorization": f"Bearer {ARSENKIN_TOKEN}"}, json=payload, timeout=200)
                                if r_set.status_code != 200:
                                    st.error(f"HTTP Error: {r_set.status_code}")
                                    continue
                                
                                tid = r_set.json().get("task_id")
                                if not tid: 
                                    st.error(f"No Task ID: {r_set.json()}")
                                    continue

                                # CHECK
                                for _ in range(15):
                                    time.sleep(1.5)
                                    r_c = requests.post("https://arsenkin.ru/api/tools/check", headers={"Authorization": f"Bearer {ARSENKIN_TOKEN}"}, json={"task_id": tid})
                                    if r_c.json().get("status") == "finish": break
                                
                                # GET
                                r_get = requests.post("https://arsenkin.ru/api/tools/get", headers={"Authorization": f"Bearer {ARSENKIN_TOKEN}"}, json={"task_id": tid})
                                final_data = r_get.json()

                                # === 🔍 ОТЛАДКА: ВЫВОДИМ JSON НА ЭКРАН ===
                                # Если здесь 0, посмотри, что внутри JSON!
                                with status_container:
                                    st.write(f"📝 Ответ сервера для '{kw}':")
                                    st.json(final_data) 
                                
                                # ПАРСИНГ
                                res_data = final_data.get("result", [])
                                found_pos_val = 0
                                
                                if res_data and isinstance(res_data, list):
                                    item = res_data[0]
                                    
                                    # Ищем позицию по всем возможным ключам
                                    keys_to_check = ["position", "pos", str(rid_int)]
                                    
                                    for key in keys_to_check:
                                        val = item.get(key)
                                        if val is not None:
                                            # Арсенкин может вернуть число 11 или строку "11"
                                            if str(val).isdigit():
                                                found_pos_val = int(val)
                                                break
                                            # Или вернуть "-" если не в топе
                                            if str(val) in ["-", "0"]:
                                                found_pos_val = 0
                                                break
                                
                                df_mon.at[i, 'Position'] = found_pos_val
                                df_mon.at[i, 'Date'] = datetime.datetime.now().strftime("%Y-%m-%d")
                                df_mon.to_csv(TRACK_FILE, sep=";", index=False)
                                
                            except Exception as e:
                                st.error(f"Crash: {e}")
                            
                            progress_bar.progress((i + 1) / total_rows)

                        status_container.update(label="✅ Готово!", state="complete", expanded=False)
                        st.rerun()

            # Таблица
            def style_pos(v):
                try:
                    i = int(v)
                    if 0 < i <= 10: return 'color: #16a34a; font-weight: bold' 
                    if 10 < i <= 30: return 'color: #ca8a04' 
                    if i == 0: return 'color: #dc2626' 
                except: pass
                return ''

            st.dataframe(
                df_mon.style.map(style_pos, subset=['Position']),
                use_container_width=True,
                height=500,
                column_config={
                    "URL": st.column_config.LinkColumn("Ссылка"),
                    "Position": st.column_config.NumberColumn("Позиция", format="%d"),
                    "Keyword": "Ключ",
                    "Date": "Дата"
                }
            )
            
            with col_del:
                if st.button("🗑️", help="Удалить базу"):
                    os.remove(TRACK_FILE); st.rerun()

# ==========================================
# TAB 5: LSI GENERATOR (FULL CYCLE + FIXES)
# ==========================================
with tab_lsi_gen:
    st.header("🏭 Массовая генерация B2B (Full Technical Mode)")
    st.markdown("Авто-цикл: **H1 (Маркер) -> SEO Анализ (фон) -> LSI -> Генерация текста под H2**.")

    # --- ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ ---
    if 'bg_tasks_queue' not in st.session_state: st.session_state.bg_tasks_queue = []
    if 'bg_results' not in st.session_state: st.session_state.bg_results = []
    if 'bg_is_running' not in st.session_state: st.session_state.bg_is_running = False
    if 'bg_batch_size' not in st.session_state: st.session_state.bg_batch_size = 3

# --- 1. НАСТРОЙКИ (ИСПРАВЛЕННОЕ СОХРАНЕНИЕ КЛЮЧА) ---
    with st.expander("⚙️ Настройки API и LSI", expanded=True):
        
# === ЖЕЛЕЗОБЕТОННОЕ СОХРАНЕНИЕ КЛЮЧА ===
        # 1. Создаем переменную, которая НЕ зависит от виджета
        if 'FINAL_GEMINI_KEY' not in st.session_state:
            st.session_state.FINAL_GEMINI_KEY = ""
            
        # 2. Пытаемся найти ключ в секретах или в старых переменных
        if not st.session_state.FINAL_GEMINI_KEY:
            try: st.session_state.FINAL_GEMINI_KEY = st.secrets["GEMINI_KEY"]
            except: pass
            
        # 3. Функция обновления при вводе
        def update_final_key():
            st.session_state.FINAL_GEMINI_KEY = st.session_state.bulk_api_key_v3

        default_lsi_text = "гарантия, звоните, консультация, купить, оплата, оптом, отгрузка, под заказ, поставка, прайс-лист, цены"

        c1, c2 = st.columns([1, 2])
        with c1:
            # 1. Достаем сохраненный ключ из глобальной памяти
            saved_key = st.session_state.get('SUPER_GLOBAL_KEY', '')
            # 2. Отрисовываем поле ввода со своим уникальным системным ключом
            current_input = st.text_input(
                "🔑 Введите API-ключ Gemini:", 
                value=saved_key, 
                type="password", 
                key="tab5_api_key_widget" # Уникальный ID, не пересекается с вкладкой 6
            )
            # 3. СРАЗУ ЖЕ (сверху вниз, до кнопки старта!) пересохраняем в глобалку
            if current_input: st.session_state['SUPER_GLOBAL_KEY'] = current_input
            
            # --- НОВОЕ: ВЫБОР РЕГИОНА ПОИСКА ДЛЯ LSI ---
            st.selectbox("Регион поиска", list(REGION_MAP.keys()), key="lsi_settings_region")

        with c2:
            # === ПОЛЕ ДЛЯ ОБЩИХ LSI ===
            st.session_state['common_lsi_input'] = st.text_area(
                "Общие LSI-слова (добавляются ко всем статьям):", 
                value=st.session_state.get('common_lsi_input', default_lsi_text),
                help="Укажите слова через запятую. Они будут объединены с 15 словами из парсинга."
            )
        # --- НОВЫЕ ЧЕК-БОКСЫ ---
        st.markdown("### Дополнительные проверки (после генерации)")
        c_chk1, c_chk2, c_chk3 = st.columns(3)
        with c_chk1:
            st.session_state['use_turgenev'] = st.checkbox("📚 Проверка по Тургеневу", value=st.session_state.get('use_turgenev', False))
            if st.session_state['use_turgenev']:
                st.session_state['turgenev_api_key'] = st.text_input("🔑 API-ключ Тургенева", value=st.session_state.get('turgenev_api_key', ''), type="password")
        with c_chk2:
            st.session_state['use_textru'] = st.checkbox("🚀 Проверка Text.ru", value=st.session_state.get('use_textru', False))
            if st.session_state['use_textru']:
                st.session_state['textru_api_key'] = st.text_input("🔑 API-ключ Text.ru", value=st.session_state.get('textru_api_key', ''), type="password")
        with c_chk3:
            st.session_state['use_deepseek_check'] = st.checkbox("🧠 Проверка темы (DeepSeek)", value=st.session_state.get('use_deepseek_check', True), help="Использует ваш текущий API-ключ от Gemini")

    # --- 2. ЗАГРУЗКА ЗАДАЧ ---
    st.subheader("1. Загрузка задач")
    
    # Выбор режима
    load_mode = st.radio(
        "Источник данных:", 
        ["📝 Вручную (Списки H1 и H2)", "🔗 Список ссылок (Авто-парсинг)"], 
        horizontal=True
    )
    
    # 2.1 РУЧНОЙ ВВОД
    if "Вручную" in load_mode:
        st.info("Введите списки. Строка 1 в левом поле соответствует Строке 1 в правом.")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            # ДОБАВИЛ KEY
            raw_h1_input = st.text_area(
                "Список H1 (МАРКЕР ДЛЯ АНАЛИЗА)", 
                height=200, 
                placeholder="Труба стальная\nЛист оцинкованный",
                key="manual_h1_input" 
            )
        with col_h2:
            # ДОБАВИЛ KEY
            raw_h2_input = st.text_area(
                "Список H2 (ЗАГОЛОВОК СТАТЬИ)", 
                height=200, 
                placeholder="Технические характеристики трубы\nПреимущества оцинкованного листа",
                key="manual_h2_input"
            )
        raw_urls_input = None

    # 2.2 ПАРСИНГ ССЫЛОК
    else:
        st.info("Скрипт зайдет на каждую ссылку, найдет там H1 (станет маркером) и H2 (станет заголовком).")
        raw_urls_input = st.text_area("Список ссылок (каждая с новой строки)", height=200, placeholder="https://site.ru/catalog/tovar1\nhttps://site.ru/catalog/tovar2", key="url_list_input")
        raw_h1_input = None; raw_h2_input = None

    # КНОПКА ЗАГРУЗКИ В ОЧЕРЕДЬ
    if st.button("📥 Загрузить задачи в очередь", use_container_width=True):
        st.session_state.bg_tasks_queue = [] # Очищаем старую очередь при новой загрузке
        st.session_state.bg_results = []
        st.session_state.bg_is_running = False
        
        # ЛОГИКА ЗАГРУЗКИ (РУЧНАЯ)
        if "Вручную" in load_mode:
            lines_h1 = [l.strip() for l in raw_h1_input.split('\n') if l.strip()]
            lines_h2 = [l.strip() for l in raw_h2_input.split('\n') if l.strip()]
            
            if len(lines_h1) != len(lines_h2):
                st.error(f"❌ Ошибка: Несовпадение строк! H1: {len(lines_h1)}, H2: {len(lines_h2)}")
            elif not lines_h1:
                st.error("❌ Списки пусты!")
            else:
                for h1, h2 in zip(lines_h1, lines_h2):
                    st.session_state.bg_tasks_queue.append({
                        'h1': h1,
                        'h2': h2,
                        'source_url': 'Manual',
                        'lsi_added': []
                    })
                st.success(f"✅ Загружено задач вручную: {len(lines_h1)}")
                time.sleep(1)
                st.rerun()

        # ЛОГИКА ЗАГРУЗКИ (ССЫЛКИ)
        else:
            urls_list = [u.strip() for u in raw_urls_input.split('\n') if u.strip()]
            if not urls_list:
                st.error("❌ Список ссылок пуст!")
            else:
                progress_bar = st.progress(0)
                status_box = st.status("🔗 Парсинг ссылок...", expanded=True)
                valid_count = 0
                for i, url in enumerate(urls_list):
                    status_box.write(f"Сканирую: {url}...")
                    h1_found, h2_found, err = scrape_h1_h2_from_url(url)
                    if h1_found:
                        st.session_state.bg_tasks_queue.append({
                            'h1': h1_found,
                            'h2': h2_found,
                            'source_url': url,
                            'lsi_added': []
                        })
                        valid_count += 1
                    else:
                        status_box.warning(f"⚠️ Сбой {url}: {err}")
                    progress_bar.progress((i + 1) / len(urls_list))
                
                status_box.update(label=f"✅ Готово! Добавлено: {valid_count}", state="complete")
                time.sleep(1)
                st.rerun()

    # --- 3. УПРАВЛЕНИЕ ПРОЦЕССОМ ---
    
    total_q = len(st.session_state.bg_tasks_queue)
    # Определяем готовые по уникальной паре
    finished_ids = set(f"{r['h1']}|{r['h2']}" for r in st.session_state.bg_results)
    
    pending_indices = []
    for i, t in enumerate(st.session_state.bg_tasks_queue):
        unique_id = f"{t['h1']}|{t['h2']}"
        if unique_id not in finished_ids:
            pending_indices.append(i)
            
    remaining_q = len(pending_indices)
    completed_q = total_q - remaining_q

    if total_q > 0:
        st.divider()
        st.subheader(f"2. Генерация (Готово: {completed_q} | Осталось: {remaining_q})")
        
        # НАСТРОЙКИ ПАЧКИ (ВЕРНУЛ ОБРАТНО)
        c_set1, c_set2 = st.columns([1, 3])
        with c_set1:
            st.session_state.bg_batch_size = st.number_input("Размер пачки (шт)", 1, 10, st.session_state.bg_batch_size)
        with c_set2:
            st.info("⚠️ Большой размер пачки может вызвать тайм-аут. Рекомендуется 2-3.")

# --- ФУНКЦИЯ-ОБРАБОТЧИК (CALLBACK) ---
        # Она выполнится ДО перезагрузки страницы, поэтому ошибки не будет
        def start_automode_callback(indices_list):
            st.session_state.lsi_automode_active = True
            if indices_list:
                idx = indices_list[0]
                task = st.session_state.bg_tasks_queue[idx]
                
                # Для LSI текстов всегда режим "Без страницы"
                st.session_state['pending_widget_updates'] = {
                    'query_input': task['h1'],
                    'my_page_source_radio': "Без страницы",
                    'my_url_input': "",
                    'competitor_source_radio': "Поиск через API Arsenkin (TOP-30)",
                    'settings_region': st.session_state.get('lsi_settings_region', 'Москва') # <--- ПЕРЕДАЕМ РЕГИОН
                }
                
                st.session_state.lsi_processing_task_id = idx
                st.session_state.start_analysis_flag = True
                st.session_state.pop('analysis_results', None)
                st.session_state.pop('analysis_done', None)
        # -------------------------------------
        c_act1, c_act2, c_act3 = st.columns([1, 1, 1])
        with c_act1:
            if not st.session_state.get('lsi_automode_active'):
                btn_label = "▶️ СТАРТ ЧЕРЕЗ ВКЛАДКУ 1" if remaining_q > 0 else "✅ ВСЕ ГОТОВО"
                lsi_api_key = st.session_state.get('SUPER_GLOBAL_KEY')
                keys_valid = bool(lsi_api_key and ARSENKIN_TOKEN)
                
                if st.button(btn_label, type="primary", disabled=(remaining_q == 0), 
                             use_container_width=True,
                             on_click=start_automode_callback if keys_valid else None,
                             args=(pending_indices,) if keys_valid else None):
                    
                    if not keys_valid:
                        if not lsi_api_key: st.error("Введите API ключ Gemini!")
                        if not ARSENKIN_TOKEN: st.error("Нужен токен Arsenkin!")
                    else:
                        st.toast("🚀 Запуск... Переход на Вкладку 1")
            else:
                # === ИЗМЕНИТЬ ВОТ ЭТОТ БЛОК ===
                st.button("⛔ ОСТАНОВИТЬ ГЕНЕРАЦИЮ", type="secondary", use_container_width=True, on_click=global_stop_callback)

        with c_act3:
            # Кнопка сброса
            if st.button("🗑️ Сброс очереди", disabled=st.session_state.get('lsi_automode_active', False), use_container_width=True):
                st.session_state.bg_tasks_queue = []
                st.session_state.bg_results = []
                st.session_state.lsi_automode_active = False
                keys_to_del = ["manual_h1_input", "manual_h2_input", "url_list_input"]
                for k in keys_to_del:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()

        # ==================================================================
        # 🔥 HOOK ДЛЯ LSI ГЕНЕРАТОРА (ВКЛАДКА 5) - ЧИСТЫЙ БЛОК
        # ==================================================================
        if st.session_state.get('lsi_automode_active'):
            # --- ФОНОВАЯ ПРОВЕРКА TEXT.RU (обновление при каждой итерации) ---
            if st.session_state.get('use_textru') and st.session_state.get('textru_api_key'):
                textru_key = st.session_state.get('textru_api_key')
                for rec in st.session_state.get('bg_results', []):
                    if rec.get('textru_uid') and "⏳" in str(rec.get('textru')):
                        t_stat = check_textru_status_sync(rec['textru_uid'], textru_key)
                        if t_stat == "processing":
                            rec['textru'] = "⏳ Проверяется..."
                        elif "Ошибка" in t_stat or t_stat == "error":
                            rec['textru'] = t_stat
                            rec['textru_uid'] = None
                        else:
                            rec['textru'] = t_stat
                            rec['textru_uid'] = None
                            
            current_idx = st.session_state.get('lsi_processing_task_id')
            
            if 'bg_tasks_queue' not in st.session_state or current_idx is None or current_idx >= len(st.session_state.bg_tasks_queue):
                st.session_state.lsi_automode_active = False
                st.success("Очередь пуста или завершена.")
                st.stop()

            task = st.session_state.bg_tasks_queue[current_idx]
            
            lsi_words = []
            results_data = st.session_state.get('analysis_results')
            if results_data and results_data.get('hybrid') is not None and not results_data['hybrid'].empty:
                lsi_words = results_data['hybrid'].head(15)['Слово'].tolist()
            
# Читаем общие LSI из поля ввода и объединяем с парсингом
            raw_common = st.session_state.get('common_lsi_input', "гарантия, звоните, консультация, купить, оплата, оптом, отгрузка, под заказ, поставка, прайс-лист, предлагаем, рассчитать, цены")
            common_lsi = [w.strip() for w in raw_common.split(",") if w.strip()]
            combined_lsi = list(set(common_lsi + lsi_words))

# 4. ГЕНЕРИРУЕМ СТАТЬЮ
            api_key_gen = st.session_state.get('SUPER_GLOBAL_KEY')
            html_out = ""
            status_code = "Error"
            
            # Переменные для результатов
            turgenev_res = "-"
            textru_res = "-"
            textru_uid = None
            ai_match_res = True # По умолчанию True

            if not api_key_gen:
                html_out = "ОШИБКА: Ключ не найден. Введите ключ на Вкладке 5!"
                st.error(html_out)
            else:
                try:
                    html_out = generate_full_article_v2(api_key_gen, task['h1'], task['h2'], combined_lsi)
                    status_code = "OK"
                    
                    # === ЗАПУСК ПРОВЕРОК ===
                    if html_out and "Error" not in status_code:
                        plain_text = BeautifulSoup(html_out, "html.parser").get_text(separator=" ")
                        
                        # Проверка на соответствие темы через DeepSeek
                        if st.session_state.get('use_deepseek_check'):
                            ai_match_res = validate_topic_deepseek(api_key_gen, task['h1'], task['h2'], plain_text)

                        # Проверка Тургенев
                        if st.session_state.get('use_turgenev'):
                            turgenev_key = st.session_state.get('turgenev_api_key', '')
                            if turgenev_key:
                                turgenev_res = check_turgenev_sync(plain_text, turgenev_key)
                            else:
                                turgenev_res = "Нет ключа API Тургенева"
                        
                        # Проверка Text.ru
                        if st.session_state.get('use_textru'):
                            textru_key = st.session_state.get('textru_api_key', '')
                            if textru_key:
                                uid = send_textru_sync(plain_text, textru_key)
                                if uid:
                                    textru_uid = uid
                                    textru_res = "⏳ Отправлено..."
                                else:
                                    textru_res = "Ошибка отправки"
                            else:
                                textru_res = "Нет ключа API Text.ru"

                except Exception as e:
                    html_out = f"Error generating: {e}"
                    status_code = "Gen Error"

            # 5. СОХРАНЯЕМ РЕЗУЛЬТАТ В СПИСОК ВКЛАДКИ 5
            if 'bg_results' not in st.session_state:
                st.session_state.bg_results = []
                
            found_existing = False
            for existing_res in st.session_state.bg_results:
                if existing_res['h1'] == task['h1'] and existing_res['h2'] == task['h2']:
                    existing_res['content'] = html_out
                    existing_res['lsi_added'] = lsi_words
                    existing_res['status'] = status_code
                    existing_res['turgenev'] = turgenev_res
                    existing_res['textru'] = textru_res
                    existing_res['textru_uid'] = textru_uid
                    existing_res['ai_match'] = ai_match_res # <-- Сохраняем флаг
                    found_existing = True
                    break
            
            if not found_existing:
                st.session_state.bg_results.append({
                    "h1": task['h1'],
                    "h2": task['h2'],
                    "source_url": task.get('source_url', '-'),
                    "lsi_added": lsi_words,
                    "content": html_out,
                    "status": status_code,
                    "turgenev": turgenev_res,
                    "textru": textru_res,
                    "textru_uid": textru_uid,
                    "ai_match": ai_match_res # <-- Сохраняем флаг
                })
            # 6. ПЕРЕХОД К СЛЕДУЮЩЕЙ ЗАДАЧЕ
            next_task_idx = current_idx + 1
            
            if next_task_idx < len(st.session_state.bg_tasks_queue):
                next_task = st.session_state.bg_tasks_queue[next_task_idx]
                st.toast(f"✅ Готово: {task['h1']}. Дальше: {next_task['h1']}")
                
                # === ТОЧЕЧНАЯ ОЧИСТКА СТАРЫХ РЕЗУЛЬТАТОВ ===
                keys_to_clear = [
                    'analysis_results', 'analysis_done', 'naming_table_df', 
                    'ideal_h1_result', 'raw_comp_data', 'full_graph_data',
                    'detected_anomalies', 'serp_trend_info', 'excluded_urls_auto'
                ]
                for k in keys_to_clear:
                    st.session_state.pop(k, None)
                
# Сохраняем обновления виджетов в буфер, чтобы применить ДО их отрисовки
                st.session_state['pending_widget_updates'] = {
                    'query_input': next_task['h1'],
                    'competitor_source_radio': "Поиск через API Arsenkin (TOP-30)",
                    'my_page_source_radio': "Без страницы",
                    'my_url_input': ""
                }
                st.session_state['lsi_processing_task_id'] = next_task_idx
                st.session_state['start_analysis_flag'] = True 
                st.session_state['analysis_done'] = False
                
                time.sleep(0.5)
                st.rerun()
                
            else:
                st.session_state.lsi_automode_active = False
                st.balloons()
                st.success("🏁 ВСЕ ЗАДАЧИ В ОЧЕРЕДИ ВЫПОЛНЕНЫ!")

# --- 4. ЭКСПОРТ И ПРОСМОТР ---
        if st.session_state.bg_results:
            st.divider()
            st.subheader("3. Результаты")

            df_res = pd.DataFrame(st.session_state.bg_results)
            
            if 'lsi_added' in df_res.columns:
                df_res['lsi_added'] = df_res['lsi_added'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
            
            # === 1. ГЕНЕРИРУЕМ СТОЛБЕЦ "КОММЕНТАРИЙ" ===
            comments = []
            for _, row in df_res.iterrows():
                row_comments = []
                
                # Проверка темы DeepSeek
                if row.get('ai_match') is False:
                    row_comments.append("Текст не соответствует заданной теме.")
                
                # Проверка Тургенева (> 5)
                turg_val = str(row.get('turgenev', ''))
                try:
                    turg_match = re.search(r'\d+\.?\d*', turg_val)
                    if turg_match:
                        risk_val = float(turg_match.group())
                        if risk_val > 5:
                            row_comments.append("Риск по Тургеневу больше 5.")
                except: pass
                
                # Проверка Text.ru (< 95%)
                txtru_val = str(row.get('textru', ''))
                if '%' in txtru_val:
                    try:
                        txtru_match = re.search(r'\d+\.?\d*', txtru_val)
                        if txtru_match:
                            uniq_val = float(txtru_match.group())
                            if uniq_val < 95:
                                row_comments.append("Уникальность от 95%, нужно перегенерировать.")
                    except: pass
                
                comments.append(" | ".join(row_comments) if row_comments else "ОК")

            df_res['Комментарий'] = comments

            # Подготовка к отображению (убираем системные колонки)
            drop_cols = ['textru_uid', 'ai_match']
            df_res_display = df_res.drop(columns=[c for c in drop_cols if c in df_res.columns])

            # === 2. ФУНКЦИЯ ЗАЛИВКИ КРАСНЫМ ЦВЕТОМ ===
            def style_failed_cells(row):
                # Инициализируем пустые стили для всех ячеек строки
                styles = [''] * len(row)
                col_idx = {name: i for i, name in enumerate(row.index)}
                err_style = 'background-color: #ffe6e6; color: #cc0000; font-weight: bold;'
                
                comment = str(row.get('Комментарий', ''))
                
                if "не соответствует заданной теме" in comment:
                    if 'h1' in col_idx: styles[col_idx['h1']] = err_style
                    if 'h2' in col_idx: styles[col_idx['h2']] = err_style
                
                if "Риск по Тургеневу больше 5" in comment:
                    if 'turgenev' in col_idx: styles[col_idx['turgenev']] = err_style
                    
                if "нужно перегенерировать" in comment: # Это маркер уникальности Text.ru
                    if 'textru' in col_idx: styles[col_idx['textru']] = err_style
                    
                # Делаем колонку комментария тоже слегка заметной, если есть ошибка
                if comment != "ОК":
                    if 'Комментарий' in col_idx: styles[col_idx['Комментарий']] = 'color: #cc0000; font-weight: bold;'
                else:
                    if 'Комментарий' in col_idx: styles[col_idx['Комментарий']] = 'color: #008000; font-weight: bold;'
                    
                return styles

            # Применяем стили к таблице
            styled_df = df_res_display.style.apply(style_failed_cells, axis=1)

            # Показываем саму таблицу (со стилями)
            st.dataframe(styled_df, use_container_width=True)
            
            # === АВТОМАТИЧЕСКАЯ ФОНОВАЯ ПРОВЕРКА (БЕЗ КНОПОК) ===
            is_processing = any("⏳" in str(row.get('textru', '')) for row in st.session_state.bg_results)
            
            if is_processing:
                st.warning("⚠️ **Генерация завершена, но идет проверка уникальности (Text.ru).** Файл будет доступен после проверки всех текстов.")
                with st.spinner("🔄 Автоматический опрос Text.ru... (обновление каждые 10 секунд)"):
                    import time
                    time.sleep(10)
                    
                    tk = st.session_state.get('textru_api_key')
                    if tk:
                        for r in st.session_state.bg_results:
                            if r.get('textru_uid') and "⏳" in str(r.get('textru')):
                                stts = check_textru_status_sync(r['textru_uid'], tk)
                                if stts not in ["processing", "error"] and "Ошибка" not in stts:
                                    r['textru'] = stts
                                    r['textru_uid'] = None
                                elif "Ошибка" in stts or stts == "error":
                                    r['textru'] = stts
                                    r['textru_uid'] = None
                                    
                    st.rerun()
            else:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                    # === ИЗМЕНЕНИЕ: сохраняем styled_df (со стилями), а не "голый" df_res_display ===
                    styled_df.to_excel(writer, index=False, sheet_name='Generated_Articles')
                
                st.success("✅ Все проверки завершены! Файл готов к скачиванию.")
                st.download_button(
                    label="💾 СКАЧАТЬ РЕЗУЛЬТАТЫ (Excel)", 
                    data=buf.getvalue(), 
                    file_name="SEO_Content_Result.xlsx", 
                    mime="application/vnd.ms-excel", 
                    type="primary"
                )
        
        st.markdown("---")
        st.markdown("#### 👁️ Просмотр статьи")
        
        # Стили для предпросмотра
        table_css = """
        <style>
            .brand-accent-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-family: 'Inter', sans-serif; font-size: 14px; }
            .brand-accent-table th { background-color: #277EFF; color: white; padding: 12px; text-align: left; }
            .brand-accent-table td { border: 1px solid #e5e7eb; padding: 10px; color: #374151; }
            .brand-accent-table tr:nth-child(even) { background-color: #f9fafb; }
            ul { margin-bottom: 15px; }
            li { margin-bottom: 5px; }
        </style>
        """
        
        opts = [f"{i+1}. {r['h2']} (H1: {r['h1']})" for i, r in enumerate(st.session_state.bg_results)]
        sel = st.selectbox("Выберите статью:", opts)
        
        if sel:
            idx = int(sel.split(".")[0]) - 1
            rec = st.session_state.bg_results[idx]
            
            lsi_str = ", ".join(rec['lsi_added']) if rec['lsi_added'] else "Нет уникальных LSI"
            st.info(f"📊 **Собранная семантика (расширение):** {lsi_str}")
            
            with st.container(border=True):
                # Вставляем стили + контент
                st.markdown(table_css + rec['content'], unsafe_allow_html=True)
            
            with st.expander("Исходный код HTML"):
                st.code(rec['content'], language='html')

# ==================================================================
# ❓ ВКЛАДКА 6: FAQ ГЕНЕРАТОР
# ==================================================================
with tab_faq_gen:
    st.markdown("### ❓ Генерация человечных FAQ по TF-IDF")
    
    c_faq1, c_faq2 = st.columns([1, 2])
    with c_faq1:
        faq_source = st.radio("Источник данных для FAQ:", ["Вручную (Списки H1)", "Список ссылок (Авто-парсинг H1)"])
        # Ползунок количества вопросов
        # Ручной ввод количества вопросов
        st.session_state['faq_questions_count'] = st.number_input(
            "Количество вопросов (от 2 до 100):", 
            min_value=2, max_value=100, value=st.session_state.get('faq_questions_count', 10), step=1
        )
        
    with c_faq2:
        st.info("Скрипт по очереди проведет SEO-анализ каждого запроса/ссылки, возьмет 15 топовых слов и сгенерирует JSON-массив с вопросами и ответами.")
        
        # Функция для безопасной синхронизации ключа между вкладками
        def sync_faq_api_key():
            if 'faq_api_key_input_unique' in st.session_state:
                st.session_state['SUPER_GLOBAL_KEY'] = st.session_state['faq_api_key_input_unique']

        # === ПОЛЕ ДЛЯ API КЛЮЧА НА 6 ВКЛАДКЕ ===
        st.text_input(
            "🔑 Gemini API Key:", 
            value=st.session_state.get('SUPER_GLOBAL_KEY', ''), 
            type="password", 
            key="faq_api_key_input_unique",
            on_change=sync_faq_api_key
        )
        
    faq_input = st.text_area("Введите H1 или URL (каждый с новой строки):", height=150)
    
    # 1. ЗАГРУЗКА ЗАДАЧ
    if st.button("📥 Загрузить задачи (FAQ)", use_container_width=True):
        tasks = []
        lines = [line.strip() for line in faq_input.split('\n') if line.strip()]
        
        if faq_source == "Вручную (Списки H1)":
            for line in lines:
                tasks.append({"h1": line, "url": "-"})
        else:
            with st.spinner("🕵️ Парсим H1 с указанных сайтов..."):
                import requests
                from bs4 import BeautifulSoup
                for url in lines:
                    try:
                        res = requests.get(url, timeout=5)
                        soup = BeautifulSoup(res.text, 'html.parser')
                        h1_tag = soup.find('h1')
                        h1_text = h1_tag.text.strip() if h1_tag else f"Без H1 ({url})"
                        tasks.append({"h1": h1_text, "url": url})
                    except:
                        tasks.append({"h1": f"Ошибка парсинга", "url": url})
                        
        st.session_state.faq_tasks_queue = tasks
        st.session_state.faq_results = []
        st.success(f"✅ В очередь добавлено задач: {len(tasks)}")

    st.markdown("---")
    
    # 2. ИНФО О ЗАДАЧАХ И КНОПКА СТАРТА
    faq_queue = st.session_state.get('faq_tasks_queue', [])
    faq_q_count = len(faq_queue)
    
    c_fstart1, c_fstart2 = st.columns([1, 1])
    with c_fstart1:
        st.markdown(f"**В очереди:** {faq_q_count} шт. | **Готово:** {len(st.session_state.get('faq_results', []))} шт.")
        
        if not st.session_state.get('faq_automode_active'):
            btn_lbl = "▶️ СТАРТ ГЕНЕРАЦИИ FAQ" if faq_q_count > 0 else "✅ ВСЕ FAQ ГОТОВЫ"
            if st.button(btn_lbl, type="primary", disabled=(faq_q_count == 0), use_container_width=True, key="faq_start_btn_unique"):
                api_key_check = st.session_state.get('SUPER_GLOBAL_KEY')
                if not api_key_check:
                    st.error("Введите API ключ Gemini (на Вкладке 5)!")
                else:
                    st.session_state.faq_automode_active = True
                    st.session_state.faq_processing_task_id = 0
                    first_t = st.session_state.faq_tasks_queue[0]
                    
                    st.session_state['pending_widget_updates'] = {
                        'query_input': first_t['h1'],
                        'competitor_source_radio': "Поиск через API Arsenkin (TOP-30)",
                        'my_page_source_radio': "Без страницы",
                        'my_url_input': ""
                    }
                    st.session_state['start_analysis_flag'] = True 
                    st.session_state['analysis_done'] = False
                    st.toast("🚀 Запуск FAQ генератора... Переход на Вкладку 1")
                    st.rerun()
        else:
            # === ИЗМЕНИТЬ ВОТ ЭТОТ БЛОК ===
            st.button("⛔ ОСТАНОВИТЬ ГЕНЕРАЦИЮ", type="secondary", use_container_width=True, on_click=global_stop_callback)

    with c_fstart2:
        if st.button("🗑️ Сбросить очередь FAQ", disabled=st.session_state.get('faq_automode_active', False), use_container_width=True):
            st.session_state.faq_tasks_queue = []
            st.session_state.faq_results = []
            st.session_state.faq_automode_active = False
            st.rerun()

# ==================================================================
    # 🔥 HOOK ДЛЯ FAQ ГЕНЕРАТОРА (СРАБАТЫВАЕТ ПОСЛЕ ПЕРВОЙ ВКЛАДКИ)
    # ==================================================================
    if st.session_state.get('faq_automode_active'):
        curr_idx = st.session_state.get('faq_processing_task_id')
        if 'faq_tasks_queue' not in st.session_state or curr_idx is None or curr_idx >= len(st.session_state.faq_tasks_queue):
            st.session_state.faq_automode_active = False
            st.stop()

        task = st.session_state.faq_tasks_queue[curr_idx]
        target_q_count = st.session_state.get('faq_questions_count', 10)
        
        lsi_words = []
        res_data = st.session_state.get('analysis_results')
        if res_data and res_data.get('hybrid') is not None and not res_data['hybrid'].empty:
            # Жестко берем ТОП-150 слов, чтобы не перегружать нейросеть мусором
            lsi_words = res_data['hybrid'].head(150)['Слово'].tolist()
        
        # ГЕНЕРАЦИЯ
        api_key_gen = str(st.session_state.get('SUPER_GLOBAL_KEY', '')).strip()
        faq_json_result = generate_faq_gemini(api_key_gen, task['h1'], lsi_words, target_q_count)
        
        if 'faq_results' not in st.session_state: st.session_state.faq_results = []

    # Инициализация для отзывов
        if 'reviews_results' not in st.session_state: st.session_state.reviews_results = []
        if 'reviews_queue' not in st.session_state: st.session_state.reviews_queue = []
        if 'reviews_automode_active' not in st.session_state: st.session_state.reviews_automode_active = False
        if 'reviews_current_index' not in st.session_state: st.session_state.reviews_current_index = 0
        if 'reviews_per_query' not in st.session_state: st.session_state.reviews_per_query = 3
        
        st.session_state.faq_results.append({
            "h1": task['h1'],
            "url": task['url'],
            "lsi": lsi_words,
            "faq_data": faq_json_result
        })

        # ПЕРЕХОД ДАЛЬШЕ
        next_idx = curr_idx + 1
        if next_idx < len(st.session_state.faq_tasks_queue):
            next_t = st.session_state.faq_tasks_queue[next_idx]
            st.toast(f"✅ FAQ готов: {task['h1']}")
            
            # Очистка мусора
            keys_to_clear = ['analysis_results', 'analysis_done', 'naming_table_df', 'ideal_h1_result', 'raw_comp_data', 'full_graph_data', 'detected_anomalies', 'serp_trend_info', 'excluded_urls_auto']
            for k in keys_to_clear: st.session_state.pop(k, None)
            
            # Буфер виджетов
            st.session_state['pending_widget_updates'] = {
                'query_input': next_t['h1'],
                'competitor_source_radio': "Поиск через API Arsenkin (TOP-30)",
                'my_page_source_radio': "Без страницы",
                'my_url_input': ""
            }
            st.session_state['faq_processing_task_id'] = next_idx
            st.session_state['start_analysis_flag'] = True 
            st.session_state['analysis_done'] = False
            import time
            time.sleep(0.5)
            st.rerun()
        else:
            st.session_state.faq_automode_active = False
            st.balloons()
            st.success("🏁 ВСЕ FAQ СГЕНЕРИРОВАНЫ!")

# 3. ВЫВОД РЕЗУЛЬТАТОВ И ЭКСПОРТ В EXCEL
        if st.session_state.get('faq_results'):
            st.markdown("### 📋 Результаты генерации")
            
            # --- ПОДГОТОВКА ДАННЫХ ДЛЯ EXCEL ---
            all_faq_rows = []
            for res in st.session_state.faq_results:
                h1_val = res['h1']
                url_val = res['url']
                
                faq_items = res['faq_data']
                if isinstance(faq_items, list):
                    for item in faq_items:
                        if isinstance(item, dict):
                            all_faq_rows.append({
                                "H1 / Маркер": h1_val,
                                "URL": url_val,
                                "Тип": item.get("Тип", "Информационный"),
                                "Вопрос": item.get("Вопрос", ""),
                                "Ответ": item.get("Ответ", "")
                            })
                            
            if all_faq_rows:
                df_export = pd.DataFrame(all_faq_rows)
                
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_export.to_excel(writer, index=False, sheet_name='FAQ_Результаты')
                    # Делаем колонки шире для удобства чтения
                    worksheet = writer.sheets['FAQ_Результаты']
                    worksheet.set_column('A:B', 30)
                    worksheet.set_column('C:C', 20)  # Ширина для "Тип"
                    worksheet.set_column('D:E', 70)  # Ширина для Вопросов/Ответов
                    
                excel_data = output.getvalue()
                st.download_button(
                    label="💾 СКАЧАТЬ ВСЕ FAQ В EXCEL",
                    data=excel_data,
                    file_name="Сгенерированные_FAQ.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
                
            st.markdown("---")
            # --- ПРЕДПРОСМОТР НА ЭКРАНЕ ---
            for res in st.session_state.faq_results:
                with st.expander(f"📌 {res['h1']} ({res['url']})"):
                    st.caption(f"**Использованы слова:** {', '.join(res['lsi'])}")
                    
                    faq_items = res['faq_data']
                    if isinstance(faq_items, list) and len(faq_items) > 0 and isinstance(faq_items[0], dict):
                        import pandas as pd
                        st.table(pd.DataFrame(faq_items))
                    else:
                        st.error("Ошибка формата ответа нейросети:")
                        st.write(faq_items)
# ==========================================
# TAB 7: ГЕНЕРАТОР ОТЗЫВОВ
# ==========================================
with tab_reviews_gen:
    st.header("💬 Генератор отзывов (Автомат)")
    
    rev_mode = st.radio("Источник запросов:", ["Список H1", "Список URL"], horizontal=True, key="rev_mode_radio")
    rev_input = st.text_area("Ввод данных (по одному на строку):", height=150, key="rev_data_input")
    rev_count_input = st.number_input("Сколько отзывов на один товар?", 1, 10, 3, key="rev_count_val")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🚀 ЗАПУСТИТЬ ГЕНЕРАЦИЮ", type="primary", use_container_width=True):
            lines =[l.strip() for l in rev_input.split('\n') if l.strip()]
            if lines:
                queue =[]
                if rev_mode == "Список URL":
                    for u in lines:
                        h1_text = get_h1_from_url(u) 
                        if not h1_text:
                            h1_text = u.split('/')[-1].replace('-', ' ').capitalize()
                        queue.append({'q': h1_text, 'url': u})
                else:
                    for q in lines: 
                        queue.append({'q': q, 'url': 'manual'})
                
                st.session_state.reviews_queue = queue
                st.session_state.reviews_results =[]
                st.session_state.reviews_current_index = 0
                st.session_state.reviews_per_query = rev_count_input
                
                # Принудительно отключаем другие генераторы на всякий случай
                st.session_state.lsi_automode_active = False
                st.session_state.faq_automode_active = False
                st.session_state.reviews_automode_active = True

                updates = {
                    'query_input': queue[0]['q'],
                    'competitor_source_radio': "Поиск через API Arsenkin (TOP-30)"
                }
                
                if rev_mode == "Список URL":
                    updates['my_page_source_radio'] = "Релевантная страница на вашем сайте"
                    updates['my_url_input'] = queue[0]['url']
                else:
                    updates['my_page_source_radio'] = "Без страницы"
                    updates['my_url_input'] = ""
                
                st.session_state['pending_widget_updates'] = updates
                st.session_state.start_analysis_flag = True
                
                # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: СБРОС СТАРОГО АНАЛИЗА ПРИ СТАРТЕ ===
                st.session_state.pop('analysis_done', None)
                st.session_state.pop('analysis_results', None)
                # =================================================================
                
                st.rerun()

    # --- ОТРРИСОВКА РЕЗУЛЬТАТОВ ---
    if 'reviews_results' in st.session_state and st.session_state.reviews_results:
        st.markdown("---")
        st.subheader("📊 Результаты")
        
        df_display = pd.DataFrame(st.session_state.reviews_results)
        st.dataframe(df_display, use_container_width=True)
        
        # Кнопка Excel тоже тут
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_display.to_excel(writer, index=False)
        
        st.download_button(
            label="📥 СКАЧАТЬ В EXCEL",
            data=buffer.getvalue(),
            file_name="reviews.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )





































