import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
import math
import inspect
import concurrent.futures
from urllib.parse import urlparse

# ==========================================
# 0. АВТОРИЗАЦИЯ
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
                border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-top: 5rem;
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
# 1. КОНФИГУРАЦИЯ И СТИЛИ (АГРЕССИВНАЯ ПЕРЕКРАСКА)
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO", page_icon="📊")

DEFAULT_EXCLUDE_DOMAINS = [
    "yandex.ru", "avito.ru", "beru.ru", "tiu.ru", "aliexpress.com", "ebay.com",
    "auto.ru", "2gis.ru", "sravni.ru", "toshop.ru", "price.ru", "pandao.ru",
    "instagram.com", "wikipedia.org", "rambler.ru", "hh.ru", "banki.ru", 
    "regmarkets.ru", "zoon.ru", "pulscen.ru", "prodoctorov.ru", "blizko.ru", 
    "domclick.ru", "satom.ru", "quto.ru", "edadeal.ru", "cataloxy.ru", 
    "irr.ru", "onliner.by", "shop.by", "deal.by", "yell.ru", "profi.ru", 
    "irecommend.ru", "otzovik.com", "ozon.ru", "ozon.by", "market.yandex.ru", 
    "youtube.com", "gosuslugi.ru", "dzen.ru", "2gis.by"
]
DEFAULT_EXCLUDE = "\n".join(DEFAULT_EXCLUDE_DOMAINS)
DEFAULT_STOPS = "рублей\nруб\nкупить\nцена\nшт\nсм\nмм\nкг\nкв\nм2\nстр\nул"
REGIONS = ["Москва", "Санкт-Петербург", "Екатеринбург", "Новосибирск", "Казань", "Нижний Новгород", "Самара", "Челябинск", "Омск", "Краснодар", "Киев (UA)", "Минск (BY)", "Алматы (KZ)"]

# Цвета
PRIMARY_COLOR = "#277EFF"
PRIMARY_DARK = "#1E63C4"
TEXT_COLOR = "#3D4858"
LIGHT_BG_MAIN = "#F1F5F9"
BORDER_COLOR = "#E2E8F0"
HEADER_BG = "#F0F7FF" 

st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        /* 1. Глобальный фон */
        .stApp {{
            background-color: #FFFFFF !important;
            color: {TEXT_COLOR} !important;
        }}
        
        html, body, p, li, h1, h2, h3, h4 {{
            font-family: 'Inter', sans-serif;
            color: {TEXT_COLOR} !important;
        }}

        /* 2. Элементы управления */
        .stButton button {{
            background-color: {PRIMARY_COLOR} !important;
            color: white !important;
            border: none;
            border-radius: 6px;
        }}
        .stButton button:hover {{
            background-color: {PRIMARY_DARK} !important;
        }}
        
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {LIGHT_BG_MAIN} !important;
            color: {TEXT_COLOR} !important;
            border: 1px solid {BORDER_COLOR} !important;
        }}

        /* 3. АГРЕССИВНЫЙ ХАК ДЛЯ ТАБЛИЦ (st.dataframe) */
        /* Принудительно красим заголовки таблицы */
        [data-testid="stDataFrame"] th {{
            background-color: {HEADER_BG} !important;
            color: {PRIMARY_COLOR} !important;
            font-weight: bold !important;
            border-bottom: 2px solid {PRIMARY_COLOR} !important;
            text-align: center !important;
        }}
        
        /* Принудительно красим ячейки таблицы */
        [data-testid="stDataFrame"] td {{
            background-color: #FFFFFF !important;
            color: {TEXT_COLOR} !important;
            border-bottom: 1px solid {BORDER_COLOR} !important;
        }}
        
        /* Индекс (левый столбец) */
        [data-testid="stDataFrame"] th[role="rowheader"] {{
            color: {PRIMARY_COLOR} !important;
            background-color: {HEADER_BG} !important;
        }}
        
        /* При наведении на строку */
        [data-testid="stDataFrame"] tr:hover td {{
            background-color: {LIGHT_BG_MAIN} !important;
        }}
        
        /* Убираем стандартные темные рамки Streamlit */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {BORDER_COLOR} !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            border-left: 1px solid {BORDER_COLOR};
        }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ЛОГИКА (БЭКЕНД)
# ==========================================

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

try:
    if not hasattr(inspect, 'getargspec'):
        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return spec.args, spec.varargs, spec.varkw, spec.defaults
        inspect.getargspec = getargspec
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    USE_NLP = True
except:
    morph = None
    USE_NLP = False

try:
    from googlesearch import search
    USE_SEARCH = True
except:
    USE_SEARCH = False

def process_text(text, settings, n_gram=1):
    pattern = r'[а-яА-ЯёЁ0-9a-zA-Z]+' if settings['numbers'] else r'[а-яА-ЯёЁa-zA-Z]+'
    words = re.findall(pattern, text.lower())
    stops = set(w.lower() for w in settings['custom_stops'])
    clean_words = []
    
    for w in words:
        if len(w) < 2 or w in stops: continue
        lemma = w
        if USE_NLP and n_gram == 1: 
            p = morph.parse(w)[0]
            if 'PREP' in p.tag or 'CONJ' in p.tag or 'PRCL' in p.tag or 'NPRO' in p.tag: continue
            lemma = p.normal_form
        clean_words.append(lemma)
    
    if n_gram > 1:
        ngrams = []
        for i in range(len(clean_words) - n_gram + 1):
            phrase = " ".join(clean_words[i:i+n_gram])
            ngrams.append(phrase)
        return ngrams
    return clean_words

def parse_page(url, settings):
    headers = {'User-Agent': settings['ua']}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200: return None
        soup = BeautifulSoup(r.text, 'html.parser')
        
        if settings['noindex']:
            for t in soup.find_all(['noindex', 'script', 'style', 'head', 'footer', 'nav']): t.decompose()
        else:
            for t in soup(['script', 'style', 'head']): t.decompose()
            
        anchors_list = [a.get_text(strip=True) for a in soup.find_all('a') if a.get_text(strip=True)]
        anchor_text = " ".join(anchors_list)
        
        extra_text = []
        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): extra_text.append(img['alt'])
            for t in soup.find_all(title=True): extra_text.append(t['title'])
        body_text = soup.get_text(separator=' ') + " " + " ".join(extra_text)
        
        return {
            'url': url, 'domain': urlparse(url).netloc, 
            'body_text': body_text, 'anchor_text': anchor_text
        }
    except: return None

def calculate_metrics(comp_data, my_data, settings):
    if not my_data or not my_data['body_text']:
        my_lemmas = []
        my_anchors = []
        my_len = 0
    else:
        my_lemmas = process_text(my_data['body_text'], settings)
        my_anchors = process_text(my_data['anchor_text'], settings)
        my_len = len(my_lemmas)
    
    comp_docs = []
    for p in comp_data:
        body = process_text(p['body_text'], settings)
        anchor = process_text(p['anchor_text'], settings)
        comp_docs.append({'body': body, 'anchor': anchor})
        
    if not comp_docs:
        return {"depth": pd.DataFrame(), "hybrid": pd.DataFrame(), "ngrams": pd.DataFrame(), "relevance_top": pd.DataFrame(), "my_score": {"width": 0, "depth": 0}}

    avg_len = np.mean([len(d['body']) for d in comp_docs])
    norm_k = (my_len / avg_len) if (settings['norm'] and avg_len > 0) else 1.0
    
    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    vocab = sorted(list(vocab))
    
    N = len(comp_docs)
    doc_freqs = Counter()
    for d in comp_docs:
        for w in set(d['body']): doc_freqs[w] += 1
        
    k1, b = 1.2, 0.75
    table_depth, table_hybrid = [], []
    
    for word in vocab:
        df = doc_freqs[word]
        if df < 2 and word not in my_lemmas: continue 
        
        my_tf = my_lemmas.count(word)
        my_anch_tf = my_anchors.count(word)
        
        c_body_tfs = [d['body'].count(word) for d in comp_docs]
        c_anch_tfs = [d['anchor'].count(word) for d in comp_docs]
        
        med_tf = np.median(c_body_tfs)
        med_anch = np.median(c_anch_tfs)
        max_tf = np.max(c_body_tfs)
        
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        target_body = int(med_tf * 1.3 * norm_k)
        diff_body = target_body - my_tf
        target_anch = int(med_anch * norm_k)
        diff_anch = target_anch - my_anch_tf
        
        if med_tf > 0.5 or my_tf > 0:
            table_depth.append({
                "Слово": word,
                "У вас (TF)": my_tf, 
                "Медиана": round(med_tf, 1),
                "Максимум": int(max_tf * norm_k),
                "Добавить/Убрать": diff_body, 
                "diff_abs": abs(diff_body),
                "Тег A у вас": my_anch_tf,
                "Тег A реком.": target_anch
            })
            table_hybrid.append({
                "Слово": word, 
                "TF-IDF ТОП": round(med_tf * idf, 2), 
                "TF-IDF у вас": round(my_tf * idf, 2),
                "Сайтов": df, 
                "Переспам": max_tf
            })

    table_ngrams = []
    if comp_docs and my_data and 'body_text' in my_data:
        try:
            my_bi = process_text(my_data['body_text'], settings, 2)
            comp_bi = [process_text(p['body_text'], settings, 2) for p in comp_data if p and 'body_text' in p]
            
            all_bi = set(my_bi)
            for c in comp_bi:
                if c: all_bi.update(c)
            bi_freqs = Counter()
            for c in comp_bi:
                if c:
                    for b_ in set(c): bi_freqs[b_] += 1

            for bg in all_bi:
                df = bi_freqs[bg]
                if df < 2 and bg not in my_bi: continue
                my_c = my_bi.count(bg)
                comp_c = [c['body'].count(bg) for c in comp_docs if 'body' in c]
                med_c = np.median(comp_c) if comp_c else 0
                
                if med_c > 0 or my_c > 0:
                    table_ngrams.append({
                        "N-грамма": bg, 
                        "Сайтов": df, 
                        "Медиана": med_c,
                        "На сайте": my_c,
                        "TF-IDF": round(my_c * math.log(N/df if df>0 else 1), 3)
                    })
        except Exception as e:
            st.error(f"Error n-grams: {e}")

    table_rel = []
    for i, p in enumerate(comp_data):
        p_lemmas = process_text(p['body_text'], settings)
        w = len(set(p_lemmas).intersection(vocab))
        table_rel.append({
            "Домен": p['domain'], "Позиция": i+1, "URL": p['url'],
            "Ширина": w, "Глубина": len(p_lemmas)
        })
        
    return {
        "depth": pd.DataFrame(table_depth), 
        "hybrid": pd.DataFrame(table_hybrid),
        "ngrams": pd.DataFrame(table_ngrams), 
        "relevance_top": pd.DataFrame(table_rel),
        "my_score": {"width": len(set(my_lemmas).intersection(vocab)), "depth": len(my_lemmas)}
    }

# ==========================================
# 3. ФУНКЦИЯ ОТОБРАЖЕНИЯ (DATAFRAME + SORT)
# ==========================================

def render_paginated_table(df, title_text, key_prefix, sort_by_col=None, use_abs_sort=False):
    """
    Рендеринг через st.dataframe (поддерживает сортировку в шапке),
    но с агрессивным CSS и Pandas Styler для белого фона.
    """
    if df.empty:
        st.info(f"{title_text}: Нет данных.")
        return

    # 1. Сортировка (Начальная)
    if sort_by_col and sort_by_col in df.columns:
        if use_abs_sort:
            df['_abs_sort'] = df[sort_by_col].abs()
            df = df.sort_values(by='_abs_sort', ascending=False).drop(columns=['_abs_sort'])
        else:
            df = df.sort_values(by=sort_by_col, ascending=False)
            
    # 2. Индекс с 1
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    
    # 3. Пагинация (10 строк)
    ROWS_PER_PAGE = 10 
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

    # 4. ПОКРАСКА ЯЧЕЕК (Pandas Styler - fallback)
    # Это красит саму таблицу изнутри данных, если CSS не сработает
    def style_dataframe(d):
        return d.style.set_properties(**{
            'background-color': '#FFFFFF',
            'color': '#3D4858',
        })

    st.markdown(f"### {title_text}")
    
    # Выводим интерактивную таблицу
    st.dataframe(
        style_dataframe(df_view),
        use_container_width=True,
        column_config={"diff_abs": None}
    )
    
    # 5. Кнопки управления
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

# ==========================================
# 4. ИНТЕРФЕЙС
# ==========================================

col_main, col_sidebar = st.columns([65, 35]) 

with col_main:
    st.title("SEO Анализатор Релевантности")

    st.markdown("### URL или код страницы Вашего сайта")
    my_input_type = st.radio(
        "Тип страницы", 
        ["Релевантная страница на вашем сайте", "Исходный код страницы или текст", "Без страницы"], 
        horizontal=True, label_visibility="collapsed", key="my_page_source_radio"
    )

    my_url = ""
    my_page_content = ""
    if my_input_type == "Релевантная страница на вашем сайте":
        my_url = st.text_input("URL страницы", placeholder="https://site.ru/catalog/tovar", label_visibility="collapsed", key="my_url_input")
    elif my_input_type == "Исходный код страницы или текст":
        my_page_content = st.text_area("Исходный код или текст", height=200, label_visibility="collapsed", placeholder="Вставьте HTML", key="my_content_input")

    st.markdown("### Поисковой запрос")
    query = st.text_input("Основной запрос", placeholder="Например: купить пластиковые окна", label_visibility="collapsed", key="query_input")

    st.markdown("### Поиск или URL страниц конкурентов")
    source_type_new = st.radio("Источник конкурентов", ["Поиск", "Список url-адресов ваших конкурентов"], horizontal=True, label_visibility="collapsed", key="competitor_source_radio")
    source_type = "Google (Авто)" if source_type_new == "Поиск" else "Ручной список" 

    if source_type == "Ручной список":
        st.markdown("### Введите список URL")
        st.text_area("Вставьте ссылки здесь (каждая с новой строки)", height=200, key="manual_urls_ui")

    st.markdown("### Редактируемые списки")
    excludes = st.text_area("Не учитывать домены", DEFAULT_EXCLUDE, height=200, key="settings_excludes")
    c_stops = st.text_area("Стоп-слова", DEFAULT_STOPS, height=200, key="settings_stops")

    st.markdown("---")
    
    if st.button("ЗАПУСТИТЬ АНАЛИЗ", type="primary", use_container_width=True, key="start_analysis_btn"):
        for key in list(st.session_state.keys()):
            if key.endswith('_page'):
                st.session_state[key] = 1
        st.session_state.start_analysis_flag = True

with col_sidebar:
    st.markdown("#####⚙️ Настройки")
    ua = st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0"], key="settings_ua")
    search_engine = st.selectbox("Поисковая система", ["Google", "Яндекс", "Яндекс + Google"], key="settings_search_engine")
    region = st.selectbox("Яндекс / Регион", REGIONS, key="settings_region")
    device = st.selectbox("Устройство", ["Desktop", "Mobile"], key="settings_device")
    top_n = st.selectbox("Анализировать ТОП", [10, 20, 30], index=1, key="settings_top_n")
    st.selectbox("Учитывать тип страниц по url", ["Все страницы", "Главные страницы", "Внутренние страницы"], key="settings_url_type")
    st.selectbox("Учитывать тип", ["Все страницы", "Коммерческие", "Информационные"], key="settings_content_type")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.checkbox("Исключать noindex/script", True, key="settings_noindex")
        st.checkbox("Учитывать Alt/Title", False, key="settings_alt")
        st.checkbox("Учитывать числа", False, key="settings_numbers")
    with col_c2:
        st.checkbox("Нормировать по длине", True, key="settings_norm")
        st.checkbox("Исключать агрегаторы", True, key="settings_agg")

# ==========================================
# 5. ОБРАБОТКА И ВЫВОД (Session State Logic)
# ==========================================

if st.session_state.get('start_analysis_flag'):
    st.session_state.start_analysis_flag = False

    if my_input_type == "Релевантная страница на вашем сайте" and not st.session_state.get('my_url_input'):
        st.error("Введите URL!")
        st.stop()
        
    if my_input_type == "Исходный код страницы или текст" and not st.session_state.get('my_content_input', '').strip():
        st.error("Введите исходный код!")
        st.stop()

    settings = {
        'noindex': st.session_state.settings_noindex, 
        'alt_title': st.session_state.settings_alt, 
        'numbers': st.session_state.settings_numbers,
        'norm': st.session_state.settings_norm, 
        'ua': st.session_state.settings_ua, 
        'custom_stops': st.session_state.settings_stops.split()
    }
    
    target_urls = []
    if source_type == "Google (Авто)":
        excl = [d.strip() for d in st.session_state.settings_excludes.split('\n') if d.strip()]
        if st.session_state.settings_agg: excl.extend(["avito", "ozon", "wildberries", "market", "tiu", "youtube"])
        
        try:
            with st.spinner(f"Сбор ТОПа..."):
                if not USE_SEARCH:
                    st.error("Нет библиотеки googlesearch")
                    st.stop()
                found = search(st.session_state.query_input, num_results=st.session_state.settings_top_n * 2, lang="ru")
                cnt = 0
                for u in found:
                    if my_input_type == "Релевантная страница на вашем сайте" and st.session_state.my_url_input in u: continue
                    if any(x in urlparse(u).netloc for x in excl): continue
                    target_urls.append(u)
                    cnt += 1
                    if cnt >= st.session_state.settings_top_n: break
        except Exception as e:
            st.error(f"Ошибка поиска: {e}")
            st.stop()
    else:
        raw_urls = st.session_state.get("manual_urls_ui", "")
        target_urls = [u.strip() for u in raw_urls.split('\n') if u.strip()]

    if not target_urls:
        st.error("Нет конкурентов.")
        st.stop()
        
    my_data = None
    if my_input_type == "Релевантная страница на вашем сайте":
        with st.spinner("Скачивание вашей страницы..."):
            my_data = parse_page(st.session_state.my_url_input, settings)
    elif my_input_type == "Исходный код страницы или текст":
        my_data = {'url': 'Local', 'domain': 'local', 'body_text': st.session_state.my_content_input, 'anchor_text': ''}

    comp_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(parse_page, u, settings): u for u in target_urls}
        done = 0
        total = len(target_urls)
        prog = st.progress(0)
        stat = st.empty()
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: comp_data.append(res)
            done += 1
            prog.progress(done / total)
            stat.text(f"Загрузка конкурентов: {done}/{total}")
    prog.empty()
    stat.empty()

    with st.spinner("Анализ данных..."):
        st.session_state.analysis_results = calculate_metrics(comp_data, my_data, settings)
        st.session_state.analysis_done = True
        st.rerun()

if st.session_state.analysis_done and st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    st.success("Анализ готов!")
    
    st.markdown(f"""
        <div style='background-color: {LIGHT_BG_MAIN}; padding: 15px; border-radius: 8px; border: 1px solid {BORDER_COLOR}; margin-bottom: 20px;'>
            <h4 style='margin:0; color: {PRIMARY_COLOR};'>Результат вашего сайта</h4>
            <p style='margin:5px 0 0 0;'>Ширина (уникальные слова): <b>{results['my_score']['width']}</b> | Глубина (всего слов): <b>{results['my_score']['depth']}</b></p>
        </div>
    """, unsafe_allow_html=True)

    render_paginated_table(
        results['depth'], 
        "1. Рекомендации по глубине (Добавить/Убрать слова)", 
        "tbl_depth_1", 
        sort_by_col="Добавить/Убрать", 
        use_abs_sort=True
    )

    render_paginated_table(
        results['hybrid'], 
        "3. Гибридный ТОП (TF-IDF)", 
        "tbl_hybrid", 
        sort_by_col="TF-IDF ТОП", 
        use_abs_sort=False
    )

    render_paginated_table(
        results['ngrams'], 
        "4. N-граммы (Фразы)", 
        "tbl_ngrams", 
        sort_by_col="TF-IDF", 
        use_abs_sort=False
    )

    render_paginated_table(
        results['relevance_top'], 
        "5. ТОП релевантности страниц конкурентов", 
        "tbl_rel", 
        sort_by_col="Ширина", 
        use_abs_sort=False
    )
