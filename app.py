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
# 1. КОНФИГУРАЦИЯ (НОВЫЙ СВЕТЛЫЙ CSS И ДВЕ КОЛОНКИ)
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO", page_icon="📊")

# Обновленный список исключаемых доменов
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

# Список регионов для имитации
REGIONS = [
    "Москва", "Санкт-Петербург", "Екатеринбург", "Новосибирск", "Казань", 
    "Нижний Новгород", "Самара", "Челябинск", "Омск", "Краснодар", 
    "Киев (UA)", "Минск (BY)", "Алматы (KZ)"
]

# Цвета
PRIMARY_COLOR = "#277EFF"    # Голубой/Синий для кнопки
PRIMARY_DARK = "#1E63C4"     
TEXT_COLOR = "#3D4858"       # Темно-серый (Основной текст)
LIGHT_BG_MAIN = "#F1F5F9"    # Светло-серый фон полей в левом блоке
LIGHT_BG_SIDEBAR = "#E9EEF4" # Очень светло-серый фон для полей в сайдбаре (как на скриншоте)
BORDER_COLOR = "#E2E8F0"     # Цвет рамки
RED_ACCENT = "#DC3545"       # Красный (для маркера выбранной кнопки)
MAROON_DIVIDER = "#8B0000"   # Темно-бордовый для разделителя

st.markdown(f"""
   <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        /* 1. ОСНОВНАЯ ТЕМА И ТИПОГРАФИКА */
        html, body, [class*="stApp"], [class*="css"] {{
            font-family: 'Inter', sans-serif;
            background-color: #FFFFFF !important;
            color: {TEXT_COLOR} !important; 
        }}
        h1, h2, h3, p, label, span, div, a {{
            color: {TEXT_COLOR} !important; 
        }}

        /* Уменьшение общего отступа */
        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 100% !important; 
        }}
        
        /* 2. Стилизация полей ввода (ЛЕВЫЙ БЛОК) */
        .stTextInput input, 
        .stTextArea textarea, 
        div[data-testid="stTextarea"] textarea {{
            color: {TEXT_COLOR} !important;
            background-color: {LIGHT_BG_MAIN} !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 6px;
        }}
        
        /* 3. Кнопка "ЗАПУСТИТЬ АНАЛИЗ" (Синий/Голубой с градиентом) */
        .stButton button {{
            background-image: linear-gradient(to right, {PRIMARY_COLOR}, {PRIMARY_DARK});
            color: white !important;
            font-weight: bold;
            border-radius: 6px;
            height: 50px;
            width: 100%;
            border: none;
        }}
        
        /* 4. Радио-кнопки (Вкладки) */
        div[data-testid="stRadio"] label {{
            background-color: #FFFFFF; 
            border-radius: 6px;
            padding: 10px 15px;
            margin-right: 5px;
            color: {TEXT_COLOR};
            border: 1px solid {BORDER_COLOR};
            font-weight: 400;
        }}
        /* Выбранный элемент (Красная рамка, Черный/Красный маркер) */
        div[data-testid="stRadio"] input:checked + div {{
            background-color: #FFFFFF !important; 
            color: {TEXT_COLOR} !important; 
            border-color: {RED_ACCENT} !important; 
            font-weight: 600;
        }}
        /* Маркеры: Стиль согласно скриншоту */
        [data-testid="stCheckbox"] svg {{ color: #000000 !important; }}
        div[data-testid="stRadio"] input:checked + div svg circle:first-child {{ stroke: {RED_ACCENT} !important; fill: {RED_ACCENT} !important; }}
        div[data-testid="stRadio"] input:not(:checked) + div svg circle:first-child {{ stroke: #000000 !important; fill: #FFFFFF !important; }}
        div[data-testid="stRadio"] input:not(:checked) + div svg circle:last-child {{ fill: #000000 !important; }}
        div[data-testid="stRadio"] input:checked + div svg circle:last-child {{ fill: #FFFFFF !important; }}

        /* ========================================================= */
        /* 5. СТИЛИЗАЦИЯ ПРАВОЙ ПАНЕЛИ (САЙДБАР) */
        /* ========================================================= */
        
        /* Фон для правой колонки (имитация сайдбара) */
        div[data-testid="column"]:nth-child(2) > div {{
            padding-left: 20px; /* Отступ от разделителя */
            padding-right: 20px;
            box-shadow: -2px 0 0 0 {MAROON_DIVIDER} inset; 
            /* Добавляем стили для прокрутки, если нужно (только на результаты) */
        }}

        /* Стилизация полей ввода в САЙДБАРЕ (ПРАВАЯ КОЛОНКА) */
        div[data-testid="column"]:nth-child(2) .stSelectbox > div:first-child,
        div[data-testid="column"]:nth-child(2) .stTextInput input,
        div[data-testid="column"]:nth-child(2) .stTextarea textarea
        {{
            background-color: {LIGHT_BG_SIDEBAR} !important; 
            border: none !important; 
            box-shadow: none !important;
            border-radius: 4px;
            padding: 10px 12px;
        }}
        
        /* Устранение лишних отступов в сайдбаре */
        div[data-testid="column"]:nth-child(2) .stSelectbox, 
        div[data-testid="column"]:nth-child(2) .stTextInput {{
            margin-bottom: 5px !important;
        }}
        
        /* Уменьшение отступа заголовков в сайдбаре */
        div[data-testid="column"]:nth-child(2) h6 {{
             margin-top: 0.5rem;
             margin-bottom: 0.5rem;
        }}
        
        /* Скрываем подписи в сайдбаре (они выглядят как лейблы) */
        div[data-testid="column"]:nth-child(2) .stCaption {{
            display: none;
        }}
        
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ЛОГИКА (БЭКЕНД - ВАШ РАБОЧИЙ КОД)
# ==========================================

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
        
        bm25_scores = []
        for i, d in enumerate(comp_docs):
            tf = c_body_tfs[i]
            dl = len(d['body'])
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_len)))
            bm25_scores.append(score)
        bm25_top = np.median(bm25_scores)
        
        bm25_my = 0
        if my_len > 0:
            bm25_my = idf * (my_tf * (k1 + 1)) / (my_tf + k1 * (1 - b + b * (my_len / avg_len)))
        
        target_body = int(med_tf * 1.3 * norm_k)
        diff_body = target_body - my_tf
        target_anch = int(med_anch * norm_k)
        diff_anch = target_anch - my_anch_tf
        
        if med_tf > 0.5 or my_tf > 0:
            table_depth.append({
                "Слово": word, "Словоформы": word, "Повторы у вас": my_tf, 
                "Минимум": np.min(c_body_tfs), "Максимум": int(max_tf * norm_k),
                "Общее Добавить/Убрать": diff_body,
                "Тег A у вас": my_anch_tf, "Тег A рекомендации": target_anch,
                "Тег A Добавить/Убрать": diff_anch,
                "Текст у вас": my_tf, "Текст рекомендации": target_body, "Текст Добавить/Убрать": diff_body,
                "Переспам": int(max_tf * norm_k), "Переспам*IDF": round(max_tf * norm_k * idf, 1),
                "diff_abs": abs(diff_body)
            })
            table_hybrid.append({
                "Слово": word, "TF-IDF ТОП": round(med_tf * idf, 2), "TF-IDF ваш сайт": round(my_tf * idf, 2),
                "BM25 ТОП": round(bm25_top, 2), "BM25 ваш сайт": round(bm25_my, 2), "IDF": round(idf, 2),
                "Кол-во сайтов": df, "Медиана": round(med_tf, 1), "Переспам": max_tf,
                "Среднее по ТОПу": round(np.mean(c_body_tfs) if c_body_tfs else 0, 1), "Ваш сайт": my_tf,
                "<a> по ТОПу": round(med_anch, 1), "<a> ваш сайт": my_anch_tf
            })

    table_ngrams = []
    if comp_docs:
        my_bi = process_text(my_data['body_text'], settings, 2) if my_data and 'body_text' in my_data else []
        comp_bi = [process_text(p['body_text'], settings, 2) for p in comp_data]
        all_bi = set(my_bi)
        for c in comp_bi: all_bi.update(c)
        bi_freqs = Counter()
        for c in comp_bi:
            for b_ in set(c): bi_freqs[b_] += 1

        for bg in all_bi:
            df = bi_freqs[bg]
            if df < 2 and bg not in my_bi: continue
            my_c = my_bi.count(bg)
            comp_c = [c.count(bg) for c in comp_docs if 'body' in c]
            med_c = np.median(comp_c) if comp_c else 0
            if med_c > 0 or my_c > 0:
                table_ngrams.append({
                    "N-грамма": bg, "Кол-во сайтов": df, "Медианное вхождение": med_c,
                    "Среднее": round(np.mean(comp_c) if comp_c else 0, 1), "На сайте": my_c,
                    "TF-IDF": round(my_c * math.log(N/df if df>0 else 1), 3)
                })

    table_rel = []
    for i, p in enumerate(comp_data):
        p_lemmas = process_text(p['body_text'], settings)
        w = len(set(p_lemmas).intersection(vocab))
        table_rel.append({
            "Домен": p['domain'], "Позиция": i+1, "URL": p['url'],
            "Ширина": w, "Глубина": len(p_lemmas)
        })
        
    return {
        "depth": pd.DataFrame(table_depth), "hybrid": pd.DataFrame(table_hybrid),
        "ngrams": pd.DataFrame(table_ngrams), "relevance_top": pd.DataFrame(table_rel),
        "my_score": {"width": len(set(my_lemmas).intersection(vocab)), "depth": len(my_lemmas)}
    }

# ==========================================
# 3. ИНТЕРФЕЙС (МАКЕТ С ДВУМЯ КОЛОНКАМИ)
# ==========================================

# --- КНОПКА ЗАПУСКА НА САМОМ ВЕРХУ ---
# Инициализация для корректного запуска
if 'last_button_click' not in st.session_state:
    st.session_state.last_button_click = ''
if 'start_analysis_flag' not in st.session_state:
    st.session_state.start_analysis_flag = False

if st.button("ЗАПУСТИТЬ АНАЛИЗ", type="primary", use_container_width=True, key="start_analysis_btn"):
    st.session_state.start_analysis_flag = True

# --- ОСНОВНОЙ МАКЕТ: ДВЕ КОЛОНКИ (65% / 35%) ---
col_main, col_sidebar = st.columns([65, 35]) 

# --- ЛЕВАЯ КОЛОНКА (Основной контент) ---
with col_main:
    st.title("SEO Анализатор Релевантности")

    # 1. URL или код страницы Вашего сайта
    st.markdown("### URL или код страницы Вашего сайта")
    my_input_type = st.radio(
        "Тип страницы", 
        ["Релевантная страница на вашем сайте", "Исходный код страницы или текст", "Без страницы"], 
        horizontal=True,
        label_visibility="collapsed",
        key="my_page_source_radio"
    )

    my_url = ""
    my_page_content = ""

    if my_input_type == "Релевантная страница на вашем сайте":
        my_url = st.text_input("URL страницы", placeholder="https://site.ru/", label_visibility="collapsed", key="my_url_input")
    elif my_input_type == "Исходный код страницы или текст":
        my_page_content = st.text_area("Исходный код или текст", height=200, label_visibility="collapsed", placeholder="Вставьте HTML-код или чистый текст страницы", key="my_content_input")
    elif my_input_type == "Без страницы":
        st.info("Выбран анализ без страницы вашего сайта.")

    # 2. Поисковой запрос
    st.markdown("### Поисковой запрос")
    query = st.text_input("Основной запрос", placeholder="Основной запрос", label_visibility="collapsed", key="query_input")
    st.checkbox("Дополнительные запросы", disabled=True, value=False)

    # 3. Поиск или URL страниц конкурентов
    st.markdown("### Поиск или URL страниц конкурентов")
    source_type_new = st.radio(
        "Источник конкурентов", 
        ["Поиск", "Список url-адресов ваших конкурентов"], 
        horizontal=True,
        label_visibility="collapsed",
        key="competitor_source_radio"
    )
    source_type = "Google (Авто)" if source_type_new == "Поиск" else "Ручной список" 

    # --- 4. Редактируемые списки (Левая колонка) ---
    st.markdown("### Редактируемые списки")

    # Не учитывать домены
    excludes = st.text_area("Не учитывать домены (каждый с новой строки)", DEFAULT_EXCLUDE, height=200, key="settings_excludes")
    st.caption("Домены, которые будут исключены из анализа конкурентов.")

    # Стоп-слова
    c_stops = st.text_area("Стоп-слова (каждое с новой строки)", DEFAULT_STOPS, height=200, key="settings_stops")
    st.caption("Слова, которые будут удалены перед лемматизацией.")

# --- ПРАВАЯ КОЛОНКА (Настройки) ---
with col_sidebar:
    st.markdown("##### ⚙️ Настройки")

    # --- Блок 1: Основные параметры (SELECTS/TEXT INPUTS) ---
    st.markdown("###### Основные параметры")
    
    # User-Agent
    ua = st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0"], key="settings_ua")
    st.caption("Определяет, как будет скачиваться страница.")
    
    # Поисковая система
    search_engine = st.selectbox("Поисковая система", ["Google", "Яндекс", "Яндекс + Google"], key="settings_search_engine")
    
    # Яндекс / Регион
    region = st.selectbox("Яндекс / Регион", REGIONS, key="settings_region")
    
    # Устройство
    device = st.selectbox("Устройство", ["Desktop", "Mobile"], key="settings_device")
    
    # Анализировать ТОП
    top_n = st.selectbox("Анализировать ТОП", [10, 20, 30], index=1, key="settings_top_n")

    # Учитывать тип страниц по url
    st.selectbox(
        "Учитывать тип страниц по url", 
        ["Все страницы", "Главные страницы", "Внутренние страницы"],
        key="settings_url_type"
    )

    # Учитывать тип
    st.selectbox(
        "Учитывать тип", 
        ["Все страницы", "Коммерческие", "Информационные"],
        key="settings_content_type"
    )
    
    # --- Блок 2: Редактируемые списки (Заглушки для сайдбара) ---
    st.markdown("###### Редактируемые списки")
    
    st.markdown("Не учитывать домены (каждый с новой строки)")
    st.text_area("Пустое поле для вида", value=DEFAULT_EXCLUDE[:100], height=100, label_visibility="collapsed", disabled=True)
    st.markdown("Стоп-слова (каждое с новой строки)")
    st.text_area("Пустое поле для вида", value=DEFAULT_STOPS[:50], height=100, label_visibility="collapsed", disabled=True)

    
    # --- Блок 3: Флажки ---
    st.markdown("###### Переключатели")
    
    col_check1_s, col_check2_s = st.columns(2)
    with col_check1_s:
        st.checkbox("Исключать noindex/script/style/head/footer/nav", True, key="settings_noindex")
        st.checkbox("Учитывать Alt/Title", False, key="settings_alt")
        st.checkbox("Учитывать числа (0-9)", False, key="settings_numbers")
    with col_check2_s:
        st.checkbox("Нормировать по длине (LSA/BM25)", True, key="settings_norm")
        st.checkbox("Исключать агрегаторы/маркетплейсы в поиске (дополнительно)", True, key="settings_agg")


# --- ЛОГИКА ЗАПУСКА (ВЫПОЛНЯЕТСЯ ПРИ НАЖАТИИ КНОПКИ) ---
if st.session_state.start_analysis_flag:
    st.session_state.start_analysis_flag = False

    # (Проверка ввода и сбор настроек)
    if my_input_type == "Релевантная страница на вашем сайте" and not st.session_state.get('my_url_input'):
        st.error("Введите URL!")
        st.stop()
        
    if my_input_type == "Исходный код страницы или текст" and not st.session_state.get('my_content_input', '').strip():
        st.error("Введите исходный код или текст!")
        st.stop()
    
    if source_type == "Google (Авто)" and st.session_state.settings_search_engine != "Google":
        st.warning(f"Анализ ТОП-а для **{st.session_state.settings_search_engine}** пока не реализован. Используется Google Search.")
        if not st.session_state.get('query_input'):
            st.error("Введите запрос для поиска конкурентов!")
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
            with st.spinner(f"Сбор ТОПа {st.session_state.settings_search_engine}..."):
                if not USE_SEARCH:
                    st.error("Библиотека 'googlesearch' не найдена. Невозможно выполнить автоматический поиск ТОПа.")
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
            st.error(f"Ошибка при поиске: {e}")
            st.stop()
    else: 
        # Ручной список (из рабочего text_area)
        target_urls = [u.strip() for u in st.session_state.settings_excludes.split('\n') if u.strip()]

    if not target_urls:
        st.error("Нет конкурентов для анализа.")
        st.stop()
        
    # --- ОБРАБОТКА ДАННЫХ ВАШЕГО САЙТА ---
    my_data = None
    if my_input_type == "Релевантная страница на вашем сайте":
        prog = st.progress(0.0)
        status = st.empty()
        s
