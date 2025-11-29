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
# 1. КОНФИГУРАЦИЯ И СТИЛИ
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="ГАР PRO: Анализ", 
    page_icon="📊",
    initial_sidebar_state="collapsed"
)

# Глобальные переменные
DEFAULT_EXCLUDE = ["yandex.ru", "avito.ru", "ozon.ru", "wildberries.ru", "youtube.com", "dzen.ru", "hh.ru", "t.me"]
DEFAULT_STOPS = ["рублей", "руб", "купить", "цена", "шт", "см", "мм", "кг", "кв", "м2", "стр", "ул"]

# ЧИСТЫЙ БЕЛЫЙ ДИЗАЙН С СИНИМИ АКЦЕНТАМИ
st.markdown("""
    <style>
        /* ПОЛНОСТЬЮ БЕЛЫЙ ФОН */
        .stApp {
            background: #ffffff !important;
            color: #333333 !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* ГРАДИЕНТНЫЕ ЗАГОЛОВКИ */
        .gradient-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            font-weight: 700 !important;
            margin-bottom: 1rem !important;
        }
        
        .gradient-subheader {
            background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            font-weight: 600 !important;
        }
        
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            text-align: center !important;
            margin-bottom: 2rem !important;
        }
        
        h2 {
            background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            font-weight: 600 !important;
            font-size: 1.5rem !important;
            margin: 1.5rem 0 1rem 0 !important;
        }
        
        h3 {
            color: #2c5282 !important;
            font-weight: 600 !important;
            font-size: 1.2rem !important;
        }
        
        /* ОСНОВНЫЕ ЭЛЕМЕНТЫ */
        .main .block-container {
            background: #ffffff !important;
            padding-top: 2rem !important;
        }
        
        p, div, span, label {
            color: #333333 !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* КАРТОЧКИ И КОНТЕЙНЕРЫ */
        .main-input-container {
            background: #ffffff !important;
            padding: 25px !important;
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
            margin-bottom: 25px !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
        }
        
        /* КНОПКИ С ГРАДИЕНТОМ */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            height: 55px !important;
            width: 100% !important;
            border: none !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
            color: white !important;
        }
        
        /* ТЕКСТОВЫЕ ПОЛЯ */
        .stTextInput input, .stTextArea textarea {
            background-color: #ffffff !important;
            color: #333333 !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 14px !important;
        }
        
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* РАДИО КНОПКИ */
        .stRadio > div {
            background-color: #ffffff !important;
            padding: 15px !important;
            border-radius: 10px !important;
            border: 2px solid #e2e8f0 !important;
            margin-bottom: 10px !important;
        }
        
        .stRadio label {
            color: #333333 !important;
            font-weight: 500 !important;
        }
        
        /* СЕЛЕКТЫ */
        .stSelectbox select {
            background-color: #ffffff !important;
            color: #333333 !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 8px !important;
        }
        
        /* ЧЕКБОКСЫ */
        .stCheckbox {
            color: #333333 !important;
        }
        
        .stCheckbox > label {
            color: #2c5282 !important;
            font-weight: 500 !important;
        }
        
        /* EXPANDER */
        .streamlit-expanderHeader {
            background: #ffffff !important;
            color: #2c5282 !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 15px !important;
        }
        
        .streamlit-expanderContent {
            background-color: #ffffff !important;
            border-radius: 0 0 8px 8px !important;
            border: 2px solid #e2e8f0 !important;
            border-top: none !important;
        }
        
        /* ПРОГРЕСС БАР */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        /* ТАБЛИЦЫ */
        .dataframe {
            border-radius: 8px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        }
        
        /* DIVIDER */
        hr {
            border-color: #e2e8f0 !important;
            margin: 2rem 0 !important;
        }
        
        /* SPINNER */
        .stSpinner > div {
            border-color: #667eea !important;
        }
        
        /* ALERTS */
        .stAlert {
            border-radius: 8px !important;
            border: 1px solid !important;
        }
        
        /* LABELS */
        .stTextInput label, .stTextArea label, .stSelectbox label, .stNumberInput label {
            color: #2c5282 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* УБИРАЕМ ЛЮБЫЕ СЛЕДЫ ТЕМНОЙ ТЕМЫ */
        .css-1d391kg, .css-1lcbmhc, .css-1outwn7 {
            background-color: #ffffff !important;
        }
        
        /* УЛУЧШЕННЫЕ РАДИО КНОПКИ */
        .st-cc, .st-cd, .st-ce, .st-cf, .st-cg {
            color: #333333 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. АВТОРИЗАЦИЯ
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]: 
        return True
    
    st.markdown("""
        <div style='
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 80vh;
            background: #ffffff;
        '>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <div style='
                background: #ffffff; 
                padding: 40px; 
                border-radius: 15px; 
                border: 2px solid #e2e8f0; 
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                text-align: center;
            '>
                <h2 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 30px;'>🔐 Авторизация</h2>
        """, unsafe_allow_html=True)
        
        pwd = st.text_input("Пароль доступа", type="password", key="auth_password")
        
        if st.button("Войти", key="auth_btn"):
            if pwd == "admin123":
                st.session_state["password_correct"] = True
                st.rerun()
            else: 
                st.error("❌ Неверный пароль")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    return False

if not check_password(): 
    st.stop()

# ==========================================
# 3. БЭКЕНД (ЛОГИКА) - БЕЗ ИЗМЕНЕНИЙ
# ==========================================

# --- Патч NLP ---
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

# --- Поиск ---
try:
    from googlesearch import search
    USE_SEARCH = True
except:
    USE_SEARCH = False

# --- Парсинг ---
def get_domain(url):
    try: 
        return urlparse(url).netloc
    except: 
        return url

def parse_page(url, settings):
    headers = {'User-Agent': settings['ua']}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200: 
            return None
        
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Мета-теги
        title = soup.title.string if soup.title else ""
        desc = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc: 
            desc = meta_desc.get("content", "")
        h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        
        # Очистка
        if settings['noindex']:
            for t in soup.find_all(['noindex', 'script', 'style']): 
                t.decompose()
        else:
            for t in soup(['script', 'style']): 
                t.decompose()
            
        # Анкоры
        anchors_list = []
        for a in soup.find_all('a'):
            txt = a.get_text(strip=True)
            if txt: 
                anchors_list.append(txt)
        anchor_text = " ".join(anchors_list)
        
        # Текст (Body)
        body_text = soup.get_text(separator=' ')
        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): 
                body_text += " " + img['alt']
            
        return {
            'url': url,
            'domain': get_domain(url),
            'title': title,
            'desc': desc,
            'h1': h1,
            'body_text': body_text,
            'anchor_text': anchor_text,
            'status': 200
        }
    except:
        return None

def process_lemmas(text, settings):
    # Токенизация
    pattern = r'[а-яА-ЯёЁ0-9a-zA-Z]+' if settings['numbers'] else r'[а-яА-ЯёЁa-zA-Z]+'
    words = re.findall(pattern, text)
    
    lemmas = []
    forms_map = {} # лемма -> список форм
    
    stops = set(w.lower() for w in settings['custom_stops'])
    
    for w in words:
        w_lower = w.lower()
        if len(w) < 2 or w_lower in stops: 
            continue
        
        lemma = w_lower
        if USE_NLP:
            p = morph.parse(w_lower)[0]
            if settings['std_stops'] and ('PREP' in p.tag or 'CONJ' in p.tag or 'PRCL' in p.tag):
                continue
            lemma = p.normal_form
            
        lemmas.append(lemma)
        
        if lemma not in forms_map: 
            forms_map[lemma] = set()
        forms_map[lemma].add(w_lower)
        
    return lemmas, forms_map

# ==========================================
# 4. ИНТЕРФЕЙС
# ==========================================

st.markdown('<h1>🎯 ГАР PRO: Анализатор Релевантности</h1>', unsafe_allow_html=True)

# ГЛАВНЫЙ БЛОК ВВОДА
with st.container():
    st.markdown('<div class="main-input-container">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        my_url = st.text_input(
            "Ваш URL (Обязательно)", 
            placeholder="https://mysite.ru/catalog/page",
            key="my_url"
        )
    with c2:
        query = st.text_input(
            "Поисковой запрос", 
            placeholder="купить товар москва",
            key="query"
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ИСТОЧНИК КОНКУРЕНТОВ
st.markdown('<h2>📊 Источник конкурентов</h2>', unsafe_allow_html=True)
source_mode = st.radio(
    "Выберите источник:",
    ["Google Поиск (Авто)", "Ручной список"], 
    horizontal=True, 
    key="source_mode"
)

competitors_final = []

if source_mode == "Google Поиск (Авто)":
    c_s1, c_s2 = st.columns([1, 3])
    with c_s1:
        top_count = st.selectbox("Анализировать ТОП:", [5, 10, 20], index=1, key="top_count")
    with c_s2:
        exclude_domains = st.text_input(
            "Исключить домены (через пробел)", 
            " ".join(DEFAULT_EXCLUDE),
            key="exclude_domains"
        )
else:
    manual_urls = st.text_area(
        "Список URL конкурентов (каждый с новой строки)", 
        height=150,
        key="manual_urls",
        placeholder="https://competitor1.com\nhttps://competitor2.com\nhttps://competitor3.com"
    )

# НАСТРОЙКИ
with st.expander("⚙️ Настройки анализа", expanded=True):
    col_set1, col_set2, col_set3 = st.columns(3)
    with col_set1:
        s_noindex = st.checkbox("Исключать noindex", True, key="s_noindex")
        s_alt = st.checkbox("Учитывать Alt/Title", False, key="s_alt")
    with col_set2:
        s_norm = st.checkbox("Нормировать по длине", True, key="s_norm")
        s_num = st.checkbox("Учитывать числа", False, key="s_num")
    with col_set3:
        s_std_stops = st.checkbox("Убирать предлоги", True, key="s_std_stops")
    
    custom_stops_text = st.text_area(
        "Стоп-слова (каждое с новой строки)", 
        "\n".join(DEFAULT_STOPS), 
        height=100,
        key="custom_stops"
    )
    user_agent = st.text_input(
        "User-Agent", 
        "Mozilla/5.0 (compatible; Hybrid-Analyzer/1.0;)",
        key="user_agent"
    )

# КНОПКА ЗАПУСКА
if st.button("🚀 ЗАПУСТИТЬ АНАЛИЗ", key="analyze_btn"):
    if not my_url:
        st.error("❌ Вы не ввели URL вашего сайта!")
        st.stop()
        
    settings = {
        'noindex': s_noindex, 
        'alt_title': s_alt, 
        'numbers': s_num,
        'norm': s_norm, 
        'std_stops': s_std_stops,
        'custom_stops': custom_stops_text.split(), 
        'ua': user_agent
    }
    
    # 1. Сбор URL
    target_urls = []
    if source_mode == "Google Поиск (Авто)":
        if not query:
            st.error("❌ Введите поисковый запрос!")
            st.stop()
        try:
            excl = exclude_domains.split()
            found = search(query, num_results=top_count*2, lang="ru")
            cnt = 0
            for u in found:
                if my_url in u: 
                    continue
                if any(x in u for x in excl): 
                    continue
                target_urls.append(u)
                cnt += 1
                if cnt >= top_count: 
                    break
        except Exception as e:
            st.error(f"❌ Ошибка поиска: {e}")
            st.stop()
    else:
        target_urls = [u.strip() for u in manual_urls.split('\n') if u.strip()]
        
    if not target_urls:
        st.error("❌ Список конкурентов пуст!")
        st.stop()
        
    # 2. Сбор данных
    all_pages_data = []
    
    # Сначала мой сайт
    with st.spinner("🔍 Анализ вашего сайта..."):
        my_page = parse_page(my_url, settings)
        if not my_page:
            st.error("❌ Ваш сайт недоступен!")
            st.stop()
            
    # Конкуренты
    progress_bar = st.progress(0)
    comp_pages = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(parse_page, u, settings): u for u in target_urls}
        done = 0
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: 
                comp_pages.append(res)
            done += 1
            progress_bar.progress(done / len(target_urls))
            
    if len(comp_pages) < 2:
        st.error("❌ Мало данных (нужно хотя бы 2 конкурента).")
        st.stop()
        
    # ==========================================
    # 5. МАТЕМАТИКА И ТАБЛИЦЫ
    # ==========================================
    
    # Лемматизация
    my_body_lemmas, my_body_forms = process_lemmas(my_page['body_text'], settings)
    my_anchor_lemmas, my_anchor_forms = process_lemmas(my_page['anchor_text'], settings)
    
    comp_stats = [] # Список словарей с леммами конкурентов
    for p in comp_pages:
        bl, _ = process_lemmas(p['body_text'], settings)
        al, _ = process_lemmas(p['anchor_text'], settings)
        comp_stats.append({'body': bl, 'anchor': al, 'len': len(bl)})
        
    # Нормировка
    avg_len = np.mean([c['len'] for c in comp_stats])
    my_len = len(my_body_lemmas)
    norm_k = (my_len / avg_len) if (settings['norm'] and avg_len > 0) else 1.0
    
    # Собираем все уникальные слова (Словарь)
    vocab = set(my_body_lemmas)
    for c in comp_stats: 
        vocab.update(c['body'])
    vocab = sorted(list(vocab))
    
    # Сборка главной таблицы (Рекомендации по глубине)
    rows = []
    
    # Для IDF
    N = len(comp_stats)
    doc_freqs = Counter()
    for c in comp_stats:
        for w in set(c['body']): 
            doc_freqs[w] += 1
    
    for word in vocab:
        # У меня
        my_body_tf = my_body_lemmas.count(word)
        my_anchor_tf = my_anchor_lemmas.count(word)
        
        # У конкурентов (массивы)
        c_body_tfs = [c['body'].count(word) for c in comp_stats]
        c_anchor_tfs = [c['anchor'].count(word) for c in comp_stats]
        
        # Статистика
        median_body = np.median(c_body_tfs)
        median_anchor = np.median(c_anchor_tfs)
        max_spam = np.max(c_body_tfs)
        
        # Целевые значения (с учетом нормировки)
        target_body = int(median_body * 1.3 * norm_k)
        target_anchor = int(median_anchor * 1.3 * norm_k)
        
        diff_body = target_body - my_body_tf
        diff_anchor = target_anchor - my_anchor_tf
        
        # IDF
        df = doc_freqs[word]
        idf = math.log((N / (df if df>0 else 1)) + 1)
        
        # Фильтр мусора (если слово есть только у одного или незначимо)
        if (median_body > 0.5 or my_body_tf > 0):
            # Сбор словоформ для отображения
            forms = []
            if word in my_body_forms: 
                forms.extend(my_body_forms[word])
            forms_str = ", ".join(list(set(forms))[:3])
            
            rows.append({
                "Слово": word,
                "Словоформы": forms_str,
                "Повторы у вас": my_body_tf,
                "Общее Добавить/Убрать": diff_body,
                
                "Тег A у вас": my_anchor_tf,
                "Тег A рекомендации": target_anchor,
                "Тег A Добавить/Убрать": diff_anchor,
                
                "Текст у вас": my_body_tf,
                "Текст рекомендации": target_body,
                "Текст Добавить/Убрать": diff_body,
                
                "Переспам": int(max_spam * norm_k),
                "Переспам*IDF": round(max_spam * norm_k * idf, 1),
                
                "diff_abs": abs(diff_body) # Скрытое поле для сортировки
            })
            
    df_main = pd.DataFrame(rows)
    
    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    
    st.divider()
    
    # 1. ТАБЛИЦА: РЕКОМЕНДАЦИИ ПО ГЛУБИНЕ
    st.markdown('<h2>📈 Рекомендации по глубине</h2>', unsafe_allow_html=True)
    if not df_main.empty:
        df_main = df_main.sort_values(by="diff_abs", ascending=False)
        
        # Стилизация (подсветка)
        def color_diff(val):
            if val > 0: 
                return 'background-color: #f0fff4; color: #22543d; border-left: 4px solid #38a169;'
            if val < 0: 
                return 'background-color: #fff5f5; color: #742a2a; border-left: 4px solid #e53e3e;'
            return ''
            
        st.dataframe(
            df_main.style.map(color_diff, subset=['Общее Добавить/Убрать', 'Тег A Добавить/Убрать', 'Текст Добавить/Убрать']),
            column_config={"diff_abs": None},
            use_container_width=True,
            height=600
        )
    else:
        st.warning("Нет данных.")

    # 2. ТАБЛИЦА: МЕТА-ТЕГИ
    st.markdown('<h2>🔍 Информация по мета-тегам</h2>', unsafe_allow_html=True)
    meta_data = []
    # Мой сайт
    meta_data.append({
        "Тип": "Ваш сайт",
        "Title": my_page['title'],
        "Description": my_page['desc'],
        "H1": my_page['h1']
    })
    # Конкуренты (первые 5)
    for i, p in enumerate(comp_pages[:5]):
        meta_data.append({
            "Тип": f"Конкурент {i+1} ({p['domain']})",
            "Title": p['title'],
            "Description": p['desc'],
            "H1": p['h1']
        })
    st.dataframe(pd.DataFrame(meta_data), use_container_width=True)

    # 3. ТАБЛИЦА: ТОП РЕЛЕВАНТНОСТИ ДОКУМЕНТОВ
    st.markdown('<h2>🏆 ТОП релевантности документов</h2>', unsafe_allow_html=True)
    top_rows = []
    for i, p in enumerate(comp_pages):
        # Простой расчет релевантности (кол-во слов из общего словаря)
        p_lemmas, _ = process_lemmas(p['body_text'], settings)
        coverage = len(set(p_lemmas).intersection(vocab))
        
        top_rows.append({
            "Домен": p['domain'],
            "Позиция": i+1,
            "URL": p['url'],
            "Ширина (Охват)": coverage,
            "Глубина (Всего слов)": len(p_lemmas)
        })
    st.dataframe(pd.DataFrame(top_rows), use_container_width=True)
