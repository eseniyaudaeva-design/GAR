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
# 1. СТИЛИЗАЦИЯ (Светлая тема + UI как на скрине)
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO: SEO Analysis", page_icon="📈")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Общий фон и шрифт */
        .stApp {
            background-color: #F3F4F6;
            font-family: 'Inter', sans-serif;
            color: #1F2937;
        }
        
        /* Блоки ввода (Карточки) */
        .input-card {
            background-color: #FFFFFF;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #E5E7EB;
        }
        
        /* Заголовки */
        h1, h2, h3 {
            color: #111827;
            font-weight: 700;
        }
        
        /* Поля ввода */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            background-color: #F9FAFB;
            border: 1px solid #D1D5DB;
            border-radius: 6px;
            color: #111827;
        }
        
        /* Кнопка (Синяя, как на скрине) */
        div.stButton > button {
            background-color: #1D4ED8; /* Ярко-синий */
            color: white;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            padding: 12px 24px;
            width: 100%;
            font-size: 16px;
            transition: all 0.2s;
        }
        div.stButton > button:hover {
            background-color: #1E40AF;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Таблицы */
        div[data-testid="stDataFrame"] {
            background-color: white;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #E5E7EB;
        }
        
        /* Убираем лишние отступы */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
        }
        
        /* Expander (Настройки) */
        .streamlit-expanderHeader {
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E5E7EB;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ЯДРО (БЭКЕНД)
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

# --- Константы ---
DEFAULT_EXCLUDE = ["yandex.ru", "avito.ru", "ozon.ru", "wildberries.ru", "youtube.com", "dzen.ru", "hh.ru", "t.me", "tiu.ru", "pulscen.ru", "satu.kz"]
DEFAULT_STOPS = ["рублей", "руб", "купить", "цена", "шт", "см", "мм", "кг", "кв", "м2", "стр", "ул", "доставка", "звоните", "заказать"]

# --- Функции Парсинга и NLP ---

def process_text(text, settings, n_gram=1):
    """Возвращает список лемм или n-грамм"""
    # 1. Токенизация
    pattern = r'[а-яА-ЯёЁ0-9a-zA-Z]+' if settings['numbers'] else r'[а-яА-ЯёЁa-zA-Z]+'
    words = re.findall(pattern, text.lower())
    
    # 2. Стоп-слова
    stops = set(w.lower() for w in settings['custom_stops'])
    clean_words = []
    
    for w in words:
        if len(w) < 2 or w in stops: continue
        
        lemma = w
        if USE_NLP and n_gram == 1: # Лемматизируем только для униграм
            p = morph.parse(w)[0]
            # Фильтр частей речи
            if 'PREP' in p.tag or 'CONJ' in p.tag or 'PRCL' in p.tag or 'NPRO' in p.tag:
                continue
            lemma = p.normal_form
        
        clean_words.append(lemma)
    
    # 3. Генерация N-грамм (если нужно)
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
        
        # Мета-теги
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        desc = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc: desc = meta_desc.get("content", "").strip()
        h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        
        # Удаление мусора (noindex и скрипты)
        if settings['noindex']:
            for t in soup.find_all(['noindex', 'script', 'style', 'head', 'footer', 'nav']): t.decompose()
        else:
            for t in soup(['script', 'style', 'head']): t.decompose()
            
        # Анкоры (текст ссылок)
        anchors_list = []
        for a in soup.find_all('a'):
            txt = a.get_text(strip=True)
            if txt: anchors_list.append(txt)
        anchor_text = " ".join(anchors_list)
        
        # Текст (Body) - добавляем alt и title
        extra_text = []
        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): extra_text.append(img['alt'])
            for t in soup.find_all(title=True): extra_text.append(t['title'])
            
        body_text = soup.get_text(separator=' ') + " " + " ".join(extra_text)
        
        return {
            'url': url,
            'domain': urlparse(url).netloc,
            'title': title,
            'desc': desc,
            'h1': h1,
            'body_text': body_text,
            'anchor_text': anchor_text,
            'full_text': body_text + " " + anchor_text # Для общего анализа
        }
    except:
        return None

# --- Математика (TF-IDF, BM25) ---
def calculate_advanced_metrics(corpus_pages, my_page, settings):
    
    # 1. Подготовка данных (Униграммы)
    my_lemmas = process_text(my_page['body_text'], settings)
    my_anchors = process_text(my_page['anchor_text'], settings)
    
    comp_docs = []
    for p in corpus_pages:
        body = process_text(p['body_text'], settings)
        anchor = process_text(p['anchor_text'], settings)
        comp_docs.append({'body': body, 'anchor': anchor, 'full': body + anchor})
        
    # Нормировка
    avg_len = np.mean([len(d['body']) for d in comp_docs])
    my_len = len(my_lemmas)
    norm_k = (my_len / avg_len) if (settings['norm'] and avg_len > 0) else 1.0
    
    # Словарь
    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    vocab = sorted(list(vocab))
    
    # --- БЛОК 1: Основные метрики (BM25, IDF) ---
    N = len(comp_docs)
    doc_freqs = Counter()
    for d in comp_docs:
        for w in set(d['body']): doc_freqs[w] += 1
        
    k1, b = 1.2, 0.75
    
    # --- Генерация ТАБЛИЦ ---
    
    # Таблица 1: Глубина (Actionable)
    table_depth = []
    # Таблица 2: Гибридный ТОП (Аналитика)
    table_hybrid = []
    
    for word in vocab:
        df = doc_freqs[word]
        if df < 2 and word not in my_lemmas: continue # Отсекаем редкий шум
        
        # Счетчики
        my_tf = my_lemmas.count(word)
        my_anch_tf = my_anchors.count(word)
        
        comp_tfs = [d['body'].count(word) for d in comp_docs]
        comp_anch_tfs = [d['anchor'].count(word) for d in comp_docs]
        
        med_tf = np.median(comp_tfs)
        mean_tf = np.mean(comp_tfs)
        max_tf = np.max(comp_tfs)
        med_anch = np.median(comp_anch_tfs)
        
        # IDF
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        # BM25 для Топа (медиана)
        bm25_scores = []
        for i, d in enumerate(comp_docs):
            tf = comp_tfs[i]
            dl = len(d['body'])
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_len)))
            bm25_scores.append(score)
        bm25_top = np.median(bm25_scores)
        
        # BM25 My
        bm25_my = idf * (my_tf * (k1 + 1)) / (my_tf + k1 * (1 - b + b * (my_len / avg_len)))
        
        # Рекомендации
        target_body = int(med_tf * 1.3 * norm_k)
        diff_body = target_body - my_tf
        
        # Заполнение таблиц
        if med_tf > 0.5 or my_tf > 0:
            # 1. Глубина
            table_depth.append({
                "Слово": word,
                "Повторы у вас": my_tf,
                "Общее Добавить/Убрать": diff_body,
                "Тег A у вас": my_anch_tf,
                "Тег A рекомендации": int(med_anch * norm_k),
                "Тег A Добавить/Убрать": int(med_anch * norm_k) - my_anch_tf,
                "Текст у вас": my_tf,
                "Текст рекомендации": target_body,
                "Текст Добавить/Убрать": diff_body,
                "Переспам": int(max_tf * norm_k),
                "Переспам*IDF": round(max_tf * norm_k * idf, 1),
                "diff_abs": abs(diff_body)
            })
            
            # 2. Гибридный ТОП
            table_hybrid.append({
                "Слово": word,
                "TF-IDF ТОП": round(med_tf * idf, 2),
                "TF-IDF ваш сайт": round(my_tf * idf, 2),
                "BM25 ТОП": round(bm25_top, 2),
                "BM25 ваш сайт": round(bm25_my, 2),
                "IDF": round(idf, 2),
                "Кол-во сайтов": df,
                "Медиана": round(med_tf, 1),
                "Переспам": max_tf,
                "Среднее по ТОПу": round(mean_tf, 1),
                "Ваш сайт": my_tf
            })

    # Таблица 3: N-граммы (Биграммы)
    my_bigrams = process_text(my_page['body_text'], settings, n_gram=2)
    comp_bigrams_list = [process_text(p['body_text'], settings, n_gram=2) for p in corpus_pages]
    
    all_bigrams = set(my_bigrams)
    for cb in comp_bigrams_list: all_bigrams.update(cb)
    
    # Считаем DF для биграмм
    bg_freqs = Counter()
    for cb in comp_bigrams_list:
        for bg in set(cb): bg_freqs[bg] += 1
        
    table_ngrams = []
    for bg in all_bigrams:
        df = bg_freqs[bg]
        if df < 2 and bg not in my_bigrams: continue
        
        my_cnt = my_bigrams.count(bg)
        comp_cnts = [cb.count(bg) for cb in comp_bigrams_list]
        med_cnt = np.median(comp_cnts)
        
        if med_cnt > 0 or my_cnt > 0:
            table_ngrams.append({
                "N-грамма": bg,
                "Кол-во сайтов": df,
                "Медиана": med_cnt,
                "Среднее": round(np.mean(comp_cnts), 1),
                "На вашем сайте": my_cnt,
                "TF-IDF": round(my_cnt * math.log(N/df if df>0 else 1), 3)
            })

    # Таблица 4: ТОП Релевантности (Сводная по конкурентам)
    table_relevance = []
    for i, p in enumerate(corpus_pages):
        p_lemmas = process_text(p['body_text'], settings)
        # Ширина (сколько слов из общего словаря есть на странице)
        common_words = set(p_lemmas).intersection(vocab)
        width = len(common_words)
        depth = len(p_lemmas)
        
        table_relevance.append({
            "Домен": p['domain'],
            "Позиция": i+1,
            "Ширина (Слов из ядра)": width,
            "Глубина (Всего слов)": depth,
            "Общая": width + (depth / 100) # Условный скор
        })
        
    # Оценка моего сайта
    my_width = len(set(my_lemmas).intersection(vocab))
    my_depth = len(my_lemmas)
    
    return {
        "depth": pd.DataFrame(table_depth),
        "hybrid": pd.DataFrame(table_hybrid),
        "ngrams": pd.DataFrame(table_ngrams),
        "relevance_top": pd.DataFrame(table_relevance),
        "my_score": {"width": my_width, "depth": my_depth}
    }

# ==========================================
# 3. ИНТЕРФЕЙС (FRONTEND)
# ==========================================

st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>SEO Анализатор Релевантности</h1>", unsafe_allow_html=True)

# --- ВЕРХНИЙ БЛОК (ВСЕГДА ВИДЕН) ---
with st.container():
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        my_url = st.text_input("Ваш URL (Обязательно)", placeholder="https://site.ru/catalog")
    with c2:
        query = st.text_input("Поисковой запрос", placeholder="пластиковые окна цена")
    st.markdown('</div>', unsafe_allow_html=True)

# --- ИСТОЧНИК И НАСТРОЙКИ ---
col_L, col_R = st.columns([2, 1])

with col_L:
    st.markdown("### 🕵️ Источник данных")
    source_type = st.radio("Тип сбора:", ["Google Поиск (Авто)", "Список URL вручную"], horizontal=True, label_visibility="collapsed")
    
    if source_type == "Google Поиск (Авто)":
        cl1, cl2 = st.columns(2)
        with cl1:
            top_n = st.selectbox("Глубина ТОПа:", [5, 10, 20], index=1)
        with cl2:
            excludes = st.text_input("Исключить домены:", " ".join(DEFAULT_EXCLUDE))
        st.caption("Поиск эмулируется. Для точности используйте ручной список.")
    else:
        manual_urls = st.text_area("URLs конкурентов (с новой строки):", height=120)

with col_R:
    st.markdown("### ⚙️ Расширенные настройки")
    with st.container():
        # Используем тогглы для современного вида
        s_noindex = st.toggle("Исключать noindex", True)
        s_alt = st.toggle("Включать alt и title", False)
        s_num = st.toggle("Обрабатывать цифры", False)
        s_norm = st.toggle("Нормировать по длине", True)
        s_agg = st.toggle("Исключать агрегаторы", True)
    
    with st.expander("Стоп-слова и User-Agent"):
        ua = st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0", "Googlebot/2.1"])
        c_stops = st.text_area("Доп. стоп-слова:", "\n".join(DEFAULT_STOPS), height=80)

# --- КНОПКА ЗАПУСКА ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Запустить анализ 🚀"):
    
    if not my_url:
        st.error("Укажите URL вашего сайта!")
        st.stop()
        
    # Сбор настроек
    settings = {
        'noindex': s_noindex, 'alt_title': s_alt, 'numbers': s_num,
        'norm': s_norm, 'ua': ua, 'custom_stops': c_stops.split(),
        'std_stops': True # Всегда вкл
    }
    
    # 1. Получение списка URL
    target_urls = []
    if source_type == "Google Поиск (Авто)":
        if not query:
            st.error("Введите запрос!")
            st.stop()
        try:
            excl = excludes.split()
            if s_agg: excl.extend(["avito", "ozon", "wildberries", "market", "tiu"])
            
            with st.spinner("Сбор ТОПа..."):
                found = search(query, num_results=top_n*2, lang="ru")
                cnt = 0
                for u in found:
                    if my_url in u: continue
                    if any(x in u for x in excl): continue
                    target_urls.append(u)
                    cnt += 1
                    if cnt >= top_n: break
        except Exception as e:
            st.error(f"Ошибка поиска: {e}")
            st.stop()
    else:
        target_urls = [u.strip() for u in manual_urls.split('\n') if u.strip()]
        
    if not target_urls:
        st.error("Нет конкурентов для анализа.")
        st.stop()
        
    # 2. Парсинг
    progress_bar = st.progress(0)
    status_txt = st.empty()
    
    # Мой сайт
    status_txt.text(f"Скачиваем ваш сайт: {my_url}...")
    my_page_data = parse_page(my_url, settings)
    
    if not my_page_data:
        st.error("Не удалось скачать ваш сайт.")
        st.stop()
        
    # Конкуренты
    comp_pages_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(parse_page, u, settings): u for u in target_urls}
        done = 0
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: comp_pages_data.append(res)
            done += 1
            progress_bar.progress(done / len(target_urls))
            status_txt.text(f"Обработано {done}/{len(target_urls)}...")
            
    progress_bar.empty()
    status_txt.empty()
    
    if len(comp_pages_data) < 2:
        st.error("Слишком мало данных от конкурентов.")
        st.stop()
        
    # 3. Расчеты
    results = calculate_advanced_metrics(comp_pages_data, my_page_data, settings)
    
    # 4. ВЫВОД РЕЗУЛЬТАТОВ
    st.success("Анализ завершен!")
    
    # Метрики сайта
    st.markdown("### 🏆 Релевантность вашего сайта")
    m1, m2, m3 = st.columns(3)
    m1.metric("Ширина (Охват слов)", results['my_score']['width'])
    m2.metric("Глубина (Всего слов)", results['my_score']['depth'])
    m3.metric("Конкурентов в анализе", len(comp_pages_data))
    
    st.divider()
    
    # ТАБЛИЦА 1: РЕКОМЕНДАЦИИ (ГЛУБИНА)
    st.subheader("1. Рекомендации по глубине (LSI)")
    df_depth = results['depth']
    if not df_depth.empty:
        df_depth = df_depth.sort_values(by="diff_abs", ascending=False)
        
        def color_table(val):
            if isinstance(val, (int, float)):
                if val > 0: return 'background-color: #dcfce7; color: #166534'
                if val < 0: return 'background-color: #fee2e2; color: #991b1b'
            return ''
            
        st.dataframe(
            df_depth.style.map(color_table, subset=['Общее Добавить/Убрать', 'Тег A Добавить/Убрать', 'Текст Добавить/Убрать']),
            column_config={"diff_abs": None},
            use_container_width=True,
            height=500
        )
        # CSV Download
        st.download_button("Скачать (CSV)", df_depth.to_csv().encode('utf-8'), "depth_recommendations.csv")
    else:
        st.info("Нет рекомендаций.")

    # ТАБЛИЦА 2: ГИБРИДНЫЙ ТОП
    with st.expander("2. Гибридный ТОП униграм на основе конкурентов", expanded=False):
        st.dataframe(results['hybrid'].sort_values(by="TF-IDF ТОП", ascending=False), use_container_width=True)

    # ТАБЛИЦА 3: N-ГРАММЫ
    with st.expander("3. N-граммы (Биграммы)", expanded=False):
        st.dataframe(results['ngrams'].sort_values(by="TF-IDF", ascending=False), use_container_width=True)
        
    # ТАБЛИЦА 4: ТОП РЕЛЕВАНТНОСТИ
    with st.expander("4. ТОП релевантности документов (Сводная)", expanded=False):
        st.dataframe(results['relevance_top'], use_container_width=True)
