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
# 1. КОНФИГУРАЦИЯ (МИНИМАЛЬНЫЙ CSS ДЛЯ ЧИТАЕМОСТИ - ТЕМНАЯ ТЕМА)
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO", page_icon="📊")

# Обновленный CSS для принудительного использования ТЕМНОЙ темы и БЕЛОГО текста
st.markdown("""
   <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        /* 1. ОБЩИЕ НАСТРОЙКИ ДЛЯ ТЕМНОЙ ТЕМЫ */
        html, body, [class*="stApp"], [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #1E293B !important; /* Темный фон */
            color: #FFFFFF !important; /* Весь текст белый */
        }
        
        /* Заголовки, текст, лейблы - все белое */
        h1, h2, h3, p, label, span, div, a {
            color: #FFFFFF !important; 
        }

        /* 2. Настройка контейнеров и полей ввода для темной темы */
        /* Стилизация верхней панели ввода */
        .main-input-container {
            background-color: #334155 !important; /* Более темный для контраста */
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #475569 !important;
            margin-bottom: 20px;
        }
        
        /* Поля ввода (фон и текст) */
        .stTextInput input, .stTextArea textarea {
            color: #FFFFFF !important;
            background-color: #475569 !important; /* Еще более темный фон для полей */
            border: 1px solid #64748B !important;
        }

        /* Кнопка (оставить яркой, текст белый) */
        .stButton button {
            background-color: #F97316;
            color: white !important;
            font-weight: bold;
            border-radius: 6px;
            height: 50px;
            width: 100%;
        }
        .stButton button:hover { background-color: #EA580C; color: white !important; }
        
        /* 3. Исправление выпадающих списков и модальных окон */
        div[data-baseweb="popover"], div[data-baseweb="menu"], li, div[role="listbox"] {
            background-color: #334155 !important; /* Темный фон для элементов выбора */
            color: #FFFFFF !important; /* Белый текст */
        }
        
        /* Заголовки таблиц */
        .table-header { 
            font-size: 18px; 
            font-weight: bold; 
            margin-top: 30px; 
            margin-bottom: 10px; 
            color: #FFFFFF !important; /* Белый */
        }
        
        /* Уменьшение отступов (убираем пустоту) */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Streamlit DataFrame: настройка для отображения белого текста на темном фоне */
        .stDataFrame > div > div {
            color: #FFFFFF !important;
            background-color: #334155 !important;
        }
        /* Заголовки колонок */
        .stDataFrame th {
            color: #FFFFFF !important;
            background-color: #475569 !important;
        }
        /* Ячейки */
        .stDataFrame td {
            color: #FFFFFF !important;
            background-color: #334155 !important;
        }
        /* Текст в статусных сообщениях (success, error и т.д.) */
        div[data-testid="stAlert"] * {
            color: #FFFFFF !important;
        }
        
        /* Стилизация радио-кнопок для имитации вкладок */
        div[data-testid="stRadio"] label {
            background-color: #334155;
            border-radius: 6px;
            padding: 10px 15px;
            margin-right: 5px;
            color: #E2E8F0;
            border: 1px solid #475569;
            transition: all 0.2s;
        }
        div[data-testid="stRadio"] label:hover {
            background-color: #475569;
        }
        /* Выбранный элемент */
        div[data-testid="stRadio"] input:checked + div {
            background-color: #F97316; 
            color: #FFFFFF;
            border-color: #F97316;
        }
        /* Скрытие исходных радио-точек */
        div[data-testid="stRadio"] input[type="radio"] {
            display: none;
        }
        div[data-testid="stRadio"] div[data-testid="stHorizontalBlock"] {
            gap: 0.5rem; /* Уменьшение пробела между "вкладками" */
        }
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

DEFAULT_EXCLUDE = "yandex.ru avito.ru ozon.ru wildberries.ru youtube.com dzen.ru hh.ru t.me tiu.ru pulscen.ru satu.kz"
DEFAULT_STOPS = "рублей\nруб\nкупить\nцена\nшт\nсм\nмм\nкг\nкв\nм2\nстр\nул"

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
    # Если my_data пуст (случай "Без страницы"), вернуть пустые метрики или обработать иначе
    if not my_data or not my_data['body_text']:
        # В этом случае можно вернуть только таблицу релевантности ТОПа, но основные таблицы будут пусты.
        # Для упрощения я создам пустой набор лемм.
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
        mean_tf = np.mean(c_body_tfs)
        
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        bm25_scores = []
        for i, d in enumerate(comp_docs):
            tf = c_body_tfs[i]
            dl = len(d['body'])
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_len)))
            bm25_scores.append(score)
        bm25_top = np.median(bm25_scores)
        
        # BM25 для вашего сайта рассчитывается только если у вас есть контент
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
                "Среднее по ТОПу": round(mean_tf, 1), "Ваш сайт": my_tf,
                "<a> по ТОПу": round(med_anch, 1), "<a> ваш сайт": my_anch_tf
            })

    # Расчет N-грамм только если есть данные конкурентов
    table_ngrams = []
    if comp_docs:
        my_bi = process_text(my_data['body_text'], settings, 2) if my_data else []
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
            comp_c = [c.count(bg) for c in comp_bi]
            med_c = np.median(comp_c)
            if med_c > 0 or my_c > 0:
                table_ngrams.append({
                    "N-грамма": bg, "Кол-во сайтов": df, "Медианное вхождение": med_c,
                    "Среднее": round(np.mean(comp_c), 1), "На сайте": my_c,
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
# 3. ИНТЕРФЕЙС (ИСПРАВЛЕННЫЙ)
# ==========================================

st.title("SEO Анализатор Релевантности")

# --- БЛОК ВВОДА (КОПИРОВАНИЕ ФУНКЦИОНАЛА СКРИНШОТОВ) ---

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
    my_url = st.text_input("URL страницы", placeholder="https://site.ru/", label_visibility="collapsed")
elif my_input_type == "Исходный код страницы или текст":
    my_page_content = st.text_area("Исходный код или текст", height=200, label_visibility="collapsed", placeholder="Вставьте HTML-код или чистый текст страницы")
elif my_input_type == "Без страницы":
    st.info("Выбран анализ без страницы вашего сайта. Результаты будут включать только метрики ТОПа.")

# 2. Поисковой запрос
st.markdown("### Поисковой запрос")
query = st.text_input("Основной запрос", placeholder="Основной запрос", label_visibility="collapsed")
st.checkbox("Дополнительные запросы", disabled=True, value=False) # Имитация чекбокса

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

# --- ИСТОЧНИК КОНКУРЕНТОВ: ДЕТАЛИ ---
if source_type == "Google (Авто)":
    # Две колонки: Глубина (узкая) и Исключения (широкая)
    cl1, cl2 = st.columns([1, 4])
    with cl1:
        top_n = st.selectbox("Глубина ТОПа", [5, 10, 20], index=1)
    with cl2:
        # Увеличенная высота для соответствия полю Стоп-слов
        excludes = st.text_area("Исключить домены (через пробел)", DEFAULT_EXCLUDE, height=150) 
else:
    manual_urls = st.text_area("Список URL (каждый с новой строки)", height=150)

# --- 4. НАСТРОЙКИ (ПОСТОЯННО ОТКРЫТЫЙ БЛОК) ---
st.markdown("##### ⚙️ Настройки")
# Удален st.expander, чтобы сделать блок постоянно открытым. 
# Используется st.container для сохранения структуры.
with st.container():
    col_set1, col_set2 = st.columns([1, 1])
    
    with col_set1:
        ua = st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0"])
        # ЗАМЕНА на text_area, теперь можно писать с новой строки
        c_stops = st.text_area("Стоп-слова (каждое с новой строки)", DEFAULT_STOPS, height=150)
        
    with col_set2:
        st.write("") # Отступ
        s_noindex = st.checkbox("Исключать noindex", True)
        s_alt = st.checkbox("Учитывать Alt/Title", False)
        s_num = st.checkbox("Учитывать числа", False)
        s_norm = st.checkbox("Нормировать по длине", True)
        s_agg = st.checkbox("Исключать агрегаторы", True)

# Кнопка запуска
if st.button("ЗАПУСТИТЬ АНАЛИЗ", type="primary", use_container_width=True):
    
    if my_input_type == "Релевантная страница на вашем сайте" and not my_url:
        st.error("Введите URL!")
        st.stop()
        
    if my_input_type == "Исходный код страницы или текст" and not my_page_content.strip():
        st.error("Введите исходный код или текст!")
        st.stop()

    settings = {
        'noindex': s_noindex, 'alt_title': s_alt, 'numbers': s_num,
        'norm': s_norm, 'ua': ua, 'custom_stops': c_stops.split()
    }
    
    target_urls = []
    if source_type == "Google (Авто)":
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
                    # Проверка на свой URL только если используется режим "Релевантная страница"
                    if my_input_type == "Релевантная страница на вашем сайте" and my_url in u: continue
                    if any(x in u for x in excl): continue
                    target_urls.append(u)
                    cnt += 1
                    if cnt >= top_n: break
        except Exception as e:
            st.error(f"Ошибка поиска: {e}")
            st.stop()
    else: # Ручной список
        manual_urls = st.session_state.get('manual_urls_text', '').split('\n')
        target_urls = [u.strip() for u in manual_urls if u.strip()]
        
    if not target_urls:
        st.error("Нет конкурентов.")
        st.stop()
        
    # --- ОБРАБОТКА ДАННЫХ ВАШЕГО САЙТА ---
    my_data = None
    if my_input_type == "Релевантная страница на вашем сайте":
        prog = st.progress(0.0)
        status = st.empty()
        status.text("Скачиваем ваш сайт...")
        my_data = parse_page(my_url, settings)
        prog.progress(0.05)
        if not my_data:
            st.error("Ошибка доступа к сайту. Проверьте URL или попробуйте 'Исходный код'.")
            st.stop()
        prog.empty()
        status.empty()
    elif my_input_type == "Исходный код страницы или текст":
        my_data = {
            'url': 'Local Content', 
            'domain': 'local.content', 
            'body_text': my_page_content, 
            'anchor_text': '' # Якоря не могут быть получены из чистого текста/кода
        }
    
    # --- СКАЧИВАНИЕ КОНКУРЕНТОВ ---
    comp_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(parse_page, u, settings): u for u in target_urls}
        done = 0
        total_tasks = len(target_urls)
        prog_comp = st.progress(0)
        status_comp = st.empty()
        
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: comp_data.append(res)
            done += 1
            prog_comp.progress(done / total_tasks)
            status_comp.text(f"Скачано {done} из {total_tasks} конкурентов...")
            
    prog_comp.empty()
    status_comp.empty()
    
    if len(comp_data) < 2 and my_input_type != "Без страницы":
        st.warning("Мало данных конкурентов для надежного анализа (менее 2). Продолжаю только с доступными данными.")

    if not my_data and my_input_type != "Без страницы":
         st.error("Не удалось получить данные для сравнения.")
         st.stop()

    results = calculate_metrics(comp_data, my_data, settings)
    st.success("Готово!")
    
    # 4. РЕЗУЛЬТАТЫ (С ПАГИНАЦИЕЙ)
    
    if my_data:
        st.markdown("### 1. Рекомендации по глубине")
        df_d = results['depth']
        if not df_d.empty:
            df_d = df_d.sort_values(by="diff_abs", ascending=False)
            
            # --- ЛОГИКА ПАГИНАЦИИ ---
            rows_per_page = 20
            total_rows = len(df_d)
            total_pages = math.ceil(total_rows / rows_per_page)
            
            if 'page_number' not in st.session_state:
                st.session_state.page_number = 1
            
            # Сброс пагинации при новом запуске
            if st.session_state.get('run_started', False) is True:
                 st.session_state.page_number = 1

            col_p1, col_p2, col_p3 = st.columns([1, 3, 1])
            with col_p1:
                if st.button("⬅️ Назад", key="prev_page_button") and st.session_state.page_number > 1:
                    st.session_state.page_number -= 1
            with col_p2:
                st.markdown(f"<div style='text-align: center; padding-top: 10px;'>Страница <b>{st.session_state.page_number}</b> из {total_pages}</div>", unsafe_allow_html=True)
            with col_p3:
                if st.button("Вперед ➡️", key="next_page_button") and st.session_state.page_number < total_pages:
                    st.session_state.page_number += 1
                    
            start_idx = (st.session_state.page_number - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            df_page = df_d.iloc[start_idx:end_idx]
            
            st.dataframe(
                df_page,
                column_config={"diff_abs": None}, 
                use_container_width=True, 
                height=800
            )
            st.download_button("Скачать ВСЮ таблицу (CSV)", df_d.to_csv().encode('utf-8'), "depth.csv")
            
            with st.expander("2. Гибридный ТОП"):
                st.dataframe(results['hybrid'].sort_values(by="TF-IDF ТОП", ascending=False), use_container_width=True)
                
            with st.expander("3. N-граммы"):
                st.dataframe(results['ngrams'].sort_values(by="TF-IDF", ascending=False), use_container_width=True)

    
    with st.expander("4. ТОП релевантности"):
        st.dataframe(results['relevance_top'], use_container_width=True)

    if not my_data:
        st.warning("Основные таблицы (Рекомендации, Гибридный ТОП, N-граммы) не отображаются, так как был выбран режим 'Без страницы'.")
