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
# 1. СТИЛИ (CSS) - CLEAN & MODERN UI
# ==========================================
st.set_page_config(layout="wide", page_title="SEO Анализатор", page_icon="🚀")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* 1. Фон страницы (Светлый серо-голубой) */
        .stApp {
            background-color: #F3F4F6;
            font-family: 'Inter', sans-serif;
            color: #1F2937;
        }
        
        /* 2. Контейнеры-карточки (Белые с тенью) */
        .css-card {
            background-color: #FFFFFF;
            padding: 25px;
            border-radius: 10px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        
        /* 3. Заголовки */
        h1, h2, h3, h4 {
            color: #111827 !important;
            font-weight: 700 !important;
            font-family: 'Inter', sans-serif;
        }
        
        /* 4. Поля ввода */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            background-color: #F9FAFB !important;
            border: 1px solid #D1D5DB !important;
            border-radius: 6px !important;
            color: #111827 !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 1px #2563EB !important;
        }
        
        /* 5. Кнопка (Ярко-синяя) */
        div.stButton > button {
            background-color: #1D4ED8 !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            padding: 0.75rem 1.5rem !important;
            border: none !important;
            width: 100%;
            transition: all 0.2s;
        }
        div.stButton > button:hover {
            background-color: #1E40AF !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* 6. Переключатели (Toggles) */
        label[data-testid="stLabel"] {
            font-weight: 500;
            color: #374151;
        }
        
        /* 7. Таблицы */
        div[data-testid="stDataFrame"] {
            background-color: white;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
        }
        
        /* Скрытие лишнего */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. БЭКЕНД (ЛОГИКА)
# ==========================================

# --- Константы ---
DEFAULT_EXCLUDE = ["yandex.ru", "avito.ru", "ozon.ru", "wildberries.ru", "youtube.com", "dzen.ru", "hh.ru", "t.me", "tiu.ru", "pulscen.ru", "satu.kz"]
DEFAULT_STOPS = ["рублей", "руб", "купить", "цена", "шт", "см", "мм", "кг", "кв", "м2", "стр", "ул", "доставка", "звоните", "заказать", "в", "на", "и", "с", "по", "к"]

# --- NLP ---
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

# --- Функции ---
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
        return [" ".join(clean_words[i:i+n_gram]) for i in range(len(clean_words) - n_gram + 1)]
    return clean_words

def parse_page(url, settings):
    headers = {'User-Agent': settings['ua']}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200: return None
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Meta
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        desc = soup.find("meta", attrs={"name": "description"})
        desc = desc.get("content", "").strip() if desc else ""
        h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        
        # Clean
        if settings['noindex']:
            for t in soup.find_all(['noindex', 'script', 'style', 'head', 'footer', 'nav']): t.decompose()
        else:
            for t in soup(['script', 'style', 'head']): t.decompose()
            
        # Anchor / Body
        anchors_list = [a.get_text(strip=True) for a in soup.find_all('a') if a.get_text(strip=True)]
        anchor_text = " ".join(anchors_list)
        
        extra = []
        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): extra.append(img['alt'])
            for t in soup.find_all(title=True): extra.append(t['title'])
        body_text = soup.get_text(separator=' ') + " " + " ".join(extra)
        
        return {
            'url': url, 'domain': urlparse(url).netloc, 'title': title, 'desc': desc, 'h1': h1,
            'body_text': body_text, 'anchor_text': anchor_text
        }
    except: return None

def calculate_metrics(comp_data, my_data, settings):
    # 1. Lemmas
    my_lemmas = process_text(my_data['body_text'], settings)
    my_anchors = process_text(my_data['anchor_text'], settings)
    
    comp_docs = []
    for p in comp_data:
        comp_docs.append({
            'body': process_text(p['body_text'], settings),
            'anchor': process_text(p['anchor_text'], settings)
        })
    
    # 2. Norm
    avg_len = np.mean([len(d['body']) for d in comp_docs])
    my_len = len(my_lemmas)
    norm_k = (my_len / avg_len) if (settings['norm'] and avg_len > 0) else 1.0
    
    # 3. Vocab
    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    vocab = sorted(list(vocab))
    
    # 4. Stats
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
        
        # BM25
        bm25_scores = []
        for i, d in enumerate(comp_docs):
            tf = c_body_tfs[i]
            dl = len(d['body'])
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_len)))
            bm25_scores.append(score)
        bm25_top = np.median(bm25_scores)
        bm25_my = idf * (my_tf * (k1 + 1)) / (my_tf + k1 * (1 - b + b * (my_len / avg_len)))
        
        # Recs
        target_body = int(med_tf * 1.3 * norm_k)
        diff_body = target_body - my_tf
        
        target_anch = int(med_anch * norm_k)
        diff_anch = target_anch - my_anch_tf
        
        if med_tf > 0.5 or my_tf > 0:
            # Таблица Глубины (Полная копия структуры)
            table_depth.append({
                "Слово": word,
                "Словоформы": word, # Упрощено для скорости
                "Повторы у вас": my_tf,
                "Минимум по реком.": 0, # Заглушка
                "Максимум по реком.": int(max_tf * norm_k),
                "Общее Добавить/Убрать": diff_body,
                "Тег A у вас": my_anch_tf,
                "Тег A рекомендации": target_anch,
                "Тег A Добавить/Убрать": diff_anch,
                "Текст у вас": my_tf,
                "Текст рекомендации": target_body,
                "Текст Добавить/Убрать": diff_body,
                "Переспам": int(max_tf * norm_k),
                "Переспам*IDF": round(max_tf * norm_k * idf, 1),
                "diff_abs": abs(diff_body)
            })
            
            # Гибридный ТОП
            table_hybrid.append({
                "Слово": word, "TF-IDF ТОП": round(med_tf * idf, 2), "TF-IDF ваш сайт": round(my_tf * idf, 2),
                "BM25 ТОП": round(bm25_top, 2), "BM25 ваш сайт": round(bm25_my, 2), "IDF": round(idf, 2),
                "Кол-во сайтов": df, "Медиана": round(med_tf, 1), "Переспам": max_tf,
                "Среднее по ТОПу (повт.)": round(mean_tf, 1), "Ваш сайт (повт.)": my_tf,
                "<a> по ТОПу (повт.)": round(med_anch, 1), "<a> ваш сайт (повт.)": my_anch_tf
            })
            
    # N-grams
    my_bi = process_text(my_data['body_text'], settings, 2)
    comp_bi = [process_text(p['body_text'], settings, 2) for p in comp_data]
    all_bi = set(my_bi)
    for c in comp_bi: all_bi.update(c)
    
    bi_freqs = Counter()
    for c in comp_bi:
        for b_ in set(c): bi_freqs[b_] += 1
        
    table_ngrams = []
    for bg in all_bi:
        df = bi_freqs[bg]
        if df < 2 and bg not in my_bi: continue
        my_c = my_bi.count(bg)
        comp_c = [c.count(bg) for c in comp_bi]
        med_c = np.median(comp_c)
        if med_c > 0 or my_c > 0:
            table_ngrams.append({
                "N-грамма": bg, "Кол-во сайтов": df, "Медианное вхождение": med_c,
                "Среднее": round(np.mean(comp_c), 1), "На нашем сайте": my_c,
                "IDF": round(math.log(N/df if df>0 else 1), 2),
                "TF-IDF": round(my_c * math.log(N/df if df>0 else 1), 3),
                "BM25": 0 # Заглушка
            })
            
    # Relevance Top
    table_rel = []
    for i, p in enumerate(comp_data):
        pl = process_text(p['body_text'], settings)
        w = len(set(pl).intersection(vocab))
        table_rel.append({
            "Домен": p['domain'], "Позиция": i+1, "ИКС": "-", "URL": p['url'],
            "Ширина": w, "Глубина": len(pl), "Общая": w + (len(pl)/100),
            "SEO": "-", "BM25": "-", "SWBM25": "-", "Общая * BM25": "-"
        })

    return {
        "depth": pd.DataFrame(table_depth),
        "hybrid": pd.DataFrame(table_hybrid),
        "ngrams": pd.DataFrame(table_ngrams),
        "relevance": pd.DataFrame(table_rel),
        "my_metrics": {"words": len(my_lemmas), "unique": len(set(my_lemmas))}
    }

# ==========================================
# 3. ИНТЕРФЕЙС (FRONTEND)
# ==========================================

st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>SEO Анализатор Релевантности</h1>", unsafe_allow_html=True)

# --- КАРТОЧКА 1: Анализ релевантности (Вкладки) ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.markdown("### 🔍 Анализ релевантности")

tab_url, tab_html, tab_comp_only = st.tabs(["🌐 По URL", "code HTML-код", "👥 Только конкуренты"])

my_url = ""
my_html = ""

with tab_url:
    my_url = st.text_input("URL вашей страницы", placeholder="https://site.ru/page")

with tab_html:
    my_html = st.text_area("Вставьте HTML код", height=100)

with tab_comp_only:
    st.info("Анализ будет проведен только по конкурентам (для сбора семантики).")

st.markdown('</div>', unsafe_allow_html=True)

# --- КАРТОЧКА 2: Конкуренты ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.markdown("### 🕵️ URL конкурентов")

# Выбор источника
col_src1, col_src2 = st.columns([1, 3])
with col_src1:
    source = st.radio("Источник:", ["Google (Авто)", "Свой список"], horizontal=True, label_visibility="collapsed")

urls_value = ""
if source == "Google (Авто)":
    c1, c2 = st.columns(2)
    with c1:
        query = st.text_input("Поисковой запрос", placeholder="пластиковые окна")
    with c2:
        top_n = st.selectbox("Глубина ТОПа", [10, 20, 30])
    excludes = st.text_input("Исключить домены", " ".join(DEFAULT_EXCLUDE))
else:
    urls_value = st.text_area("Список URL (каждый с новой строки)", height=150)

st.markdown('</div>', unsafe_allow_html=True)

# --- КАРТОЧКА 3: Настройки (Две колонки) ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.markdown("### ⚙️ Расширенные настройки")

col_set_left, col_set_right = st.columns(2)

with col_set_left:
    st.markdown("**Параметры парсинга**")
    ua = st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0", "Googlebot/2.1"])
    custom_stops = st.text_area("Дополнительные стоп-слова", "\n".join(DEFAULT_STOPS), height=120)

with col_set_right:
    st.markdown("**Фильтры**")
    # Используем st.toggle для красивых переключателей
    s_noindex = st.toggle("Исключать контент в noindex", True)
    s_alt = st.toggle("Включать alt и title", False)
    s_num = st.toggle("Обрабатывать цифры", False)
    s_norm = st.toggle("Нормировать по длине", True)
    s_agg = st.toggle("Исключать агрегаторы", True)

st.markdown('</div>', unsafe_allow_html=True)

# --- КНОПКА ---
if st.button("ЗАПУСТИТЬ АНАЛИЗ 🚀"):
    
    # Сбор настроек
    settings = {
        'noindex': s_noindex, 'alt_title': s_alt, 'numbers': s_num,
        'norm': s_norm, 'ua': ua, 'custom_stops': custom_stops.split()
    }
    
    # 1. Получение URL конкурентов
    target_urls = []
    if source == "Google (Авто)":
        if not query:
            st.error("Введите запрос!")
            st.stop()
        try:
            excl = excludes.split()
            if s_agg: excl.extend(["avito", "ozon", "wildberries", "tiu", "satu", "market"])
            with st.spinner("Парсинг Google..."):
                found = search(query, num_results=top_n*2, lang="ru")
                cnt = 0
                for u in found:
                    if my_url and my_url in u: continue
                    if any(x in u for x in excl): continue
                    target_urls.append(u)
                    cnt += 1
                    if cnt >= top_n: break
        except:
            st.error("Ошибка поиска Google. Используйте ручной список.")
            st.stop()
    else:
        target_urls = [u.strip() for u in urls_value.split('\n') if u.strip()]
        
    if not target_urls:
        st.error("Нет конкурентов.")
        st.stop()
        
    # 2. Сбор данных
    prog = st.progress(0)
    status = st.empty()
    
    # My Page
    my_page_data = {'body_text': "", 'anchor_text': ""}
    if my_url:
        status.text("Скачиваем ваш сайт...")
        d = parse_page(my_url, settings)
        if d: my_page_data = d
    elif my_html:
        # Simple HTML parse
        s = BeautifulSoup(my_html, 'html.parser')
        my_page_data['body_text'] = s.get_text(separator=' ')
        my_page_data['anchor_text'] = " ".join([a.get_text() for a in s.find_all('a')])
        
    # Comp Pages
    comp_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(parse_page, u, settings): u for u in target_urls}
        done = 0
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: comp_data.append(res)
            done += 1
            prog.progress(done / len(target_urls))
            status.text(f"Обработано {done}/{len(target_urls)}")
            
    prog.empty()
    status.empty()
    
    if len(comp_data) < 2:
        st.error("Мало данных конкурентов.")
        st.stop()
        
    # 3. Расчет
    results = calculate_metrics(comp_data, my_page_data, settings)
    
    # 4. Вывод
    st.markdown("## 📊 Результаты анализа")
    
    # Метрики
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего слов", results['my_metrics']['words'])
    c2.metric("Уникальных слов", results['my_metrics']['unique'])
    c3.metric("Конкурентов", len(comp_data))
    c4.metric("Средняя длина (ТОП)", int(np.mean([len(process_text(p['body_text'], settings)) for p in comp_data])))
    
    st.divider()
    
    st.subheader("1. Рекомендации по глубине")
    df_d = results['depth']
    if not df_d.empty:
        df_d = df_d.sort_values(by="diff_abs", ascending=False)
        def color(v):
            if isinstance(v, (int, float)):
                if v > 0: return 'background-color: #DCFCE7; color: #166534'
                if v < 0: return 'background-color: #FEE2E2; color: #991B1B'
            return ''
        st.dataframe(
            df_d.style.map(color, subset=['Общее Добавить/Убрать', 'Тег A Добавить/Убрать', 'Текст Добавить/Убрать']),
            column_config={"diff_abs": None}, use_container_width=True, height=600
        )
        st.download_button("Скачать CSV", df_d.to_csv().encode('utf-8'), "depth.csv")
    
    with st.expander("2. Гибридный ТОП униграм на основе конкурентов"):
        st.dataframe(results['hybrid'].sort_values(by="TF-IDF ТОП", ascending=False), use_container_width=True)
        
    with st.expander("3. N-граммы (включая все словоформы)"):
        st.dataframe(results['ngrams'].sort_values(by="TF-IDF", ascending=False), use_container_width=True)
        
    with st.expander("4. Информация по всем сайтам ТОПа"):
        st.dataframe(results['relevance'], use_container_width=True)
