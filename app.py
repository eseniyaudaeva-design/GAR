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
# 1. СТИЛЬ (ПРИНУДИТЕЛЬНО БЕЛЫЙ + КОМПАКТНЫЙ)
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO", page_icon="💎")

st.markdown("""
    <style>
        /* ПРИНУДИТЕЛЬНАЯ СВЕТЛАЯ ТЕМА ДЛЯ ВСЕХ ЭЛЕМЕНТОВ */
        :root {
            --primary-color: #2563EB;
            --background-color: #F0F4F8;
            --secondary-background-color: #FFFFFF;
            --text-color: #1E293B;
            --font: 'Inter', sans-serif;
        }
        
        /* Глобальный сброс */
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #F0F4F8 !important;
        }
        
        /* Убираем черноту в выпадающих списках и хедерах */
        div[data-baseweb="select"] > div, div[data-baseweb="popover"], ul[data-baseweb="menu"] {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
        }
        li[data-baseweb="option"] {
            color: #1E293B !important;
        }
        div[data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
            border: 1px solid #E2E8F0;
        }
        .streamlit-expanderHeader {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
        }
        
        /* Текст */
        h1, h2, h3, h4, label, span, p {
            color: #1E293B !important;
        }
        
        /* Поля ввода (Input/Textarea) */
        .stTextInput input, .stTextArea textarea {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
            border: 1px solid #CBD5E1 !important;
        }
        
        /* Уменьшаем ширину колонок в настройках */
        [data-testid="column"] {
            padding: 0 10px;
        }
        
        /* Кнопка */
        div.stButton > button {
            background-color: #2563EB !important;
            color: white !important;
            border: none;
            padding: 10px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. БЭКЕНД
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

DEFAULT_EXCLUDE = """yandex.ru
avito.ru
ozon.ru
wildberries.ru
youtube.com
dzen.ru
hh.ru
t.me
tiu.ru"""

DEFAULT_STOPS = """рублей
руб
купить
цена
шт
см
мм
кг
кв
м2
стр
ул"""

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
        
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        desc = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc: desc = meta_desc.get("content", "").strip()
        h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        
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
            'url': url, 'domain': urlparse(url).netloc, 'title': title, 'desc': desc, 'h1': h1,
            'body_text': body_text, 'anchor_text': anchor_text
        }
    except: return None

def calculate_metrics(comp_data, my_data, settings):
    my_lemmas = process_text(my_data['body_text'], settings)
    my_anchors = process_text(my_data['anchor_text'], settings)
    
    comp_docs = []
    for p in comp_data:
        body = process_text(p['body_text'], settings)
        anchor = process_text(p['anchor_text'], settings)
        comp_docs.append({'body': body, 'anchor': anchor})
        
    avg_len = np.mean([len(d['body']) for d in comp_docs])
    my_len = len(my_lemmas)
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
# 3. ИНТЕРФЕЙС
# ==========================================

st.markdown("<h2 style='text-align: center; color: #1D4ED8;'>SEO Анализатор</h2>", unsafe_allow_html=True)

# 1. ВВОД
st.markdown("##### 📝 Ввод данных")
c1, c2 = st.columns(2)
with c1:
    my_url = st.text_input("Ваш URL", placeholder="https://site.ru/page")
with c2:
    query = st.text_input("Запрос", placeholder="купить окна")

# 2. КОНКУРЕНТЫ
st.markdown("##### 🕵️ Источник")
source_type = st.radio("Источник:", ["Google (Авто)", "Ручной список"], horizontal=True, label_visibility="collapsed")

if source_type == "Google (Авто)":
    cl1, cl2 = st.columns([1, 2]) # Сделал колонку для доменов шире, чтобы не обрезалось
    with cl1:
        top_n = st.selectbox("Глубина ТОПа", [5, 10, 20], index=1)
    with cl2:
        # Используем text_area с малой высотой, чтобы писать в столбик, но компактно
        excludes = st.text_area("Исключить домены (каждый с новой строки)", DEFAULT_EXCLUDE, height=100)
else:
    manual_urls = st.text_area("URL конкурентов (с новой строки)", height=150)

# 3. НАСТРОЙКИ
st.markdown("##### ⚙️ Настройки")
with st.expander("Открыть параметры", expanded=False):
    col_set1, col_set2 = st.columns([1, 1]) # Две равные колонки
    
    with col_set1:
        ua = st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0"])
        # Сделал поле стоп-слов уже (в пределах колонки) и многострочным
        c_stops = st.text_area("Стоп-слова (с новой строки)", DEFAULT_STOPS, height=150)
        
    with col_set2:
        st.write("") # Отступ
        s_noindex = st.checkbox("Исключать noindex", True)
        s_alt = st.checkbox("Учитывать Alt/Title", False)
        s_num = st.checkbox("Учитывать числа", False)
        s_norm = st.checkbox("Нормировать по длине", True)
        s_agg = st.checkbox("Исключать агрегаторы", True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("ЗАПУСТИТЬ АНАЛИЗ 🚀"):
    if not my_url:
        st.error("Введите URL!")
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
                    if my_url in u: continue
                    if any(x in u for x in excl): continue
                    target_urls.append(u)
                    cnt += 1
                    if cnt >= top_n: break
        except:
            st.error("Ошибка поиска.")
            st.stop()
    else:
        target_urls = [u.strip() for u in manual_urls.split('\n') if u.strip()]
        
    if not target_urls:
        st.error("Нет конкурентов.")
        st.stop()
        
    prog = st.progress(0)
    status = st.empty()
    status.text("Скачиваем ваш сайт...")
    my_data = parse_page(my_url, settings)
    
    if not my_data:
        st.error("Ошибка доступа к сайту.")
        st.stop()
        
    comp_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(parse_page, u, settings): u for u in target_urls}
        done = 0
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: comp_data.append(res)
            done += 1
            prog.progress(done / len(target_urls))
            
    prog.empty()
    status.empty()
    
    if len(comp_data) < 2:
        st.error("Мало данных.")
        st.stop()
        
    results = calculate_metrics(comp_data, my_data, settings)
    st.success("Готово!")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Ширина", results['my_score']['width'])
    m2.metric("Глубина", results['my_score']['depth'])
    m3.metric("Конкурентов", len(comp_data))
    
    st.markdown("### 1. Рекомендации")
    df_d = results['depth']
    if not df_d.empty:
        df_d = df_d.sort_values(by="diff_abs", ascending=False)
        def color(v):
            if isinstance(v, (int, float)):
                if v > 0: return 'background-color: #DCFCE7; color: #14532D'
                if v < 0: return 'background-color: #FEE2E2; color: #7F1D1D'
            return ''
        st.dataframe(
            df_d.style.map(color, subset=['Общее Добавить/Убрать', 'Тег A Добавить/Убрать', 'Текст Добавить/Убрать']),
            column_config={"diff_abs": None}, use_container_width=True, height=600
        )
        st.download_button("Скачать CSV", df_d.to_csv().encode('utf-8'), "depth.csv")
    
    with st.expander("2. Гибридный ТОП"):
        st.dataframe(results['hybrid'].sort_values(by="TF-IDF ТОП", ascending=False), use_container_width=True)
        
    with st.expander("3. N-граммы"):
        st.dataframe(results['ngrams'].sort_values(by="TF-IDF", ascending=False), use_container_width=True)
        
    with st.expander("4. ТОП релевантности"):
        st.dataframe(results['relevance_top'], use_container_width=True)
