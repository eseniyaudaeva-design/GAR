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
# 1. СТИЛЬ (ULTIMATE LIGHT THEME)
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO", page_icon="💎")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');
        
        /* --- 1. ГЛОБАЛЬНЫЙ ФОН (ПРИНУДИТЕЛЬНО СВЕТЛЫЙ) --- */
        [data-testid="stAppViewContainer"] {
            background-color: #F3F6F9 !important;
            font-family: 'Manrope', sans-serif;
        }
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        /* --- 2. ТЕКСТ (ВСЕГДА ТЕМНЫЙ/ЧЕРНЫЙ) --- */
        h1, h2, h3, h4, h5, h6, p, span, label, div, .stMarkdown {
            color: #1E293B !important;
        }
        h1, h2 {
            color: #0F172A !important; /* Очень темно-синий для заголовков */
            font-weight: 800 !important;
        }
        
        /* --- 3. КАРТОЧКИ (БЕЛЫЕ БЛОКИ С ТЕНЬЮ) --- */
        .css-card {
            background-color: #FFFFFF;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            border: 1px solid #E2E8F0;
            margin-bottom: 24px;
        }
        
        /* --- 4. ПОЛЯ ВВОДА (БЕЛЫЙ ФОН, ЧЕРНЫЙ ТЕКСТ) --- */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            caret-color: #000000 !important; /* Курсор ввода */
            border: 2px solid #E2E8F0 !important;
            border-radius: 8px !important;
            font-size: 15px !important;
        }
        
        /* Фокус на поле */
        .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        }
        
        /* Выпадающие списки (меню внутри) */
        ul[data-baseweb="menu"] {
            background-color: #FFFFFF !important;
        }
        li[data-baseweb="option"] {
            color: #000000 !important;
        }
        
        /* --- 5. КНОПКА (СИНИЙ ГРАДИЕНТ) --- */
        div.stButton > button {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 16px 32px !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(37, 99, 235, 0.5) !important;
        }
        div.stButton > button:active {
            color: #FFFFFF !important; /* Чтобы текст не пропадал при нажатии */
        }
        
        /* --- 6. ТАБЛИЦЫ (ЧИТАЕМЫЕ) --- */
        div[data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* Заголовки таблиц */
        [data-testid="stDataFrame"] th {
            background-color: #F8FAFC !important;
            color: #475569 !important;
            font-weight: 700 !important;
            border-bottom: 1px solid #E2E8F0 !important;
        }
        [data-testid="stDataFrame"] td {
            color: #334155 !important;
            background-color: #FFFFFF !important;
            border-bottom: 1px solid #F1F5F9 !important;
        }
        
        /* --- 7. ЧЕКБОКСЫ И РАДИОКНОПКИ --- */
        label[data-testid="stLabel"] {
            font-size: 14px;
            font-weight: 600 !important;
            color: #334155 !important;
        }
        /* Сами чекбоксы */
        span[data-baseweb="checkbox"] div {
            background-color: #FFFFFF !important;
        }
        
        /* --- 8. EXPANDER (НАСТРОЙКИ) --- */
        .streamlit-expanderHeader {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            color: #0F172A !important;
        }
        
        /* Убираем лишние отступы */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ЛОГИКА (БЭКЕНД - БЕЗ ИЗМЕНЕНИЙ)
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

DEFAULT_EXCLUDE = ["yandex.ru", "avito.ru", "ozon.ru", "wildberries.ru", "youtube.com", "dzen.ru", "hh.ru", "t.me", "tiu.ru", "pulscen.ru", "satu.kz"]
DEFAULT_STOPS = ["рублей", "руб", "купить", "цена", "шт", "см", "мм", "кг", "кв", "м2", "стр", "ул", "доставка", "звоните", "заказать"]

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
            'body_text': body_text, 'anchor_text': anchor_text,
            'full_text': body_text + " " + anchor_text
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
        mean_tf = np.mean(c_body_tfs)
        max_tf = np.max(c_body_tfs)
        med_anch = np.median(c_anch_tfs)
        
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
        
        if med_tf > 0.5 or my_tf > 0:
            table_depth.append({
                "Слово": word, "Словоформы": word, "Повторы у вас": my_tf, 
                "Минимум": np.min(c_body_tfs), "Максимум": int(max_tf * norm_k),
                "Общее Добавить/Убрать": diff_body,
                "Тег A у вас": my_anch_tf, "Тег A рекомендации": int(med_anch * norm_k),
                "Тег A Добавить/Убрать": int(med_anch * norm_k) - my_anch_tf,
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

    my_bigrams = process_text(my_page['body_text'], settings, n_gram=2)
    comp_bigrams_list = [process_text(p['body_text'], settings, n_gram=2) for p in comp_data]
    all_bigrams = set(my_bigrams)
    for cb in comp_bigrams_list: all_bigrams.update(cb)
    
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
                "N-грамма": bg, "Кол-во сайтов": df, "Медианное вхождение": med_cnt,
                "Среднее": round(np.mean(comp_cnts), 1), "На вашем сайте": my_cnt,
                "TF-IDF": round(my_cnt * math.log(N/df if df>0 else 1), 3)
            })

    table_relevance = []
    for i, p in enumerate(comp_data):
        p_lemmas = process_text(p['body_text'], settings)
        w = len(set(p_lemmas).intersection(vocab))
        table_relevance.append({
            "Домен": p['domain'], "Позиция": i+1, "URL": p['url'],
            "Ширина": w, "Глубина": len(p_lemmas)
        })
        
    return {
        "depth": pd.DataFrame(table_depth), "hybrid": pd.DataFrame(table_hybrid),
        "ngrams": pd.DataFrame(table_ngrams), "relevance_top": pd.DataFrame(table_relevance),
        "my_score": {"width": len(set(my_lemmas).intersection(vocab)), "depth": len(my_lemmas)}
    }

# ==========================================
# 3. ИНТЕРФЕЙС (КРАСИВЫЙ И ПОНЯТНЫЙ)
# ==========================================

st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>💎 GAR PRO: SEO Аналитика</h1>", unsafe_allow_html=True)

# --- БЛОК 1: ДАННЫЕ ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.markdown("### 📝 Ввод данных")
c1, c2 = st.columns(2)
with c1:
    my_url = st.text_input("URL вашей страницы", placeholder="https://site.ru/catalog/okna")
with c2:
    query = st.text_input("Поисковой запрос", placeholder="купить пластиковые окна")
st.markdown('</div>', unsafe_allow_html=True)

# --- БЛОК 2: КОНКУРЕНТЫ ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.markdown("### 🕵️ Источник конкурентов")

# Радиокнопки как большие кнопки (pills)
source_type = st.radio("Источник:", ["Google Поиск (Авто)", "Список URL вручную"], horizontal=True)

if source_type == "Google Поиск (Авто)":
    cl1, cl2 = st.columns([1, 3])
    with cl1:
        top_n = st.selectbox("Глубина ТОПа", [5, 10, 20], index=1)
    with cl2:
        excludes = st.text_input("Исключить домены (через пробел)", " ".join(DEFAULT_EXCLUDE))
else:
    manual_urls = st.text_area("Список URL (каждый с новой строки)", height=150)
st.markdown('</div>', unsafe_allow_html=True)

# --- БЛОК 3: НАСТРОЙКИ ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.markdown("### ⚙️ Настройки")

c_left, c_right = st.columns(2)

with c_left:
    st.markdown("**Парсинг**")
    ua = st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0", "Googlebot/2.1"])
    custom_stops = st.text_area("Стоп-слова", "\n".join(DEFAULT_STOPS), height=120)

with c_right:
    st.markdown("**Фильтры**")
    # Используем чекбоксы, но они будут выглядеть четко
    s_noindex = st.checkbox("🚫 Исключать noindex", True)
    s_alt = st.checkbox("🖼️ Учитывать Alt/Title", False)
    s_num = st.checkbox("🔢 Обрабатывать цифры", False)
    s_norm = st.checkbox("📏 Нормировать по длине", True)
    s_agg = st.checkbox("🛒 Исключать агрегаторы", True)

st.markdown('</div>', unsafe_allow_html=True)

# --- КНОПКА ЗАПУСКА ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("ЗАПУСТИТЬ АНАЛИЗ 🚀"):
    
    if not my_url:
        st.error("Введите URL вашего сайта!")
        st.stop()
        
    settings = {
        'noindex': s_noindex, 'alt_title': s_alt, 'numbers': s_num,
        'norm': s_norm, 'ua': ua, 'custom_stops': custom_stops.split()
    }
    
    # 1. Получение URL
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
        st.error("Нет конкурентов.")
        st.stop()
        
    # 2. Сбор данных
    prog = st.progress(0)
    status = st.empty()
    
    status.text(f"Скачиваем ваш сайт...")
    my_data = parse_page(my_url, settings)
    if not my_data:
        st.error("Ваш сайт недоступен.")
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
        st.error("Мало данных конкурентов.")
        st.stop()
        
    # 3. Расчет
    results = calculate_metrics(comp_data, my_data, settings)
    st.success("Готово!")
    
    # Метрики
    m1, m2, m3 = st.columns(3)
    m1.metric("Ширина (Охват)", results['my_score']['width'])
    m2.metric("Глубина (Слов)", results['my_score']['depth'])
    m3.metric("Конкурентов", len(comp_data))
    
    st.divider()
    
    # Таблицы
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
    
    with st.expander("2. Гибридный ТОП униграм"):
        st.dataframe(results['hybrid'].sort_values(by="TF-IDF ТОП", ascending=False), use_container_width=True)
        
    with st.expander("3. N-граммы (Биграммы)"):
        st.dataframe(results['ngrams'].sort_values(by="TF-IDF", ascending=False), use_container_width=True)
        
    with st.expander("4. ТОП релевантности документов"):
        st.dataframe(results['relevance_top'], use_container_width=True)

