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
# 1. СТИЛИЗАЦИЯ (Светло-голубая тема)
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO: SEO Analysis", page_icon="💎")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        
        /* 1. Глобальный фон приложения (Светло-голубой градиент) */
        .stApp {
            background: linear-gradient(135deg, #E0F7FA 0%, #E3F2FD 100%);
            font-family: 'Roboto', sans-serif;
            color: #0F172A; /* Темно-синий текст для контраста */
        }
        
        /* 2. Заголовки */
        h1, h2, h3, h4 {
            color: #0277BD !important; /* Насыщенный голубой */
            font-weight: 700;
        }
        
        /* 3. Белые карточки для контента (чтобы читалось) */
        .block-container {
            padding-top: 2rem;
        }
        
        /* Стилизация полей ввода (белый фон, голубая рамка) */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            background-color: #FFFFFF !important;
            color: #333333 !important;
            border: 1px solid #81D4FA !important;
            border-radius: 6px;
        }
        
        /* Лейблы над полями */
        .stTextInput label, .stTextArea label, .stSelectbox label, .stRadio label {
            color: #01579B !important;
            font-weight: 600;
        }
        
        /* 4. Кнопка (Яркая, контрастная) */
        div.stButton > button {
            background: linear-gradient(90deg, #0288D1 0%, #01579B 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: bold;
            font-size: 16px;
            box-shadow: 0 4px 10px rgba(2, 136, 209, 0.3);
            transition: 0.3s;
            width: 100%;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #039BE5 0%, #0277BD 100%);
            box-shadow: 0 6px 14px rgba(2, 136, 209, 0.5);
            transform: translateY(-2px);
        }
        
        /* 5. Таблицы (Белый фон, четкие границы) */
        div[data-testid="stDataFrame"] {
            background-color: white;
            border: 1px solid #B3E5FC;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        /* 6. Настройки (Expander) */
        .streamlit-expanderHeader {
            background-color: #E1F5FE !important;
            color: #0277BD !important;
            border: 1px solid #81D4FA;
            border-radius: 8px;
        }
        div[data-testid="stExpander"] {
            background-color: rgba(255,255,255,0.6);
            border-radius: 8px;
        }
        
        /* Убираем верхнюю полосу декора */
        header[data-testid="stHeader"] {
            background-color: transparent;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. БЭКЕНД (ЛОГИКА - ТА ЖЕ САМАЯ)
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

# --- Функции Парсинга ---

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

# --- Математика ---
def calculate_advanced_metrics(corpus_pages, my_page, settings):
    my_lemmas = process_text(my_page['body_text'], settings)
    my_anchors = process_text(my_page['anchor_text'], settings)
    
    comp_docs = []
    for p in corpus_pages:
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
        comp_tfs = [d['body'].count(word) for d in comp_docs]
        comp_anch_tfs = [d['anchor'].count(word) for d in comp_docs]
        
        med_tf = np.median(comp_tfs)
        mean_tf = np.mean(comp_tfs)
        max_tf = np.max(comp_tfs)
        med_anch = np.median(comp_anch_tfs)
        
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        bm25_scores = []
        for i, d in enumerate(comp_docs):
            tf = comp_tfs[i]
            dl = len(d['body'])
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_len)))
            bm25_scores.append(score)
        bm25_top = np.median(bm25_scores)
        bm25_my = idf * (my_tf * (k1 + 1)) / (my_tf + k1 * (1 - b + b * (my_len / avg_len)))
        
        target_body = int(med_tf * 1.3 * norm_k)
        diff_body = target_body - my_tf
        
        if med_tf > 0.5 or my_tf > 0:
            table_depth.append({
                "Слово": word, "Повторы у вас": my_tf, "Общее Добавить/Убрать": diff_body,
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
                "Среднее по ТОПу": round(mean_tf, 1), "Ваш сайт": my_tf
            })

    # N-граммы
    my_bigrams = process_text(my_page['body_text'], settings, n_gram=2)
    comp_bigrams_list = [process_text(p['body_text'], settings, n_gram=2) for p in corpus_pages]
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
                "N-грамма": bg, "Кол-во сайтов": df, "Медиана": med_cnt,
                "Среднее": round(np.mean(comp_cnts), 1), "На вашем сайте": my_cnt,
                "TF-IDF": round(my_cnt * math.log(N/df if df>0 else 1), 3)
            })

    table_relevance = []
    for i, p in enumerate(corpus_pages):
        p_lemmas = process_text(p['body_text'], settings)
        common = set(p_lemmas).intersection(vocab)
        table_relevance.append({
            "Домен": p['domain'], "Позиция": i+1, "Ширина": len(common), "Глубина": len(p_lemmas)
        })
        
    return {
        "depth": pd.DataFrame(table_depth), "hybrid": pd.DataFrame(table_hybrid),
        "ngrams": pd.DataFrame(table_ngrams), "relevance_top": pd.DataFrame(table_relevance),
        "my_score": {"width": len(set(my_lemmas).intersection(vocab)), "depth": len(my_lemmas)}
    }

# ==========================================
# 3. ИНТЕРФЕЙС (UI)
# ==========================================

st.markdown("<h1 style='text-align: center;'>SEO Анализатор Релевантности</h1>", unsafe_allow_html=True)

# Блок ввода URL и Запроса
st.markdown("#### 📝 Ввод данных")
c1, c2 = st.columns(2)
with c1:
    my_url = st.text_input("Ваш URL", placeholder="https://site.ru")
with c2:
    query = st.text_input("Поисковой запрос", placeholder="пластиковые окна")

st.markdown("#### 🕵️ Источник конкурентов")
source_type = st.radio("Источник:", ["Google Поиск (Авто)", "Список URL вручную"], horizontal=True, label_visibility="collapsed")

if source_type == "Google Поиск (Авто)":
    cl1, cl2 = st.columns(2)
    with cl1: top_n = st.selectbox("Глубина ТОПа:", [5, 10, 20], index=1)
    with cl2: excludes = st.text_input("Исключить:", " ".join(DEFAULT_EXCLUDE))
else:
    manual_urls = st.text_area("Список URL (построчно):", height=100)

st.markdown("#### ⚙️ Настройки")
with st.expander("Открыть расширенные настройки", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        s_noindex = st.toggle("Исключать noindex", True)
        s_alt = st.toggle("Включать alt/title", False)
    with col2:
        s_norm = st.toggle("Нормировать по длине", True)
        s_num = st.toggle("Учитывать цифры", False)
    with col3:
        s_agg = st.toggle("Без агрегаторов", True)
    
    st.markdown("---")
    ua = st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0", "Googlebot/2.1"])
