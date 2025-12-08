import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import re
from collections import Counter, defaultdict
import math
import concurrent.futures
from urllib.parse import urlparse
import inspect
import time
import json

# ==========================================
# 1. КОНФИГУРАЦИЯ И ПАТЧИ
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO", page_icon="📊")

if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

# ==========================================
# 2. АВТОРИЗАЦИЯ
# ==========================================
def check_password():
    if st.session_state.get("authenticated"): return True
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h3 style='text-align: center; margin-top: 5rem;'>📊 GAR PRO Вход</h3>", unsafe_allow_html=True)
        pwd = st.text_input("Пароль", type="password", label_visibility="collapsed")
        if st.button("ВОЙТИ", type="primary", use_container_width=True):
            if pwd == "jfV6Xel-Q7vp-_s2UYPO":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Неверный пароль")
    return False

if not check_password(): st.stop()

# ==========================================
# 3. НАСТРОЙКИ
# ==========================================
ARSENKIN_TOKEN = "43acbbb60cb7989c05914ff21be45379"

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
    "Краснодар": {"ya": 35, "go": 1011894},
    "Киев (UA)": {"ya": 143, "go": 1012852},
    "Минск (BY)": {"ya": 157, "go": 1001493},
    "Алматы (KZ)": {"ya": 162, "go": 1014601}
}

DEFAULT_EXCLUDE = """yandex.ru\navito.ru\nberu.ru\ntiu.ru\naliexpress.com\nebay.com\nauto.ru\n2gis.ru\nsravni.ru\ntoshop.ru\nprice.ru\npandao.ru\ninstagram.com\nwikipedia.org\nrambler.ru\nhh.ru\nbanki.ru\nregmarkets.ru\nzoon.ru\npulscen.ru\nprodoctorov.ru\nblizko.ru\ndomclick.ru\nsatom.ru\nquto.ru\nedadeal.ru\ncataloxy.ru\nirr.ru\nonliner.by\nshop.by\ndeal.by\nyell.ru\nprofi.ru\nirecommend.ru\notzovik.com\nozon.ru\nozon.by\nmarket.yandex.ru\nyoutube.com\ngosuslugi.ru\ndzen.ru\n2gis.by\nwildberries.ru\nrutube.ru\nvk.com\nfacebook.com"""
DEFAULT_STOPS = "рублей\nруб\nкупить\nцена\nшт\nсм\nмм\nкг\nкв\nм2\nстр\nул"

# Стили
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .stApp { background-color: #FFFFFF !important; color: #3D4858 !important; }
        html, body, p, li, h1, h2, h3, h4 { font-family: 'Inter', sans-serif; color: #3D4858 !important; }
        div[data-testid="stDataFrame"] { border: 2px solid #277EFF !important; border-radius: 8px !important; }
        div[data-testid="stDataFrame"] div[role="columnheader"] { background-color: #F0F7FF !important; color: #277EFF !important; font-weight: 700 !important; border-bottom: 2px solid #277EFF !important; }
        .legend-box { padding: 10px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 5px; font-size: 14px; margin-bottom: 10px; }
        .text-red { color: #D32F2F; font-weight: bold; }
        .text-bold { font-weight: 600; }
        .sort-container { background-color: #F1F5F9; padding: 10px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. ФУНКЦИИ (API, NLP, Parse)
# ==========================================
try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    USE_NLP = True
except:
    morph = None
    USE_NLP = False

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def get_arsenkin_urls(query, engine_type, region_name, depth_val=30):
    url_set = "https://arsenkin.ru/api/tools/set"
    url_check = "https://arsenkin.ru/api/tools/check"
    url_get = "https://arsenkin.ru/api/tools/get"
    headers = {"Authorization": f"Bearer {ARSENKIN_TOKEN}", "Content-type": "application/json"}
    
    reg_ids = REGION_MAP.get(region_name, {"ya": 213, "go": 1011969})
    se_params = []
    if "Яндекс" in engine_type: se_params.append({"type": 2, "region": reg_ids['ya']})
    if "Google" in engine_type: se_params.append({"type": 11, "region": reg_ids['go']})
        
    payload = {"tools_name": "check-top", "data": {"queries": [query], "is_snippet": False, "noreask": True, "se": se_params, "depth": depth_val}}
    
    try:
        r = requests.post(url_set, headers=headers, json=payload, timeout=15)
        if "task_id" not in r.json(): return []
        task_id = r.json()["task_id"]
    except: return []
    
    status, attempts = "process", 0
    while status == "process" and attempts < 40:
        time.sleep(3)
        attempts += 1
        try:
            r_check = requests.post(url_check, headers=headers, json={"task_id": task_id})
            if r_check.json().get("status") == "finish": status = "done"
        except: pass
        
    if status != "done": return []
    
    try:
        r_final = requests.post(url_get, headers=headers, json={"task_id": task_id}, timeout=30)
        collect = r_final.json()['result']['result']['collect']
        final_url_list = collect[0][0] # Simple structure
        return [{'url': u, 'pos': i+1} for i, u in enumerate(final_url_list)]
    except: return []

def process_text_detailed(text, settings, n_gram=1):
    pattern = r'[а-яА-ЯёЁ0-9a-zA-Z]+' if settings['numbers'] else r'[а-яА-ЯёЁa-zA-Z]+'
    words = re.findall(pattern, text.lower())
    stops = set(w.lower() for w in settings['custom_stops'])
    lemmas, forms_map = [], defaultdict(set)
    
    for w in words:
        if len(w) < 2 or w in stops: continue
        lemma = w
        if USE_NLP and n_gram == 1: 
            p = morph.parse(w)[0]
            if not any(t in p.tag for t in ['PREP', 'CONJ', 'PRCL', 'NPRO']):
                lemma = p.normal_form
        lemmas.append(lemma)
        forms_map[lemma].add(w)
    
    if n_gram > 1:
        return [" ".join(lemmas[i:i+n_gram]) for i in range(len(lemmas)-n_gram+1)], {}
    return lemmas, forms_map

def parse_page(url, settings):
    try:
        r = requests.get(url, headers={'User-Agent': settings['ua']}, timeout=10)
        if r.status_code != 200: return None
        soup = BeautifulSoup(r.text, 'html.parser')
        
        tags_rm = ['script', 'style', 'head']
        if settings['noindex']: tags_rm.extend(['noindex', 'nav', 'footer', 'header', 'aside'])
        for t in soup.find_all(tags_rm): t.decompose()
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)): c.extract()
            
        anchors = [a.get_text(" ", strip=True) for a in soup.find_all('a')]
        anchor_txt = " ".join(anchors)
        
        extra = []
        if settings['alt_title']:
            extra.extend([img['alt'] for img in soup.find_all('img', alt=True)])
            extra.extend([t['title'] for t in soup.find_all(title=True)])
            
        body = re.sub(r'\s+', ' ', soup.get_text(" ", strip=True) + " " + " ".join(extra)).strip()
        if not body: return None
        return {'url': url, 'domain': urlparse(url).netloc, 'body_text': body, 'anchor_text': anchor_txt}
    except: return None

def calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, original_results):
    all_forms_map = defaultdict(set)
    
    if my_data and my_data.get('body_text'):
        my_lemmas, my_forms = process_text_detailed(my_data['body_text'], settings)
        my_anchors, _ = process_text_detailed(my_data['anchor_text'], settings)
        for k, v in my_forms.items(): all_forms_map[k].update(v)
    else:
        my_lemmas, my_forms, my_anchors = [], {}, []

    comp_docs = []
    for p in comp_data_full:
        if not p.get('body_text'): continue
        body, c_forms = process_text_detailed(p['body_text'], settings)
        anchor, _ = process_text_detailed(p['anchor_text'], settings)
        comp_docs.append({'body': body, 'anchor': anchor})
        for k, v in c_forms.items(): all_forms_map[k].update(v)
    
    if not comp_docs:
        # Fallback table if no competitors downloaded
        tbl = [{"Домен": urlparse(x['url']).netloc, "Позиция": x['pos'], "Ширина (балл)": 0, "Глубина (балл)": 0} for x in original_results]
        my_l = f"{my_data['domain']} (Вы)" if my_data else "Ваш сайт"
        tbl.append({"Домен": my_l, "Позиция": my_serp_pos, "Ширина (балл)": 0, "Глубина (балл)": 0})
        return {"depth": pd.DataFrame(), "hybrid": pd.DataFrame(), "ngrams": pd.DataFrame(), "relevance_top": pd.DataFrame(tbl).sort_values('Позиция'), "my_score": {"width": 0, "depth": 0}}

    avg_len = np.mean([len(d['body']) for d in comp_docs])
    norm_k = (len(my_lemmas) / avg_len) if (settings['norm'] and len(my_lemmas) > 0 and avg_len > 0) else 1.0
    
    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    vocab = sorted(list(vocab))
    N = len(comp_docs)
    doc_freqs = Counter([w for d in comp_docs for w in set(d['body'])])
        
    table_depth, table_hybrid = [], []
    for word in vocab:
        df = doc_freqs[word]
        if df < 2 and word not in my_lemmas: continue 
        
        my_tf = my_lemmas.count(word)
        my_tf_a = my_anchors.count(word)
        c_tfs = [d['body'].count(word) for d in comp_docs]
        c_anchor_tfs = [d['anchor'].count(word) for d in comp_docs]
        
        med_total, max_total = np.median(c_tfs), np.max(c_tfs)
        rec_min = int(round(min(np.mean(c_tfs), med_total) * norm_k))
        rec_max = int(round(max_total * norm_k))
        
        idf = max(0.1, math.log((N - df + 0.5)/(df + 0.5) + 1))
        diff = rec_min - my_tf if my_tf < rec_min else (rec_max - my_tf if my_tf > rec_max else 0)
        
        if med_total > 0.5 or my_tf > 0:
            table_depth.append({
                "Слово": word, "Словоформы": ", ".join(sorted(list(all_forms_map.get(word, set())))), 
                "Повторы у вас": my_tf, "Минимум (рек)": rec_min, "Максимум (рек)": rec_max, "Добавить/Убрать": diff,
                "is_missing": (my_tf == 0), "diff_abs": abs(diff)
            })
            table_hybrid.append({
                "Слово": word, "TF-IDF ТОП": round(med_total * idf, 2), "TF-IDF у вас": round(my_tf * idf, 2), "Сайтов": df
            })

    # N-Grams logic (skipped for brevity, structure remains similar)
    table_ngrams = [] 
    
    # Top Relevance
    table_rel = []
    comp_stats = []
    for item in original_results:
        parsed = next((d for d in comp_data_full if d['url'] == item['url']), None)
        w, d = 0, 0
        if parsed:
            p_lem = process_text_detailed(parsed['body_text'], settings)[0]
            rel = [x for x in p_lem if x in vocab]
            w, d = len(set(rel)), len(rel)
        comp_stats.append({'d': urlparse(item['url']).netloc, 'pos': item['pos'], 'w': w, 'dd': d})
    
    mx_w = max([c['w'] for c in comp_stats]) if comp_stats else 1
    mx_d = max([c['dd'] for c in comp_stats]) if comp_stats else 1
    
    for c in comp_stats:
        table_rel.append({"Домен": c['d'], "Позиция": c['pos'], "Ширина (балл)": int(c['w']/mx_w*100), "Глубина (балл)": int(c['dd']/mx_d*100)})
        
    my_rel = [x for x in my_lemmas if x in vocab]
    my_w, my_d = len(set(my_rel)), len(my_rel)
    my_sw, my_sd = int(my_w/mx_w*100), int(my_d/mx_d*100)
    
    table_rel.append({"Домен": f"{my_data['domain']} (Вы)" if my_data else "Вы", "Позиция": my_serp_pos, "Ширина (балл)": my_sw, "Глубина (балл)": my_sd})
    
    return {
        "depth": pd.DataFrame(table_depth), "hybrid": pd.DataFrame(table_hybrid), "ngrams": pd.DataFrame(table_ngrams),
        "relevance_top": pd.DataFrame(table_rel).sort_values('Позиция'), "my_score": {"width": my_sw, "depth": my_sd}
    }

def render_table(df, title):
    if df.empty: return st.info(f"{title}: Нет данных.")
    st.markdown(f"### {title}")
    st.dataframe(df, use_container_width=True, hide_index=True)

# ==========================================
# 5. ИНТЕРФЕЙС
# ==========================================
c1, c2 = st.columns([65, 35])
with c1:
    st.title("SEO Анализатор")
    mode = st.radio("Режим", ["URL вашего сайта", "Текст/Код"], horizontal=True, label_visibility="collapsed")
    my_val = st.text_input("URL", key="my_url") if mode == "URL вашего сайта" else st.text_area("Код", key="my_code", height=150)
    
    st.markdown("### Запрос и Конкуренты")
    query = st.text_input("Поисковой запрос")
    src = st.radio("Источник", ["API Arsenkin", "Свой список"], horizontal=True)
    man_urls = st.text_area("Список URL", height=150) if src == "Свой список" else None
    
    if st.button("ЗАПУСТИТЬ", type="primary", use_container_width=True):
        st.session_state.start = True

with c2:
    st.markdown("##### Настройки")
    se = st.selectbox("ПС", ["Яндекс", "Google", "Яндекс + Google"])
    reg = st.selectbox("Регион", list(REGION_MAP.keys()))
    top_n = st.selectbox("Кол-во конкурентов", [10, 20, 30])
    excl = st.text_area("Исключить домены", DEFAULT_EXCLUDE, height=200)

# ==========================================
# 6. ЛОГИКА ЗАПУСКА
# ==========================================
if st.session_state.get('start'):
    st.session_state.start = False
    
    # 1. Сбор вашего сайта
    my_data = None
    if mode == "URL вашего сайта" and my_val:
        my_data = parse_page(my_val, {'ua': 'Mozilla/5.0', 'noindex': True, 'alt_title': False})
    elif mode == "Текст/Код" and my_val:
        my_data = {'url': 'local', 'domain': 'local', 'body_text': my_val, 'anchor_text': ''}
        
    if not my_data: st.error("Ошибка: нет данных вашего сайта"); st.stop()
    my_domain = my_data['domain']
    
    # 2. Сбор конкурентов
    target = []
    my_pos = 0
    
    if src == "API Arsenkin":
        if not query: st.error("Введите запрос"); st.stop()
        with st.spinner("API запрос..."):
            raw = get_arsenkin_urls(query, se, reg, 30) # Макс глубина 30
            
        excludes = [x.strip() for x in excl.split('\n') if x.strip()]
        clean = []
        
        for r in raw:
            d = urlparse(r['url']).netloc
            if my_domain in d: 
                if my_pos == 0: my_pos = r['pos']
                continue
            if any(e in d for e in excludes): continue
            clean.append(r)
            
        # Re-ranking: берем N штук и ставим им позиции 1..N
        target = clean[:top_n]
        for i, t in enumerate(target): t['pos'] = i + 1
            
        st.success(f"Найдено {len(raw)}. После фильтрации: {len(target)}. Ваша позиция: {my_pos}")
        
    else:
        if not man_urls: st.error("Введите список"); st.stop()
        target = [{'url': u.strip(), 'pos': i+1} for i, u in enumerate(man_urls.split('\n')) if u.strip()]

    # 3. Парсинг
    full_data = []
    with st.spinner("Скачивание конкурентов..."):
        with concurrent.futures.ThreadPoolExecutor(10) as ex:
            futs = {ex.submit(parse_page, t['url'], {'ua': 'Mozilla/5.0', 'noindex': True, 'alt_title': False}): t for t in target}
            for f in concurrent.futures.as_completed(futs):
                if f.result(): full_data.append(f.result())
                
    # 4. Анализ
    res = calculate_metrics(full_data, my_data, {'numbers': False, 'custom_stops': DEFAULT_STOPS.split(), 'norm': True}, my_pos, target)
    
    st.markdown(f"### Результат: Ширина {res['my_score']['width']} | Глубина {res['my_score']['depth']}")
    render_table(res['relevance_top'], "ТОП Релевантности")
    render_table(res['depth'], "Анализ слов")
    render_table(res['hybrid'], "TF-IDF")
