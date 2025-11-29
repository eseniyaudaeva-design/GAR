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
st.set_page_config(layout="wide", page_title="ГАР PRO: Анализ", page_icon="📊")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Основные стили */
        .main {
            background: linear-gradient(135deg, #E6F3FF 0%, #F0F9FF 50%, #E6F7FF 100%);
        }
        
        html, body, [class*="css"] { 
            font-family: 'Inter', sans-serif;
            background: #f8fcff;
        }
        
        /* Стилизация верхней панели ввода */
        .main-input-container {
            background: linear-gradient(135deg, #ffffff 0%, #f8fcff 100%);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #e1f0ff;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0, 120, 215, 0.08);
        }
        
        /* Кнопка с градиентом */
        .stButton button {
            background: linear-gradient(135deg, #1890ff 0%, #096dd9 100%);
            color: white;
            font-weight: 600;
            border-radius: 10px;
            height: 55px;
            width: 100%;
            border: none;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(24, 144, 255, 0.3);
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #096dd9 0%, #0050b3 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(24, 144, 255, 0.4);
        }
        
        /* Заголовки */
        h1 {
            color: #1890ff;
            font-weight: 700;
            margin-bottom: 20px;
        }
        
        h2, h3 {
            color: #096dd9;
            font-weight: 600;
        }
        
        /* Заголовки таблиц */
        .table-header { 
            font-size: 20px; 
            font-weight: 600; 
            margin-top: 35px; 
            margin-bottom: 15px; 
            color: #096dd9;
            padding: 10px 0;
            border-bottom: 2px solid #e6f7ff;
        }
        
        /* Радио кнопки */
        .stRadio > div {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e1f0ff;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f0f9ff 0%, #e6f7ff 100%);
            color: #096dd9;
            font-weight: 600;
            border-radius: 10px;
            border: 1px solid #bae7ff;
        }
        
        /* Прогресс бар */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #1890ff 0%, #36cfc9 100%);
        }
        
        /* Таблицы */
        .dataframe {
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 120, 215, 0.1);
        }
        
        /* Улучшенные текстовые поля */
        .stTextInput input, .stTextArea textarea {
            border: 1px solid #bae7ff;
            border-radius: 8px;
            padding: 12px;
            background: #fafdff;
        }
        
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #1890ff;
            box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
        }
        
        /* Селекты */
        .stSelectbox select {
            border: 1px solid #bae7ff;
            border-radius: 8px;
            background: #fafdff;
        }
        
        /* Чекбоксы */
        .stCheckbox > label {
            color: #096dd9;
            font-weight: 500;
        }
        
        /* Успешные сообщения */
        .stAlert {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. АВТОРИЗАЦИЯ
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]: return True
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #ffffff 0%, #f8fcff 100%); 
                     border-radius: 15px; border: 1px solid #e1f0ff; box-shadow: 0 4px 12px rgba(0, 120, 215, 0.08);'>
                <h2 style='color: #1890ff; margin-bottom: 30px;'>🔐 Авторизация</h2>
        """, unsafe_allow_html=True)
        pwd = st.text_input("Пароль доступа", type="password")
        if st.button("Войти", key="auth_btn"):
            if pwd == "admin123":
                st.session_state["password_correct"] = True
                st.rerun()
            else: 
                st.error("Неверный пароль")
        st.markdown('</div>', unsafe_allow_html=True)
    return False

if not check_password(): st.stop()

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

DEFAULT_EXCLUDE = ["yandex.ru", "avito.ru", "ozon.ru", "wildberries.ru", "youtube.com", "dzen.ru", "hh.ru", "t.me"]
DEFAULT_STOPS = ["рублей", "руб", "купить", "цена", "шт", "см", "мм", "кг", "кв", "м2", "стр", "ул"]

# --- Парсинг ---
def get_domain(url):
    try: return urlparse(url).netloc
    except: return url

def parse_page(url, settings):
    headers = {'User-Agent': settings['ua']}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200: return None
        
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Мета-теги
        title = soup.title.string if soup.title else ""
        desc = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc: desc = meta_desc.get("content", "")
        h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        
        # Очистка
        if settings['noindex']:
            for t in soup.find_all(['noindex', 'script', 'style']): t.decompose()
        else:
            for t in soup(['script', 'style']): t.decompose()
            
        # Анкоры
        anchors_list = []
        for a in soup.find_all('a'):
            txt = a.get_text(strip=True)
            if txt: anchors_list.append(txt)
        anchor_text = " ".join(anchors_list)
        
        # Текст (Body)
        body_text = soup.get_text(separator=' ')
        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): body_text += " " + img['alt']
            
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
        if len(w) < 2 or w_lower in stops: continue
        
        lemma = w_lower
        if USE_NLP:
            p = morph.parse(w_lower)[0]
            if settings['std_stops'] and ('PREP' in p.tag or 'CONJ' in p.tag or 'PRCL' in p.tag):
                continue
            lemma = p.normal_form
            
        lemmas.append(lemma)
        
        if lemma not in forms_map: forms_map[lemma] = set()
        forms_map[lemma].add(w_lower)
        
    return lemmas, forms_map

# ==========================================
# 4. ИНТЕРФЕЙС: ВВОД ДАННЫХ (ВСЕГДА ВИДЕН)
# ==========================================

st.title("🎯 ГАР PRO: Анализатор Релевантности")

# ГЛАВНЫЙ БЛОК ВВОДА
with st.container():
    st.markdown('<div class="main-input-container">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        my_url = st.text_input("Ваш URL (Обязательно)", placeholder="https://mysite.ru/catalog/page")
    with c2:
        query = st.text_input("Поисковой запрос", placeholder="купить товар москва")
    st.markdown('</div>', unsafe_allow_html=True)

# ИСТОЧНИК КОНКУРЕНТОВ
st.subheader("📊 Источник конкурентов")
source_mode = st.radio("", ["Google Поиск (Авто)", "Ручной список"], horizontal=True, label_visibility="collapsed")

competitors_final = []

if source_mode == "Google Поиск (Авто)":
    c_s1, c_s2 = st.columns([1, 3])
    with c_s1:
        top_count = st.selectbox("Анализировать ТОП:", [5, 10, 20], index=1)
    with c_s2:
        exclude_domains = st.text_input("Исключить домены (через пробел)", " ".join(DEFAULT_EXCLUDE))
else:
    manual_urls = st.text_area("Список URL конкурентов (каждый с новой строки)", height=150)

# НАСТРОЙКИ (СНИЗУ)
with st.expander("⚙️ Настройки анализа", expanded=True):
    col_set1, col_set2, col_set3 = st.columns(3)
    with col_set1:
        s_noindex = st.checkbox("Исключать noindex", True)
        s_alt = st.checkbox("Учитывать Alt/Title", False)
    with col_set2:
        s_norm = st.checkbox("Нормировать по длине", True)
        s_num = st.checkbox("Учитывать числа", False)
    with col_set3:
        s_std_stops = st.checkbox("Убирать предлоги", True)
    
    custom_stops_text = st.text_area("Стоп-слова", "\n".join(DEFAULT_STOPS), height=60)
    user_agent = st.text_input("User-Agent", "Mozilla/5.0 (compatible; Hybrid-Analyzer/1.0;)")

# КНОПКА ЗАПУСКА
if st.button("🚀 ЗАПУСТИТЬ АНАЛИЗ"):
    if not my_url:
        st.error("❌ Вы не ввели URL вашего сайта!")
        st.stop()
        
    settings = {
        'noindex': s_noindex, 'alt_title': s_alt, 'numbers': s_num,
        'norm': s_norm, 'std_stops': s_std_stops,
        'custom_stops': custom_stops_text.split(), 'ua': user_agent
    }
    
    # 1. Сбор URL
    target_urls = []
    if source_mode == "Google Поиск (Авто)":
        if not query:
            st.error("Введите запрос!")
            st.stop()
        try:
            excl = exclude_domains.split()
            found = search(query, num_results=top_count*2, lang="ru")
            cnt = 0
            for u in found:
                if my_url in u: continue
                if any(x in u for x in excl): continue
                target_urls.append(u)
                cnt += 1
                if cnt >= top_count: break
        except Exception as e:
            st.error(f"Ошибка поиска: {e}")
            st.stop()
    else:
        target_urls = [u.strip() for u in manual_urls.split('\n') if u.strip()]
        
    if not target_urls:
        st.error("Список конкурентов пуст!")
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
            if res: comp_pages.append(res)
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
    for c in comp_stats: vocab.update(c['body'])
    vocab = sorted(list(vocab))
    
    # Сборка главной таблицы (Рекомендации по глубине)
    rows = []
    
    # Для IDF
    N = len(comp_stats)
    doc_freqs = Counter()
    for c in comp_stats:
        for w in set(c['body']): doc_freqs[w] += 1
    
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
            if word in my_body_forms: forms.extend(my_body_forms[word])
            # (в идеале нужно собирать формы и с конкурентов, но для скорости пока так)
            forms_str = ", ".join(list(set(forms))[:3])
            
            rows.append({
                "Слово": word,
                "Словоформы": forms_str,
                "Повторы у вас": my_body_tf,
                # "Минимум": np.min(c_body_tfs), # Можно добавить если надо
                # "Максимум": max_spam,
                "Общее Добавить/Убрать": diff_body,
                
                "Тег A у вас": my_anchor_tf,
                "Тег A рекомендации": target_anchor,
                "Тег A Добавить/Убрать": diff_anchor,
                
                "Текст у вас": my_body_tf, # Для Body
                "Текст рекомендации": target_body,
                "Текст Добавить/Убрать": diff_body,
                
                "Переспам": int(max_spam * norm_k),
                "Переспам*IDF": round(max_spam * norm_k * idf, 1),
                
                "diff_abs": abs(diff_body) # Скрытое поле для сортировки
            })
            
    df_main = pd.DataFrame(rows)
    
    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    
    st.divider()
    
    # 1. ТАБЛИЦА: РЕКОМЕНДАЦИИ ПО ГЛУБИНЕ (Main Table)
    st.markdown('<div class="table-header">📈 Рекомендации по глубине</div>', unsafe_allow_html=True)
    if not df_main.empty:
        df_main = df_main.sort_values(by="diff_abs", ascending=False)
        
        # Стилизация (подсветка)
        def color_diff(val):
            if val > 0: return 'background-color: #e6fffb; color: #006d75' # Сине-зеленый
            if val < 0: return 'background-color: #fff2e8; color: #ad4e00' # Оранжевый
            return ''
            
        st.dataframe(
            df_main.style.map(color_diff, subset=['Общее Добавить/Убрать', 'Тег A Добавить/Убрать', 'Текст Добавить/Убрать']),
            column_config={"diff_abs": None}, # Скрыть сортировочную колонку
            use_container_width=True,
            height=600
        )
    else:
        st.warning("Нет данных.")

    # 2. ТАБЛИЦА: МЕТА-ТЕГИ
    st.markdown('<div class="table-header">🔍 Информация по мета-тегам</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="table-header">🏆 ТОП релевантности документов</div>', unsafe_allow_html=True)
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
