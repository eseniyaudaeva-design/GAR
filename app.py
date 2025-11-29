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

# ==========================================
# 1. КОНФИГУРАЦИЯ И СТИЛИ
# ==========================================

st.set_page_config(
    page_title="ГАР PRO: Релевантность",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS для визуального соответствия
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #171717;
        }
        
        /* Заголовки */
        h1, h2, h3 { font-weight: 700; color: #0F172A; }

        /* Поля ввода и кнопки */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            border-radius: 6px;
            border: 1px solid #CBD5E1;
        }
        
        /* Акцентная кнопка */
        div.stButton > button {
            background-color: #F97316; /* Оранжевый как в оригинале */
            color: white;
            border-radius: 6px;
            font-weight: 600;
            border: none;
            padding: 0.5rem 1rem;
            transition: 0.2s;
        }
        div.stButton > button:hover {
            background-color: #EA580C;
            color: white;
        }

        /* Таблица */
        div[data-testid="stDataFrame"] {
            font-size: 14px;
        }
        
        /* Скрытие меню */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. БЛОК АВТОРИЗАЦИИ
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h3 style='text-align: center;'>Вход в систему</h3>", unsafe_allow_html=True)
        pwd = st.text_input("Пароль", type="password", label_visibility="collapsed")
        if st.button("Войти"):
            if pwd == "admin123":  
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Неверный пароль")
    return False

if not check_password():
    st.stop()

# ==========================================
# 3. ЛОГИКА ГАР (BACKEND)
# ==========================================

# --- Патч Pymorphy2 ---
try:
    if not hasattr(inspect, 'getargspec'):
        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return spec.args, spec.varargs, spec.varkw, spec.defaults
        inspect.getargspec = getargspec
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    USE_NLP = True
except Exception:
    morph = None
    USE_NLP = False

# --- Поиск Google ---
try:
    from googlesearch import search
    USE_SEARCH = True
except ImportError:
    USE_SEARCH = False

# --- Дефолтные данные ---
DEFAULT_EXCLUDE = """yandex.ru
avito.ru
ozon.ru
wildberries.ru
wikipedia.org
youtube.com
dzen.ru
rutube.ru
hh.ru
t.me"""

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

STANDARD_STOP_WORDS = {
    'и', 'в', 'на', 'с', 'к', 'по', 'за', 'от', 'до', 'это', 'мы', 'вы', 'он', 'она', 'они', 'их', 'ее', 'его', 'мне',
    'тебе', 'себе', 'для', 'что', 'как', 'так', 'но', 'или', 'а', 'чтобы', 'же', 'бы', 'да', 'нет', 'у', 'без', 'под',
    'над', 'перед', 'при', 'через', 'между', 'среди', 'после', 'вместо', 'около', 'вокруг', 'со', 'из', 'из-за', 'из-под'
}

# --- Функции обработки ---

def get_lemmas(text, settings):
    """Возвращает список лемм"""
    # 1. Очистка от HTML тегов (грубая, если текст уже не чист)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Токенизация
    pattern = r'[а-яА-ЯёЁ0-9a-zA-Z]+' if settings['numbers'] else r'[а-яА-ЯёЁa-zA-Z]+'
    words = re.findall(pattern, text)
    
    # 3. Фильтрация и лемматизация
    clean_lemmas = []
    custom_stops = set(w.lower() for w in settings['custom_stops'])
    
    for w in words:
        w_lower = w.lower()
        if len(w) < 2 or w_lower in custom_stops: continue
        
        lemma = w_lower
        if USE_NLP:
            p = morph.parse(w_lower)[0]
            if settings['std_stops']:
                if 'PREP' in p.tag or 'CONJ' in p.tag or 'PRCL' in p.tag or 'NPRO' in p.tag:
                    continue
            lemma = p.normal_form
        
        clean_lemmas.append(lemma)
        
    return clean_lemmas

def parse_html(html, settings):
    """Парсит HTML и отдает Текст и Текст Ссылок раздельно"""
    if not html: return "", ""
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Удаление мусора
    if settings['noindex']:
        for tag in soup.find_all(['noindex', 'script', 'style', 'head', 'footer', 'nav', 'header', 'aside']):
            tag.decompose()
    else:
        for tag in soup(['script', 'style', 'head']):
            tag.decompose()

    # 1. Извлекаем текст ссылок (Anchor)
    anchors = []
    for a in soup.find_all('a'):
        txt = a.get_text(strip=True)
        if txt:
            anchors.append(txt)
    anchor_text = " ".join(anchors)

    # 2. Извлекаем весь текст (Body)
    # Добавляем Alt и Title, если нужно
    extra_text = []
    if settings['alt_title']:
        for img in soup.find_all('img', alt=True):
            extra_text.append(img['alt'])
        for t in soup.find_all(title=True):
            extra_text.append(t['title'])
            
    body_text = soup.get_text(separator=' ') + " " + " ".join(extra_text)
    
    return body_text, anchor_text

def get_page_data(url, settings):
    headers = {'User-Agent': settings['ua']}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return parse_html(r.text, settings)
    except:
        return "", ""
    return "", ""

# --- Математика (TF-IDF, BM25) ---

def calculate_metrics(corpus_data, my_data, settings):
    """
    corpus_data: list of dicts {'body': [lemmas], 'anchor': [lemmas]}
    my_data: dict {'body': [lemmas], 'anchor': [lemmas]}
    """
    
    # 1. Собираем словарь (все уникальные слова)
    vocab = set(my_data['body'])
    for doc in corpus_data:
        vocab.update(doc['body'])
    vocab = sorted(list(vocab))
    
    N = len(corpus_data)
    
    # Структуры данных
    stats = []
    
    # Средняя длина документа в корпусе (для BM25)
    avgdl = np.mean([len(doc['body']) for doc in corpus_data]) if N > 0 else 1
    
    # Параметры BM25
    k1 = 1.2
    b = 0.75

    # Предварительный подсчет DF (Document Frequency)
    doc_freqs = Counter()
    for doc in corpus_data:
        unique_words = set(doc['body'])
        for w in unique_words:
            doc_freqs[w] += 1

    # Подсчет векторов
    for word in vocab:
        # --- Базовые счетчики ---
        # Мой сайт
        my_tf = my_data['body'].count(word)
        my_anchor_tf = my_data['anchor'].count(word)
        
        # Конкуренты (массивы значений)
        comp_tfs = [doc['body'].count(word) for doc in corpus_data]
        comp_anchor_tfs = [doc['anchor'].count(word) for doc in corpus_data]
        
        # --- Метрики корпуса ---
        df = doc_freqs[word] # Кол-во сайтов
        
        # IDF (Standard: log(N/df))
        idf = math.log((N / (df if df > 0 else 1)) + 1)
        
        # Медиана, Максимум, Среднее
        median_tf = np.median(comp_tfs)
        max_tf = np.max(comp_tfs)
        mean_tf = np.mean(comp_tfs)
        
        median_anchor = np.median(comp_anchor_tfs)
        
        # --- TF-IDF ---
        # Для топа берем медианный TF
        tfidf_top = median_tf * idf # Упрощенно, как часто делают в SEO тулзах
        tfidf_my = my_tf * idf
        
        # --- BM25 ---
        # Score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (|D| / avgdl)))
        # Считаем BM25 для каждого конкурента и берем медиану
        bm25_scores = []
        for i, doc in enumerate(corpus_data):
            tf = comp_tfs[i]
            dl = len(doc['body'])
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
            bm25_scores.append(score)
        
        bm25_top = np.median(bm25_scores)
        
        my_dl = len(my_data['body'])
        bm25_my = idf * (my_tf * (k1 + 1)) / (my_tf + k1 * (1 - b + b * (my_dl / avgdl)))

        # --- Нормировка (если включена) ---
        # Если мой текст длиннее/короче среднего, корректируем целевые значения
        norm_factor = 1.0
        if settings['norm'] and avgdl > 0:
            norm_factor = my_dl / avgdl
        
        # Фильтр "значимости" (чтобы не выводить мусор)
        if (median_tf > 0 or my_tf > 0):
            stats.append({
                "Слова": word,
                "TF-IDF ТОП": round(tfidf_top, 2),
                "TF-IDF ваш сайт": round(tfidf_my, 2),
                "BM25 ТОП": round(bm25_top, 2),
                "BM25 ваш сайт": round(bm25_my, 2),
                "IDF": round(idf, 2),
                "Кол-во сайтов": df,
                "Медиана": median_tf * norm_factor, # С учетом нормировки
                "Переспам": max_tf * norm_factor,
                "Среднее по ТОПу (повт.)": round(mean_tf * norm_factor, 1),
                "Ваш сайт (повт.)": my_tf,
                "<a> по ТОПу (повт.)": round(median_anchor * norm_factor, 1),
                "<a> ваш сайт (повт.)": my_anchor_tf,
                # Скрытое поле для сортировки по важности (разница с медианой)
                "diff": abs((median_tf * norm_factor) - my_tf) 
            })
            
    return pd.DataFrame(stats)

# ==========================================
# 4. ФУНКЦИЯ ОТРИСОВКИ НАСТРОЕК
# ==========================================
def render_settings_block(key_suffix):
    """Рисует блок настроек и возвращает словарь параметров"""
    st.markdown("---")
    st.markdown("### Параметры анализа")
    
    with st.expander("⚙️ Настройки парсинга и фильтрации", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            noindex = st.checkbox("Исключать noindex", True, key=f"noindex_{key_suffix}")
            alt = st.checkbox("Учитывать Alt/Title", False, key=f"alt_{key_suffix}")
            num = st.checkbox("Учитывать числа", False, key=f"num_{key_suffix}")
        with c2:
            norm = st.checkbox("Нормировать по длине", True, key=f"norm_{key_suffix}", help="Корректирует рекомендации с учетом разницы в объеме текста")
            std_stops = st.checkbox("Убирать предлоги/союзы", True, key=f"std_{key_suffix}")
            
    with st.expander("🛑 Стоп-слова и User-Agent", expanded=False):
        c_stops = st.text_area("Свои стоп-слова (с новой строки)", DEFAULT_STOPS, height=100, key=f"stops_{key_suffix}")
        ua = st.text_input("User-Agent", "Mozilla/5.0 (compatible; Hybrid-Analyzer/1.0;)", key=f"ua_{key_suffix}")

    return {
        'noindex': noindex, 'alt_title': alt, 'numbers': num, 
        'norm': norm, 'std_stops': std_stops, 
        'custom_stops': c_stops.split(), 'ua': ua
    }

# ==========================================
# 5. ИНТЕРФЕЙС
# ==========================================

st.title("SEO Анализатор Релевантности")

# Вкладки
tab_task, tab_comp = st.tabs(["📄 Постановка задачи", "🕵️ Конкуренты"])

# --- ВКЛАДКА 1: ЗАДАЧА ---
with tab_task:
    col1, col2 = st.columns(2)
    with col1:
        my_url = st.text_input("Ваш URL", placeholder="https://site.ru/page")
    with col2:
        query = st.text_input("Поисковой запрос", placeholder="пластиковые окна")
    
    st.info("Введите URL своей страницы для сравнения. Если оставить пустым, будет проанализирован только ТОП.")
    
    # Настройки внизу блока
    settings_task = render_settings_block("task")
    
    if st.button("ЗАПУСТИТЬ АНАЛИЗ 🚀", key="btn_task"):
        # Логика запуска (для Таба 1 нужен автопоиск)
        if not query:
            st.error("Введите поисковой запрос!")
            st.stop()
            
        with st.spinner("Сбор позиций в Google..."):
            try:
                # Эмуляция поиска (исключаем домены)
                excl_list = DEFAULT_EXCLUDE.split()
                found_urls = search(query, num_results=20, lang="ru")
                competitors = []
                count = 0
                for u in found_urls:
                    if my_url and u in my_url: continue # Пропуск своего сайта
                    if any(x in u for x in excl_list): continue
                    competitors.append(u)
                    count += 1
                    if count >= 10: break # Берем ТОП-10
                
                if not competitors:
                    st.error("Конкуренты не найдены. Попробуйте ручной режим.")
                else:
                    st.session_state['run_data'] = {
                        'my_url': my_url, 
                        'competitors': competitors,
                        'settings': settings_task
                    }
                    st.rerun() # Перезагрузка для отображения результата
            except Exception as e:
                st.error(f"Ошибка поиска: {e}")

# --- ВКЛАДКА 2: КОНКУРЕНТЫ ---
with tab_comp:
    manual_urls_text = st.text_area("Список URL конкурентов (каждый с новой строки)", height=150, placeholder="https://site1.ru\nhttps://site2.ru")
    
    # Настройки внизу блока
    settings_comp = render_settings_block("comp")
    
    if st.button("ЗАПУСТИТЬ АНАЛИЗ (По списку) 🚀", key="btn_comp"):
        comps = [u.strip() for u in manual_urls_text.split('\n') if u.strip()]
        if not comps:
            st.error("Список пуст!")
        else:
            st.session_state['run_data'] = {
                'my_url': my_url, # Берем из первой вкладки, если заполнен
                'competitors': comps,
                'settings': settings_comp
            }
            st.rerun()

# ==========================================
# 6. ВЫПОЛНЕНИЕ И ВЫВОД РЕЗУЛЬТАТОВ
# ==========================================

if 'run_data' in st.session_state:
    data = st.session_state['run_data']
    st.divider()
    st.header("📊 Результаты анализа")
    
    status = st.empty()
    bar = st.progress(0)
    
    # 1. Сбор данных
    my_body_lemmas = []
    my_anchor_lemmas = []
    
    # Если URL своего сайта задан - качаем
    if data['my_url']:
        status.info(f"Скачиваем ваш сайт: {data['my_url']}")
        b_txt, a_txt = get_page_data(data['my_url'], data['settings'])
        if b_txt:
            my_body_lemmas = get_lemmas(b_txt, data['settings'])
            my_anchor_lemmas = get_lemmas(a_txt, data['settings'])
        else:
            st.warning("Не удалось скачать ваш сайт. Таблица будет построена только по конкурентам.")
    
    # Качаем конкурентов
    corpus_data = []
    comps = data['competitors']
    
    status.info(f"Обработка {len(comps)} конкурентов...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(get_page_data, url, data['settings']): url for url in comps}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_url):
            try:
                b_txt, a_txt = future.result()
                if len(b_txt) > 100:
                    corpus_data.append({
                        'body': get_lemmas(b_txt, data['settings']),
                        'anchor': get_lemmas(a_txt, data['settings'])
                    })
            except: pass
            completed += 1
            bar.progress(completed / len(comps))
            
    if len(corpus_data) < 2:
        st.error("Недостаточно данных от конкурентов (меньше 2 успешных загрузок).")
    else:
        status.success("Расчет метрик...")
        bar.empty()
        
        # 2. Расчет таблицы
        df_result = calculate_metrics(
            corpus_data, 
            {'body': my_body_lemmas, 'anchor': my_anchor_lemmas}, 
            data['settings']
        )
        
        # 3. Вывод
        if not df_result.empty:
            # Сортировка по TF-IDF ТОП по умолчанию
            df_result = df_result.sort_values(by="TF-IDF ТОП", ascending=False)
            
            # Подсветка (условное форматирование)
            # Если "Ваш сайт" сильно меньше "Медианы" - красный, если больше Переспама - желтый
            st.dataframe(
                df_result,
                column_config={
                    "diff": None # Скрываем тех. колонку
                },
                use_container_width=True,
                height=800
            )
            
            # Кнопка скачивания
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Скачать в Excel (CSV)",
                csv,
                "gar_analysis.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.warning("Нет данных для отображения.")
