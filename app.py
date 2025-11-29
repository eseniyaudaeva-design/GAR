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
from googlesearch import search # Импорт для поиска, хотя в интерфейсе не используется

# ==========================================
# 1. НАСТРОЙКА СТРАНИЦЫ И СТИЛИ
# ==========================================

st.set_page_config(
    page_title="SEO Анализатор Релевантности",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Внедряем АКТУАЛЬНЫЕ CSS СТИЛИ (Manrope, светло-серый фон, карточки с тенью)
st.markdown("""
    <style>
        /* --- 0. Подключение шрифта Manrope --- */
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
        
        /* --- 1. ГЛОБАЛЬНЫЙ ФОН И ШРИФТ (СВЕТЛЫЙ) --- */
        [data-testid="stAppViewContainer"] {
            background-color: #F3F6F9 !important; /* Светло-серый фон */
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
            font-weight: 800 !important; /* Экстра-жирный */
        }
        
        /* --- 3. КАРТОЧКИ (БЕЛЫЕ БЛОКИ С ТЕНЬЮ) --- */
        /* Универсальный класс для белых блоков */
        .css-card {
            background-color: #FFFFFF;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            border: 1px solid #E2E8F0;
            margin-bottom: 24px;
        }
        
        /* Карточки метрик (Результаты анализа) */
        div[data-testid="metric-container"] {
            background-color: #FFFFFF !important;
            padding: 15px;
            border-radius: 16px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        }

        /* --- 4. ПОЛЯ ВВОДА (БЕЛЫЙ ФОН, ЧЕРНЫЙ ТЕКСТ) --- */
        .stTextInput input, 
        .stTextArea textarea, 
        .stSelectbox div[data-baseweb="select"] {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: 2px solid #E2E8F0 !important;
            border-radius: 8px !important;
            font-size: 15px !important;
        }
        /* Фокус на поле */
        .stTextInput input:focus, 
        .stTextArea textarea:focus, 
        .stSelectbox div[data-baseweb="select"]:focus-within {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        }
        
        /* --- 5. КНОПКИ (СИНИЙ ГРАДИЕНТ) --- */
        /* Кнопка "ЗАПУСТИТЬ АНАЛИЗ" */
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
            width: 100% !important; 
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(37, 99, 235, 0.5) !important;
        }
        /* Кнопка входа */
        div[data-testid="stForm"] div.stButton > button {
            box-shadow: none !important;
            padding: 0.6rem 1.2rem !important;
            font-size: 16px !important;
            text-transform: none;
            width: auto !important;
        }

        /* --- 6. ТАБЛИЦЫ (ЧИТАЕМЫЕ) --- */
        div[data-testid="stDataFrame"] {
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        
        /* --- 7. ДОПОЛНИТЕЛЬНЫЕ ЭЛЕМЕНТЫ --- */
        /* Заголовок-экспандер */
        .streamlit-expanderHeader {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            color: #0F172A !important;
            font-weight: 700;
        }
        /* Вкладки (Tabs) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 0 16px;
            font-weight: 600;
            color: #64748B;
            transition: all 0.2s;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3B82F6 !important;
            color: white !important;
            border-color: #3B82F6 !important;
        }
        /* Убираем лишние отступы */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
            max-width: 1200px;
        }
        /* Скрытие стандартного меню Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. БЛОК АВТОРИЗАЦИИ
# ==========================================
def check_password():
    """Возвращает True, если пароль верный."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    # Центрируем форму входа
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Применяем стили 'css-card' для блока входа
        st.markdown(f'<div class="css-card" style="margin-top: 50px;">'
                    f"<h2 style='text-align: center; color: #0F172A !important; font-weight: 800 !important;'>🔒 Вход в систему</h2>", unsafe_allow_html=True)
        
        st.info("Введите пароль доступа к SEO Анализатору")
        password = st.text_input("Пароль", type="password", label_visibility="collapsed", placeholder="Введите пароль...")
        
        if st.button("Войти в систему"):
            # === ПАРОЛЬ (меняйте здесь) ===
            if password == "admin123":  
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("⛔ Неверный пароль")
        
        st.markdown('</div>', unsafe_allow_html=True) # Закрываем div
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
            return spec.args, spec.varargs, spec.varkw, inspect.getfullargspec(func).defaults
        inspect.getargspec = getargspec
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    USE_NLP = True
except Exception:
    morph = None
    USE_NLP = False

# --- Поиск Google ---
# Доступность поиска все еще полезна для бэкенд-логики, даже если не используется в UI
try:
    #from googlesearch import search
    USE_SEARCH = True
except ImportError:
    USE_SEARCH = False

# --- Списки по умолчанию ---
DEFAULT_EXCLUDE = ["yandex.ru", "avito.ru", "ozon.ru", "wildberries.ru", "wikipedia.org", "youtube.com", "dzen.ru", "rutube.ru", "hh.ru"]
DEFAULT_STOPS = ["рублей", "руб", "купить", "цена", "шт", "см", "мм", "кг", "кв", "м2"]
STANDARD_STOP_WORDS = {
    'и', 'в', 'на', 'с', 'к', 'по', 'за', 'от', 'до', 'это', 'мы', 'вы', 'он', 'она', 'они', 'их', 'ее', 'его', 'мне',
    'тебе', 'себе', 'для', 'что', 'как', 'так', 'но', 'или', 'а', 'чтобы', 'же', 'бы', 'да', 'нет', 'у', 'без', 'под',
    'над', 'перед', 'при', 'через', 'между', 'среди', 'после', 'вместо', 'около', 'вокруг', 'со', 'из', 'из-за', 'из-под'
}

# --- Функции ---
def get_word_forms(lemma):
    if not USE_NLP or not morph: return lemma
    parses = morph.parse(lemma)
    if not parses: return lemma
    forms = {tag.word for tag in parses[0].lexeme}
    return ", ".join(list(forms)[:5])

def clean_text(html, settings):
    soup = BeautifulSoup(html, 'html.parser')
    
    if settings['noindex']:
        for tag in soup.find_all(['noindex', 'script', 'style', 'head', 'footer', 'nav', 'header', 'aside']):
            tag.decompose()
    else:
        for tag in soup(['script', 'style', 'head']):
            tag.decompose()
            
    text = soup.get_text(separator=' ')
    
    if settings['alt_title']:
        for img in soup.find_all('img', alt=True):
            text += " " + img['alt']
        for t in soup.find_all(title=True):
            text += " " + t['title']
            
    pattern = r'[а-яА-ЯёЁ0-9a-zA-Z]+' if settings['numbers'] else r'[а-яА-ЯёЁa-zA-Z]+'
    words = re.findall(pattern, text)
    
    clean_words = []
    custom_stop_list = set(w.lower() for w in settings['custom_stops'])
    
    for w in words:
        w_lower = w.lower()
        if len(w) < 2 or w_lower in custom_stop_list: continue
        
        if USE_NLP:
            p = morph.parse(w_lower)[0]
            if settings['std_stops']:
                if 'PREP' in p.tag or 'CONJ' in p.tag or 'PRCL' in p.tag:
                    continue
            clean_words.append(p.normal_form)
        else:
            clean_words.append(w_lower)
            
    return " ".join(clean_words)

def get_page(url, settings):
    headers = {'User-Agent': settings['ua']}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return clean_text(r.text, settings)
    except:
        return ""
    return ""

def run_analysis(my_url, competitors, settings):
    # Используем контейнеры для чистоты UI
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    status_container.info(f"📥 Скачиваем ваш сайт: {my_url}")
    my_text = get_page(my_url, settings)
    
    if not my_text:
        status_container.error("❌ Не удалось скачать ваш сайт! Проверьте URL.")
        return None
        
    corpus = []
    status_container.info(f"🚀 Анализ {len(competitors)} конкурентов...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(get_page, url, settings): url for url in competitors}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_url):
            txt = future.result()
            if len(txt) > 50:
                corpus.append(txt)
            completed += 1
            progress_bar.progress(completed / len(competitors))
            
    if len(corpus) < 2:
        status_container.error("❌ Мало данных (менее 2 доступных конкурентов).")
        return None
        
    status_container.success("✅ Данные собраны! Расчет TF-IDF...")
    progress_bar.empty()
    
    # Расчеты
    all_words = set(my_text.split())
    for doc in corpus:
        all_words.update(doc.split())
    all_words = sorted(list(all_words))
    
    def count_vec(text, vocab):
        cnt = Counter(text.split())
        return [cnt[w] for w in vocab]
    
    my_vec = np.array(count_vec(my_text, all_words))
    comp_vecs = np.array([count_vec(doc, all_words) for doc in corpus])
    
    medians = np.median(comp_vecs, axis=0)
    
    data = []
    # Исправлена логика расчета norm, чтобы избежать деления на ноль при пустом корпусе, хотя это уже обработано выше.
    comp_lengths = [len(d.split()) for d in corpus]
    avg_comp_len = np.mean(comp_lengths) if comp_lengths else 1
    norm = len(my_text.split()) / avg_comp_len if settings['norm'] else 1.0
    
    for i, word in enumerate(all_words):
        med = medians[i]
        my_val = my_vec[i]
        
        target = int(med * 1.3 * norm) # Коэффициент 1.3
        diff = target - my_val
        
        if (med > 0 or my_val > 0):
            # Форматирование для вывода
            rec_text = "✅ OK"
            if diff > 0: rec_text = f"➕ Добавить {diff}"
            elif diff < 0: rec_text = f"➖ Убрать {abs(diff)}"
            
            # Фильтр "мусора"
            if med >= 0.5 or my_val >= 1:
                data.append({
                    "Слово": word,
                    "Медиана (ТОП)": round(med, 1),
                    "На сайте": int(my_val),
                    "Рекомендация": rec_text,
                    "Сортировка": abs(diff) # Скрытое поле для сортировки
                })
                
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by="Сортировка", ascending=False).drop(columns=["Сортировка"])
        return df
    return None

# ==========================================
# 4. ИНТЕРФЕЙС ПРИЛОЖЕНИЯ
# ==========================================

st.title("SEO Анализатор Релевантности")
st.markdown("Профессиональный инструмент TF-IDF анализа для оптимизации контента")
st.markdown("---") 

# --- ЗАКРЕПЛЕННЫЙ ВЕРХНИЙ БЛОК: МОЙ URL И ЗАПРОС ---

st.markdown('<div class="css-card">', unsafe_allow_html=True) 
st.markdown("### 📋 URL и Ключевой Запрос")
col1, col2 = st.columns(2)
with col1:
    my_url = st.text_input("URL вашей страницы", placeholder="https://site.ru/page", key="my_url_input")
with col2:
    query = st.text_input("Поисковой запрос", placeholder="Например: купить окна", key="query_input")
st.markdown('</div>', unsafe_allow_html=True)


# --- БЛОКИ ВХОДНЫХ ДАННЫХ И НАСТРОЕК ---
tab1, tab2 = st.tabs(["🕵️ Конкуренты", "⚙️ Настройки Парсинга"])

with tab1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True) 
    st.markdown("### Источник конкурентов")
    
    # Одно поле для ввода URL конкурентов
    manual_urls = st.text_area(
        "Список URL конкурентов (каждый с новой строки):", 
        height=300, 
        placeholder="https://comp1.ru\nhttps://comp2.ru\n..."
    )

    # --- НАСТРОЙКИ ПАРСИНГА - ЗАКРЕПЛЕНЫ ВНИЗУ БЛОКА ---
    with st.expander("Расширенные настройки User-Agent и Нормирования"):
        col_ua1, col_ua2 = st.columns(2)
        with col_ua1:
            ua = st.text_input("User-Agent бота:", "Mozilla/5.0 (compatible; Hybrid-Analyzer/1.0;)")
        with col_ua2:
            s_norm = st.checkbox(
                "Нормировать по длине текста", 
                True, 
                help="Корректирует медиану, если ваш текст длиннее или короче среднего по ТОПу"
            )

    st.markdown('</div>', unsafe_allow_html=True) # Закрываем css-card


with tab2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True) 
    st.markdown("### Параметры очистки текста (Content Filtering)")
    
    # --- НАСТРОЙКИ ОЧИСТКИ ТЕКСТА ---
    with st.expander("Фильтрация контента", expanded=True):
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            s_noindex = st.checkbox("Исключать noindex", True)
            s_alt = st.checkbox("Учитывать Alt/Title", False)
        with col_opt2:
            s_num = st.checkbox("Учитывать числа", False)
            s_std_stops = st.checkbox("Убирать предлоги/союзы", True)
    
    with st.expander("Управление Стоп-словами"):
        custom_stops = st.text_area("Свои стоп-слова (каждое с новой строки):", "\n".join(DEFAULT_STOPS))

    st.markdown('</div>', unsafe_allow_html=True) # Закрываем css-card


# Нижняя панель с кнопкой
st.divider()

if st.button("ЗАПУСТИТЬ АНАЛИЗ 🚀"):
    if not my_url or not query:
        st.error("❌ Вы не ввели URL своего сайта и/или Поисковой запрос!")
        st.stop()
        
    # Сбор настроек в словарь
    settings = {
        "noindex": s_noindex, "alt_title": s_alt, "numbers": s_num,
        "norm": s_norm, "std_stops": s_std_stops, "ua": ua,
        "custom_stops": custom_stops.split()
    }
    
    # Логика получения списка конкурентов
    comps = [u.strip() for u in manual_urls.split('\n') if u.strip()]
    
    # Здесь можно добавить логику Google Search, если пользователь захочет (закомментировано)
    # Если вы хотите вернуть авто-поиск, добавьте radio-кнопку и соответствующую логику.
    # Сейчас мы просто используем ручной список.
    
    if not comps:
        st.error("❌ Список конкурентов пуст. Введите URL конкурентов в соответствующее поле.")
    else:
        # ЗАПУСК БЭКЕНДА
        df_res = run_analysis(my_url, comps, settings)
        
        if df_res is not None:
            st.markdown("### 📊 Результаты анализа")
            
            # Подсветка строк для красоты
            def highlight_rec(val):
                if "Добавить" in str(val): return 'color: #166534; font-weight: bold; background-color: #dcfce7' # Зеленый
                if "Убрать" in str(val): return 'color: #991b1b; font-weight: bold; background-color: #fee2e2' # Красный
                return ''

            st.dataframe(
                df_res.style.map(highlight_rec, subset=['Рекомендация']),
                use_container_width=True, 
                height=600
            )
