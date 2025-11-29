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
# 1. НАСТРОЙКА СТРАНИЦЫ И СТИЛИ
# ==========================================

st.set_page_config(
    page_title="SEO Анализатор Релевантности",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Внедряем CSS стили на основе шрифтов из вашего файла (Inter)
st.markdown("""
    <style>
        /* Подключение шрифта Inter */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        /* Основной шрифт приложения */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #171717;
        }
        
        /* Заголовки */
        h1, h2, h3 {
            font-weight: 700;
            color: #0F172A;
        }

        /* Настройка контейнера приложения */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* Стилизация полей ввода */
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            padding: 10px 12px;
            font-size: 16px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #3B82F6;
            box-shadow: 0 0 0 1px #3B82F6;
        }
        
        /* Стилизация текстовых областей */
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 1px solid #E2E8F0;
        }

        /* Стилизация кнопок (акцентная) */
        div.stButton > button {
            background-color: #2563EB; /* Синий профессиональный */
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            border: none;
            width: 100%;
            transition: all 0.2s;
        }
        div.stButton > button:hover {
            background-color: #1D4ED8;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        /* Стилизация таблиц (Dataframe) */
        div[data-testid="stDataFrame"] {
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            overflow: hidden;
        }

        /* Карточки метрик */
        div[data-testid="metric-container"] {
            background-color: #F8FAFC;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #F1F5F9;
        }

        /* Вкладки */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            font-weight: 600;
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
        st.markdown("<h2 style='text-align: center;'>🔒 Вход в систему</h2>", unsafe_allow_html=True)
        st.info("Введите пароль доступа к SEO Анализатору")
        password = st.text_input("Пароль", type="password", label_visibility="collapsed", placeholder="Введите пароль...")
        
        if st.button("Войти в систему"):
            # === ПАРОЛЬ (меняйте здесь) ===
            if password == "admin123":  
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("⛔ Неверный пароль")
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
    norm = len(my_text.split()) / np.mean([len(d.split()) for d in corpus]) if settings['norm'] else 1.0
    
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
st.markdown("TF-IDF анализ контента на основе ТОП выдачи")

# Создаем вкладки
tab1, tab2, tab3 = st.tabs(["📋 Задача", "🕵️ Конкуренты", "⚙️ Настройки"])

with tab1:
    st.markdown("### Постановка задачи")
    col1, col2 = st.columns(2)
    with col1:
        my_url = st.text_input("URL вашего сайта", placeholder="https://site.ru/page")
    with col2:
        query = st.text_input("Поисковой запрос", placeholder="Например: купить окна")
    
    st.info("💡 Введите URL страницы, которую нужно оптимизировать, и основной ключевой запрос.")

with tab2:
    st.markdown("### Источник конкурентов")
    search_method = st.radio("Как собрать конкурентов?", ["Google Поиск (Авто)", "Свой список URL"], horizontal=True)
    
    competitors_list = []
    
    if search_method == "Google Поиск (Авто)":
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            top_n = st.selectbox("Глубина ТОПа:", [5, 10, 15, 20], index=1)
        with col_s2:
            st.warning("⚠️ Google может блокировать частые авто-запросы.")
        excludes = st.text_area("Исключить домены (каждый с новой строки):", "\n".join(DEFAULT_EXCLUDE), height=100)
    else:
        manual_urls = st.text_area("Список URL конкурентов (каждый с новой строки):", height=200, placeholder="https://comp1.ru\nhttps://comp2.ru")

with tab3:
    st.markdown("### Параметры анализа")
    
    with st.expander("Основные настройки", expanded=True):
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            s_noindex = st.checkbox("Исключать noindex", True)
            s_alt = st.checkbox("Учитывать Alt/Title", False)
            s_num = st.checkbox("Учитывать числа", False)
        with col_opt2:
            s_norm = st.checkbox("Нормировать по длине текста", True, help="Корректирует медиану, если ваш текст длиннее или короче среднего по ТОПу")
            s_std_stops = st.checkbox("Убирать предлоги/союзы", True)
    
    with st.expander("Стоп-слова и User-Agent"):
        custom_stops = st.text_area("Свои стоп-слова:", "\n".join(DEFAULT_STOPS))
        ua = st.text_input("User-Agent бота:", "Mozilla/5.0 (compatible; Hybrid-Analyzer/1.0;)")

# Нижняя панель с кнопкой
st.divider()

if st.button("ЗАПУСТИТЬ АНАЛИЗ 🚀"):
    if not my_url:
        st.error("❌ Вы не ввели URL своего сайта!")
        st.stop()
        
    # Сбор настроек в словарь
    settings = {
        "noindex": s_noindex, "alt_title": s_alt, "numbers": s_num,
        "norm": s_norm, "std_stops": s_std_stops, "ua": ua,
        "custom_stops": custom_stops.split()
    }
    
    # Логика получения списка конкурентов
    comps = []
    if search_method == "Google Поиск (Авто)":
        if not query:
            st.error("❌ Для поиска нужен запрос!")
            st.stop()
        try:
            excl_list = excludes.split()
            # Пробуем искать
            found = search(query, num_results=top_n*2, lang="ru")
            count = 0
            for u in found:
                if u == my_url: continue
                if any(x in u for x in excl_list): continue
                comps.append(u)
                count += 1
                if count >= top_n: break
        except Exception as e:
            st.error(f"Ошибка поиска: {e}. Попробуйте ручной список.")
    else:
        if manual_urls:
            comps = [u.strip() for u in manual_urls.split('\n') if u.strip()]
        
    if not comps:
        st.error("❌ Список конкурентов пуст.")
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