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
import os # Для работы с файловой системой

# ==========================================
# 0. ПАТЧ СОВМЕСТИМОСТИ (Для NLP)
# ==========================================
if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

# ==========================================
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO", page_icon="📊")

# ==========================================
# 2. АВТОРИЗАЦИЯ
# ==========================================
def check_password():
    if st.session_state.get("authenticated"):
        return True
   
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <style>
            .auth-container {
                display: flex; flex-direction: column; align-items: center;
                justify-content: center; padding: 2rem; background-color: white;
                border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-top: 5rem;
            }
            </style>
            <div class="auth-container">
                <h3>📊 GAR PRO</h3>
                <h3>Вход в систему</h3>
            </div>
        """, unsafe_allow_html=True)
        
        
        password = st.text_input("Пароль", type="password", key="password_input", label_visibility="collapsed")
        
        if st.button("ВОЙТИ", type="primary", use_container_width=True):
            if password == "jfV6Xel-Q7vp-_s2UYPO":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Неверный пароль")
    return False

if not check_password():
    st.stop()

# ==========================================
# 3. НАСТРОЙКИ API И РЕГИОНОВ
# ==========================================
# Убедитесь, что этот токен актуален!
ARSENKIN_TOKEN = "43acbbb60cb7989c05914ff21be45379"

# Словарь регионов (Название -> {yandex_id, google_id})
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

DEFAULT_EXCLUDE_DOMAINS = [
    "yandex.ru", "avito.ru", "beru.ru", "tiu.ru", "aliexpress.com", "ebay.com",
    "auto.ru", "2gis.ru", "sravni.ru", "toshop.ru", "price.ru", "pandao.ru",
    "instagram.com", "wikipedia.org", "rambler.ru", "hh.ru", "banki.ru", 
    "regmarkets.ru", "zoon.ru", "pulscen.ru", "prodoctorov.ru", "blizko.ru", 
    "domclick.ru", "satom.ru", "quto.ru", "edadeal.ru", "cataloxy.ru", 
    "irr.ru", "onliner.by", "shop.by", "deal.by", "yell.ru", 
    "profi.ru", 
    "irecommend.ru", "otzovik.com", "ozon.ru", "ozon.by", "market.yandex.ru", 
    "youtube.com", "gosuslugi.ru", "dzen.ru", "2gis.by", "wildberries.ru", 
    "rutube.ru", "vk.com", "facebook.com"
]
DEFAULT_EXCLUDE = "\n".join(DEFAULT_EXCLUDE_DOMAINS)
DEFAULT_STOPS = "рублей\nруб\nкупить\nцена\nшт\nсм\nмм\nкг\nкв\nм2\nстр\nул"

# Цвета
PRIMARY_COLOR = "#277EFF"
PRIMARY_DARK = "#1E63C4"
TEXT_COLOR = "#3D4858"
LIGHT_BG_MAIN = "#F1F5F9"
BORDER_COLOR = "#E2E8F0"
HEADER_BG = "#F0F7FF"
ROW_BORDER_COLOR = "#DBEAFE" 

st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .stApp {{ background-color: #FFFFFF !important; color: {TEXT_COLOR} !important; }}
        html, body, p, li, h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR} !important; }}
        .stButton button {{ background-color: {PRIMARY_COLOR} !important; color: white !important;
        border: none; border-radius: 6px; }}
        .stButton button:hover {{ background-color: {PRIMARY_DARK} !important;
        }}
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {LIGHT_BG_MAIN} !important;
        color: {TEXT_COLOR} !important; border: 1px solid {BORDER_COLOR} !important;
        }}
        div[data-testid="stDataFrame"] {{ border: 2px solid {PRIMARY_COLOR} !important;
        border-radius: 8px !important; }}
        div[data-testid="stDataFrame"] div[role="columnheader"] {{
            background-color: {HEADER_BG} !important;
        color: {PRIMARY_COLOR} !important; font-weight: 700 !important; border-bottom: 2px solid {PRIMARY_COLOR} !important;
        }}
        div[data-testid="stDataFrame"] div[role="gridcell"] {{
            background-color: #FFFFFF !important;
        color: {TEXT_COLOR} !important; border-bottom: 1px solid {ROW_BORDER_COLOR} !important;
        }}
        .legend-box {{ padding: 10px;
        background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 5px; font-size: 14px; margin-bottom: 10px;
        }}
        .text-red {{ color: #D32F2F; font-weight: bold;
        }}
        .text-bold {{ font-weight: 600;
        }}
        .sort-container {{ background-color: {LIGHT_BG_MAIN}; padding: 10px; border-radius: 8px; margin-bottom: 10px;
        border: 1px solid {BORDER_COLOR}; }}
        section[data-testid="stSidebar"] {{ background-color: #FFFFFF !important;
        border-left: 1px solid {BORDER_COLOR} !important; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. ЛОГИКА (БЭКЕНД)
# ==========================================

# Инициализация NLP
try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    USE_NLP = True
except Exception as e:
    morph = None
    USE_NLP = False
    st.sidebar.error(f"Ошибка загрузки NLP: {e}")

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'app_page' not in st.session_state:
    st.session_state.app_page = "Анализ" # Добавление состояния для навигации

# --- ФУНКЦИИ ДЛЯ ИСТОРИИ ЗАДАЧ ---
RESULTS_FILE = "gar_pro_results.json" # Файл для сохранения истории

def load_results():
    """Загружает историю результатов из JSON файла."""
    if not os.path.path.exists(RESULTS_FILE):
        return []
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_results(data):
    """Сохраняет историю результатов в JSON файл."""
    try:
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        st.error(f"Ошибка сохранения результатов: {e}")
        return False
        
# --- ФУНКЦИИ ДЛЯ СКАЧИВАНИЯ ---
def convert_df_to_csv(df):
    """Преобразование DataFrame в CSV строку (с разделителем ';')."""
    return df.to_csv(index=False, sep=';', encoding='utf-8')

def convert_df_to_xml(df, root_name="Results", row_name="Item"):
    """Преобразование DataFrame в простую XML строку."""
    data = df.to_dict(orient='records')
    xml_string = f'<?xml version="1.0" encoding="utf8"?>\n<{root_name}>\n'
    
    for record in data:
        xml_string += f'  <{row_name}>\n'
        for key, value in record.items():
            # Замена недопустимых символов в именах тегов
            tag_name = re.sub(r'[^a-zA-Z0-9_]', '', key.replace(' ', '_'))
            # Экранирование значений
            safe_value = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
            xml_string += f'    <{tag_name}>{safe_value}</{tag_name}>\n'
        xml_string += f'  </{row_name}>\n'
    
    xml_string += f'</{root_name}>'
    return xml_string


# --- ФУНКЦИЯ РАБОТЫ С API ARSENKIN ---
def get_arsenkin_urls(query, engine_type, region_name, depth_val=10):
# ... (оставшаяся часть функции get_arsenkin_urls остается без изменений) ...

# --- ФУНКЦИЯ АНАЛИЗА МЕТРИК ---
def calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, original_results):
    all_forms_map = defaultdict(set)
    
    # 1. Ваш сайт
    if not my_data or not my_data.get('body_text'):
        my_lemmas, my_forms, my_anchors, my_len = [], {}, [], 0
    else:
        my_lemmas, my_forms = process_text_detailed(my_data['body_text'], settings)
        my_anchors, _ = process_text_detailed(my_data['anchor_text'], settings)
        my_len = len(my_lemmas)
        for k, v in my_forms.items():
            all_forms_map[k].update(v)

    # Разделяем успешно скачанные данные для анализа лемм и статистики
    comp_data_parsed = [d for d in comp_data_full if d.get('body_text')]
    
    # 2. Конкуренты (только успешно скачанные)
    comp_docs = []
    for p in comp_data_parsed:
        body, c_forms = process_text_detailed(p['body_text'], settings)
        anchor, _ = process_text_detailed(p['anchor_text'], settings)
        comp_docs.append({'body': body, 'anchor': anchor})
        for k, v in c_forms.items():
            all_forms_map[k].update(v)
    
    # Если нет успешно скачанных конкурентов, мы не можем рассчитать релевантность
    if not comp_docs:
        # Тем не менее, нам нужна таблица релевантности, чтобы показать, кто был в ТОПе
        
        table_rel_fallback = []
        # Добавляем все URL, которые пришли из API/ручного списка, чтобы показать их позиции
        for item in original_results:
            domain = urlparse(item['url']).netloc
            table_rel_fallback.append({
                "Домен": domain, 
                "Позиция": item['pos'],
                "Ширина (балл)": 0, "Глубина (балл)": 0
            })
        
        if my_data and my_data.get('domain'):
            my_label = f"{my_data['domain']} (Вы)"
        else:
            my_label = "Ваш сайт"
        
        # Добавляем Ваш сайт
        table_rel_fallback.append({
            "Домен": my_label, 
            "Позиция": my_serp_pos if my_serp_pos > 0 else len(original_results) + 1,
            "Ширина (балл)": 0, "Глубина (балл)": 0
        })
        
        table_rel_df = pd.DataFrame(table_rel_fallback).sort_values(by='Позиция', ascending=True).reset_index(drop=True)
        # --- ИЗМЕНЕНИЕ 1: Добавление относительного ранга (№) в начало ---
        table_rel_df.insert(0, '№', table_rel_df.index + 1)
        # -----------------------------------------------------------------
        
        return {"depth": pd.DataFrame(), "hybrid": pd.DataFrame(), "ngrams": pd.DataFrame(), "relevance_top": table_rel_df, "my_score": {"width": 0, "depth": 0}}


    # Дальше расчеты идут только по успешно скачанным comp_docs
    avg_len = np.mean([len(d['body']) for d in comp_docs])
    norm_k = (my_len / avg_len) if (settings['norm'] and my_len > 0 and avg_len > 0) else 1.0
    
    vocab = set(my_lemmas)
    for d in comp_docs: vocab.update(d['body'])
    vocab = sorted(list(vocab))
    N = len(comp_docs) # N - количество успешно скачанных документов
    doc_freqs = Counter()
    for d in comp_docs:
        for w in set(d['body']): doc_freqs[w] += 1
        
    # ... (Остальная часть расчета table_depth, table_hybrid, table_ngrams) ...
    
    # Расчет релевантности для таблицы TOP
    # 2. Баллы для конкурентов (по успешно скачанным)
    competitor_stats_raw = []
    # ... (Расчет raw_width, raw_depth) ...
    # ... (Определение max_width_top, max_depth_top) ...
    # 3. Баллы конкурентов (рассчитываем по всем, кто был в original_results)
    table_rel = []
    for c in competitor_stats_raw:
        score_w = int(round((c['raw_w'] / max_width_top) * 100))
        score_d = int(round((c['raw_d'] / max_depth_top) * 100))
        table_rel.append({
            "Домен": c['domain'],
            "Позиция": c['pos'], # Это фактическая позиция в SERP
            "Ширина (балл)": score_w,
            "Глубина (балл)": score_d
        })
        
    # 4. Баллы для ВАШЕГО сайта
    # ... (Расчет my_score_w, my_score_d) ...
    
    # Добавляем ВАШ сайт в таблицу
    if my_data and my_data.get('domain'):
        my_label = f"{my_data['domain']} (Вы)"
    else:
        my_label = "Ваш сайт"
        
    table_rel.append({
        "Домен": my_label,
        "Позиция": my_serp_pos if my_serp_pos > 0 else len(original_results) + 1, # Ставим после последнего конкурента
        "Ширина (балл)": my_score_w,
        "Глубина (балл)": my_score_d
    })

    # Сортируем таблицу релевантности по позиции
    table_rel_df = pd.DataFrame(table_rel)
    table_rel_df = table_rel_df.sort_values(by='Позиция', ascending=True).reset_index(drop=True)

    # --- ИЗМЕНЕНИЕ 1: Добавление относительного ранга (№) в начало ---
    table_rel_df.insert(0, '№', table_rel_df.index + 1)
    # -----------------------------------------------------------------
    
    return {
        "depth": pd.DataFrame(table_depth),
        "hybrid": pd.DataFrame(table_hybrid),
        "ngrams": pd.DataFrame(table_ngrams),
        "relevance_top": table_rel_df,
        "my_score": {"width": my_score_w, "depth": my_score_d}
    }

# ==========================================
# 5. ФУНКЦИЯ ОТОБРАЖЕНИЯ (FINAL)
# ==========================================

def render_paginated_table(df, title_text, key_prefix, default_sort_col=None, use_abs_sort_default=False):
    # --- ИЗМЕНЕНИЕ 3: Добавление кнопок скачивания CSV/XML ---
    st.markdown(f"#### {title_text}")

    df_for_download = df.copy() 
    # Удаляем служебные колонки перед скачиванием
    if 'diff_abs' in df_for_download.columns:
        df_for_download = df_for_download.drop(columns=['diff_abs'])
    if 'is_missing' in df_for_download.columns:
        df_for_download = df_for_download.drop(columns=['is_missing'])
        
    csv_data = convert_df_to_csv(df_for_download)
    xml_data = convert_df_to_xml(df_for_download, root_name=key_prefix, row_name="item")

    c_dl1, c_dl2, c_dl_spacer = st.columns([1, 1, 8])

    with c_dl1:
        st.download_button(
            label="⬇️ CSV",
            data=csv_data,
            file_name=f"{key_prefix}.csv",
            mime="text/csv",
            key=f"{key_prefix}_dl_csv",
            use_container_width=True
        )

    with c_dl2:
        st.download_button(
            label="⬇️ XML",
            data=xml_data,
            file_name=f"{key_prefix}.xml",
            mime="text/xml",
            key=f"{key_prefix}_dl_xml",
            use_container_width=True
        )
    # -------------------------------------------------------------------------
    
    # Оригинальная логика сортировки и пагинации
    if default_sort_col and default_sort_col in df.columns:
        # ... (логика сортировки)
        if use_abs_sort_default:
            df = df.sort_values(by='diff_abs', ascending=False).reset_index(drop=True)
        else:
            df = df.sort_values(by=default_sort_col, ascending=False).reset_index(drop=True)
            
    df = df.reset_index(drop=True)
    df.index = df.index + 1 # Стандартный индекс для отображения в Streamlit (1, 2, 3...)
    ROWS_PER_PAGE = 20 
    # ... (Остальная часть пагинации и отрисовки) ...
    
    # ВЫВОД ТАБЛИЦЫ
    dynamic_height = (len(df_view) * 35) + 40
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=dynamic_height,
        column_config={c: None for c in cols_to_hide}
    )
    # ... (Кнопки переключения страниц) ...
    st.markdown("---")

# --- ФУНКЦИЯ ОТОБРАЖЕНИЯ СТРАНИЦЫ ИСТОРИИ ---
def render_history_page():
    st.title("📊 История Анализов")
    st.markdown("Здесь хранятся результаты всех ваших предыдущих задач.")
    
    all_results = load_results()
    
    if not all_results:
        st.info("История задач пуста.")
        return
        
    for idx, task in enumerate(all_results):
        # Используем expanser для каждого результата
        header = f"[{task['date_str']}] {task['query']} ({task['url']} / {task['region']})"
        with st.expander(header):
            st.markdown(f"**Запрос:** {task['query']}")
            st.markdown(f"**URL:** {task['url']}")
            st.markdown(f"**Регион/ПС:** {task['region']} / {task['engine']}")
            
            # Отображение ключевых метрик
            st.markdown(f""" 
                <div style='background-color: {LIGHT_BG_MAIN}; padding: 10px; border-radius: 6px; border: 1px solid {BORDER_COLOR};'>
                    <h5 style='margin:0; color: {PRIMARY_COLOR};'>Результат вашего сайта (в баллах)</h5>
                    <p style='margin:5px 0 0 0;'>Ширина (охват): <b>{task['my_score']['width']}</b> | Глубина (оптимизация): <b>{task['my_score']['depth']}</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Кнопка для загрузки данных из истории в текущий анализ
            if st.button(f"Показать детальные таблицы", key=f"show_details_{idx}"):
                # Конвертируем JSON-структуру обратно в DataFrames
                st.session_state.analysis_results = {
                    'depth': pd.DataFrame.from_records(task['depth']),
                    'hybrid': pd.DataFrame.from_records(task['hybrid']),
                    'ngrams': pd.DataFrame.from_records(task['ngrams']),
                    'relevance_top': pd.DataFrame.from_records(task['relevance_top']),
                    'my_score': task['my_score']
                }
                st.session_state.analysis_done = True
                st.session_state.app_page = "Анализ" # Переключаем на страницу анализа
                st.rerun()

# ==========================================
# 6. ИНТЕРФЕЙС
# ==========================================

col_main, col_sidebar = st.columns([65, 35])

with col_sidebar:
    st.session_state.app_page = st.radio(
        "Навигация",
        ["Анализ", "История"],
        index=0 if st.session_state.app_page == "Анализ" else 1,
        key="app_page_select"
    )
    
if st.session_state.app_page == "История":
    render_history_page()
    
elif st.session_state.app_page == "Анализ":
    
    with col_main:
        st.title("SEO Анализатор Релевантности")
        # ... (Остальная часть интерфейса: ввод URL, запроса, настроек) ...

        # ... (Настройки в сайдбаре - остаются там же, но после навигации) ...
        
        with col_sidebar:
            st.markdown("#####⚙️ Настройки")
            ua = st.selectbox("User-Agent", ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "YandexBot/3.0"], key="settings_ua")
            search_engine = st.selectbox("Поисковая система", ["Яндекс", "Google", "Яндекс + Google"], key="settings_search_engine")
            region = st.selectbox("Регион поиска", list(REGION_MAP.keys()), key="settings_region")
            device = st.selectbox("Устройство", ["Desktop", "Mobile"], key="settings_device")
            # Максимальная глубина, которую позволяет API - 30. 
            top_n = st.selectbox("Глубина сбора (ТОП)", [10, 20, 30], index=0, key="settings_top_n")
            # ... (Остальная часть настроек) ...
            
        # ... (Оригинальная секция 7. ВЫПОЛНЕНИЕ) ...
        # ==========================================
        # 7. ВЫПОЛНЕНИЕ (СКОРРЕКТИРОВАННАЯ ЛОГИКА СБОРА)
        # ==========================================
        if st.session_state.get('start_analysis_flag'):
            st.session_state.start_analysis_flag = False
            # ... (Проверки входных данных) ... 
            
            # ... (Сбор данных, парсинг, и т.д.) ...
            
            # ... (ВЫЗОВ calculate_metrics) ...
            with st.spinner("Анализ данных..."):
                results = calculate_metrics(
                    comp_data_full, my_data, settings, my_serp_pos, target_urls_raw # Используем список URL:pos, которые мы отобрали
                )
                
            st.session_state.analysis_results = results
            st.session_state.analysis_done = True
            
            # --- ИЗМЕНЕНИЕ 2: Сохранение результата в историю ---
            if st.session_state.analysis_results:
                # Конвертируем DataFrames в JSON-совместимый формат (список словарей)
                new_result = {
                    "timestamp": time.time(),
                    "date_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "query": st.session_state.get('query_input', 'N/A'),
                    "url": st.session_state.get('my_url_input', 'N/A'),
                    "region": st.session_state.settings_region,
                    "engine": st.session_state.settings_search_engine,
                    "depth": results['depth'].to_dict(orient='records'),
                    "hybrid": results['hybrid'].to_dict(orient='records'),
                    "ngrams": results['ngrams'].to_dict(orient='records'),
                    "relevance_top": results['relevance_top'].to_dict(orient='records'),
                    "my_score": results['my_score']
                }
                
                all_results = load_results()
                all_results.insert(0, new_result) # Добавляем в начало
                save_results(all_results)
            # ---------------------------------------------------
            
            st.rerun()

        if st.session_state.analysis_done and st.session_state.analysis_results:
            results = st.session_state.analysis_results
            st.success("Анализ готов!")
            # ... (Отображение результатов) ...
            
            # ... (Вызовы render_paginated_table - остаются без изменений, но теперь они включают кнопки скачивания) ...
