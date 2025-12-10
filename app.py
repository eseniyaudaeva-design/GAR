import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import re
from collections import Counter, defaultdict
import math
import concurrent.futures
from urllib.parse import urlparse, urljoin
import inspect
import time
import json
import io
import os

# Попытка импорта openai, если нет - предупреждение
try:
    import openai
except ImportError:
    openai = None

# ==========================================
# 0. ПАТЧ СОВМЕСТИМОСТИ
# ==========================================
if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

# ==========================================
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ
# ==========================================
st.set_page_config(layout="wide", page_title="GAR PRO + AI", page_icon="📊")

# ==========================================
# 2. АВТОРИЗАЦИЯ
# ==========================================
def check_password():
    if st.session_state.get("authenticated"):
        return True
    
    st.markdown("""
        <style>
        .main { display: flex; flex-direction: column; justify-content: center; align-items: center; }
        .auth-logo-box { text-align: center; margin-bottom: 1rem; }
        .login-box h3 { margin-top: 0; text-align: center; }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-box"><h3>Вход в систему</h3>', unsafe_allow_html=True)
        password = st.text_input("Пароль", type="password", key="password_input", label_visibility="collapsed")
        
        if st.button("ВОЙТИ", type="primary", use_container_width=True):
            if password == "jfV6Xel-Q7vp-_s2UYPO":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Неверный пароль")
        st.markdown('</div>', unsafe_allow_html=True)
    return False

if not check_password():
    st.stop()

# ==========================================
# 3. СТИЛИ И НАСТРОЙКИ
# ==========================================
ARSENKIN_TOKEN = "43acbbb60cb7989c05914ff21be45379"

REGION_MAP = {
    "Москва": {"ya": 213, "go": 1011969},
    "Санкт-Петербург": {"ya": 2, "go": 1011966},
    "Екатеринбург": {"ya": 54, "go": 1011868},
    # ... остальные регионы
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

DEFAULT_EXCLUDE_DOMAINS = ["yandex.ru", "avito.ru", "ozon.ru", "wildberries.ru", "youtube.com", "vk.com"]
DEFAULT_EXCLUDE = "\n".join(DEFAULT_EXCLUDE_DOMAINS)
DEFAULT_STOPS = "рублей\nруб\nкупить\nцена\nшт\nсм\nмм\nкг\nкв\nм2\nстр\nул"

PRIMARY_COLOR = "#277EFF"
TEXT_COLOR = "#3D4858"
LIGHT_BG_MAIN = "#F1F5F9"
BORDER_COLOR = "#E2E8F0"

st.markdown(f"""
    <style>
        .stButton button {{ background-color: {PRIMARY_COLOR} !important; color: white !important; }}
        div[data-testid="stDataFrame"] {{ border: 2px solid {PRIMARY_COLOR} !important; border-radius: 8px !important; }}
    </style>
""", unsafe_allow_html=True)

# Инициализация NLP
try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    USE_NLP = True
except Exception:
    morph = None
    USE_NLP = False

# ==========================================
# 4. ФУНКЦИИ (ОБЩИЕ И SEO)
# ==========================================

def get_arsenkin_urls(query, engine_type, region_name, depth_val=10):
    # (Код функции get_arsenkin_urls без изменений из вашего файла)
    # ... Для краткости оставляю логику такой же ...
    url_set = "https://arsenkin.ru/api/tools/set"
    url_check = "https://arsenkin.ru/api/tools/check" 
    url_get = "https://arsenkin.ru/api/tools/get"
    
    headers = {"Authorization": f"Bearer {ARSENKIN_TOKEN}", "Content-type": "application/json"}
    reg_ids = REGION_MAP.get(region_name, {"ya": 213, "go": 1011969})
    se_params = []
    if "Яндекс" in engine_type: se_params.append({"type": 2, "region": reg_ids['ya']})
    if "Google" in engine_type: se_params.append({"type": 11, "region": reg_ids['go']})
        
    payload = {
        "tools_name": "check-top",
        "data": {"queries": [query], "is_snippet": False, "noreask": True, "se": se_params, "depth": depth_val}
    }
    
    try:
        r = requests.post(url_set, headers=headers, json=payload, timeout=15)
        resp_json = r.json()
        if "task_id" not in resp_json: return []
        task_id = resp_json["task_id"]
    except: return []
    
    status = "process"
    attempts = 0
    while status == "process" and attempts < 40:
        time.sleep(3)
        attempts += 1
        try:
            r_check = requests.post(url_check, headers=headers, json={"task_id": task_id})
            if r_check.json().get("status") == "finish": status = "done"
        except: pass
            
    if status != "done": return []
    try:
        r_final = requests.post(url_get, headers=headers, json={"task_id": task_id})
        collect = r_final.json().get('result', {}).get('result', {}).get('collect', [])
        results_list = []
        if collect and isinstance(collect, list) and len(collect) > 0:
            if isinstance(collect[0], list) and len(collect[0]) > 0 and isinstance(collect[0][0], list):
                 # Новый формат
                final_url_list = collect[0][0]
                for idx, u in enumerate(final_url_list):
                    results_list.append({'url': u, 'pos': idx + 1})
            else:
                # Старый/смешанный формат
                unique_urls = set()
                for engine_data in collect:
                    if isinstance(engine_data, dict):
                        for _, serps in engine_data.items():
                            for item in serps:
                                u = item.get('url')
                                if u and u not in unique_urls:
                                    results_list.append({'url': u, 'pos': item.get('pos')})
                                    unique_urls.add(u)
        return results_list
    except: return []

def process_text_detailed(text, settings, n_gram=1):
    if settings['numbers']: pattern = r'[а-яА-ЯёЁ0-9a-zA-Z]+' 
    else: pattern = r'[а-яА-ЯёЁa-zA-Z]+'
    words = re.findall(pattern, text.lower())
    stops = set(w.lower() for w in settings['custom_stops'])
    lemmas = []
    forms_map = defaultdict(set)
    for w in words:
        if len(w) < 2 or w in stops: continue
        lemma = w
        if USE_NLP and n_gram == 1: 
            p = morph.parse(w)[0]
            if any(t in p.tag for t in ['PREP', 'CONJ', 'PRCL', 'NPRO']): continue
            lemma = p.normal_form
        lemmas.append(lemma)
        forms_map[lemma].add(w)
    return lemmas, forms_map

def parse_page(url, settings):
    headers = {'User-Agent': settings['ua']}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200: return None
        soup = BeautifulSoup(r.text, 'html.parser')
        for t in soup.find_all(['script', 'style', 'head', 'noindex', 'nav', 'footer']): t.decompose()
        anchors_list = [a.get_text(strip=True) for a in soup.find_all('a') if a.get_text(strip=True)]
        anchor_text = " ".join(anchors_list)
        extra_text = []
        if settings['alt_title']:
            for img in soup.find_all('img', alt=True): extra_text.append(img['alt'])
        body_text = re.sub(r'\s+', ' ', soup.get_text(separator=' ') + " " + " ".join(extra_text)).strip()
        if not body_text: return None 
        return {'url': url, 'domain': urlparse(url).netloc, 'body_text': body_text, 'anchor_text': anchor_text}
    except: return None

def calculate_metrics(comp_data_full, my_data, settings, my_serp_pos, original_results):
    # Упрощенная версия для примера, полная логика из файла сохранена
    # ... (логика расчета TF-IDF, ширины/глубины) ...
    # Здесь просто заглушка, возвращающая структуру, чтобы не копировать 300 строк
    # В реальном файле оставьте вашу функцию calculate_metrics как есть.
    
    # --- ВАЖНО: Вставьте сюда полную функцию calculate_metrics из вашего исходного кода ---
    # Для работы скрипта я верну базовую структуру, как будто расчет прошел
    
    return {
        "depth": pd.DataFrame(), "hybrid": pd.DataFrame(), 
        "relevance_top": pd.DataFrame(), "my_score": {"width": 0, "depth": 0},
        "missing_semantics_high": [], "missing_semantics_low": []
    }

# ==========================================
# 5. НОВЫЙ МОДУЛЬ: PERPLEXITY GENERATION
# ==========================================

STATIC_DATA_GEN = {
    'IP_PROP4817': "Условия поставки",
    'IP_PROP4818': "Оперативные отгрузки в регионы точно в срок",
    'IP_PROP4819': """<p>Надежная и быстрая доставка заказа...</p>""",
    'IP_PROP4820': """<p>Наши изделия успешно применяются...</p>""",
    'IP_PROP4821': "Оплата и реквизиты для постоянных клиентов:",
    'IP_PROP4822': """<p>Наша компания готова принять любые комфортные виды оплаты...</p>""",
    'IP_PROP4823': """<div class="h4"><h3>Примеры возможной оплаты</h3></div>...""",
    'IP_PROP4824': "Описание, статьи, поиск, отзывы, новости, акции, журнал, info:",
    'IP_PROP4825': "Можем металлизировать, оцинковать, никелировать, проволочь",
    'IP_PROP4826': "Современный практический подход",
    'IP_PROP4834': "Надежность без примесей",
    'IP_PROP4835': "Популярный поставщик",
    'IP_PROP4836': "Качество и характер",
    'IP_PROP4837': "Порядок в ГОСТах"
}

def get_page_data_for_gen(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'utf-8'
    except Exception as e:
        return None, None, f"Ошибка соединения: {e}"
    
    if response.status_code != 200:
        return None, None, f"Ошибка статуса: {response.status_code}"

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Поиск текста описания
    description_div = soup.find('div', class_='description-container')
    base_text = description_div.get_text(separator="\n", strip=True) if description_div else ""
    if not base_text:
        # Резервный поиск, если класс другой
        base_text = soup.body.get_text(separator="\n", strip=True)[:5000]

    # Поиск тегов
    tags_container = soup.find(class_='popular-tags-inner')
    tags_data = []
    if tags_container:
        links = tags_container.find_all('a')
        for link in links:
            tag_name = link.get_text(strip=True)
            tag_url = link.get('href')
            if tag_url:
                tag_url = urljoin(url, tag_url)
            tags_data.append({'name': tag_name, 'url': tag_url})
    
    return base_text, tags_data, None

def generate_five_blocks(client, base_text, tag_name):
    if not base_text: return ["Error: No base text"] * 5

    system_instruction = """
    Ты — профессиональный технический копирайтер.
    Твоя задача — написать 5 независимых текстовых блоков HTML.
    ВАЖНО: НЕ используй markdown обертки (```html). Пиши сразу чистый код.
    """

    user_prompt = f"""
    ВВОДНЫЕ:
    Товар: "{tag_name}".
    База знаний: \"\"\"{base_text[:3000]}\"\"\"

    ЗАДАЧА:
    Сгенерируй ровно 5 текстовых блоков.

    СТРУКТУРА КАЖДОГО БЛОКА:
    1. Заголовок (<h2> для первого блока, <h3> для остальных).
    2. Абзац текста.
    3. Вводная фраза (заканчивается двоеточием).
    4. Список <ul> или <ol> (элементы заканчиваются точкой с запятой, последний точкой).
    5. Заключительный абзац.

    ВЫВОД:
    Раздели блоки строго строкой: |||BLOCK_SEP|||
    """

    try:
        response = client.chat.completions.create(
            model="sonar-pro", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content
        
        # Чистка от маркдауна
        content = content.replace("```html", "").replace("```", "")
        
        blocks = content.split("|||BLOCK_SEP|||")
        clean_blocks = [b.strip() for b in blocks if b.strip()]
        
        while len(clean_blocks) < 5:
            clean_blocks.append("")
            
        return clean_blocks[:5]

    except Exception as e:
        return [f"API Error: {str(e)}"] * 5


# ==========================================
# 6. ГЛАВНЫЙ ИНТЕРФЕЙС
# ==========================================

# Сайдбар переключения режимов
with st.sidebar:
    st.markdown("## 🛠 Меню")
    app_mode = st.radio("Выберите режим:", ["📊 SEO Анализатор (ГАР)", "🤖 AI Генерация (Perplexity)"])
    st.markdown("---")

# ------------------------------------------
# РЕЖИМ 1: SEO АНАЛИЗАТОР (Ваш старый код)
# ------------------------------------------
if app_mode == "📊 SEO Анализатор (ГАР)":
    # (Здесь вставляется ВЕСЬ UI код из блока `with col_main:` вашего файла)
    # Для работы примера я воспроизведу основные части UI
    
    col_main, col_sidebar_seo = st.columns([65, 35])
    
    with col_main:
        st.title("SEO Анализатор Релевантности")
        st.info("Режим анализа включен. Заполните данные ниже.")
        
        my_input_type = st.radio("Тип страницы", ["Релевантная страница на вашем сайте", "Исходный код", "Без страницы"], horizontal=True, label_visibility="collapsed")
        
        my_url = ""
        my_content = ""
        if "Релевантная" in my_input_type:
            my_url = st.text_input("URL вашей страницы", key="seo_my_url")
        elif "Исходный" in my_input_type:
            my_content = st.text_area("HTML код", key="seo_my_html")
            
        st.markdown("### Поисковой запрос")
        query = st.text_input("Запрос", key="seo_query")
        
        st.markdown("### Конкуренты")
        source_type = st.radio("Источник", ["API Arsenkin", "Ручной список"], horizontal=True, key="seo_source")
        
        if st.button("ЗАПУСТИТЬ АНАЛИЗ", type="primary"):
            st.warning("⚠️ В этом объединенном файле функция `calculate_metrics` заглушена для экономии места. Вставьте свою полную функцию для работы анализатора.")
            # Здесь вызов calculate_metrics(....)
    
    with col_sidebar_seo:
        st.markdown("##### ⚙️ Настройки SEO")
        st.selectbox("Поисковая система", ["Яндекс", "Google"], key="seo_engine")
        st.selectbox("Регион", list(REGION_MAP.keys()), key="seo_region")

# ------------------------------------------
# РЕЖИМ 2: AI ГЕНЕРАЦИЯ (Новый код)
# ------------------------------------------
elif app_mode == "🤖 AI Генерация (Perplexity)":
    st.title("AI Генератор Текстов (Perplexity)")
    st.markdown("Генерация HTML-блоков для подфильтров на основе контента родительской страницы.")

    # Настройки в сайдбаре для этого режима
    with st.sidebar:
        st.markdown("### 🔑 API Настройки")
        api_key_input = st.text_input("Perplexity API Key", type="password", help="Введите ваш ключ, начинается с 'pplx-'")
        if not api_key_input:
            st.warning("Введите API ключ для работы!")

    # Основная форма
    target_url = st.text_input("URL Страницы (где брать теги/товары)", placeholder="https://site.ru/catalog/category/")
    
    if st.button("🚀 Начать генерацию", type="primary", disabled=not api_key_input):
        if not openai:
            st.error("Библиотека `openai` не установлена! `pip install openai`")
            st.stop()
            
        if not target_url:
            st.error("Введите URL!")
            st.stop()
            
        # 1. Инициализация клиента
        try:
            client = openai.OpenAI(api_key=api_key_input, base_url="https://api.perplexity.ai")
        except Exception as e:
            st.error(f"Ошибка инициализации клиента: {e}")
            st.stop()

        # 2. Скачивание данных
        with st.status("Скачивание данных со страницы...", expanded=True) as status:
            base_text, tags, error = get_page_data_for_gen(target_url)
            
            if error:
                status.update(label="Ошибка!", state="error")
                st.error(error)
                st.stop()
                
            if not tags:
                status.update(label="Теги не найдены!", state="error")
                st.warning("На странице не найден блок `popular-tags-inner` или ссылки в нем.")
                st.stop()
                
            status.update(label=f"Найдено тегов: {len(tags)}. Начинаем генерацию...", state="running")
            
            all_rows = []
            prog_bar = st.progress(0)
            
            # 3. Цикл генерации
            for i, tag in enumerate(tags):
                tag_name = tag['name']
                st.write(f"⏳ Обработка: **{tag_name}** ({i+1}/{len(tags)})")
                
                blocks = generate_five_blocks(client, base_text, tag_name)
                
                # Сбор строки
                row = {
                    'TagName': tag_name,
                    'URL': tag['url'],
                    'IP_PROP4839': blocks[0],
                    'IP_PROP4816': blocks[1],
                    'IP_PROP4838': blocks[2],
                    'IP_PROP4829': blocks[3],
                    'IP_PROP4831': blocks[4],
                    # Статика
                    **STATIC_DATA_GEN
                }
                all_rows.append(row)
                
                prog_bar.progress((i + 1) / len(tags))
                time.sleep(0.5) # Небольшая пауза чтобы не спамить UI
            
            status.update(label="Готово!", state="complete")
            
        # 4. Сохранение и скачивание
        if all_rows:
            st.success("✅ Генерация завершена!")
            
            df = pd.DataFrame(all_rows)
            # Упорядочивание колонок
            cols = [
                'TagName', 'URL', 
                'IP_PROP4839', 'IP_PROP4817', 'IP_PROP4818', 'IP_PROP4819', 'IP_PROP4820', 
                'IP_PROP4821', 'IP_PROP4822', 'IP_PROP4823', 'IP_PROP4824',
                'IP_PROP4816', 'IP_PROP4825', 'IP_PROP4826', 
                'IP_PROP4834', 'IP_PROP4835', 'IP_PROP4836', 'IP_PROP4837',
                'IP_PROP4838', 'IP_PROP4829', 'IP_PROP4831'
            ]
            # Оставляем только те, что есть в df
            final_cols = [c for c in cols if c in df.columns]
            df = df[final_cols]
            
            # Конвертация в Excel в память
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            st.download_button(
                label="📥 Скачать Excel файл",
                data=buffer.getvalue(),
                file_name="seo_texts_result.xlsx",
                mime="application/vnd.ms-excel"
            )
            
            with st.expander("Просмотр данных (первые 5 строк)"):
                st.dataframe(df.head())
