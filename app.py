# ==========================================
# ШАГ 0: АУТЕНТИФИКАЦИЯ
# ==========================================

CORRECT_PASSWORD = "garpro"

password_input = widgets.Password(
    placeholder='Введите пароль для доступа',
    description='Пароль:',
    layout=widgets.Layout(width='300px')
)
login_button = widgets.Button(
    description='Войти',
    button_style='info',
    layout=widgets.Layout(width='100px')
)
auth_output = widgets.Output()

# Контейнер для основного UI (изначально скрыт)
main_ui_container = widgets.VBox([], layout=widgets.Layout(display='none', border='1px solid #CCC', padding='15px', background_color='#F7F7F7'))
bn_run = widgets.Button(
    description='АНАЛИЗИРОВАТЬ 🚀',
    button_style='warning',
    layout=widgets.Layout(width='99%', height='50px', margin='20px 0', display='none')
)
output_log = widgets.Output()


def check_password(b):
    with auth_output:
        clear_output()
        if password_input.value == CORRECT_PASSWORD:
            print("✅ Авторизация успешна. Загрузка интерфейса...")
            
            # Скрываем логин-форму
            password_input.layout.display = 'none'
            login_button.layout.display = 'none'
            
            # Отображаем основной UI и кнопку запуска
            main_ui_container.layout.display = 'block'
            bn_run.layout.display = 'block'
            
            # Выводим главный интерфейс
            display(widgets.HTML("<h2>Гибридный Анализ Релевантности PRO</h2>"))
            display(main_ui_container)
            display(bn_run)
            display(output_log)
            
        else:
            print("❌ Неверный пароль. Попробуйте снова.")

login_button.on_click(check_password)

# Выводим сначала только форму логина
display(widgets.HTML("<h2>Гибридный Анализ Релевантности PRO: Вход</h2>"))
display(widgets.HBox([password_input, login_button]))
display(auth_output)

# ==========================================
# ШАГ 1: УСТАНОВКА И ГАРАНТИЯ РАБОТЫ PYMORPHY2
# ==========================================
print("⏳ Запуск установки необходимых библиотек...")
!pip install googlesearch-python beautifulsoup4 requests pandas numpy ipywidgets -q
!pip install pymorphy2 --upgrade --force-reinstall -q

import requests
from bs4 import BeautifulSoup
try:
    from googlesearch import search
    USE_SEARCH = True
except ImportError:
    USE_SEARCH = False

import pandas as pd
import numpy as np
import re
import ipywidgets as widgets
from IPython.display import display, clear_output
from collections import Counter
import math
import warnings
import inspect
import sys

try:
    if sys.version_info >= (3, 10):
        if not hasattr(inspect, 'getargspec'):
            def getargspec(func):
                spec = inspect.getfullargspec(func)
                return spec.args, spec.varargs, spec.varkw, spec.defaults
            inspect.getargspec = getargspec

    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    print("✅ Pymorphy2 успешно инициализирован.")
    USE_NLP = True
except Exception as e:
    print("❌ Критическая ошибка: Pymorphy2 не работает. Будет использовано простое токенизирование.")
    morph = None
    USE_NLP = False

warnings.filterwarnings("ignore")

BLACKLIST_DOMAINS = [
    'avito.ru', 'wikipedia.org', 'yandex.ru', 'ozon.ru', 'wildberries.ru', 'tiu.ru',
    'beru.ru', 'aliexpress.com', 'youtube.com', 'dzen.ru', 'hh.ru',
    'market.yandex.ru', 'sbermegamarket.ru', 'rutube.ru', 't.me', 'instagram.com',
    'gosuslugi.ru', 'rambler.ru', '2gis.ru', 'sravni.ru', 'toshop.ru', 'price.ru',
    'pandao.ru', 'banki.ru', 'regmarkets.ru', 'zoon.ru', 'pulscen.ru', 'prodoctorov.ru',
    'blizko.ru', 'domclick.ru', 'satom.ru', 'quto.ru', 'edadeal.ru', 'cataloxy.ru', 
    'irr.ru', 'onliner.by', 'shop.by', 'deal.by', 'yell.ru', 'profi.ru', 
    'irecommend.ru', 'otzovik.com', 'auto.ru'
]

# ==========================================
# ШАГ 2: ЛОГИКА (BACKEND) - ИЗ ФАЙЛА
# (Все функции с get_word_forms до run_analysis)
# ==========================================

def get_word_forms(lemma):
    if not USE_NLP or not morph:
        return f"Токен: {lemma}"
    if not lemma: return ""
    forms = []
    parses = morph.parse(lemma)
    if not parses: return ""
    base_parse = parses[0]
    for tag in base_parse.lexeme:
        forms.append(tag.word)
        if len(forms) >= 5:
            break
    return ", ".join(list(set(forms)))

def process_words(word_list, settings):
    base_stop_words = {
        'и', 'в', 'на', 'с', 'к', 'по', 'за', 'от', 'до', 'это', 'мы', 'вы', 'он', 'она', 'они', 'их', 'ее', 'его', 'мне',
        'тебе', 'себе', 'для', 'что', 'как', 'так', 'но', 'или', 'а', 'чтобы', 'же', 'бы', 'да', 'нет', 'у', 'без', 'под',
        'над', 'перед', 'при', 'через', 'между', 'среди', 'после', 'вместо', 'около', 'вокруг', 'со', 'из', 'из-за', 'из-под',
        'только', 'даже', 'хоть', 'ли', 'ни', 'разве', 'уже', 'еще', 'всё', 'все', 'когда', 'где', 'куда', 'откуда', 'почему',
        'зачем', 'какой', 'который', 'кто', 'что', 'весь', 'свой', 'такой', 
        'самый', 'много', 'мало', 'несколько', 'немного',
        'очень', 'просто', 'совсем', 'опять', 'снова', 'здесь', 'там', 'сюда', 'туда', 'никогда', 'всегда', 'обычно', 'часто',
        'редко', 'почти', 'поэтому', 'потом', 'раньше', 'позже', 'ранний', 'поздний', 'новый', 'старый', 'большой', 'маленький',
        'хороший', 'плохой', 'лучший', 'худший', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять',
        'рублей', 'руб', 'стр', 'ул', 'шт', 'см', 'мм', 'мл', 'кг', 'кв', 'м', 'м2', 'см2', 'м²', 'см²'
    }
    
    if settings.get('custom_stops'):
        base_stop_words.update(set(settings['custom_stops']))

    if not USE_NLP or not morph:
        return [w.lower() for w in word_list if len(w) > 2 and w.lower() not in base_stop_words]

    lemmas = []
    for word in word_list:
        word_lower = word.lower()
        if len(word) > 2 and word_lower not in base_stop_words:
            p = morph.parse(word_lower)[0]
            if 'PREP' not in p.tag and 'CONJ' not in p.tag and 'NUMR' not in p.tag and 'PRCL' not in p.tag:
                lemmas.append(p.normal_form)
    return lemmas

def clean_and_tokenize(html_content, settings):
    soup = BeautifulSoup(html_content, 'html.parser')

    if settings.get('noindex', True):
        for noindex in soup.find_all('noindex'):
            noindex.decompose()

    for script in soup(["script", "style", "head", "footer", "nav", "header", "aside"]):
        script.extract()

    text_parts = [soup.get_text(separator=' ')]

    if settings.get('alt_title', False):
        for img in soup.find_all('img', alt=True):
            text_parts.append(img['alt'])
        for tag in soup.find_all(title=True):
            text_parts.append(tag['title'])

    full_text = " ".join(text_parts)

    if settings.get('numbers', False):
         words = re.findall(r'[а-яА-ЯёЁ0-9]+', full_text)
    else:
         words = re.findall(r'[а-яА-ЯёЁ]+', full_text)

    return " ".join(process_words(words, settings))

def clean_anchor_text(html_content, settings):
    soup = BeautifulSoup(html_content, 'html.parser')
    anchor_words = []
    for a_tag in soup.find_all('a'):
        text = a_tag.get_text(strip=True)
        if text:
            words = re.findall(r'[а-яА-ЯёЁ]+', text)
            anchor_words.extend(process_words(words, settings))

    return " ".join(anchor_words)

def get_page_data(url, user_agent, settings):
    headers = {'User-Agent': user_agent}
    try:
        response = requests.get(url.strip(), headers=headers, timeout=20)
        response.raise_for_status() 
        html = response.text
        return clean_and_tokenize(html, settings), clean_anchor_text(html, settings) 
    except Exception as e:
        return "", ""

def manual_vectorize_and_analyze(corpus_body, corpus_anchor, my_idx):

    all_tokens_list = [token for doc in corpus_body for token in doc.split()]
    feature_names = sorted(list(set(all_tokens_list)))

    count_vectors = []
    doc_freq = Counter()
    N_docs = len(corpus_body)

    for i, doc in enumerate(corpus_body):
        counts = Counter(doc.split())
        vector = [counts.get(token, 0) for token in feature_names]
        count_vectors.append(vector)

        for token in set(doc.split()):
            doc_freq[token] += 1

    anchor_vectors = []
    for doc in corpus_anchor:
        counts = Counter(doc.split())
        vector = [counts.get(token, 0) for token in feature_names]
        anchor_vectors.append(vector)


    idf_values = {}
    for token in feature_names:
        df = doc_freq[token]
        idf = math.log(N_docs / df) + 1
        idf_values[token] = idf

    tfidf_vectors = []
    for count_vector, doc in zip(count_vectors, corpus_body):
        doc_tokens = doc.split()
        doc_len = len(doc_tokens)

        tf_vector = [count / doc_len if doc_len > 0 else 0 for count in count_vector]
        tfidf_vector = [tf * idf_values.get(token, 0) for tf, token in zip(tf_vector, feature_names)]
        tfidf_vectors.append(tfidf_vector)

    dense_tfidf = np.array(tfidf_vectors)
    dense_count = np.array(count_vectors)
    dense_anchor = np.array(anchor_vectors)

    comp_tfidf_matrix = dense_tfidf[:my_idx]
    my_tfidf_vector = dense_tfidf[my_idx]

    comp_count_matrix = dense_count[:my_idx]
    my_count_vector = dense_count[my_idx]

    comp_anchor_matrix = dense_anchor[:my_idx]
    my_anchor_vector = dense_anchor[my_idx]

    return feature_names, comp_tfidf_matrix, my_tfidf_vector, comp_count_matrix, my_count_vector, comp_anchor_matrix, my_anchor_vector

def run_analysis(my_url_id, competitors_urls, settings, my_body_content=None, my_anchor_content=None):
    
    if my_url_id == "No_Page_Mode":
        my_body, my_anchor = "", ""
    elif my_body_content is not None:
        my_body, my_anchor = my_body_content, my_anchor_content
    else:
        print(f"📥 Скачивание Вашего сайта: {my_url_id}...")
        my_body, my_anchor = get_page_data(my_url_id, settings['user_agent'], settings)
    
    if not my_body and my_url_id not in ["No_Page_Mode", "Manual_Code_Input"]:
        print("❌ Ошибка: Не удалось получить контент вашего сайта. Проверьте URL или User-Agent.")
        return None

    corpus_body, corpus_anchor = [], []

    print(f"📥 Обработка {len(competitors_urls)} конкурентов...")
    for url in competitors_urls:
        body_text, anchor_text = get_page_data(url, settings['user_agent'], settings)
        if len(body_text) > 50:
            corpus_body.append(body_text)
            corpus_anchor.append(anchor_text)

    if len(corpus_body) < 2:
        print("❌ Недостаточно данных для сравнения (скачано менее 2 конкурентов).")
        return None

    my_idx = len(corpus_body)

    my_body_len = len(my_body.split())
    comp_body_lengths = [len(doc.split()) for doc in corpus_body] 
    avg_comp_body_len = np.mean(comp_body_lengths) if comp_body_lengths else 1.0

    length_normalization_factor = 1.0
    if settings.get('normalize', False) and avg_comp_body_len > 0:
        length_normalization_factor = my_body_len / avg_comp_body_len 
        
    corpus_body.append(my_body)
    corpus_anchor.append(my_anchor)

    feature_names, comp_tfidf_matrix, my_tfidf_vector, comp_count_matrix, my_count_vector, comp_anchor_matrix, my_anchor_vector = \
        manual_vectorize_and_analyze(corpus_body, corpus_anchor, my_idx)

    results = []
    TARGET_FACTOR = 1.3
    PERCENT_OUTPUT = settings.get('percent_output', False)

    for col in range(len(feature_names)):
        token = feature_names[col]

        comp_tfidf_col = comp_tfidf_matrix[:, col]
        comp_count_col = comp_count_matrix[:, col]
        comp_anchor_col = comp_anchor_matrix[:, col]

        median_tfidf = float(np.median(comp_tfidf_col)) if comp_tfidf_col.size > 0 else 0
        median_count = float(np.median(comp_count_col)) if comp_count_col.size > 0 else 0
        median_anchor_count = float(np.median(comp_anchor_col)) if comp_anchor_col.size > 0 else 0

        my_tfidf = float(my_tfidf_vector[col])
        my_count = float(my_count_vector[col])
        my_anchor_count = float(my_anchor_vector[col])

        # --- BODY COUNT CALCULATION ---
        target_body_count = int(median_count * TARGET_FACTOR * length_normalization_factor)
        rec_body_count = target_body_count - int(my_count)

        rec_body_text = "0"
        if PERCENT_OUTPUT:
            if target_body_count > 0:
                current_coverage_percent = (my_count / target_body_count) * 100
                if current_coverage_percent < 100: rec_body_text = f" +{abs(100 - current_coverage_percent):.0f}%"
                elif current_coverage_percent > 100: rec_body_text = f" -{abs(current_coverage_percent - 100):.0f}%"
                else: rec_body_text = "0%"
            elif my_count > 0: rec_body_text = f" -100%" 
            else: rec_body_text = "0%"
        else:
            if rec_body_count > 0: rec_body_text = f" +{abs(rec_body_count)}"
            elif rec_body_count < 0: rec_body_text = f" {rec_body_count}"
            else: rec_body_text = "0"


        # --- ANCHOR COUNT CALCULATION ---
        target_anchor_count = int(median_anchor_count * TARGET_FACTOR * length_normalization_factor)
        rec_anchor_count = target_anchor_count - int(my_anchor_count)

        rec_anchor_text = "0"
        if PERCENT_OUTPUT:
            if target_anchor_count > 0:
                current_coverage_percent = (my_anchor_count / target_anchor_count) * 100
                if current_coverage_percent < 100: rec_anchor_text = f" +{abs(100 - current_coverage_percent):.0f}%"
                elif current_coverage_percent > 100: rec_anchor_text = f" -{abs(current_coverage_percent - 100):.0f}%"
                else: rec_anchor_text = "0%"
            elif my_anchor_count > 0: rec_anchor_text = f" -100%" 
            else: rec_anchor_text = "0%"
        else:
            if rec_anchor_count > 0: rec_anchor_text = f" +{abs(rec_anchor_count)}"
            elif rec_anchor_count < 0: rec_anchor_text = f" {rec_anchor_count}"
            else: rec_anchor_text = "0"


        # 6. Фильтрация и сбор результатов
        is_relevant = median_tfidf > 0.05
        is_actionable = (rec_body_text != '0' and rec_body_text != '0%') or \
                        (rec_anchor_text != '0' and rec_anchor_text != '0%')

        if is_relevant or is_actionable:

            lemma_name = token if USE_NLP else f"Токен: {token}"

            results.append({
                "Слово (Лемма)": lemma_name,
                "Словоформы": get_word_forms(token),
                "TF-IDF (Вы)": round(my_tfidf, 3),
                "Медиана (ТОП)": round(median_tfidf, 3),
                "Текст (Рек.)": rec_body_text,
                "Текст (Повт.)": int(my_count),
                "Тег <a> (Рек.)": rec_anchor_text,
                "Тег <a> (Повт.)": int(my_anchor_count)
            })

    # 7. Финальная сортировка и фильтрация
    df = pd.DataFrame(results)
    if not df.empty:
        def extract_abs_value(rec_str):
            if rec_str == '0' or rec_str == '0%': return 0
            return abs(float(re.sub(r'[+\- %]', '', rec_str)))

        df['Sort_Body_Abs'] = df['Текст (Рек.)'].apply(extract_abs_value)
        df['Sort_Anchor_Abs'] = df['Тег <a> (Рек.)'].apply(extract_abs_value)

        df = df.sort_values(by=['Sort_Body_Abs', 'Sort_Anchor_Abs', 'Медиана (ТОП)'], ascending=[False, False, False])
        df = df.drop(columns=['Sort_Body_Abs', 'Sort_Anchor_Abs'])

        df_filtered = df[(df['Текст (Рек.)'] != '0') & (df['Текст (Рек.)'] != '0%') |
                         (df['Тег <a> (Рек.)'] != '0') & (df['Тег <a> (Рек.)'] != '0%')]

        return df_filtered
    return None


# ==========================================
# ШАГ 3: ИНТЕРФЕЙС (UI) - УЛУЧШЕННЫЙ ДИЗАЙН
# ==========================================

style_header = "font-size: 16px; font-weight: bold; margin-top: 10px; margin-bottom: 5px; color: #1E293B;"
w_layout = widgets.Layout(width='99%')
w_half_layout = widgets.Layout(width='50%')

# --- 1. Секция: Постановка задачи ---
html_task = widgets.HTML(f"<div style='{style_header}'>1️⃣ Ваша страница и запрос</div>")
r_input_type = widgets.RadioButtons(options=['Релевантная страница на вашем сайте', 'Исходный код страницы или текст', 'Без страницы'], value='Релевантная страница на вашем сайте', layout=widgets.Layout(width='100%'))
w_my_url = widgets.Text(placeholder="https://site.ru/catalog/page", layout=w_layout) 
w_source_code = widgets.Textarea(placeholder="Вставьте сюда HTML код страницы или текст статьи...", layout=widgets.Layout(width='99%', height='200px', display='none')) 
w_query = widgets.Text(placeholder="Основной поисковой запрос", layout=w_layout)
chk_extra_queries = widgets.Checkbox(value=False, description='Дополнительные запросы')
w_extra_queries_text = widgets.Textarea(placeholder="Каждый запрос с новой строки", layout=widgets.Layout(width='99%', height='60px', display='none'))

def toggle_input_mode(change):
    mode = change['new']
    w_my_url.layout.display = 'block' if mode == 'Релевантная страница на вашем сайте' else 'none'
    w_source_code.layout.display = 'block' if mode == 'Исходный код страницы или текст' else 'none'
    chk_norm.disabled = mode == 'Без страницы'
    chk_norm.value = not chk_norm.disabled

r_input_type.observe(toggle_input_mode, names='value')
def toggle_extra_queries(change): w_extra_queries_text.layout.display = 'block' if change['new'] else 'none'
chk_extra_queries.observe(toggle_extra_queries, names='value')

task_box = widgets.VBox([
    html_task, 
    r_input_type, 
    w_my_url, 
    w_source_code, 
    w_query,
    chk_extra_queries,
    w_extra_queries_text,
    widgets.HTML("<hr style='border-top: 1px solid #DDD;'>")
], layout=widgets.Layout(border='1px solid #CCC', padding='10px', margin='0 0 10px 0', background_color='#FFFFFF'))


# --- 2. Секция: Конкуренты ---
html_comp = widgets.HTML(f"<div style='{style_header}'>2️⃣ Источник данных и фильтры</div>")
r_comp_source = widgets.RadioButtons(options=['Поиск', 'Список url-адресов ваших конкурентов'], value='Поиск', layout=widgets.Layout(width='100%'))
w_engine = widgets.Dropdown(options=['Google', 'Яндекс (Не работает!)'], value='Google', description='Система:', layout=w_half_layout)
w_region = widgets.Dropdown(options=['Москва', 'Санкт-Петербург', 'Россия', 'СНГ'], value='Москва', description='Регион:', layout=w_half_layout)
w_device = widgets.Dropdown(options=['Desktop', 'Mobile'], value='Desktop', description='Устройство:', layout=w_half_layout)
w_top_count = widgets.Dropdown(options=[5, 10, 15, 20, 30], value=10, description='ТОП:', layout=w_half_layout)
w_exclude_domains = widgets.Textarea(value="\n".join(BLACKLIST_DOMAINS), description='Не учитывать:', placeholder='Домены для исключения', layout=w_layout)
w_manual_comps = widgets.Textarea(placeholder="https://competitor1.ru\nhttps://competitor2.ru", layout=widgets.Layout(width='99%', height='150px', display='none'))

comp_settings_col = widgets.VBox([
    widgets.HTML("<b>Настройки поиска</b>"),
    widgets.HBox([w_engine, w_region], layout=w_layout),
    widgets.HBox([w_device, w_top_count], layout=w_layout),
], layout=w_half_layout)

comp_exclude_col = widgets.VBox([
    widgets.HTML("<b>Домены для исключения (по домену/URL)</b>"),
    w_exclude_domains,
], layout=w_half_layout)

comp_search_settings = widgets.VBox([
    widgets.HBox([comp_settings_col, comp_exclude_col]),
], layout=widgets.Layout(display='block'))

def toggle_comp_source(change):
    comp_search_settings.layout.display = 'block' if change['new'] == 'Поиск' else 'none'
    w_manual_comps.layout.display = 'block' if change['new'] == 'Список url-адресов ваших конкурентов' else 'none'
r_comp_source.observe(toggle_comp_source, names='value')

comp_box = widgets.VBox([
    html_comp, 
    r_comp_source,
    comp_search_settings,
    w_manual_comps,
    widgets.HTML("<hr style='border-top: 1px solid #DDD;'>")
], layout=widgets.Layout(border='1px solid #CCC', padding='10px', margin='0 0 10px 0', background_color='#FFFFFF'))


# --- 3. Секция: Настройки (ДВЕ КОЛОНКИ) ---
html_settings = widgets.HTML(f"<div style='{style_header}'>3️⃣ Детализация и технические настройки</div>")
w_perfect_url = widgets.Text(placeholder="https://site.ru/ (Главный конкурент)", layout=w_layout)
chk_norm = widgets.Checkbox(value=True, description='Нормировать общие значения (Медиана, переспам)')
chk_percent = widgets.Checkbox(value=False, description='Выводить общие значения в процентах')
chk_aggr = widgets.Checkbox(value=True, description='Исключить агрегаторы и type-in трафик (список выше)')
chk_noindex = widgets.Checkbox(value=True, description='Исключать текст в теге noindex')
chk_alt = widgets.Checkbox(value=False, description='Учитывать атрибуты alt и title')
chk_num = widgets.Checkbox(value=False, description='Учитывать числа')
chk_stop_pos = widgets.Checkbox(value=True, description='Исключать служебные части речи') 
chk_extra_data = widgets.Checkbox(value=False, description='Дополнительные данные (экспертные)')

main_settings_col = widgets.VBox([
    widgets.HTML("<b>Основные параметры</b>"),
    w_perfect_url,
    chk_norm, 
    chk_percent,
    chk_aggr,
    chk_noindex, 
    chk_alt, 
    chk_num,
    chk_extra_data,
], layout=widgets.Layout(width='50%', padding='5px'))

# Колонка Б: Технические и Стоп-слова
w_user_agent = widgets.Text(value="Mozilla/5.0 (compatible; Artur2k/1.0;)", description='User-Agent:', layout=w_layout)
chk_stop_custom = widgets.Checkbox(value=True, description='Исключать свой список слов')
default_stops = "рублей\nруб\nстр\nул\nшт\nсм\nмм\nмл\nкг\nкв\nм²\nсм²\nм2\nсм2"
w_stop_custom_text = widgets.Textarea(value=default_stops, layout=w_layout)
chk_depth_formula = widgets.Checkbox(value=False, description='Параметры формулы глубины')
w_depth_top = widgets.Dropdown(options=['ТОП3', 'ТОП5', 'ТОП10', 'ТОП20'], value='ТОП5', description='Слов:', layout=widgets.Layout(display='none', width='100%'))
w_depth_count = widgets.Checkbox(value=True, description='Учитывать кол-во повторов', layout=widgets.Layout(display='none', width='100%'))

def toggle_stop_custom(change): w_stop_custom_text.layout.display = 'block' if change['new'] else 'none'
chk_stop_custom.observe(toggle_stop_custom, names='value')

def toggle_depth(change):
    vis = 'block' if change['new'] else 'none'
    w_depth_top.layout.display = vis
    w_depth_count.layout.display = vis
chk_depth_formula.observe(toggle_depth, names='value')

tech_settings_col = widgets.VBox([
    widgets.HTML("<b>Технические настройки</b>"),
    w_user_agent,
    chk_stop_pos,
    chk_stop_custom, 
    w_stop_custom_text,
    chk_depth_formula, 
    w_depth_top, 
    w_depth_count,
], layout=widgets.Layout(width='50%', padding='5px'))

settings_hbox = widgets.HBox([main_settings_col, tech_settings_col], layout=widgets.Layout(justify_content='space-between', width='100%'))

settings_box = widgets.VBox([
    html_settings,
    settings_hbox
], layout=widgets.Layout(border='1px solid #CCC', padding='10px', margin='0 0 10px 0', background_color='#FFFFFF'))


# --- ФИНАЛЬНАЯ СБОРКА ИНТЕРФЕЙСА ---
ui_elements = widgets.VBox([
    task_box,
    comp_box,
    settings_box
])

main_ui_container.children = [ui_elements]

# --- ОБРАБОТЧИК ЗАПУСКА ---

def on_btn_click(b):
    with output_log:
        clear_output()
        print("⚙️ Сбор данных задачи...")

        settings = {
            'top': w_top_count.value,
            'noindex': chk_noindex.value,
            'alt_title': chk_alt.value,
            'numbers': chk_num.value,
            'normalize': chk_norm.value,
            'percent_output': chk_percent.value,
            'user_agent': w_user_agent.value,
            'exclude': [x.strip() for x in w_exclude_domains.value.split('\n') if x.strip()],
            'custom_stops': [x.strip() for x in w_stop_custom_text.value.split('\n') if x.strip()] if chk_stop_custom.value else []
        }
        
        if chk_aggr.value: 
            settings['exclude'].extend(BLACKLIST_DOMAINS)
        settings['exclude'] = list(set([item for item in settings['exclude'] if item]))

        # 2. Определение "Моей Страницы"
        input_mode = r_input_type.value
        my_body_content = None
        my_anchor_content = None
        my_url_id = "" 

        if input_mode == 'Релевантная страница на вашем сайте':
            my_url_id = w_my_url.value
            if not my_url_id:
                print("❌ Ошибка: Введите URL вашего сайта!")
                return
        elif input_mode == 'Исходный код страницы или текст':
            raw_code = w_source_code.value
            if not raw_code:
                print("❌ Ошибка: Вставьте HTML код или текст!")
                return
            my_url_id = "Manual_Code_Input"
            my_body_content = clean_and_tokenize(raw_code, settings)
            my_anchor_content = clean_anchor_text(raw_code, settings) 
        elif input_mode == 'Без страницы':
            my_url_id = "No_Page_Mode"
            my_body_content = "" 
            my_anchor_content = ""
        
        # 3. Сбор конкурентов
        competitors_urls = []
        if r_comp_source.value == 'Поиск':
            query = w_query.value
            if not query:
                print("❌ Ошибка: Введите поисковой запрос!")
                return
            
            if USE_SEARCH and w_engine.value == 'Google':
                try:
                    print(f"🔎 Поиск в Google по запросу: {query}")
                    raw_urls = search(query, num_results=settings['top'] + 10, lang="ru") 

                    count_collected = 0
                    for u in raw_urls:
                        if u == my_url_id or any(ex in u for ex in settings['exclude']):
                            continue

                        competitors_urls.append(u)
                        count_collected += 1

                        if count_collected >= settings['top']: break
                    
                    if not competitors_urls:
                        print("❌ Google Search не вернул результатов.")
                        return

                except Exception as e:
                    print(f"⚠️ Ошибка поиска Google: {e}.")
                    return
            else:
                 print("❌ Поиск недоступен или Яндекс не поддерживается.")
                 return

        else: # Список URL вручную
            raw_list = w_manual_comps.value.split('\n')
            
            for comp_url in [u.strip() for u in raw_list if u.strip()]:
                if comp_url == my_url_id or any(ex in comp_url for ex in settings['exclude']):
                    continue
                competitors_urls.append(comp_url)


        if not competitors_urls:
            print("❌ Список конкурентов пуст.")
            return

        # 4. Запуск анализа
        print(f"\n🚀 Старт анализа. Будет обработано {len(competitors_urls)} URL.")
        
        df = run_analysis(
            my_url_id, 
            competitors_urls, 
            settings, 
            my_body_content=my_body_content, 
            my_anchor_content=my_anchor_content
        )

        if df is not None:
            print("\n📊 РЕЗУЛЬТАТЫ ГИБРИДНОГО АНАЛИЗА:")
            display(df)
        else:
            print("\n⚠️ Анализ не дал значимых результатов.")

bn_run.on_click(on_btn_click)
