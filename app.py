import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
try:
    from googlesearch import search
    USE_SEARCH = True
except ImportError:
    USE_SEARCH = False
    
import re
from collections import Counter
import math
import inspect
import concurrent.futures
from urllib.parse import urlparse
import time 

# --- ФИНАЛЬНЫЙ БРОНЕБОЙНЫЙ ПАТЧ ДЛЯ PYMORPHY2 ---
# (Патч для обеспечения совместимости с некоторыми средами)
try:
    if not hasattr(inspect, 'getargspec'):
        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return spec.args, spec.varargs, spec.varkw, spec.defaults
        inspect.getargspec = getargspec

    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    USE_MORPH = True
except ImportError:
    USE_MORPH = False
except Exception:
    USE_MORPH = False


# ==========================================
# 1. НАСТРОЙКА СТРАНИЦЫ И СТИЛИ
# ==========================================

st.set_page_config(
    page_title="SEO Анализатор Релевантности",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Внедряем CSS стили (шрифт Inter)
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
            max-width: 1400px;
        }

        /* Заголовок H1 */
        h1 {
            color: #1E40AF; /* Синий для акцента */
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        /* Карточки ввода */
        .stTextInput > div > div > input, .stTextArea > div > textarea, .stSelectbox > div > button {
            border-radius: 0.5rem;
            border: 1px solid #E5E7EB;
            padding: 0.75rem 1rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }

        /* Кнопки */
        .stButton>button {
            background-color: #1E40AF;
            color: white;
            font-weight: 600;
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            transition: all 0.2s;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1D4ED8;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        }
        
        /* Таблицы Pandas */
        .stDataFrame {
            border-radius: 0.75rem;
            overflow: hidden;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        }

        /* Подсветка для рекомендаций */
        .dataframe td {
             vertical-align: middle !important;
        }

    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. ФУНКЦИИ БЭКЕНДА
# ==========================================

# 2.1. Извлечение контента
def fetch_content(url):
    """Получает HTML-контент страницы и извлекает текст и анкоры."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Устанавливаем таймаут 10 секунд
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Проверка на ошибки HTTP

        soup = BeautifulSoup(response.content, 'lxml')
        
        # Удаляем скрипты, стили, и другие ненужные элементы
        for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
            script_or_style.decompose()

        # Извлечение анкоров
        anchors = [a.get_text(separator=' ', strip=True).lower() for a in soup.find_all('a') if a.get_text(strip=True)]
        
        # Извлечение основного текста (теги p, h1-h6, li, td, span, div)
        main_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'span', 'div'])
        text_content = ' '.join([tag.get_text(separator=' ', strip=True) for tag in main_tags])
        
        # Удаление лишних пробелов и переносов строк
        text_content = re.sub(r'\s+', ' ', text_content).strip()

        return text_content, ' '.join(anchors)

    except requests.exceptions.RequestException:
        return None, None
    except Exception:
        return None, None

# 2.2. Очистка и лемматизация текста
def process_text(text):
    """Очищает текст, удаляет стоп-слова и лемматизирует."""
    if not text:
        return []
        
    # Удаление небуквенных символов и перевод в нижний регистр
    text = re.sub(r'[^а-яa-z\s]', ' ', text.lower())
    words = text.split()
    
    # Очень простой список стоп-слов
    stopwords = set([
        'и', 'в', 'на', 'по', 'с', 'к', 'от', 'до', 'для', 'из', 'за', 'под', 
        'не', 'да', 'это', 'то', 'как', 'так', 'же', 'мы', 'вы', 'он', 'она', 
        'оно', 'они', 'их', 'все', 'что', 'который', 'при', 'у', 'я', 'но'
    ])
    
    lemmas = []
    if USE_MORPH:
        for word in words:
            if word in stopwords or len(word) < 3:
                continue
            p = morph.parse(word)[0]
            # Лемматизируем только существительные, глаголы, прилагательные, наречия
            if p.tag.POS in ('NOUN', 'VERB', 'ADJF', 'ADJS', 'ADVB'):
                lemmas.append(p.normal_form)
    else:
        # Если pymorphy2 недоступен, просто фильтруем по стоп-словам
        lemmas = [word for word in words if word not in stopwords and len(word) >= 3]
            
    return lemmas

# 2.3. Расчет метрик (TF, IDF, BM25)
def calculate_metrics(word_freqs, N, idf_db, D, avg_D, k1, b):
    """
    Рассчитывает метрики TF, TF-IDF, BM25 для каждого слова.
    """
    metrics = {}
    
    for word, freq in word_freqs.items():
        if word not in idf_db:
            # Если слово не найдено в IDF базе, используем максимальный IDF (min_doc_freq = 1)
            idf_value = math.log(N / 1) if N > 0 else 0 
        else:
            idf_value = idf_db[word]
            
        # 1. Term Frequency (TF)
        tf = freq / D if D > 0 else 0

        # 2. TF-IDF
        tfidf = tf * idf_value

        # 3. BM25
        # Расчет K
        K = k1 * ( (1 - b) + b * (D / avg_D) )
        # Формула BM25
        bm25 = idf_value * ( (freq * (k1 + 1)) / (freq + K) )

        metrics[word] = {
            'tf': tf,
            'tfidf': tfidf,
            'bm25': bm25,
            'idf': idf_value,
            'count': freq, # частота слова в текущем документе
        }
    return metrics

# 2.4. Основная функция анализа
def run_analysis(my_url, competitors_urls, settings):
    """
    Запускает полный гибридный анализ релевантности.
    """
    # 1. Сбор контента в параллельном режиме
    all_data = {}
    
    urls_to_fetch = [my_url] + competitors_urls
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(fetch_content, url): url for url in urls_to_fetch}
        
        fetch_status = st.empty()
        fetch_status.info("⏳ Идет сбор контента со страниц. Пожалуйста, подождите...")
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            url = future_to_url[future]
            
            try:
                body_content, anchor_content = future.result()
            except Exception:
                body_content, anchor_content = None, None
            
            if body_content:
                
                # Задержка 1 секунда, чтобы снизить нагрузку на целевые сайты (кроме своего)
                if url != my_url:
                    time.sleep(1) 
                
                lemmas = process_text(body_content)
                anchor_lemmas = process_text(anchor_content)
                
                all_data[url] = {
                    'body_lemmas': lemmas,
                    'anchor_lemmas': anchor_lemmas,
                    'D_body': len(lemmas),
                    'D_anchor': len(anchor_lemmas),
                    'domain': urlparse(url).netloc
                }
                
                fetch_status.text(f"✅ Обработано {i+1}/{len(urls_to_fetch)} страниц. Текущая: {urlparse(url).netloc}")
            else:
                fetch_status.warning(f"❌ Не удалось обработать страницу: {url}")


    if my_url not in all_data:
        st.error("❌ Не удалось получить контент с вашего URL. Анализ невозможен.")
        return None

    my_data = all_data.pop(my_url)
    comp_data = all_data

    N_comps = len(comp_data)
    if N_comps == 0:
        st.error("❌ Не удалось собрать данные ни с одного конкурента. Анализ невозможен.")
        return None

    # 2. База данных частоты документов (DF) и IDF
    
    all_comp_words = set()
    for data in comp_data.values():
        all_comp_words.update(set(data['body_lemmas']))

    doc_freq = Counter()
    for word in all_comp_words:
        for data in comp_data.values():
            if word in data['body_lemmas']:
                doc_freq[word] += 1
    
    idf_db = {
        word: math.log(N_comps / count) for word, count in doc_freq.items() if N_comps > 0 and count > 0
    }
    
    comp_lengths = [data['D_body'] for data in comp_data.values()]
    avg_D = np.mean(comp_lengths) if comp_lengths else 1

    # 3. Расчет метрик
    
    all_word_metrics = {}
    
    # --- 3.1. Анализ конкурентов ---
    for data in comp_data.values():
        word_freqs = Counter(data['body_lemmas'])
        
        metrics = calculate_metrics(
            word_freqs, N_comps, idf_db, data['D_body'], avg_D, 
            settings['bm25_k1'], settings['bm25_b']
        )

        for word, m in metrics.items():
            if word not in all_word_metrics:
                all_word_metrics[word] = {
                    'tfidf_comp': [], 'bm25_comp': [], 'count_comp': [], 'count_sites': doc_freq.get(word, 0)
                }
            all_word_metrics[word]['tfidf_comp'].append(m['tfidf'])
            all_word_metrics[word]['bm25_comp'].append(m['bm25'])
            all_word_metrics[word]['count_comp'].append(m['count'])

    # --- 3.2. Анализ нашего сайта ---
    my_body_freqs = Counter(my_data['body_lemmas'])
    my_anchor_freqs = Counter(my_data['anchor_lemmas'])
    
    my_body_metrics = calculate_metrics(
        my_body_freqs, N_comps, idf_db, my_data['D_body'], avg_D, 
        settings['bm25_k1'], settings['bm25_b']
    )

    # 4. Формирование финальной таблицы
    
    final_data = []
    
    all_words = set(all_word_metrics.keys()) | set(my_body_metrics.keys())
    
    for word in all_words:
        
        comp_data_word = all_word_metrics.get(word, {'tfidf_comp': [], 'bm25_comp': [], 'count_comp': [], 'count_sites': 0})
        my_m = my_body_metrics.get(word, {})

        # Метрики конкурентов (медианы)
        tfidf_top_median = np.median(comp_data_word['tfidf_comp']) if comp_data_word['tfidf_comp'] else 0
        bm25_top_median = np.median(comp_data_word['bm25_comp']) if comp_data_word['bm25_comp'] else 0
        count_top_avg = np.mean(comp_data_word['count_comp']) if comp_data_word['count_comp'] else 0
        
        # Метрики нашего сайта
        tfidf_my = my_m.get('tfidf', 0)
        bm25_my = my_m.get('bm25', 0)
        idf_val = my_m.get('idf', comp_data_word.get('idf', 0)) # Берем IDF из нашего словаря или из IDF базы конкурентов
        count_my = my_m.get('count', 0)
        
        # Анкорные повторы
        anchor_my = my_anchor_freqs.get(word, 0)
        anchor_top_avg = 0 # Анкорный анализ у конкурентов отключен для упрощения

        # 4.1. Фильтрация и Расчеты
        
        # Фильтрация по частоте у конкурентов
        if comp_data_word['count_sites'] < settings['min_sites']:
             continue

        # Логика для подсветки и рекомендаций
        # 1. Повторы в основном тексте
        if count_my == 0 and count_top_avg > 0:
            rec_text = f"Добавить {math.ceil(count_top_avg):.0f} (avg) - {comp_data_word['count_sites']}/{N_comps} сайтов"
        elif count_my > count_top_avg * settings['max_spam_factor'] and count_top_avg > 0:
            rec_text = f"Убрать {math.ceil(count_my - count_top_avg):.0f} (spam)"
        else:
            rec_text = "OK"

        # 2. Повторы в анкорах
        if anchor_my == 0 and anchor_top_avg > 0:
             rec_anchor = f"Добавить {math.ceil(anchor_top_avg):.0f} (avg)"
        elif anchor_my > anchor_top_avg * settings['max_spam_factor'] and anchor_top_avg > 0:
             rec_anchor = f"Убрать {math.ceil(anchor_my - anchor_top_avg):.0f} (spam)"
        else:
             rec_anchor = "OK"
             
        # 3. Общая рекомендация (на основе BM25)
        # Усилить BM25, если сильно отстает, и слово популярно у большинства
        if (bm25_my < bm25_top_median * 0.5) and (comp_data_word['count_sites'] >= N_comps * 0.5):
            rec_total = f"Добавить (BM25: {bm25_my:.2f} < {bm25_top_median:.2f})"
        # Убрать, если сильно переспамлено по BM25
        elif bm25_my > bm25_top_median * settings['max_spam_factor'] and bm25_top_median > 0:
            rec_total = f"Убрать (BM25: {bm25_my:.2f} > {bm25_top_median:.2f})"
        else:
            rec_total = "OK"

        
        final_data.append({
            'Слово': word,
            'TF-IDF ТОП': tfidf_top_median,
            'TF-IDF ваш сайт': tfidf_my,
            'BM25 ТОП': bm25_top_median,
            'BM25 ваш сайт': bm25_my,
            'IDF': idf_val,
            'Кол-во сайтов': comp_data_word['count_sites'],
            'Медиана': np.median(comp_data_word['count_comp']) if comp_data_word['count_comp'] else 0, # Медиана повторов в тексте
            'Переспам': rec_total, # Итоговая рекомендация
            'Среднее по ТОПу (повт.)': count_top_avg,
            'Ваш сайт (повт.)': count_my,
            '<a/> по ТОПу (повт.)': anchor_top_avg,
            '<a/> ваш сайт (повт.)': anchor_my,
            'Текст Добавить/Убрать': rec_text,
            'Тег A Добавить/Убрать': rec_anchor,
        })


    df = pd.DataFrame(final_data)
    
    # 5. Фильтрация и Сортировка
    
    if df.empty:
        return None
        
    numeric_cols = [
        'TF-IDF ТОП', 'TF-IDF ваш сайт', 'BM25 ТОП', 'BM25 ваш сайт', 'IDF', 
        'Медиана', 'Среднее по ТОПу (повт.)', 'Ваш сайт (повт.)', '<a/> по ТОПу (повт.)', '<a/> ваш сайт (повт.)'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


    df = df[df['TF-IDF ТОП'] > 0]
    
    df['Разница BM25'] = df['BM25 ТОП'] - df['BM25 ваш сайт']
    
    df_sorted = df.sort_values(
        by=['BM25 ТОП', 'Разница BM25'], 
        ascending=[False, False]
    ).drop(columns=['Разница BM25'])

    for col in numeric_cols:
        df_sorted[col] = df_sorted[col].round(3)
        
    return df_sorted


# ==========================================
# 3. ИНТЕРФЕЙС STREAMLIT
# ==========================================

st.title("💎 Гибридный Анализ Релевантности PRO")
st.markdown("""
    Профессиональный SEO-инструмент для сравнения текстовых метрик (TF-IDF, BM25) 
    вашего сайта с конкурентами из ТОПа.
""")

# --- 3.1. БЛОК ВВОДА ДАННЫХ ---
with st.container(border=True):
    col1, col2 = st.columns([3, 1])

    with col1:
        my_url = st.text_input("🚀 Ваш URL для анализа", 
                               placeholder="https://vash-site.ru/stranitsa",
                               help="Введите URL страницы, которую вы хотите проанализировать.")
    with col2:
        mode = st.radio("Источник конкурентов", ["Google ТОП", "Ввести вручную"], 
                        index=0, horizontal=True)

    # Настройки
    st.markdown("---")
    st.subheader("⚙️ Настройки")
    
    col_set1, col_set2, col_set3 = st.columns(3)

    with col_set1:
        query = st.text_input("Поисковый запрос (для Google ТОПа)", 
                              placeholder="купить дом в москве",
                              disabled=(mode != "Google ТОП"))
        
        min_sites = st.slider("Мин. кол-во сайтов в ТОПе для слова", 
                              min_value=1, max_value=10, value=2, step=1,
                              help="Слово должно встречаться минимум на N сайтах конкурентов, чтобы попасть в анализ.")

    with col_set2:
        top_n = st.slider("Кол-во конкурентов из ТОПа", 
                          min_value=5, max_value=20, value=10, step=1,
                          disabled=(mode != "Google ТОП"),
                          help="Сколько сайтов из ТОПа Google учитывать в расчете.")
        
        max_spam_factor = st.slider("Коэф. переспама", 
                                    min_value=1.0, max_value=5.0, value=2.0, step=0.1,
                                    help="Во сколько раз повторов вашего сайта должно превышать среднее ТОПа, чтобы считать это переспамом.")

    with col_set3:
        excludes = st.text_area("Список исключений (по домену/URL)", 
                                placeholder="yandex.ru\nwikipedia.org\nprofi.ru", height=100,
                                help="Укажите части URL или домены, которые нужно исключить из списка конкурентов.")
        
        st.caption("Параметры BM25 (k1=1.2, b=0.75)")


    if mode == "Ввести вручную":
        manual_urls = st.text_area("Список URL конкурентов (каждый с новой строки)", 
                                   placeholder="https://comp1.ru/page\nhttps://comp2.ru/page\n...", 
                                   height=150)
    else:
        manual_urls = ""

# --- 3.2. БЛОК ЗАПУСКА И АНАЛИЗА ---

if st.button("📈 Запустить анализ", type="primary", use_container_width=True):
    
    if not my_url:
        st.error("⚠️ Введите Ваш URL для анализа!")
        st.stop()
        
    if mode == "Google ТОП" and not query:
        st.error("⚠️ Для поиска нужен поисковый запрос!")
        st.stop()
        
    if mode == "Ввести вручную" and not manual_urls:
        st.error("⚠️ Введите список URL конкурентов вручную!")
        st.stop()


    st.markdown("---")
    st.subheader("🔍 Процесс анализа")

    settings = {
        'exclude': [x.strip() for x in excludes.split() if x.strip()],
        'min_sites': min_sites,
        'max_spam_factor': max_spam_factor,
        'bm25_k1': 1.2,
        'bm25_b': 0.75,
    }
    
    # 1. Получение списка конкурентов
    comps = []
    
    if mode == "Google ТОП":
        if not USE_SEARCH:
            st.error("❌ Библиотека googlesearch-python недоступна. Пожалуйста, используйте ручной ввод.")
            st.stop()
            
        with st.spinner(f"Ищем {top_n} конкурентов по запросу '{query}' в Google..."):
            try:
                excl_list = settings['exclude']
                # Ищем больше, чтобы отфильтровать ненужные
                found = search(query, num_results=top_n * 2, lang="ru")
                count = 0
                for u in found:
                    if u == my_url: continue 
                    if any(x in u for x in excl_list): continue 
                    comps.append(u)
                    count += 1
                    if count >= top_n: break
            except Exception as e:
                st.error(f"Ошибка поиска Google: {e}. Попробуйте ручной список.")
                st.stop()
    else:
        if manual_urls:
            comps_raw = [u.strip() for u in manual_urls.split('\n') if u.strip()]
            for u in comps_raw:
                if u == my_url: continue
                if any(x in u for x in settings['exclude']): continue
                comps.append(u)
        
    if not comps:
        st.error("❌ Список конкурентов пуст или не удалось собрать данные.")
        st.stop()
    else:
        st.success(f"✅ Найдены конкуренты ({len(comps)} URL):")
        st.dataframe(pd.DataFrame({'URL': comps}), use_container_width=True, height=200)

        # 2. ЗАПУСК БЭКЕНДА
        with st.spinner("🚀 Запуск гибридного анализа..."):
            df_res = run_analysis(my_url, comps, settings)
        
        if df_res is not None and not df_res.empty:
            st.markdown("### 📊 Результаты анализа")
            
            # --- Логика для подсветки строк ---
            def highlight_rec(val):
                val_str = str(val)
                # Зеленый - для "Добавить"
                if "Добавить" in val_str: 
                    return 'color: #166534; font-weight: bold; background-color: #DCFCE7' 
                # Красный - для "Убрать"
                if "Убрать" in val_str: 
                    return 'color: #991B1B; font-weight: bold; background-color: #FEE2E2' 
                return ''
            
            # Применяем стили к колонкам с рекомендациями
            styled_df = df_res.style.applymap(highlight_rec, subset=['Переспам', 'Текст Добавить/Убрать', 'Тег A Добавить/Убрать'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Скачать результаты в CSV",
                data=csv,
                file_name=f'seo_relevance_analysis_{urlparse(my_url).netloc}.csv',
                mime='text/csv',
                type="secondary"
            )

        else:
            st.warning("⚠️ Анализ не дал значимых результатов. Проверьте Ваш URL и список конкурентов.")
