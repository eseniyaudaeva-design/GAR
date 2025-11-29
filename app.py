st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        :root {{
            --primary-color: {PRIMARY_COLOR};
            --text-color: {TEXT_COLOR};
        }}
        
        /* 1. БАЗОВЫЙ ТЕКСТ */
        html, body, .stApp {{
            font-family: 'Inter', sans-serif;
            background-color: #FFFFFF !important;
            color: {TEXT_COLOR} !important;
        }}
        
        h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, div[data-testid="stMarkdownContainer"] p {{
            color: {TEXT_COLOR} !important;
        }}

        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 100% !important; 
        }}
        
        /* ======================================================= */
        /* ПОЛЯ ВВОДА                                              */
        /* ======================================================= */
        
        .stTextInput input, 
        .stTextArea textarea, 
        .stSelectbox div[data-baseweb="select"] > div {{
            color: {TEXT_COLOR} !important;
            background-color: {LIGHT_BG_MAIN} !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 6px;
        }}

        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="textarea"]:focus-within,
        div[data-baseweb="select"] > div:focus-within {{
            border-color: {PRIMARY_COLOR} !important;
            box-shadow: 0 0 0 1px {PRIMARY_COLOR} !important;
        }}

        .stTextInput input:focus,
        .stTextArea textarea:focus {{
            outline: none !important;
            border-color: transparent !important;
            box-shadow: none !important;
        }}
        
        input, textarea {{
            caret-color: {PRIMARY_COLOR} !important;
            color: {TEXT_COLOR} !important;
        }}
        
        ::placeholder {{
            color: #94a3b8 !important;
            opacity: 1;
        }}
        
        .stSelectbox svg {{
            fill: {TEXT_COLOR} !important;
        }}

        /* ======================================================= */
        /* !!! ИСПРАВЛЕНИЕ ВЫПАДАЮЩЕГО СПИСКА (POPOVER) !!!        */
        /* ======================================================= */
        
        /* Фон самого выпадающего окна и списка */
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] ul {{
            background-color: #FFFFFF !important;
        }}

        /* Опции (строки) внутри списка */
        div[data-baseweb="menu"] li {{
            background-color: #FFFFFF !important;
            color: {TEXT_COLOR} !important;
        }}
        
        /* Контейнер для текста опции */
        div[data-baseweb="menu"] li span, 
        div[data-baseweb="menu"] li div {{
            color: {TEXT_COLOR} !important;
        }}

        /* При наведении курсора на опцию */
        div[data-baseweb="menu"] li:hover {{
            background-color: {LIGHT_BG_MAIN} !important;
        }}

        /* Выбранный элемент в списке (активный) */
        div[data-baseweb="menu"] li[aria-selected="true"] {{
            background-color: {LIGHT_BG_MAIN} !important;
            color: {PRIMARY_COLOR} !important;
            font-weight: 600;
        }}
        
        /* Цвет текста выбранного элемента */
        div[data-baseweb="menu"] li[aria-selected="true"] * {{
            color: {PRIMARY_COLOR} !important;
        }}

        /* ======================================================= */
        /* РАДИО И ЧЕКБОКСЫ                                        */
        /* ======================================================= */
        
        div[role="radiogroup"] label {{
            background-color: #FFFFFF !important;
            border: 1px solid {BORDER_COLOR};
            margin-right: 5px;
        }}
        
        div[role="radiogroup"] p {{
            color: {TEXT_COLOR} !important;
        }}
        
        div[role="radiogroup"] label div[data-baseweb="radio"] > div {{
            background-color: #FFFFFF !important;
            border: 2px solid {DARK_BORDER} !important;
        }}
        div[role="radiogroup"] label input:checked + div[data-baseweb="radio"] > div {{
            background-color: {PRIMARY_COLOR} !important;
            border-color: {PRIMARY_COLOR} !important;
        }}
        div[role="radiogroup"] label input:checked + div[data-baseweb="radio"] > div > div {{
            background-color: #FFFFFF !important;
        }}
        div[role="radiogroup"] label:has(input:checked) {{
            border-color: {PRIMARY_COLOR} !important;
        }}

        /* Чекбоксы */
        div[data-baseweb="checkbox"] label, div[data-baseweb="checkbox"] p {{
            color: {TEXT_COLOR} !important;
        }}
        div[data-baseweb="checkbox"] > div:first-child {{
            background-color: #FFFFFF !important;
            border: 2px solid {DARK_BORDER} !important;
        }}
        div[data-baseweb="checkbox"] input:checked + div:first-child {{
            background-color: {PRIMARY_COLOR} !important;
            border-color: {PRIMARY_COLOR} !important;
        }}
        div[data-baseweb="checkbox"] input:checked + div:first-child svg {{
            fill: #FFFFFF !important;
        }}

        /* ======================================================= */
        /* КНОПКА                                                  */
        /* ======================================================= */
        .stButton button {{
            background-image: linear-gradient(to right, {PRIMARY_COLOR}, {PRIMARY_DARK});
            color: white !important;
            border: none;
            height: 50px;
        }}
        .stButton button:focus {{
            border-color: {PRIMARY_COLOR} !important;
            box-shadow: 0 0 0 1px {PRIMARY_COLOR} !important;
            color: white !important;
        }}
        .stButton button p {{
            color: white !important;
        }}

        /* ======================================================= */
        /* САЙДБАР                                                 */
        /* ======================================================= */
        .st-emotion-cache-1cpxwwu {{ 
            width: 65% !important;
            max-width: 65% !important;
        }}
        div[data-testid="column"]:nth-child(2) {{
            position: fixed !important;
            right: 0 !important;
            top: 0 !important;
            width: 35% !important; 
            height: 100vh !important;
            overflow-y: auto !important; 
            background-color: #FFFFFF !important; 
            padding: 1rem 1rem 2rem 1.5rem !important; 
            z-index: 100;
            box-shadow: -1px 0 0 0 {MAROON_DIVIDER} inset; 
            border-left: 1px solid {BORDER_COLOR};
        }}
        div[data-testid="column"]:nth-child(2) .stSelectbox div[data-baseweb="select"] > div,
        div[data-testid="column"]:nth-child(2) .stTextInput input,
        div[data-testid="column"]:nth-child(2) .stTextarea textarea {{
            background-color: {LIGHT_BG_MAIN} !important; 
            color: {TEXT_COLOR} !important;
            border: 1px solid {BORDER_COLOR} !important;
        }}
        div[data-testid="column"]:nth-child(2) .stCaption {{ display: none; }}

    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ФУНКЦИИ
# ==========================================

# Инициализация state для пагинации
if 'page_number' not in st.session_state:
    st.session_state.page_number = 1

def parse_url(url):
    """Извлекает домен из URL."""
    try:
        return urlparse(url).netloc
    except:
        return ""

def process_url(url, exclude_domains, stop_words):
    """
    Обрабатывает один URL: получает текст, чистит, извлекает N-граммы и TF-IDF.
    Имитация.
    """
    domain = parse_url(url)
    if domain in exclude_domains or not domain:
        return None

    try:
        # Имитация запроса
        # response = requests.get(url, timeout=5)
        # response.raise_for_status()
        # soup = BeautifulSoup(response.content, 'html.parser')
        # text = soup.get_text()
        
        # Имитация текста
        text = f"Заголовок страницы для {url}. Лучшая цена, купить товар, 1000 рублей. Описание товара: {domain}, шт. Продажа, акции, скидки. {url}."
        
        # Очистка текста
        text = text.lower()
        text = re.sub(r'[^а-яa-z0-9\s]', '', text)
        
        tokens = text.split()
        
        # Удаление стоп-слов
        filtered_tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        # Имитация TF-IDF и N-грамм
        word_counts = Counter(filtered_tokens)
        
        # Имитация глубины
        depth = math.ceil(1 + np.random.rand() * 4) # Случайная глубина от 1 до 5
        
        # Имитация TF-IDF для примера
        tf_idf_value = np.random.rand() * 10 
        
        return {
            "URL": url,
            "Домен": domain,
            "Глубина": depth,
            "Текст": text[:100] + "...",
            "Кол-во слов": len(filtered_tokens),
            "TF-IDF": tf_idf_value,
            "Слова": word_counts.most_common(5)
        }
    except Exception as e:
        # st.error(f"Ошибка при обработке {url}: {e}")
        return None

# ==========================================
# 3. ОСНОВНОЙ ИНТЕРФЕЙС
# ==========================================

st.title("📊 GAR PRO: Анализ Поисковой Выдачи")

# ==========================================
# КОНТЕЙНЕР ДЛЯ РЕЗУЛЬТАТОВ (ОСНОВНОЕ ОКНО)
# ==========================================
col_main, col_sidebar = st.columns([0.65, 0.35])

with col_main:
    st.header("1. Ввод данных и запуск")
    
    # Имитация ввода данных
    search_query = st.text_input("Поисковый запрос (например, 'купить ноутбук')", "купить ноутбук msi", key="query_input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region = st.selectbox("Регион поиска (имитация)", REGIONS, index=REGIONS.index("Москва"), key="region_select")
        
    with col2:
        num_results = st.slider("Количество результатов (имитация)", 10, 100, 30, 10, key="num_results_slider")

    with col3:
        concurrency = st.number_input("Потоки (Threads) для парсинга", 1, 10, 5, key="concurrency_input")

    # Имитация ссылок
    if 'urls' not in st.session_state:
        st.session_state.urls = [f"https://example.com/item/{i}" for i in range(num_results)]
    
    st.markdown("---")

    if st.button(f"🚀 Запустить анализ ({search_query})", key="run_analysis_button", use_container_width=True):
        
        # Имитация парсинга и обработки
        with st.spinner("Анализ поисковой выдачи..."):
            
            # Подготовка данных
            exclude_domains = set(st.session_state.exclude_text.split())
            stop_words = set(st.session_state.stop_words_text.split())
            
            # Имитация обработки URL
            results = []
            urls_to_process = st.session_state.urls[:num_results]
            
            # Используем ThreadPoolExecutor для имитации параллельной обработки
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_url = {
                    executor.submit(process_url, url, exclude_domains, stop_words): url 
                    for url in urls_to_process
                }
                
                for future in st.as_completed(future_to_url):
                    result = future.result()
                    if result:
                        results.append(result)
            
            if not results:
                st.warning("Не удалось обработать ни одной ссылки (возможно, все домены были исключены).")
            else:
                df_results = pd.DataFrame(results)
                
                # Имитация метрик
                df_results['TF-IDF ТОП'] = np.random.rand(len(df_results))
                df_results['N-граммы ТОП'] = np.random.rand(len(df_results))
                
                # Имитация расчета глубины
                df_results['Глубина'] = df_results['Глубина'].astype(int)
                
                # Расчет среднего для имитации
                avg_depth = df_results['Глубина'].mean()
                
                # Расчет TF-IDF различий (имитация)
                df_results['TF-IDF'] = np.random.rand(len(df_results)) * 10
                df_results['Avg_TFIDF'] = df_results['TF-IDF'].mean()
                df_results['diff'] = df_results['TF-IDF'] - df_results['Avg_TFIDF']
                df_results['diff_abs'] = df_results['diff'].abs()
                
                # Подготовка разделов
                results_data = {
                    'depth': df_results.sort_values(by="Глубина", ascending=True),
                    'hybrid': df_results[['URL', 'Домен', 'TF-IDF ТОП', 'N-граммы ТОП']].sort_values(by="TF-IDF ТОП", ascending=False),
                    'ngrams': df_results[['URL', 'Домен', 'TF-IDF']].sort_values(by="TF-IDF", ascending=False),
                    'top_domains': df_results['Домен'].value_counts().reset_index().rename(columns={'index': 'Домен', 'Домен': 'Кол-во URL'})
                }
                
                st.session_state.analysis_results = results_data
                st.session_state.avg_depth = avg_depth
                st.session_state.is_processed = True
                st.session_state.page_number = 1
                st.success(f"Анализ завершен. Обработано {len(results)} из {num_results} URL.")


# ==========================================
# КОНТЕЙНЕР ДЛЯ НАСТРОЕК (САЙДБАР)
# ==========================================
with col_sidebar:
    st.header("2. Настройки")
    
    st.markdown(f"**Исключить домены (исключено {len(DEFAULT_EXCLUDE_DOMAINS)}):**")
    exclude_text = st.text_area(
        "Домены (каждый с новой строки):", 
        DEFAULT_EXCLUDE, 
        height=200, 
        key="exclude_text"
    )

    st.markdown("**Стоп-слова для очистки:**")
    stop_words_text = st.text_area(
        "Стоп-слова (каждое с новой строки):", 
        DEFAULT_STOPS, 
        height=150, 
        key="stop_words_text"
    )

# ==========================================
# 4. ВЫВОД РЕЗУЛЬТАТОВ
# ==========================================

with col_main:
    if st.session_state.get('is_processed'):
        st.subheader("3. Результаты анализа")
        results = st.session_state.analysis_results
        
        # Общая статистика
        st.markdown(f"""
        <div style="padding: 10px; background-color: {LIGHT_BG_MAIN}; border-radius: 6px; border: 1px solid {BORDER_COLOR}; margin-bottom: 20px;">
            <p style='color:{TEXT_COLOR}; font-weight: 600; margin: 0;'>
                Средняя глубина (имитация): 
                <span style='color:{PRIMARY_COLOR}; font-size: 1.2em;'>
                    {st.session_state.avg_depth:.2f}
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Детальный анализ (Глубина)", "Общие метрики"])
        
        with tab1:
            st.markdown("#### 3.1. Детальный анализ (по глубине)")
            df_d = results['depth'].reset_index(drop=True)
            
            rows_per_page = 20
            total_pages = math.ceil(len(df_d) / rows_per_page)
            
            # Пагинация
            col_p1, col_p2, col_p3 = st.columns([1, 1.5, 1])
            with col_p1:
                if st.button("⬅️ Назад", key="prev_page_button") and st.session_state.page_number > 1:
                    st.session_state.page_number -= 1
            with col_p2:
                st.markdown(f"<div style='text-align: center; padding-top: 10px; color: {TEXT_COLOR};'>Страница <b>{st.session_state.page_number}</b> из {total_pages}</div>", unsafe_allow_html=True)
            with col_p3:
                if st.button("Вперед ➡️", key="next_page_button") and st.session_state.page_number < total_pages:
                    st.session_state.page_number += 1
                        
            start_idx = (st.session_state.page_number - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            df_page = df_d.iloc[start_idx:end_idx]
            
            st.dataframe(df_page, column_config={"diff_abs": None}, use_container_width=True, height=800)
            st.download_button("Скачать ВСЮ таблицу (CSV)", df_d.to_csv(index=False).encode('utf-8'), "depth.csv")
            
            with st.expander("2. Гибридный ТОП"):
                st.dataframe(results['hybrid'].sort_values(by="TF-IDF ТОП", ascending=False), use_container_width=True)
                
            with st.expander("3. N-граммы"):
                st.dataframe(results['ngrams'].sort_values(by="TF-IDF", ascending=False), use_container_width=True)

        
        with tab2:
            st.markdown("#### 3.2. Общие метрики")
            
            st.markdown("##### 4. ТОП домены")
            st.dataframe(results['top_domains'], use_container_width=True)
            
            st.markdown("##### 5. Распределение TF-IDF")
            st.line_chart(results['depth'][['TF-IDF', 'Avg_TFIDF']].set_index(results['depth'].index))

# Запуск функции для обработки клика (необходимо для работы кнопок пагинации)
def run_app():
    # Эта часть имитирует запуск приложения и нужна для корректной работы state
    if st.session_state.get('is_processed'):
        pass

if __name__ == '__main__':
    run_app()

