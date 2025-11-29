st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');
        
        /* --- 1. ГЛОБАЛЬНЫЙ ФОН (ПРИНУДИТЕЛЬНО СВЕТЛЫЙ) --- */
        [data-testid="stAppViewContainer"] {
            background-color: #F3F6F9 !important;
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
            font-weight: 800 !important;
        }
        
        /* --- 3. КАРТОЧКИ (БЕЛЫЕ БЛОКИ С ТЕНЬЮ) --- */
        .css-card {
            background-color: #FFFFFF;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            border: 1px solid #E2E8F0;
            margin-bottom: 24px;
        }
        
        /* --- 4. ПОЛЯ ВВОДА (БЕЛЫЙ ФОН, ЧЕРНЫЙ ТЕКСТ) --- */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            caret-color: #000000 !important; /* Курсор ввода */
            border: 2px solid #E2E8F0 !important;
            border-radius: 8px !important;
            font-size: 15px !important;
        }
        
        /* Фокус на поле */
        .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        }
        
        /* Выпадающие списки (меню внутри) */
        ul[data-baseweb="menu"] {
            background-color: #FFFFFF !important;
        }
        li[data-baseweb="option"] {
            color: #000000 !important;
        }
        
        /* --- 5. КНОПКА (СИНИЙ ГРАДИЕНТ) --- */
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
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px -5px rgba(37, 99, 235, 0.5) !important;
        }
        div.stButton > button:active {
            color: #FFFFFF !important; /* Чтобы текст не пропадал при нажатии */
        }
        
        /* --- 6. ТАБЛИЦЫ (ЧИТАЕМЫЕ) --- */
        div[data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* Заголовки таблиц */
        [data-testid="stDataFrame"] th {
            background-color: #F8FAFC !important;
            color: #475569 !important;
            font-weight: 700 !important;
            border-bottom: 1px solid #E2E8F0 !important;
        }
        [data-testid="stDataFrame"] td {
            color: #334155 !important;
            background-color: #FFFFFF !important;
            border-bottom: 1px solid #F1F5F9 !important;
        }
        
        /* --- 7. ЧЕКБОКСЫ И РАДИОКНОПКИ --- */
        label[data-testid="stLabel"] {
            font-size: 14px;
            font-weight: 600 !important;
            color: #334155 !important;
        }
        /* Сами чекбоксы */
        span[data-baseweb="checkbox"] div {
            background-color: #FFFFFF !important;
        }
        
        /* --- 8. EXPANDER (НАСТРОЙКИ) --- */
        .streamlit-expanderHeader {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            color: #0F172A !important;
        }
        
        /* Убираем лишние отступы */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
        }
    </style>
""", unsafe_allow_html=True)
