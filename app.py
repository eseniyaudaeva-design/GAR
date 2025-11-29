st.markdown(f"""
   <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        /* ГЛОБАЛЬНОЕ ПЕРЕОПРЕДЕЛЕНИЕ ТЕМЫ STREAMLIT */
        :root {{
            --primary-color: {PRIMARY_COLOR};
            --text-color: {TEXT_COLOR};
        }}
        
        /* 1. ОСНОВНАЯ ТИПОГРАФИКА */
        html, body, [class*="stApp"], [class*="css"] {{
            font-family: 'Inter', sans-serif;
            background-color: #FFFFFF !important;
            color: {TEXT_COLOR} !important; 
        }}
        h1, h2, h3, p, label, span, div, a {{
            color: {TEXT_COLOR} !important; 
        }}

        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 100% !important; 
        }}
        
        /* ======================================================= */
        /* 2. ПОЛЯ ВВОДА (Input, Textarea, Selectbox)              */
        /* ======================================================= */
        
        .stTextInput input, 
        .stTextArea textarea, 
        .stSelectbox div[data-baseweb="select"] > div {{
            color: {TEXT_COLOR} !important;
            background-color: {LIGHT_BG_MAIN} !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 6px;
        }}

        /* ФОКУС: СИНЯЯ РАМКА */
        .stTextInput input:focus,
        .stTextArea textarea:focus,
        .stSelectbox div[data-baseweb="select"] > div:focus-within {{
            border-color: {PRIMARY_COLOR} !important;
            box-shadow: 0 0 0 1px {PRIMARY_COLOR} !important;
        }}
        
        /* Иконки Selectbox */
        .stSelectbox svg {{
            fill: {TEXT_COLOR} !important;
        }}

        /* ======================================================= */
        /* ИСПРАВЛЕНИЕ: ЦВЕТА ВЫПАДАЮЩЕГО СПИСКА (Скриншот 3)      */
        /* ======================================================= */
        
        /* Само всплывающее окно меню */
        ul[data-baseweb="menu"] {{
            background-color: #FFFFFF !important; /* Белый фон списка */
            border: 1px solid {BORDER_COLOR} !important;
        }}
        
        /* Элементы списка */
        li[data-baseweb="option"] {{
            color: {TEXT_COLOR} !important; /* Темный текст */
            background-color: #FFFFFF !important;
        }}
        
        /* Элемент списка при наведении или выборе */
        li[data-baseweb="option"]:hover,
        li[data-baseweb="option"][aria-selected="true"] {{
            background-color: {LIGHT_BG_MAIN} !important; /* Светло-серый фон при наведении */
            color: {PRIMARY_COLOR} !important; /* Синий текст */
            font-weight: 600;
        }}
        
        /* ======================================================= */
        /* 3. РАДИО-КНОПКИ (ИСПРАВЛЕНИЕ - Скриншот 1 и 2)          */
        /* ======================================================= */

        /* Контейнер радио-кнопок */
        div[role="radiogroup"] label {{
            background-color: #FFFFFF !important;
            border: 1px solid {BORDER_COLOR};
            border-radius: 6px;
            padding: 10px 15px;
            margin-right: 5px;
            transition: border 0.2s;
        }}

        /* 3.1. КРУЖОК - НЕ ВЫБРАН (Белый фон, ТОНКАЯ ЧЕРНАЯ рамка) */
        div[role="radiogroup"] label div[data-baseweb="radio"] > div {{
            background-color: #FFFFFF !important;
            border: 1px solid #222222 !important; /* Исправлено на почти черный */
        }}

        /* 3.2. КРУЖОК - ВЫБРАН (СИНИЙ фон, СИНЯЯ рамка) */
        div[role="radiogroup"] label input:checked + div[data-baseweb="radio"] > div {{
            background-color: {PRIMARY_COLOR} !important;
            border-color: {PRIMARY_COLOR} !important;
            border-width: 1px !important;
        }}
        
        /* 3.3. ВНУТРЕННЯЯ ТОЧКА (Белая) */
        div[role="radiogroup"] label input:checked + div[data-baseweb="radio"] > div > div {{
            background-color: #FFFFFF !important;
        }}

        /* Текст выбранной радио-кнопки */
        div[role="radiogroup"] label input:checked + div {{
             color: {TEXT_COLOR} !important;
             font-weight: 600;
        }}
        
        /* Рамка вокруг выбранного блока (синяя) */
        div[role="radiogroup"] label:has(input:checked) {{
            border-color: {PRIMARY_COLOR} !important;
            background-color: #F8FAFC !important; /* Чуть подсвечиваем фон блока */
        }}


        /* ======================================================= */
        /* 4. ЧЕКБОКСЫ (ИСПРАВЛЕНИЕ - Скриншот 1 и 2)              */
        /* ======================================================= */

        /* 4.1. КВАДРАТ - НЕ ВЫБРАН (Белый фон, ТОНКАЯ ЧЕРНАЯ рамка) */
        div[data-baseweb="checkbox"] > div:first-child {{
            background-color: #FFFFFF !important;
            border: 1px solid #222222 !important; /* Исправлено на почти черный */
        }}

        /* 4.2. КВАДРАТ - ВЫБРАН (СИНИЙ фон, СИНЯЯ рамка) */
        div[data-baseweb="checkbox"] input:checked + div:first-child {{
            background-color: {PRIMARY_COLOR} !important;
            border-color: {PRIMARY_COLOR} !important;
        }}
        
        /* ГАЛОЧКА (Белая) */
        div[data-baseweb="checkbox"] input:checked + div:first-child svg {{
            fill: #FFFFFF !important;
        }}
        
        /* Убираем ховер-эффект (чтобы не краснел/оранжевел) */
        div[data-baseweb="checkbox"]:hover > div:first-child {{
            border-color: {PRIMARY_COLOR} !important;
        }}


        /* ======================================================= */
        /* 5. КНОПКА ЗАПУСКА                                       */
        /* ======================================================= */
        .stButton button {{
            background-image: linear-gradient(to right, {PRIMARY_COLOR}, {PRIMARY_DARK});
            color: white !important;
            font-weight: bold;
            border-radius: 6px;
            height: 50px;
            width: 100%;
            border: none;
            margin-top: 10px;
        }}
        .stButton button:focus {{
            border-color: {PRIMARY_COLOR} !important;
            box-shadow: 0 0 0 1px {PRIMARY_COLOR} !important;
            color: white !important;
        }}


        /* ======================================================= */
        /* 6. САЙДБАР                                              */
        /* ======================================================= */
        .st-emotion-cache-1cpxwwu {{ 
            width: 65% !important;
            padding-right: 20px; 
            max-width: 65% !important;
            padding-left: 0 !important;
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
        }}

        div[data-testid="column"]:nth-child(2) .stSelectbox div[data-baseweb="select"] > div,
        div[data-testid="column"]:nth-child(2) .stTextInput input,
        div[data-testid="column"]:nth-child(2) .stTextarea textarea {{
            background-color: {LIGHT_BG_MAIN} !important; 
            color: {TEXT_COLOR} !important;
            border: 1px solid {BORDER_COLOR} !important;
            box-shadow: none !important;
        }}

        div[data-testid="column"]:nth-child(2) .stCaption {{
            display: none;
        }}

    </style>
""", unsafe_allow_html=True)
