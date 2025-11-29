try:
            with st.spinner(f"Сбор ТОПа {st.session_state.settings_search_engine}..."):
                if not USE_SEARCH:
                    st.error("Библиотека 'googlesearch' не найдена.")
                    st.stop()

                found = search(st.session_state.query_input, num_results=st.session_state.settings_top_n * 2, lang="ru")
                cnt = 0
                for u in found:
                    if my_input_type == "Релевантная страница на вашем сайте" and st.session_state.my_url_input in u: continue
                    if any(x in urlparse(u).netloc for x in excl): continue
                    target_urls.append(u)
                    cnt += 1
                    if cnt >= st.session_state.settings_top_n: break
        except Exception as e:
            st.error(f"Ошибка при поиске: {e}")
            st.stop()
    else:
        # Здесь мы используем данные из поля ввода, которое определено в интерфейсе
        raw_urls = st.session_state.get("manual_urls_ui", "")
        target_urls = [u.strip() for u in raw_urls.split('\n') if u.strip()]

    if not target_urls:
        st.error("Нет конкурентов для анализа.")
        st.stop()
        
    # 3. Скачивание/Обработка своей страницы
    my_data = None
    if my_input_type == "Релевантная страница на вашем сайте":
        with st.spinner(f"Скачивание страницы: {st.session_state.my_url_input}..."):
            my_data = parse_page(st.session_state.my_url_input, settings)
        if not my_data:
            st.error(f"Не удалось скачать страницу: {st.session_state.my_url_input}")
            st.stop()
    elif my_input_type == "Исходный код страницы или текст":
        my_data = {
            'url': 'Local Content', 'domain': 'local.content',
            'body_text': st.session_state.my_content_input, 'anchor_text': ''
        }

    # 4. Скачивание страниц конкурентов (многопоточность)
    comp_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(parse_page, u, settings): u for u in target_urls}
        done = 0
        total_tasks = len(target_urls)
        prog_comp = st.progress(0)
        status_comp = st.empty()
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: comp_data.append(res)
            done += 1
            prog_comp.progress(done / total_tasks)
            status_comp.text(f"Скачано {done} из {total_tasks} конкурентов...")

    prog_comp.empty()
    status_comp.empty()

    if len(comp_data) < 2 and my_input_type != "Без страницы":
        st.warning(f"Мало данных конкурентов (менее 2). Продолжаю с {len(comp_data)}.")
        
    if not comp_data and my_input_type == "Без страницы":
        st.error("Не удалось скачать ни одну страницу конкурентов. Невозможно продолжить анализ.")
        st.stop()

    # 5. Расчет метрик
    with st.spinner("Расчет метрик релевантности..."):
        results = calculate_metrics(comp_data, my_data, settings)

    st.success("✅ Анализ завершен!")

    # 6. Вывод результатов

    # 6.1. Релевантность ТОПа
    if not results['relevance_top'].empty:
        st.markdown("## 4. Обзор ТОПа")
        st.dataframe(results['relevance_top'], use_container_width=True)
        st.markdown(f"""
            <div style='background-color: {LIGHT_BG_MAIN}; padding: 10px; border-radius: 5px;'>
                <b>Ваша страница:</b> Ширина (кол-во уникальных слов) = {results['my_score']['width']} | Глубина (общее кол-во слов) = {results['my_score']['depth']}
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
    # 6.2. Детализация
    if not results['depth'].empty:
        st.markdown("## 5. Детализация по словам (Обязательные работы)")
        st.caption("Показаны слова, которые есть у вас (TF>0) или встречаются минимум у 2 конкурентов (DF>=2).")
        
        # Пагинация для таблицы Depth
        rows_per_page = 15
        df_d = results['depth'].sort_values(by=["Общее Добавить/Убрать", "diff_abs"], ascending=[True, True]).reset_index(drop=True)
        total_pages = math.ceil(len(df_d) / rows_per_page)
        
        if 'page_number' not in st.session_state:
            st.session_state.page_number = 1
            
        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
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
        st.download_button("Скачать ВСЮ таблицу (CSV)", df_d.to_csv().encode('utf-8'), "depth.csv")
        
        with st.expander("2. Гибридный ТОП"):
            st.dataframe(results['hybrid'].sort_values(by="TF-IDF ТОП", ascending=False), use_container_width=True)
            
        with st.expander("3. N-граммы"):
            st.dataframe(results['ngrams'].sort_values(by="TF-IDF", ascending=False), use_container_width=True)

    
    with st.expander("4. ТОП релевантных страниц конкурентов"):
        st.dataframe(results['relevance_top'], use_container_width=True)

