# Antes (con fondos blancos):
with col_mod1:
    if modelos.get('ann') is not None:
        st.markdown(
            """
            <div style="
                background-color: #e8f5e8;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
            ">
                <strong>✅ ANN</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                background-color: #ffebee;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
            ">
                <strong>❌ ANN</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

# Después (sin fondos):
with col_mod1:
    if modelos.get('ann') is not None:
        st.markdown("✅ ANN")
    else:
        st.markdown("❌ ANN")
