import streamlit as st

st.set_page_config(page_title="CreditGuard", layout="wide")

navigation = st.navigation(
    [
        st.Page("views/risk_assessment.py", title="Risk Assessment", default=True),
        st.Page("views/model_performance.py", title="Model Performance"),
        st.Page("views/about.py", title="About"),
    ]
)
navigation.run()
