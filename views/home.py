"""ClearMind home: choose between taking the survey and the admin backend."""
import streamlit as st

from ui import brand_header

brand_header("technologies")
st.markdown("### Understand your exam stress, and do something about it.")
st.write("ClearMind uses machine learning to estimate how much pressure you're under, shows you "
         "exactly what's driving it, and helps you build a calmer plan for test day.")
st.write("")

c1, c2 = st.columns(2, gap="medium")
with c1:
    with st.container(border=True):
        st.markdown("#### :material/edit_note: Take the survey")
        st.write("Six quick questions, rated 1 to 10. Get your stress-risk score, what's behind it, "
                 "and a personal AI coach. Use an account to track yourself over time, or go as a guest.")
        if st.button("Start the survey", type="primary", width="stretch"):
            st.switch_page("views/survey.py")
with c2:
    with st.container(border=True):
        st.markdown("#### :material/admin_panel_settings: Admin login")
        st.write("For the ClearMind team: student data access and deletion requests, plus the EEG "
                 "cognitive-workload research model.")
        if st.button("Admin login", width="stretch"):
            st.switch_page("views/admin.py")

st.divider()
st.markdown("#### How it works")
s1, s2, s3 = st.columns(3)
s1.markdown("**1. Check in**  \nRate your sleep, anxiety, workload, extracurriculars and how classes are going.")
s2.markdown("**2. See the why**  \nFind out which of your answers raised your stress risk and which lowered it.")
s3.markdown("**3. Make a plan**  \nGet tips for your top stressors, chat with the coach, and watch your trend on your dashboard.")

st.write("")
st.page_link("views/science.py", label="The science behind ClearMind", icon=":material/science:")

st.write("")
st.caption("ClearMind is an educational tool, not a medical or psychological assessment, and not a "
           "crisis service. If you're struggling, call or text **988** (US), text **HOME** to "
           "**741741**, or visit **findahelpline.com**.")
