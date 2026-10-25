"""ClearMind platform entry point.

Run: streamlit run app.py

Routes between the home page, the student survey (/survey), the science page
(/science) and the password-protected admin backend (/admin).
"""
import streamlit as st

import storage
from ui import setup_page

setup_page("ClearMind")

page = st.navigation([
    st.Page("views/home.py", title="ClearMind", icon=":material/home:", default=True),
    st.Page("views/survey.py", title="Take the survey", icon=":material/edit_note:", url_path="survey"),
    st.Page("views/science.py", title="The science behind", icon=":material/science:", url_path="science"),
    st.Page("views/admin.py", title="Admin", icon=":material/admin_panel_settings:", url_path="admin"),
], position="hidden")

# Links from reminder emails: ?resume=<token> reopens a guest's history,
# ?unsubscribe=<token> stops reminders. Tokens are removed from the address bar.
params = st.query_params
if "unsubscribe" in params:
    done = storage.unsubscribe(params["unsubscribe"])
    st.query_params.clear()
    st.session_state.flash = ("You're unsubscribed. Your email address has been deleted."
                              if done else "That unsubscribe link has already been used.")
elif "resume" in params:
    uid = storage.resume_user(params["resume"])
    st.query_params.clear()
    if uid:
        st.session_state.user = {"id": uid, "name": "Guest", "guest": True}
        st.switch_page("views/survey.py")
    st.session_state.flash = "That check-in link has expired. You can still take the survey as a new guest."

if msg := st.session_state.pop("flash", None):
    st.info(msg)
page.run()
