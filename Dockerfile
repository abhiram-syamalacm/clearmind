# Container for hosts that run Docker (Hugging Face Spaces, Render, Railway).
FROM python:3.12-slim

# Hugging Face Spaces runs containers as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

EXPOSE 8501
# Railway and Render pass the port in $PORT; Hugging Face uses 8501.
# Spaces shows the app inside an iframe on huggingface.co, where Streamlit's
# XSRF check blocks file uploads, so it is turned off here.
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.enableXsrfProtection=false --server.enableCORS=false
