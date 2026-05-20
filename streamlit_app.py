"""
Streamlit Cloud entrypoint wrapper.

Streamlit Cloud often expects `streamlit_app.py` as the main file. This wrapper
imports and runs the `main()` function from `main.py` so you can select
`streamlit_app.py` in the Streamlit Cloud UI without changing `main.py`.
"""
from main import main


if __name__ == "__main__":
    main()
