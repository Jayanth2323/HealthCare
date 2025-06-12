def bootstrap_font() -> str:
    os.makedirs(FONT_DIR, exist_ok=True)
    st.write(f"bootstrap_font: checking {FONT_PATH}")
    if not _is_valid_ttf(FONT_PATH):
        st.write("Font not valid or missing; attempting download...")
        try:
            urllib.request.urlretrieve(RAW_URL, FONT_PATH)
            st.write("Downloaded font to:", FONT_PATH)
        except Exception as download_exc:
            raise RuntimeError(f"Font download failed → {download_exc}") from download_exc
        if not _is_valid_ttf(FONT_PATH):
            raise RuntimeError("Downloaded file is not a valid TrueType font.")
    else:
        st.write("Font already present and valid at:", FONT_PATH)
    return FONT_PATH
