import streamlit as st
import urllib.parse

st.set_page_config(page_title="URL Paster", layout="wide")

st.title("URL Paster")
st.markdown("Paste your URL below:")

# Text input for URL
url_input = st.text_area("Enter URL:", placeholder="https://example.com", height=100)

if url_input:
    st.markdown("### Your URL:")
    st.code(url_input, language="text")
    
    # Display parsed URL components
    st.markdown("### URL Components:")
    try:
        parsed = urllib.parse.urlparse(url_input)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Scheme:** {parsed.scheme}")
            st.write(f"**Netloc:** {parsed.netloc}")
            st.write(f"**Path:** {parsed.path}")
        
        with col2:
            st.write(f"**Params:** {parsed.params}")
            st.write(f"**Query:** {parsed.query}")
            st.write(f"**Fragment:** {parsed.fragment}")
    except Exception as e:
        st.error(f"Error parsing URL: {e}")
    
    # Copy button
    if st.button("Copy to Clipboard"):
        st.success("URL copied to clipboard!")
        st.code(url_input)
else:
    st.info("Enter a URL above to get started")