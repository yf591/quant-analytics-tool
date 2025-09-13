"""
Cache Management utility page for Quant Analytics Tool
"""

import streamlit as st
import gc

try:
    import psutil
except ImportError:
    psutil = None


def get_system_status():
    """Get current system status"""
    try:
        if psutil:
            # Check memory usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        else:
            memory_usage = 0

        # Check cache sizes
        data_cache_size = len(st.session_state.get("data_cache", {}))
        feature_cache_size = len(st.session_state.get("feature_cache", {}))
        model_cache_size = len(st.session_state.get("model_cache", {}))

        return {
            "memory_mb": memory_usage,
            "data_cache_size": data_cache_size,
            "feature_cache_size": feature_cache_size,
            "model_cache_size": model_cache_size,
            "status": "healthy" if memory_usage < 1000 else "warning",
        }
    except Exception:
        return {
            "memory_mb": 0,
            "data_cache_size": 0,
            "feature_cache_size": 0,
            "model_cache_size": 0,
            "status": "unknown",
        }


def show_cache_management_page():
    """Display cache management utilities"""

    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    st.header("🗄️ Cache Management")

    st.markdown(
        """
    ### 📊 Cache Overview
    Manage system caches and monitor data flow across the platform.
    """
    )

    # Cache statistics
    cache_stats = {
        "Data Cache": len(st.session_state.get("data_cache", {})),
        "Feature Cache": len(st.session_state.get("feature_cache", {})),
        "Model Cache": len(st.session_state.get("model_cache", {})),
        "Backtest Cache": len(st.session_state.get("backtest_cache", {})),
        "Analysis Cache": len(st.session_state.get("analysis_cache", {})),
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📊 Cache Sizes")
        for cache_name, size in cache_stats.items():
            st.metric(cache_name, size)

    with col2:
        st.markdown("#### 🔄 Cache Operations")
        if st.button("🗑️ Clear Data Cache", use_container_width=True):
            st.session_state.data_cache = {}
            st.success("Data cache cleared!")
            st.rerun()

        if st.button("🗑️ Clear Feature Cache", use_container_width=True):
            st.session_state.feature_cache = {}
            st.success("Feature cache cleared!")
            st.rerun()

        if st.button("🗑️ Clear Model Cache", use_container_width=True):
            st.session_state.model_cache = {}
            st.success("Model cache cleared!")
            st.rerun()

    with col3:
        st.markdown("#### 🖥️ System Info")
        system_status = get_system_status()
        st.json(system_status)

        if st.button("🔄 Force Garbage Collection", use_container_width=True):
            gc.collect()
            st.success("Garbage collection completed!")

    # Detailed cache inspection
    st.markdown("---")
    st.markdown("### 🔍 Cache Inspector")

    selected_cache = st.selectbox(
        "Select cache to inspect:",
        [
            "Data Cache",
            "Feature Cache",
            "Model Cache",
            "Backtest Cache",
            "Analysis Cache",
        ],
    )

    cache_mapping = {
        "Data Cache": "data_cache",
        "Feature Cache": "feature_cache",
        "Model Cache": "model_cache",
        "Backtest Cache": "backtest_cache",
        "Analysis Cache": "analysis_cache",
    }

    cache_key = cache_mapping[selected_cache]
    cache_data = st.session_state.get(cache_key, {})

    if cache_data:
        st.markdown(f"#### {selected_cache} Contents")

        # Show cache keys and basic info
        for key, value in cache_data.items():
            with st.expander(f"📁 {key}"):
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        st.write(f"**{sub_key}**: {type(sub_value).__name__}")
                        if hasattr(sub_value, "shape"):
                            st.write(f"Shape: {sub_value.shape}")
                        elif hasattr(sub_value, "__len__"):
                            st.write(f"Length: {len(sub_value)}")
                else:
                    st.write(f"Type: {type(value).__name__}")
                    if hasattr(value, "shape"):
                        st.write(f"Shape: {value.shape}")
                    elif hasattr(value, "__len__"):
                        st.write(f"Length: {len(value)}")
    else:
        st.info(f"No data in {selected_cache}")

    st.markdown("</div>", unsafe_allow_html=True)
