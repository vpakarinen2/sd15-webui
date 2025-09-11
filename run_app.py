import os

from dotenv import load_dotenv
from app import build_ui


if __name__ == "__main__":
    load_dotenv()

    try:
        from gradio_client import utils as _gc_utils
        _orig_get_type = getattr(_gc_utils, "get_type", None)
        _orig_json_to_py = getattr(_gc_utils, "_json_schema_to_python_type", None)

        def _patched_get_type(schema):
            if isinstance(schema, bool):
                return "object" if schema else "never"
            return _orig_get_type(schema) if _orig_get_type else "object"

        def _patched_json_to_py(schema, defs):
            if isinstance(schema, bool):
                return "Any" if schema else "None"
            return _orig_json_to_py(schema, defs) if _orig_json_to_py else {}

        if _orig_get_type:
            _gc_utils.get_type = _patched_get_type
        if _orig_json_to_py:
            _gc_utils._json_schema_to_python_type = _patched_json_to_py
    except Exception:
        pass

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    grsd = build_ui()
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    
    grsd.queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_api=False,
        max_threads=1,
        app_kwargs={"docs_url": None, "redoc_url": None}
    )
