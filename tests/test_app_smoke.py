from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_title() -> None:
    app = AppTest.from_file("snowcat_demo/app.py")

    app.run(timeout=45)

    assert not app.exception
    assert any(title.value == "Snowcat GEMM Explorer" for title in app.title)
