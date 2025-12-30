from godmode.webapp.app import create_app


def test_webapp_create_app_smoke() -> None:
    app = create_app()
    assert app is not None


