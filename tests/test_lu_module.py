from fastapi.testclient import TestClient

from app.main import app
from app.services.lu_service import LuService


client = TestClient(app)


def test_lu_service_layer_separation():
    three_flows = LuService.get_three_flows_snapshot()
    four_matrix = LuService.get_four_matrix_snapshot()
    discipline = LuService.get_334_discipline_snapshot()
    candidates = LuService.get_candidate_pool()

    assert three_flows["layer"] == "战略层"
    assert four_matrix["layer"] == "战略层"
    assert discipline["layer"] == "仓位纪律层"
    assert candidates["layer"] == "执行层"


def test_lu_api_endpoints_shape():
    paths = [
        "/api/lu/three-flows",
        "/api/lu/four-matrix",
        "/api/lu/discipline",
        "/api/lu/candidates",
    ]

    for path in paths:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["is_mock"] is True
        assert "as_of" in data
        assert "mode" in data


def test_lu_page_and_existing_pages_available():
    for path in ["/lu", "/pro", "/home", "/strategies", "/knowledge"]:
        response = client.get(path)
        assert response.status_code == 200
