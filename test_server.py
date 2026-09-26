from fastapi.testclient import TestClient
from paged_infer.server.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["free_blocks"] > 0
    print("--- [1] Health check endpoint passed ---")
    print(f"Status: {data['status']} | Free Blocks: {data['free_blocks']}/{data['total_blocks']}")


def test_generate():
    payload = {
        "prompt_tokens": [101, 102, 103],
        "max_new_tokens": 4,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200, f"Generate request failed: {response.text}"
    data = response.json()
    assert "seq_id" in data
    assert len(data["output_tokens"]) == 4
    assert data["num_generated"] == 4
    print("\n--- [2] Generate endpoint passed ---")
    print(f"Sequence ID: {data['seq_id']}")
    print(f"Generated Tokens: {data['output_tokens']}")
    print(f"Tokens Generated: {data['num_generated']}")


if __name__ == "__main__":
    print("=== Running PagedInfer Server Test Suite ===")
    test_health()
    test_generate()
    print("\nAll server API tests passed successfully.")