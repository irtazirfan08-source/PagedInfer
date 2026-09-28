import json
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


def test_generate_stream():
    payload = {
        "prompt_tokens": [201, 202],
        "max_new_tokens": 3,
    }
    received_tokens = []
    done_received = False

    with client.stream("POST", "/generate/stream", json=payload) as response:
        assert response.status_code == 200, f"Streaming failed: {response.status_code}"
        assert "text/event-stream" in response.headers.get("content-type", "")

        for line in response.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data: "):
                payload_str = line[len("data: "):]
                if payload_str == "[DONE]":
                    done_received = True
                    break
                data = json.loads(payload_str)
                received_tokens.append(data["token_id"])

    assert done_received, "Stream did not emit [DONE] sentinel"
    assert len(received_tokens) == 3, f"Expected 3 streamed tokens, got {len(received_tokens)}"
    print("\n--- [3] SSE Token Streaming endpoint passed ---")
    print(f"Streamed Tokens Received: {received_tokens}")
    print("Stream Termination Sentinel [DONE] verified.")


if __name__ == "__main__":
    print("=== Running PagedInfer Server Test Suite ===")
    test_health()
    test_generate()
    test_generate_stream()
    print("\nAll server API tests passed successfully.")