"""Print the OpenAPI schema without starting the server (used by `make types`)."""
import json

from .main import app

if __name__ == "__main__":
    print(json.dumps(app.openapi(), indent=2))
