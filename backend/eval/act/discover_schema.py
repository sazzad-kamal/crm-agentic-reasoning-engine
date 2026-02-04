"""Discover Act! API schema by calling endpoints and inspecting responses.

Run with: python -m backend.eval.act.discover_schema
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.act_fetch import ACT_API_URL, _auth, _get


def discover_endpoint_schema(endpoint: str, params: dict | None = None) -> dict:
    """Call an endpoint and extract field names from response."""
    try:
        data = _get(endpoint, params or {})

        if not data:
            return {"endpoint": endpoint, "error": "No data returned", "fields": []}

        # Get first item to extract fields
        if isinstance(data, list):
            sample = data[0] if data else {}
            count = len(data)
        else:
            sample = data
            count = 1

        # Extract field names and types
        fields = {}
        for key, value in sample.items():
            field_type = type(value).__name__
            if value is None:
                field_type = "null"
            elif isinstance(value, dict):
                field_type = f"object({len(value)} keys)"
            elif isinstance(value, list):
                field_type = f"array({len(value)} items)"
            fields[key] = {
                "type": field_type,
                "sample": str(value)[:100] if value else None
            }

        return {
            "endpoint": endpoint,
            "params": params,
            "count": count,
            "fields": fields,
            "sample": sample
        }
    except Exception as e:
        return {"endpoint": endpoint, "error": str(e), "fields": {}}


def main():
    """Discover schema for all relevant endpoints."""
    print("=" * 60)
    print("Act! API Schema Discovery")
    print("=" * 60)
    print(f"API URL: {ACT_API_URL}")

    # Authenticate first
    print("\nAuthenticating...")
    try:
        _auth()
        print("[OK] Authenticated successfully")
    except Exception as e:
        print(f"[ERR] Authentication failed: {e}")
        return

    # Endpoints to discover
    import time
    today = time.strftime("%Y-%m-%d", time.gmtime())

    endpoints: list[tuple[str, dict[str, str | int]]] = [
        ("/api/contacts", {"$top": 5}),
        ("/api/opportunities", {"$top": 5}),
        ("/api/activities", {"$top": 5}),
        ("/api/history", {"$top": 5}),
        ("/api/companies", {"$top": 5}),
        ("/api/groups", {"$top": 5}),
        ("/api/notes", {"$top": 5}),
        ("/api/calendar", {"startDate": today, "$top": 5}),
    ]

    results = {}

    for endpoint, params in endpoints:
        print(f"\nDiscovering {endpoint}...")
        result = discover_endpoint_schema(endpoint, params)
        results[endpoint] = result

        if "error" in result and result.get("fields") == {}:
            print(f"  [ERR] Error: {result['error']}")
        else:
            print(f"  [OK] Found {len(result.get('fields', {}))} fields, {result.get('count', 0)} records")
            for field, info in result.get("fields", {}).items():
                print(f"    - {field}: {info['type']}")

    # Save to file
    output_path = Path(__file__).parent / "schema.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Schema saved to {output_path}")

    # Also generate Python constant
    print("\n" + "=" * 60)
    print("ACT_API_SCHEMA for prompts:")
    print("=" * 60)

    schema_text = "ACT_API_SCHEMA = '''\n## Act! CRM Web API Schema (discovered from live API)\n\n"

    for endpoint, result in results.items():
        if result.get("fields"):
            schema_text += f"### {endpoint}\n"
            schema_text += "Fields:\n"
            for field, info in result["fields"].items():
                schema_text += f"- {field} ({info['type']})\n"
            schema_text += "\n"

    schema_text += "'''"
    print(schema_text)

    # Save schema.py
    schema_py_path = Path(__file__).parent / "schema.py"
    with open(schema_py_path, "w") as f:
        f.write('"""Act! API schema discovered from live API."""\n\n')
        f.write(schema_text)
        f.write("\n")
    print(f"\n[OK] Schema constant saved to {schema_py_path}")


if __name__ == "__main__":
    main()
