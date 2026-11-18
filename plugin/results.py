"""Publish native DSA annotations and small downloadable results."""

from __future__ import annotations

import json
from pathlib import Path

GEOMETRY = "40x_512px_384px_overlap"
OWNER = "CorticalMedullaryIdentification"


def attach_refined_heatmap(client, item_id: str, path: Path) -> list[str]:
    documents = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(documents, list) or len(documents) != 2:
        raise ValueError("Refined heatmap must contain cortex and medulla annotation records")
    names = {"cortical heatmap", "medullary heatmap"}
    if {doc.get("name") for doc in documents} != names:
        raise ValueError("Unexpected refined heatmap class names")
    existing = client.get("annotation", parameters={"itemId": item_id, "limit": 0})
    ids = []
    for doc in documents:
        if not isinstance(doc.get("elements"), list):
            raise ValueError("Refined heatmap has no annotation elements")
        attributes = doc.setdefault("attributes", {})
        attributes["sourcePlugin"] = OWNER
        attributes["sourceItemId"] = item_id
        attributes["layer"] = "refined_heatmap"
        owned = [entry for entry in existing
                 if entry.get("annotation", {}).get("name") == doc["name"]
                 and entry.get("annotation", {}).get("attributes", {}).get("sourcePlugin") == OWNER
                 and entry.get("annotation", {}).get("attributes", {}).get("sourceItemId") == item_id]
        if len(owned) > 1:
            raise ValueError(f"Multiple managed {doc['name']} annotations; resolve manually")
        if owned:
            response = client.put(f"annotation/{owned[0]['_id']}", json=doc)
        else:
            response = client.post("annotation", parameters={"itemId": item_id}, json=doc)
        ids.append(str(response["_id"]))
    return ids


def upload_result(client, folder_id: str, item_id: str, run_id: str,
                  refined: Path, status: Path, manifest: Path | None = None) -> dict:
    folder = client.createFolder(folder_id, GEOMETRY, parentType="folder", reuseExisting=True)
    name = f"{OWNER}_{item_id}_{run_id}"
    result_item = client.createItem(folder["_id"], name, reuseExisting=False)
    client.uploadFileToItem(result_item["_id"], str(refined), filename=refined.name)
    client.uploadFileToItem(result_item["_id"], str(status), filename="status.json")
    if manifest:
        client.uploadFileToItem(result_item["_id"], str(manifest), filename="manifest.json")
    return {"folderId": str(folder["_id"]), "itemId": str(result_item["_id"])}
