"""Validate Girder resources before any slide is staged to HiPerGator."""

from __future__ import annotations

import re
from pathlib import PurePath

SLIDE_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi"}


def select_items(client, file_ids: list[str], folder_id: str | None,
                 pattern: str) -> list[dict]:
    if not file_ids and not folder_id:
        raise ValueError("Select at least one slide file or an input folder")
    try:
        expression = re.compile(pattern)
    except re.error as error:
        raise ValueError(f"Invalid item filter regular expression: {error}") from error
    if len(file_ids) != len(set(file_ids)):
        raise ValueError("Duplicate slide files were selected")
    selected_files = {file_id: client.getFile(file_id) for file_id in file_ids}
    ids = [str(file["itemId"]) for file in selected_files.values()]
    if folder_id:
        ids.extend(str(item["_id"]) for item in client.listItem(folder_id))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate slide items were selected")
    selected = []
    for item_id in ids:
        item = client.getItem(item_id)
        if not expression.search(item["name"]):
            continue
        file_id = item.get("largeImage", {}).get("fileId")
        if not file_id:
            raise ValueError(f"Item is not a DSA large image: {item['name']}")
        source_file = client.getFile(str(file_id))
        if PurePath(source_file["name"]).suffix.lower() not in SLIDE_EXTENSIONS:
            raise ValueError(f"Unsupported slide file type: {source_file['name']}")
        if not item.get("folderId"):
            raise ValueError(f"Slide has no parent folder: {item['name']}")
        if any(str(file["itemId"]) == item_id and str(file["_id"]) != str(file_id)
               for file in selected_files.values()):
            raise ValueError(f"Selected file is not the large-image source for {item['name']}")
        selected.append({"item_id": str(item["_id"]), "folder_id": str(item["folderId"]),
                         "file_id": str(file_id), "name": item["name"],
                         "filename": source_file["name"]})
    if not selected:
        raise ValueError("Item filter matched no selectable slide items")
    return selected


def validate_checkpoints(client, cortical_id: str, medullary_id: str) -> dict:
    if not cortical_id or not medullary_id:
        raise ValueError("Select both cortical and medullary model checkpoints")
    files = [client.getFile(file_id) for file_id in (cortical_id, medullary_id)]
    for file in files:
        if PurePath(file["name"]).suffix.lower() != ".pkl":
            raise ValueError(f"Model checkpoint must be a .pkl DSA file: {file['name']}")
    if cortical_id != medullary_id:
        raise ValueError(
            "This repository trains one binary cortex/medulla SVC checkpoint. "
            "Select the same shared .pkl file in both fields. Separate class models "
            "need a documented calibration and score-combination interface.")
    return files[0]
