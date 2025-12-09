#!/cbica/home/salot/miniconda3/envs/salot/bin/python
"""Anonymize subject IDs in dataset."""

import os
from pathlib import Path

# NOTE: This ID mapping used to include the actual subject IDs, but they are PII,
# so we cannot share them here.
ID_MAPPER = {
    "sub-01": "sub-01",
    "sub-02": "sub-02",
    "sub-03": "sub-03",
    "sub-04": "sub-04",
    "sub-05": "sub-05",
    "sub-06": "sub-06",
    "sub-07": "sub-07",
    "sub-08": "sub-08",
}

if __name__ == "__main__":
    dset_dir = "/cbica/projects/executive_function/mebold_trt/dset"

    # Rename files and folders with new subject IDs
    for orig_id, new_id in ID_MAPPER.items():
        all_objects = list(
            filter(
                lambda path: not any(
                    (part for part in path.parts if part.startswith("."))
                )
                and orig_id in path.name,
                Path(dset_dir).rglob("*"),
            )
        )
        all_objects = [o.absolute() for o in all_objects]
        # Sort files and folders from deep to shallow
        sorted_objects = sorted(all_objects, key=lambda x: len(x.parts), reverse=True)
        for obj in sorted_objects:
            fname = obj.name
            fpath = obj.parent
            new_fname = fname.replace(orig_id, new_id)
            new_obj = fpath.joinpath(new_fname)
            obj.rename(new_obj)

    # Replace subject IDs inside JSON and TSV files
    all_files = list(
        filter(
            lambda path: not any((part for part in path.parts if part.startswith("."))),
            Path(dset_dir).rglob("*"),
        )
    )
    exts = [".tsv", ".json"]
    all_files = [f.absolute() for f in all_files if f.suffix in exts]
    for file_ in all_files:
        data = file_.read_text()
        os.remove(file_)

        for orig_id, new_id in ID_MAPPER.items():
            data = data.replace(orig_id, new_id)

        with open(str(file_), "w") as fo:
            fo.write(data)
