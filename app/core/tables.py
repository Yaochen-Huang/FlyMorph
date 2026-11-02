"""Table I/O for the eye task: immutable prefill table + regenerated final table."""
import csv
import os

PREFILL = "prefill.csv"
FINAL = "final.csv"
CORRECTIONS = "corrections.csv"
FIELDS = ["file", "area_px", "conf", "thr", "edited"]


def read_table(path):
    """Return (order, rows) where rows is {file: dict}."""
    order, rows = [], {}
    if not os.path.exists(path):
        return order, rows
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            f = (r.get("file") or "").strip()
            if not f:
                continue
            if f not in rows:
                order.append(f)
            rows[f] = {k: r.get(k, "") for k in FIELDS}
    return order, rows


def write_table(path, order, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for f in order:
            w.writerow({k: rows[f].get(k, "") for k in FIELDS})


def rebuild_final(out_dir):
    """final.csv = prefill.csv with corrections applied (areas/corrections.csv if present)."""
    order, base = read_table(os.path.join(out_dir, PREFILL))
    _, corr = read_table(os.path.join(out_dir, CORRECTIONS))
    for f, c in corr.items():
        if f not in base:
            base[f] = {k: "" for k in FIELDS}
            base[f]["file"] = f
            order.append(f)
        if c.get("area_px"):
            base[f]["area_px"] = c["area_px"]
        base[f]["edited"] = "1"
    write_table(os.path.join(out_dir, FINAL), order, base)
    return order, base
