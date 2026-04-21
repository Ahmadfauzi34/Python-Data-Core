"""
Backward-compat shim.

Dataset asli sekarang ada di YAML: data/datasets/default.yaml.
Modul ini hanya re-export `fhrr_research_dataset` agar kode lama tetap jalan.

Untuk dataset baru, **jangan tambah dict di sini** — buat file
`data/datasets/<nama>.yaml` lalu panggil:

    from fhrr_project.data.loader import load_dataset
    ds = load_dataset("<nama>")
"""
from fhrr_project.data.loader import load_dataset

fhrr_research_dataset = load_dataset("default", strict=False)
