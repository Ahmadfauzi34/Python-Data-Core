# 🧠 Panduan Penggunaan dan Perawatan FHRR Cognitive System

Dokumen ini menjelaskan alur kerja (*workflow*) arsitektur kognitif FHRR, cara menggunakannya melalui Antarmuka Pengguna (UI), dan protokol perawatan data agar sistem tetap beroperasi secara optimal dan aman.

---

## 1. Alur Penggunaan Kognitif (Cognitive Workflow)

Sistem ini dirancang tidak hanya untuk sekadar memproses tanya-jawab statis, tetapi untuk **belajar dan berevolusi**. Berikut adalah 3 pilar operasional utama:

### A. Ingesti Teks Tak Terstruktur (Text Ingestor)
Sistem memiliki modul `TextIngestorBlueprint` yang dapat membaca paragraf teks bebas.
*   **Cara Kerja:** Agen memecah paragraf menjadi kalimat, menggunakan *Finite-State Machine* dan pemicu preposisi (misal: "di", "oleh") untuk mengubah kata-kata menjadi vektor semantik (`bindings`).
*   **Out-Of-Vocabulary (OOTV):** Jika agen menemui kata baru, ia akan menebak kategorinya berdasarkan posisi struktural kata tersebut (misal: kata setelah "di" dianggap sebagai *tempat*) dan otomatis mendaftarkannya ke dalam kamus memori.
*   **Penyimpanan:** Hasilnya disimpan sebagai memori jangka pendek (*Episodic Buffer*) dan memori jangka panjang (*Knowledge Graph*).

### B. Simulasi Masa Depan (Simulation Space / Mental Sandbox)
Sebelum mengambil keputusan, agen dapat "membayangkan" dampak dari tindakannya.
*   **Cara Kerja:** Melalui `SimulationSpace`, agen melakukan *Epistemic Fork* dengan menumpuk vektor aksi di atas vektor keadaan saat ini secara matematis (`c_state + c_action`).
*   **Evaluasi:** Skenario dievaluasi berdasarkan dua hal:
    1.  **Epistemic Reward:** Seberapa dekat hasil simulasi dengan *Goal* (Tujuan).
    2.  **Topological Coherence:** Seberapa konsisten hasil tersebut terhadap hukum logika global (*Sheaf Constraints*).
*   **UI:** Fitur ini dapat diakses di sidebar aplikasi melalui bagian **Otonomi Kognitif -> 👁️ Simulasi Sandbox**. Anda dapat memasukkan *Goal* dan opsi aksi secara dinamis.

### C. Konsolidasi Memori (Sleep Phase / Meta-Learning)
Agen dirancang untuk merefleksikan kejadian masa lalu untuk menemukan hukum logika baru tanpa diajari manusia.
*   **Cara Kerja:** `MetaCognitiveConsolidator` membaca riwayat kejadian di *Episodic Buffer*. Jika ia menemukan dua kejadian yang terjadi berurutan (Temporal Causation), ia akan menghitung selisih vektornya (Phase Difference). Jika selisih ini konsisten di berbagai memori, agen akan menginduksi aturan logis baru (misal: "Hujan" selalu diikuti "Basah").
*   **UI:** Fitur ini dapat diakses di sidebar melalui **Otonomi Kognitif -> 💤 Fase Tidur (Konsolidasi)**.

---

## 2. Perawatan Data (Data Maintenance)

Data adalah nyawa dari sistem FHRR. Dataset disimpan secara modular di folder `fhrr_project/data/datasets/default/`.

### A. Kurasi Manusia vs Aturan Otomatis
*   **File Utama (Kurasi Manusia):** File seperti `vocab.yaml`, `observations.yaml`, dan `reasoning_patterns.yaml` adalah *sumber kebenaran absolut*. Anda (sebagai *Engineer*) dapat menambah kosa kata atau kejadian baru langsung ke file ini.
*   **File Otomatis (Auto-Induced):** Saat agen melakukan *Sleep Phase* (Konsolidasi), ia tidak akan menyentuh file utama. Agen akan menyimpan aturan yang ia pelajari sendiri ke dalam file `reasoning_patterns.auto.yaml`.

### B. Protokol Rollback (Pemulihan Kesalahan)
Jika agen mengalami "halusinasi" logika dan mempelajari aturan yang salah selama Fase Tidur:
1.  Buka folder `fhrr_project/data/datasets/default/`.
2.  Buka file `reasoning_patterns.auto.yaml`.
3.  Anda dapat menghapus blok aturan (YAML block) yang spesifik (misalnya aturan yang tidak masuk akal).
4.  Atau, Anda dapat **menghapus seluruh file** `reasoning_patterns.auto.yaml` untuk mereset seluruh hasil belajar otonom agen kembali ke titik 0.
5.  Tekan tombol **"🔄 Re-init"** di sidebar UI untuk memuat ulang dataset.

### C. Mencegah Kebocoran State
Sistem telah dilengkapi perlindungan *stale state*. Jika Anda memiliki beberapa dataset berbeda dan berpindah dataset melalui dropdown di UI, Anda harus menekan tombol **"🔄 Re-init"** agar memori *cache* agen (termasuk rancangan *pending rules* dari Fase Tidur) dibersihkan dengan sempurna.

---

## 3. Pengembangan Lanjutan (Tuning)

Bagi pengembang (Developer) yang ingin menyesuaikan perilaku mesin:
*   **Parameter Toleransi (Similarity Threshold):** Jika agen terlalu sulit menemukan pola baru saat Konsolidasi, Anda dapat menurunkan batas toleransi `self.similarity_threshold` di dalam `fhrr_project/memory/consolidation.py` (saat ini 0.35). Semakin rendah, agen semakin "kreatif" (tapi berisiko halusinasi).
*   **Stopwords Ingestor:** Jika agen salah menangkap predikat (misal: menangkap kata hubung baru), Anda harus menambahkan kata tersebut ke dalam *set* `stopwords` di metode `_extract_entities_and_roles` dalam file `fhrr_project/agents/text_ingestor.py`.
*   **Kecepatan Mesin:** Optimasi besar-besaran dengan Aljabar Linier (Vectorization) telah diterapkan di file `topology.py` dan `discoverer.py`. Harap **hindari penggunaan Python `for` loops** yang bersarang (nested) untuk operasi yang melibatkan perbandingan seluruh token. Selalu gunakan komputasi matriks NumPy.
