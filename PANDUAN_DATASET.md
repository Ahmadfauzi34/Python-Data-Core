# 📚 Aturan Pembuatan Dataset FHRR

Dataset FHRR tidak menggunakan format *teks mentah* seperti LLM standar, melainkan menggunakan representasi **Simbolik Terstruktur** (Structured Symbolic Representation) berformat YAML.

Dataset disimpan di dalam `fhrr_project/data/datasets/default/`. Saat Anda menjalankan `load_dataset("default")`, sistem akan secara otomatis menggabungkan seluruh file YAML di dalam folder tersebut menjadi satu *Knowledge Graph* dan ruang Vektor yang utuh.

Berikut adalah aturan dan contoh pembuatan masing-masing komponen dataset:

---

## 1. `vocab.yaml` (Kosa Kata & Ontologi)
File ini mendefinisikan seluruh kosa kata yang diizinkan untuk digunakan oleh agen, beserta kategori ontologisnya (kata benda, kata kerja, sifat, dsb). Jika sebuah kata tidak ada di sini, agen tidak akan mengenali maknanya secara *default* (meskipun `TextIngestor` bisa menebaknya).

**Aturan:**
*   Setiap kunci (*key*) di dalam `categories` mewakili sebuah *Stalk* dalam Topologi.
*   Item di dalam *list* harus berupa string huruf kecil tanpa spasi (gunakan *underscore* `_` untuk kata majemuk, misal `ruang_tamu`).

**Contoh:**
```yaml
vocab:
  categories:
    entitas:
      - budi
      - siti
      - kucing
    aksi:
      - makan
      - tidur
      - lari
    tempat:
      - sekolah
      - pasar
    keadaan:
      - pintar
      - basah
      - lapar
```

---

## 2. `observations.yaml` (Memori Episodik)
File ini berisi "ingatan" atau kejadian yang pernah dilihat oleh agen. Observasi inilah yang membentuk pemahaman dasar agen tentang dunia (Sebab-Akibat).

**Aturan:**
*   Harus memiliki `id` yang unik (misal: `o1`, `o2`).
*   `type` saat ini didukung adalah `event`.
*   `bindings` berisi pasangan `Role: Token`. Token **harus** sudah didefinisikan di `vocab.yaml`. Role standar yang disarankan: `agen` (subjek), `predikat` (kata kerja), `pasien` (objek), `lokasi`, `instrumen`, `waktu`, `atribut` (sifat).

**Contoh (Mewakili kejadian: "Budi makan apel di sekolah"):**
```yaml
observations:
  - id: o1
    type: event
    bindings:
      agen: budi
      predikat: makan
      pasien: apel
      lokasi: sekolah
```

---

## 3. `reasoning_patterns.yaml` (Aturan Logika & Inferensi)
File ini adalah "Otak Logis" dari agen. Ia berisi aturan Deduksi (Transformasi) atau Kausalitas Temporal.

**Aturan:**
*   `mechanism`: Gunakan `'transform'` untuk logika deduktif murni (misal: "besar adalah lawan kecil"). Gunakan `'temporal_causation'` untuk hukum sebab-akibat (misal: "hujan menyebabkan basah").
*   `premise` dan `conclusion`: Adalah *dictionary* dari pola Role dan Token.
*   `confidence`: Nilai $0.0$ hingga $1.0$.

**Contoh (Aturan Kausalitas: "Membaca menyebabkan pintar"):**
```yaml
reasoning_patterns:
  - id: r1
    name: causal_read_smart
    premise:
      predikat: baca
    conclusion:
      atribut: pintar
    mechanism: temporal_causation
    confidence: 0.85
    explanation: membaca menyebabkan seseorang menjadi pintar
```

> **Catatan Penting:** Anda tidak perlu menulis semua aturan ini secara manual! Agen dapat memikirkan dan menginduksi aturan baru secara mandiri selama "Fase Tidur", dan menyimpannya ke file `reasoning_patterns.auto.yaml`.

---

## 4. `qa_pairs.yaml` (Evaluasi & Tanya Jawab Dasar)
File ini digunakan untuk menguji kemampuan agen dalam menarik informasi langsung dari memori observasi.

**Aturan:**
*   `q_focus`: Daftar Role yang diberikan oleh pertanyaan.
*   `answer_role`: Role target yang dicari oleh pertanyaan.
*   `source`: Merujuk pada `id` di `observations.yaml`.

**Contoh (Pertanyaan: "Di mana budi makan?"):**
```yaml
qa_pairs:
  - id: qa1
    question: di mana budi makan?
    q_focus:
      - agen
      - predikat
    answer_role: lokasi
    source: o1
    reasoning: cari lokasi dari peristiwa makan oleh budi
```

---

## 5. `comprehension_tasks.yaml` (Evaluasi Multi-Hop / Tingkat Lanjut)
File ini menguji apakah agen mampu **menggabungkan observasi dan aturan logika** untuk menjawab sesuatu yang tidak tersurat secara eksplisit.

**Aturan:**
*   `inference_needed`: Set ke `true` jika jawaban membutuhkan logika tambahan.
*   `inference_rule`: Rujuk ke `id` yang ada di `reasoning_patterns.yaml`.

**Contoh (Pertanyaan: "Apakah budi menjadi pintar?"):**
*(Membutuhkan gabungan dari Observasi O2: "Budi baca buku" + Aturan R1: "baca -> pintar")*
```yaml
comprehension_tasks:
  - id: c1
    context: o2
    question: apakah budi menjadi pintar?
    expected_answer: ya
    required_bindings:
      - agen
      - predikat
    inference_needed: true
    inference_rule: r1
```
