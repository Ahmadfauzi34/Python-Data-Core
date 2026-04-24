import yaml
import sys

def main():
    new_qa_str = """
qa_pairs:
- id: qa1
  question: siapa yang makan mangga?
  q_focus:
  - predikat
  - pasien
  answer_role: agen
  source: o1
  reasoning: cari agen dari peristiwa makan mangga
- id: qa2
  question: apa yang dimakan anak?
  q_focus:
  - agen
  - predikat
  answer_role: pasien
  source: o1
  reasoning: cari pasien dari aksi makan oleh anak
- id: qa3
  question: di mana anak makan?
  q_focus:
  - agen
  - predikat
  answer_role: lokasi
  source: o1
  reasoning: cari lokasi dari peristiwa makan
- id: qa4
  question: kapan anak makan mangga?
  q_focus:
  - agen
  - predikat
  - pasien
  answer_role: waktu
  source: o1
  reasoning: cari waktu dari peristiwa
- id: qa5
  question: apa yang ditulis guru?
  q_focus:
  - agen
  - predikat
  - instrumen
  answer_role: pasien
  source: o2
  reasoning: cari pasien dari aksi tulis oleh guru dengan kapur
- id: qa6
  question: apa yang digunakan ayah untuk memotong?
  q_focus:
  - agen
  - predikat
  answer_role: instrumen
  source: o7
  reasoning: cari instrumen dari aksi potong
- id: qa7
  question: apakah anak kenyang setelah makan?
  q_focus:
  - agen
  - predikat
  answer_role: atribut
  source: o1
  reasoning: inferensi kekenyangan dari aksi makan
- id: qa8
  question: apa yang dikendarai polisi?
  q_focus:
  - agen
  - predikat
  answer_role: pasien
  source: o13
  reasoning: cari pasien dari aksi kendarai oleh polisi
- id: qa9
  question: di mana dosen membaca?
  q_focus:
  - agen
  - predikat
  answer_role: lokasi
  source: o12
  reasoning: cari lokasi dari aksi baca oleh dosen
- id: qa10
  question: di mana penulis membeli buku?
  q_focus:
  - agen
  - predikat
  - pasien
  answer_role: lokasi
  source: o16
  reasoning: cari lokasi dari aksi beli oleh penulis
- id: qa11
  question: apa yang dibawa penyanyi?
  q_focus:
  - agen
  - predikat
  answer_role: pasien
  source: o18
  reasoning: cari pasien dari aksi bawa oleh penyanyi
- id: qa12
  question: siapa yang lari di pantai?
  q_focus:
  - predikat
  - lokasi
  answer_role: agen
  source: o17
  reasoning: cari agen dari aksi lari di pantai
- id: qa13
  question: apa yang turun di luar?
  q_focus:
  - lokasi
  answer_role: pasien
  source: o15
  reasoning: cari pasien dari peristiwa turun di luar
- id: qa14
  question: siapa yang tidur dengan tenang?
  q_focus:
  - predikat
  - atribut
  answer_role: agen
  source: o6
  reasoning: cari agen dari aksi tidur dengan atribut tenang
- id: qa15
  question: bagaimana perasaan kucing saat tidur?
  q_focus:
  - agen
  - predikat
  answer_role: atribut
  source: o6
  reasoning: cari atribut dari aksi tidur oleh kucing
- id: qa16
  question: siapa yang lari dengan cepat?
  q_focus:
  - predikat
  - atribut
  answer_role: agen
  source: o17
  reasoning: cari agen dari aksi lari dengan atribut cepat
- id: qa17
  question: di mana gajah berjalan?
  q_focus:
  - agen
  - predikat
  answer_role: lokasi
  source: o21
  reasoning: cari lokasi dari aksi jalan oleh gajah
"""

    new_comp_str = """
comprehension_tasks:
- id: c1
  context: o1
  question: apakah anak makan mangga di kebun?
  expected_answer: ya
  required_bindings:
  - agen
  - predikat
  - pasien
  - lokasi
  inference_needed: false
- id: c2
  context: o1
  question: apakah anak kenyang setelah makan?
  expected_answer: ya
  required_bindings:
  - agen
  - predikat
  inference_needed: true
  inference_rule: r1
- id: c3
  context: o4
  question: apakah burung berada di atas?
  expected_answer: ya
  required_bindings:
  - agen
  - predikat
  inference_needed: true
  inference_rule: r3
- id: c4
  context: o1
  question: apakah anak makan pisang?
  expected_answer: tidak
  required_bindings:
  - agen
  - predikat
  - pasien
  inference_needed: false
  contradiction_check: true
- id: c5
  context:
  - o1
  - o7
  question: apa persamaan antara anak dan ayah?
  expected_answer: keduanya adalah agen
  required_bindings:
  - agen
  inference_needed: true
  integration: compare_roles
- id: c6
  context: o12
  question: apakah dosen menjadi pintar?
  expected_answer: ya
  required_bindings:
  - agen
  - predikat
  inference_needed: true
  inference_rule: r7
- id: c7
  context: o12
  question: apakah dosen berada di sekolah?
  expected_answer: ya
  required_bindings:
  - agen
  - lokasi
  inference_needed: true
  inference_rule: r9
- id: c8
  context: o20
  question: apakah pelukis menjadi kaya?
  expected_answer: ya
  required_bindings:
  - agen
  - predikat
  inference_needed: true
  inference_rule: r10
- id: c9
  context: o18
  question: apakah penyanyi membawa gitar?
  expected_answer: ya
  required_bindings:
  - agen
  - predikat
  - pasien
  inference_needed: false
- id: c10
  context: o19
  question: apakah sapi makan rumput?
  expected_answer: ya
  required_bindings:
  - agen
  - predikat
  - pasien
  inference_needed: true
  inference_rule: r12
- id: c11
  context: o17
  question: apakah kuda berlari dengan cepat?
  expected_answer: ya
  required_bindings:
  - agen
  - predikat
  inference_needed: true
  inference_rule: r13
- id: c12
  context: o6
  question: apakah kucing tenang saat tidur?
  expected_answer: ya
  required_bindings:
  - agen
  - predikat
  inference_needed: true
  inference_rule: r14
- id: c13
  context: o8
  question: apakah dapur bersih saat ibu memasak?
  expected_answer: ya
  required_bindings:
  - agen
  - predikat
  - lokasi
  inference_needed: false
- id: c14
  context: o15
  question: apakah hujan turun di luar?
  expected_answer: ya
  required_bindings:
  - pasien
  - predikat
  - lokasi
  inference_needed: false
- id: c15
  context: o21
  question: apakah gajah besar?
  expected_answer: ya
  required_bindings:
  - agen
  - atribut
  inference_needed: false
"""

    qa_data = yaml.safe_load(new_qa_str)
    comp_data = yaml.safe_load(new_comp_str)

    with open('fhrr_project/data/datasets/default/qa_pairs.yaml', 'w') as f:
        yaml.dump(qa_data, f, sort_keys=False, allow_unicode=True)

    with open('fhrr_project/data/datasets/default/comprehension_tasks.yaml', 'w') as f:
        yaml.dump(comp_data, f, sort_keys=False, allow_unicode=True)

if __name__ == '__main__':
    main()
