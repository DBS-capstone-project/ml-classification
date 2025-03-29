import pandas as pd
import random

column_mapping = {
    "Bagaimana perasaanmu hari ini?": "Mood",
    "Seberapa puaskah kamu dengan harimu hari ini?": "Kepuasan_Hari",
    "Apa aktivitas favoritmu hari ini?": "Aktivitas_Favorit",
    "Apa yang ingin kamu lakukan besok?": "Rencana_Besok",
    "Bagaimana perasaanmu tentang sekolah/mata pelajaranmu hari ini?": "Perasaan_Sekolah",
    "Bagaimana perasaanmu tentang pekerjaan/kariermu hari ini?": "Perasaan_Karier",
    "Seberapa sering kamu berbicara dengan teman atau keluarga hari ini?": "Interaksi_Teman_Keluarga",
    "Seberapa sering kamu berinteraksi dengan orang lain hari ini?": "Interaksi_Orang_Lain",
    "Seberapa sering kamu bermain dengan teman hari ini?": "Frekuensi_Bermain",
    "Apakah kamu merasa senang di sekolah hari ini?": "Senang_Sekolah",
    "Apakah kamu merasa stres akhir-akhir ini?": "Stres_Tingkat",
    "Apakah kamu merasa kesulitan belajar pelajaran tertentu hari ini?": "Kesulitan_Belajar",
    "Apakah kamu bertengkar dengan teman/saudaramu hari ini?": "Tingkat_Bertengkar",
    "Seberapa sering kamu membantu orang lain hari ini?": "Frekuensi_Membantu",
    "Apakah kamu merasa didukung oleh orang-orang di sekitarmu?": "Dukungan_Sosial",
    "Seberapa yakin kamu dengan tujuan hidupmu saat ini?": "Keyakinan_Tujuan_Hidup",
    "Apa yang bisa kamu lakukan lebih baik besok?": "Peningkatan_Besok",
}

categories = {
    "Anak-Anak": {
        "Mood": ["Senang", "Biasa saja", "Sedikit sedih", "Sangat sedih"],
        "Senang_Sekolah": ["Sangat senang", "Cukup senang", "Tidak senang"],
        "Frekuensi_Bermain": ["Banyak sekali", "Hanya sebentar", "Tidak sama sekali"],
        "Kesulitan_Belajar": ["Tidak ada kesulitan", "Sedikit kesulitan", "Sangat sulit"],
        "Tingkat_Bertengkar": ["Tidak sama sekali", "Sedikit bertengkar", "Bertengkar"],
        "Frekuensi_Membantu": ["Selalu", "Kadang-kadang", "Tidak pernah"],
        "Aktivitas_Favorit": ["Bermain", "Belajar", "Bermain aktif", "Istirahat"],
        "Rencana_Besok": ["Bermain", "Belajar", "Bermain aktif", "Istirahat"],
        "Kepuasan_Hari": ["Sangat puas", "Cukup puas", "Biasa saja", "Kurang puas", "Tidak puas sama sekali"],
        "Peningkatan_Besok": ["Lebih fokus", "Lebih banyak bermain", "Lebih baik kepada orang lain", "Lebih banyak istirahat"],
    },
    "Remaja": {
        "Mood": ["Senang", "Biasa saja", "Sedikit sedih", "Sangat sedih"],
        "Kepuasan_Hari": ["Sangat puas", "Cukup puas", "Biasa saja", "Kurang puas", "Tidak puas sama sekali"],
        "Perasaan_Sekolah": ["Sangat tertarik", "Cukup tertarik", "Tidak tertarik"],
        "Stres_Tingkat": ["Jarang sekali", "Kadang-kadang", "Sering"],
        "Aktivitas_Favorit": ["Mendengarkan musik", "Media sosial", "Main game", "Olahraga"],
        "Interaksi_Teman_Keluarga": ["Banyak sekali", "Hanya sebentar", "Tidak sama sekali"],
        "Dukungan_Sosial": ["Selalu", "Kadang-kadang", "Tidak pernah"],
        "Rencana_Besok": ["Mendengarkan musik", "Media sosial", "Main game", "Olahraga"],
        "Keyakinan_Tujuan_Hidup": ["Sangat yakin", "Cukup yakin", "Biasa saja", "Kurang yakin", "Tidak yakin sama sekali"],
        "Peningkatan_Besok": ["Lebih fokus", "Lebih aktif", "Lebih baik kepada orang lain", "Lebih banyak istirahat"],
    },
    "Dewasa": {
        "Mood": ["Senang", "Biasa saja", "Sedikit sedih", "Sangat sedih"],
        "Kepuasan_Hari": ["Sangat puas", "Cukup puas", "Biasa saja", "Kurang puas", "Tidak puas sama sekali"],
        "Perasaan_Karier": ["Sangat puas", "Cukup puas", "Tidak puas"],
        "Stres_Tingkat": ["Jarang sekali", "Kadang-kadang", "Sering"],
        "Aktivitas_Favorit": ["Minum kopi/teh", "Membaca buku", "Mendengarkan musik", "Olahraga"],
        "Interaksi_Orang_Lain": ["Banyak sekali", "Hanya sebentar", "Tidak sama sekali"],
        "Dukungan_Sosial": ["Selalu", "Kadang-kadang", "Tidak pernah"],
        "Rencana_Besok": ["Minum kopi/teh", "Membaca buku", "Mendengarkan musik", "Olahraga"],
        "Keyakinan_Tujuan_Hidup": ["Sangat yakin", "Cukup yakin", "Biasa saja", "Kurang yakin", "Tidak yakin sama sekali"],
        "Peningkatan_Besok": ["Lebih fokus", "Lebih aktif", "Lebih baik kepada orang lain", "Lebih banyak istirahat"],
    }
}

num_samples_per_group = 5000
data = []

for category, questions in categories.items():
    for _ in range(num_samples_per_group):
        entry = {"Kategori": category}
        for question, answers in questions.items():
            entry[question] = random.choice(answers)
        data.append(entry)

df = pd.DataFrame(data)

category_files = {}
base_path = "./"

for category in categories.keys():
    df_category = df[df["Kategori"] == category]
    filename = f"{base_path}synthetic_survey_{category.lower().replace('-', '_')}.csv"
    df_category.to_csv(filename, index=False)
    category_files[category] = filename

category_files
