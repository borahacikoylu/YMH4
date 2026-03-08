import pandas as pd
import numpy as np

# DataFrame(): Pandas'ın tablo verilerini tuttuğu ana yapısıdır.
# Satırlar (index) ve sütunlardan oluşur, Excel'deki bir sheet gibi düşünebilirsin.
# Sözlük (dict) veya listelerden DataFrame oluşturulabilir.

# Sözlük ile tabloyu oluştur (her sütun için liste verilir)
df = pd.DataFrame({
    'Gün': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14'],
    'Hava Durumu': ['Güneşli', 'Güneşli', 'Kapalı', 'Yağmurlu', 'Yağmurlu', 'Yağmurlu', 'Kapalı', 'Güneşli', 'Güneşli', 'Yağmurlu', 'Güneşli', 'Kapalı', 'Kapalı', 'Yağmurlu'],
    'Sıcaklık': ['Sıcak', 'Sıcak', 'Sıcak', 'Ilıman', 'Soğuk', 'Soğuk', 'Soğuk', 'Ilıman', 'Soğuk', 'Ilıman', 'Ilıman', 'Ilıman', 'Sıcak', 'Ilıman'],
    'Nem': ['Yüksek', 'Yüksek', 'Yüksek', 'Yüksek', 'Normal', 'Normal', 'Normal', 'Yüksek', 'Normal', 'Normal', 'Normal', 'Yüksek', 'Normal', 'Yüksek'],
    'Yağış': ['Seyrek', 'Aşırı', 'Seyrek', 'Seyrek', 'Seyrek', 'Aşırı', 'Aşırı', 'Seyrek', 'Seyrek', 'Seyrek', 'Aşırı', 'Aşırı', 'Seyrek', 'Aşırı'],
    'Oyun': ['Yok', 'Yok', 'Var', 'Var', 'Var', 'Yok', 'Var', 'Var', 'Yok', 'Var', 'Var', 'Yok', 'Var', 'Yok']
})

print("=== Tablo ===")
print(df)

# Betimleyici (açıklayıcı) istatistikler
# describe(): sayısal sütunlar için ortalama, std, min, max vb. verir
# include='all': kategorik sütunlar için de (count, unique, top, freq) gösterir
print("\n=== Betimleyici İstatistikler ===")
print(df.describe(include='all'))

# info(): sütun tipleri, null değer sayısı
print("\n=== Tablo Bilgisi (sütun tipleri, null sayısı) ===")
print(df.info())

# -----------------------------------------------------------------------
# (3,4) boyutunda dizi oluştur ve (6,2) boyutuna dönüştür
# NumPy: sayısal dizilerle çalışmak için kullanılır (Pandas'tan farklı ama birlikte kullanılır)
# -----------------------------------------------------------------------

# 1) (3,4) boyutunda dizi: 3 satır x 4 sütun = 12 eleman
# np.arange(12): 0'dan 11'e kadar sayılar üretir, reshape ile (3,4) matrise çeviririz
dizi = np.arange(12).reshape(3, 4)
print("\n=== (3,4) Boyutundaki Dizi ===")
print(dizi)
print("Boyut:", dizi.shape)  # (3, 4)

# 2) Boyutu (6,2) olacak şekilde değiştir
# reshape(): eleman sayısı aynı kalmalı! 3*4=12, 6*2=12 ✓
dizi_degisti = dizi.reshape(6, 2)
print("\n=== (6,2) Boyutuna Dönüştürülmüş Dizi ===")
print(dizi_degisti)
print("Boyut:", dizi_degisti.shape)  # (6, 2)