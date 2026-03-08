# =============================================================================
# ADIM 0: Gerekli Kütüphaneler
# =============================================================================
# NumPy: Sayısal hesaplamalar, array/matris işlemleri için.
# Pandas: Tablo verisi okuma, işleme, analiz için.
# LabelEncoder: Metin/kategorik sınıfları sayıya çevirir (0, 1, 2...).
# StandardScaler: Özellikleri ölçekler (ortalama≈0, standart sapma≈1).
# train_test_split: Veriyi eğitim/test olarak ayırır.
# One-hot encoding: np.eye ile (TensorFlow to_categorical yerine, Python 3.14 uyumluluğu için).
# Matplotlib: Grafik çizmek için (eğitim/doğrulama başarım ve kayıp grafikleri).
# Keras (TensorFlow): Sinir ağı modeli oluşturmak, derlemek ve eğitmek için.
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import kagglehub
from kagglehub import KaggleDatasetAdapter

# NOT: Model eğitimi ve grafikler için TensorFlow kurulu olmalı (Python 3.12 önerilir).

# =============================================================================
# ADIM 1: Veri Setini Yükle
# =============================================================================
# Kaggle'dan Pima Indians Diabetes veri seti indirilip DataFrame'e çevriliyor.
# Sütunlar: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
#           DiabetesPedigreeFunction, Age, Outcome (0=diyabet yok, 1=diyabet var)
# =============================================================================
file_path = "diabetes.csv"
veri = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "uciml/pima-indians-diabetes-database",
    file_path,
)

print("=== Ham Veri (ilk 5 satır) ===")
print(veri.head())

# =============================================================================
# ADIM 2: Sınıfları Etiketle (Label Encoding)
# =============================================================================
# Outcome sütunu zaten 0 ve 1 içeriyor ama LabelEncoder kullanırsak:
# - Tüm benzersiz değerleri otomatik tespit eder (0, 1)
# - classes listesiyle orijinal sınıf adlarını saklarız
# - İleride farklı veri setlerinde "Var/Yok" gibi metin etiketleri de sayıya döner
# =============================================================================
label_encoder = LabelEncoder().fit(veri.Outcome)
labels = label_encoder.transform(veri.Outcome)
classes = list(label_encoder.classes_)

print("\n=== Sınıflar (classes) ===")
print(classes)  # [0, 1] → Diyabet yok / Diyabet var

# =============================================================================
# ADIM 3: Girdi (X) ve Çıktı (y) Hazırla + Standardizasyon
# =============================================================================
# X: Outcome sütunu hariç tüm özellikler (modelin girdi verisi)
# y: Etiketlenmiş hedef değişken (modelin tahmin etmesi gereken)
#
# StandardScaler: Her özelliği "Z-score" ile ölçekler.
#   - Ortalama ≈ 0, standart sapma ≈ 1 yapar
#   - Nedeni: Glikoz 50–200, Yaş 20–80 gibi farklı ölçekler modeli bozar.
#     Hepsi aynı skaleye gelince (SVM, k-NN, sinir ağları) daha iyi öğrenir.
# fit_transform: Önce veriden ort. ve std öğrenir, sonra X'i dönüştürür.
# =============================================================================
X = veri.drop(["Outcome"], axis=1)
y = labels

sc = StandardScaler()
X = sc.fit_transform(X)

print("\n=== Standartlaştırılmış X (ilk 3 satır) ===")
print(X[:3])

# =============================================================================
# ADIM 4: Eğitim ve Test Verisi Ayır + One-Hot Encoding
# =============================================================================
# train_test_split: Veriyi %80 eğitim, %20 test olarak rastgele böler.
#   - Model sadece X_train, y_train ile eğitilir
#   - X_test, y_test ile hiç görmediği veri üzerinde değerlendirilir
#   - test_size=0.2 → %20 test demek
#
# One-hot encoding (np.eye): Sayısal etiketleri one-hot vektörlere çevirir.
#   - 0 → [1, 0]   (diyabet yok)
#   - 1 → [0, 1]   (diyabet var)
#   Sinir ağları genelde softmax çıkışıyla her sınıfın olasılığını verir;
#   bu yüzden hedef de one-hot olmalı.
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# np.eye(2)[y]: her etiket için [1,0] veya [0,1] üretir (to_categorical ile aynı)
y_train = np.eye(2)[y_train]
y_test = np.eye(2)[y_test]

print("\n=== Eğitim seti boyutu ===")
print("X_train:", X_train.shape, "| y_train:", y_train.shape)
print("X_test:", X_test.shape, "| y_test:", y_test.shape)

# =============================================================================
# ADIM 5: Sinir Ağı Modelinin Oluşturulması
# =============================================================================
# Sequential: Katmanları sırayla üst üste koyan model yapısı.
# Dense: Tam bağlı katman; her nöron bir önceki katmandaki tüm nöronlara bağlıdır.
# - İlk katman: 64 nöron, relu aktivasyonu, input_shape=(8,) çünkü 8 özellik var.
# - İkinci katman: 32 nöron, relu (gizli katman).
# - Çıkış katmanı: 2 nöron (2 sınıf), softmax → her sınıf için olasılık verir (toplamı 1).
# =============================================================================
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(2, activation="softmax"),
])

# =============================================================================
# ADIM 6: Modelin Derlenmesi
# =============================================================================
# compile(): Hangi kayıp fonksiyonu, optimizasyon ve metrik kullanılacağını belirler.
# - loss="categorical_crossentropy": Çok sınıflı (one-hot) sınıflandırma için uygun kayıp.
# - optimizer="adam": Adaptif öğrenme oranlı, pratikte çok kullanılan optimizasyon.
# - metrics=["accuracy"]: Eğitim sırasında doğruluk (başarım) takip edilir.
# =============================================================================
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# =============================================================================
# ADIM 7: Modelin Eğitilmesi
# =============================================================================
# fit(): Modeli eğitim verisi üzerinde belirtilen epok sayısı kadar eğitir.
# - X_train, y_train: Eğitim girdileri ve etiketleri.
# - validation_data=(X_test, y_test): Her epok sonunda test verisi üzerinde de değerlendirilir.
#   Böylece ezberleme (overfitting) olup olmadığı grafikten takip edilir.
# - epochs=150: Tüm eğitim verisi 150 kez baştan sona kullanılır.
# Dönen history: Her epoktaki loss, accuracy, val_loss, val_accuracy değerleri saklanır;
# grafiklerde model.history.history ile kullanılır.
# =============================================================================
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=150,
)

# =============================================================================
# ADIM 8: Eğitim ve Doğrulama Başarımlarının Grafikte Gösterilmesi
# =============================================================================
# model.history.history: fit() sırasında kaydedilen metrikler (accuracy, val_accuracy, loss, val_loss).
# plt.plot(): Epok sayısına göre başarım eğrilerini çizer.
# Eğitim (training) ve test/doğrulama (validation) eğrileri birlikte görülür;
# ikisi birbirine çok yakınsa model genelleme yapıyor, ayrılıyorsa overfitting olabilir.
# =============================================================================
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımları")
plt.ylabel("Başarım")
plt.xlabel("Epok sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()

# =============================================================================
# ADIM 9: Eğitim ve Doğrulama Kayıplarının Grafikte Gösterilmesi
# =============================================================================
# Kayıp (loss): Model tahmininin gerçek etiketten ne kadar saptığını ölçer.
# Eğitim kaybı düşerken doğrulama kaybı artıyorsa model ezberlemeye başlamış demektir.
# =============================================================================
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kayıpları")
plt.ylabel("Kayıp")
plt.xlabel("Epok sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()

