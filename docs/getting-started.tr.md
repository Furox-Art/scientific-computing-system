# Başlangıç (Türkçe)

> Bu, [İngilizce Getting Started](getting-started.md) belgesinin Türkçe çevirisidir. Çeviri güncel olmayabilir — İngilizce versiyon kanonik kaynaktır.

## Kurulum

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Hızlı Başlangıç

### Komut Satırı Arayüzü (CLI)

```bash
# Mevcut komutları listele
cds --help

# Fiziksel sabitleri listele
cds constants

# Etkileşimli fizik hesaplayıcı
cds calc ke       # kinetik enerji
cds calc gravity
cds calc wave
cds calc gas

# Hipotez üret
cds hypothesize "Hubble gerilimine ne sebep olur?" --domain cosmology
```

### Python API

```python
# Kuantum simülasyonu
from cds.quantum import bell_state, is_entangled, ghz_state

reg = bell_state(0)
print(is_entangled(reg))  # True

# İstatistik ve hipotez testi
from cds.stats import chi_square_independence, one_way_anova

sonuc = chi_square_independence([[10, 20], [30, 40]])
print(sonuc.p_value)

# Monte Carlo
from cds.montecarlo import estimate_pi

pi_tahmin = estimate_pi(samples=100_000, seed=42)

# Sinyal işleme
from cds.signals import fft, ifft

frekans = fft([1.0, 0.0, -1.0, 0.0])
```

## Modüller

| Modül | Ne İşe Yarar |
|---|---|
| `cds.quantum` | Tek ve çok kübitli kuantum simülasyonu, dolanıklık testleri |
| `cds.stats` | Hipotez testleri, ANOVA, regresyon, tanımlayıcı istatistik |
| `cds.montecarlo` | Pi tahmini, Buffon iğnesi, paralel Monte Carlo |
| `cds.signals` | FFT, konvolüsyon, filtreleme, güç spektrumu |
| `cds.optimization` | Gradyan inişi, Newton, BFGS, line search |
| `cds.probability` | Dağılımlar, örnekleme, CDF/PDF |
| `cds.ml` | Algılayıcı, lineer regresyon, backprop (pure Python) |
| `cds.math_utils` | Lineer cebir, matris işlemleri, LU/QR/Cholesky |
| `cds.diffeq` | RK4, Euler, sistem ODE'leri |
| `cds.numerical_integration` | Simpson, Romberg, Gaussian quadrature |
| `cds.data_analysis` | Moving average, korelasyon, hipotez değerlendirme |
| `cds.scientific` | Fizik formülleri, sabitler |
| `cds.hypothesis` | Yapılandırılmış hipotez üretimi ve değerlendirme |
| `cds.graph` | Graf algoritmaları (BFS, DFS, Dijkstra) |

## Sonraki Adımlar

- [Tutorial: Quantum](tutorials/quantum_demo.md) — Bell durumları, GHZ, ölçüm
- [Tutorial: Hipotez Üretimi](tutorials/hypothesis_demo.md) — Araştırma sorularınızı yapılandırın
- [Vaka Çalışması: Hubble](CASE_STUDY_HUBBLE.md) — Gerçek bilim problemi örneği
- [Tutorial: Quick Start](tutorials/quick_start.md) — 5 dakikada temel kullanım

## Sıkça Sorulan Sorular

**S: Neden "cds" olarak import ediliyor, paket adı farklıyken?**
Y: Paket adı `cognitive-discovery-system` (PyPI için uzun açıklayıcı isim), import ismi `cds` (kısa ve kullanışlı). `pip install cognitive-discovery-system` ile kurulur, `import cds` ile kullanılır.

**S: NumPy/SciPy yok mu?**
Y: CDS saf Python. Performans için ödün vermeden eğitim ve araştırma için ideal. NumPy tabanlı iş istasyonları için: ileride `cds[numpy]` gibi opsiyonel bir ekstra planlanıyor (ROADMAP).

**S: Üretim ortamında kullanabilir miyim?**
Y: v1.2.0 stabil sürüm. API kararlı, %100 test coverage (statement + branch), 1441 test, CI multi-OS yeşil. Production/Stable sınıflandırmasıyla yayınlandı.

## Topluluk

- [Issue aç](https://github.com/Furox88/cognitive-discovery-system/issues)
- [Tartışmalar](https://github.com/Furox88/cognitive-discovery-system/discussions)
- İletişim: furkanarkn1451@gmail.com

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Bakınız: [CONTRIBUTING.md](https://github.com/Furox88/cognitive-discovery-system/blob/main/CONTRIBUTING.md). Türkçe konuşan katkıcılar için: issue/PR'ları Türkçe açabilirsiniz, kod yorumları İngilizce tercih edilir.
