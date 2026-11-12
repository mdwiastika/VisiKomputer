# Supervision LineZone: Implementasi Siap Pakai

**Durasi Baca**: 12 Menit | **Level**: Pemula-Menengah

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan mampu:
- Menggunakan `sv.LineZone` untuk counting yang robust
- Mengkonfigurasi triggering anchors yang tepat
- Menangani multiple lines untuk multi-lane counting
- Melakukan filtering berdasarkan class
- Debugging dan troubleshooting LineZone

---

## 🎬 Pendahuluan

Library **Supervision** menyediakan `LineZone` yang sudah menghandle semua matematika kompleks dari artikel sebelumnya. Dengan beberapa baris kode, Anda bisa mendapatkan:
- Line crossing detection
- Direction discrimination (IN/OUT)
- Double counting prevention
- Beautiful visualization

---

## 1. Instalasi

```bash
pip install supervision ultralytics opencv-python
```

---

## 2. Konsep LineZone

### 2.1 Apa Itu LineZone?

`LineZone` adalah kelas yang:
- Menyimpan koordinat garis counting
- Melacak objek yang melewatinya
- Membedakan arah pergerakan (IN/OUT)
- Otomatis menghindari double counting

### 2.2 Cara Kerja Internal

```
┌──────────────────────────────────────────────────────────────┐
│                      LineZone Internals                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   1. Simpan posisi sebelumnya per tracker_id                │
│   2. Setiap frame, bandingkan posisi baru dengan lama       │
│   3. Tentukan apakah crossing terjadi (cross product)        │
│   4. Tentukan arah crossing                                  │
│   5. Increment counter yang sesuai                          │
│   6. Mark ID sebagai "sudah dihitung"                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Kode Dasar

### 3.1 Minimal Example

```python
import cv2
from ultralytics import YOLO
import supervision as sv

# Setup model dan tracker
model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()

# Definisikan garis counting
LINE_START = sv.Point(0, 500)
LINE_END = sv.Point(1920, 500)

line_zone = sv.LineZone(
    start=LINE_START,
    end=LINE_END,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

# Annotators untuk visualisasi
box_annotator = sv.BoxAnnotator()
line_annotator = sv.LineZoneAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

# Video loop
cap = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect + Track
    results = model(frame, conf=0.1, classes=[2, 3, 5, 7], verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    
    # Update LineZone (ini yang menghitung crossing)
    crossed_in, crossed_out = line_zone.trigger(detections=detections)
    
    # Annotate
    frame = box_annotator.annotate(frame, detections)
    frame = line_annotator.annotate(frame, line_counter=line_zone)
    
    # Print counts
    print(f"IN: {line_zone.in_count}, OUT: {line_zone.out_count}")
    
    cv2.imshow("Counting", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 3.2 Return Values dari trigger()

```python
crossed_in, crossed_out = line_zone.trigger(detections)

# crossed_in: boolean array, True untuk objek yang crossing IN di frame ini
# crossed_out: boolean array, True untuk objek yang crossing OUT di frame ini

# Contoh penggunaan
for i, tracker_id in enumerate(detections.tracker_id):
    if crossed_in[i]:
        print(f"Vehicle {tracker_id} entered!")
    if crossed_out[i]:
        print(f"Vehicle {tracker_id} exited!")
```

---

## 4. Triggering Anchors

### 4.1 Pilihan Anchor

Anchor menentukan titik mana dari bounding box yang digunakan untuk crossing detection:

```python
from supervision import Position

# Opsi yang tersedia:
Position.CENTER           # Tengah box
Position.BOTTOM_CENTER    # Bawah tengah (RECOMMENDED untuk traffic)
Position.TOP_CENTER       # Atas tengah
Position.BOTTOM_LEFT      # Pojok kiri bawah
Position.BOTTOM_RIGHT     # Pojok kanan bawah
Position.TOP_LEFT         # Pojok kiri atas
Position.TOP_RIGHT        # Pojok kanan atas
Position.CENTER_LEFT      # Tengah kiri
Position.CENTER_RIGHT     # Tengah kanan
```

### 4.2 Rekomendasi untuk Traffic

```python
# RECOMMENDED: Bottom center
# Karena merepresentasikan posisi "roda menyentuh jalan"
line_zone = sv.LineZone(
    start=sv.Point(0, 500),
    end=sv.Point(1920, 500),
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)
```

### 4.3 Multiple Anchors

Bisa menggunakan beberapa anchor sekaligus:

```python
line_zone = sv.LineZone(
    start=sv.Point(0, 500),
    end=sv.Point(1920, 500),
    triggering_anchors=(
        sv.Position.BOTTOM_LEFT,
        sv.Position.BOTTOM_RIGHT
    )
)
```

**Behavior**: Objek dianggap crossing jika **SALAH SATU** anchor melewati garis.

### 4.4 Perbandingan Anchor

| Anchor | Use Case | Pros | Cons |
|--------|----------|------|------|
| `BOTTOM_CENTER` | General traffic | Akurat untuk ground level | - |
| `CENTER` | Drone/aerial view | Representasi visual baik | Tidak akurat untuk ground |
| `BOTTOM_LEFT + BOTTOM_RIGHT` | Wide vehicles | Catch partial crossing | Might count twice |

---

## 5. Multiple Lines

### 5.1 Multi-Lane Road

```python
# Lane 1 (Leftmost)
line_zone_1 = sv.LineZone(
    start=sv.Point(0, 400),
    end=sv.Point(640, 400),
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

# Lane 2 (Middle)
line_zone_2 = sv.LineZone(
    start=sv.Point(640, 400),
    end=sv.Point(1280, 400),
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

# Lane 3 (Rightmost)
line_zone_3 = sv.LineZone(
    start=sv.Point(1280, 400),
    end=sv.Point(1920, 400),
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

# Dalam loop:
line_zone_1.trigger(detections)
line_zone_2.trigger(detections)
line_zone_3.trigger(detections)

total_in = line_zone_1.in_count + line_zone_2.in_count + line_zone_3.in_count
total_out = line_zone_1.out_count + line_zone_2.out_count + line_zone_3.out_count

print(f"Total IN: {total_in}, Total OUT: {total_out}")
```

### 5.2 Bidirectional Road with Entry/Exit

```python
# Entry line (top of frame)
entry_line = sv.LineZone(
    start=sv.Point(0, 200),
    end=sv.Point(1920, 200)
)

# Exit line (bottom of frame)
exit_line = sv.LineZone(
    start=sv.Point(0, 800),
    end=sv.Point(1920, 800)
)

# Track both
entry_line.trigger(detections)
exit_line.trigger(detections)

vehicles_entered = entry_line.in_count
vehicles_exited = exit_line.out_count
vehicles_in_zone = vehicles_entered - vehicles_exited
```

### 5.3 Helper Class for Multiple Lines

```python
class MultiLineCounter:
    """Manage multiple counting lines"""
    
    def __init__(self, lines_config: list):
        """
        Args:
            lines_config: List of dicts with 'name', 'start', 'end' keys
        """
        self.lines = {}
        self.annotators = {}
        
        for config in lines_config:
            name = config['name']
            self.lines[name] = sv.LineZone(
                start=sv.Point(*config['start']),
                end=sv.Point(*config['end']),
                triggering_anchors=(sv.Position.BOTTOM_CENTER,)
            )
            self.annotators[name] = sv.LineZoneAnnotator()
    
    def update(self, detections):
        """Update all lines"""
        results = {}
        for name, line in self.lines.items():
            line.trigger(detections)
            results[name] = {
                'in': line.in_count,
                'out': line.out_count
            }
        return results
    
    def annotate(self, frame):
        """Annotate all lines on frame"""
        for name, line in self.lines.items():
            frame = self.annotators[name].annotate(frame, line_counter=line)
        return frame
    
    def get_totals(self):
        """Get total counts across all lines"""
        total_in = sum(line.in_count for line in self.lines.values())
        total_out = sum(line.out_count for line in self.lines.values())
        return {'total_in': total_in, 'total_out': total_out}

# Usage
config = [
    {'name': 'Lane_1', 'start': (0, 400), 'end': (640, 400)},
    {'name': 'Lane_2', 'start': (640, 400), 'end': (1280, 400)},
    {'name': 'Lane_3', 'start': (1280, 400), 'end': (1920, 400)},
]

multi_counter = MultiLineCounter(config)
```

---

## 6. Filter by Class

### 6.1 Hanya Hitung Kendaraan Tertentu

```python
import numpy as np

# Vehicle classes: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

# Dalam loop:
results = model(frame, conf=0.1, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results)
detections = tracker.update_with_detections(detections)

# Filter sebelum trigger
mask = np.isin(detections.class_id, VEHICLE_CLASSES)
vehicle_detections = detections[mask]

# Trigger dengan filtered detections
line_zone.trigger(vehicle_detections)
```

### 6.2 Separate Counts per Class

```python
class ClassBasedCounter:
    """Count each vehicle class separately"""
    
    def __init__(self, line_start, line_end, class_names: dict):
        self.class_names = class_names
        self.lines = {
            class_id: sv.LineZone(
                start=sv.Point(*line_start),
                end=sv.Point(*line_end)
            )
            for class_id in class_names.keys()
        }
    
    def update(self, detections):
        results = {}
        
        for class_id, line in self.lines.items():
            # Filter untuk class ini saja
            mask = detections.class_id == class_id
            class_detections = detections[mask]
            
            line.trigger(class_detections)
            results[self.class_names[class_id]] = {
                'in': line.in_count,
                'out': line.out_count
            }
        
        return results

# Usage
class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
class_counter = ClassBasedCounter(
    line_start=(0, 400),
    line_end=(1920, 400),
    class_names=class_names
)

# Dalam loop
counts = class_counter.update(detections)
# {'car': {'in': 10, 'out': 5}, 'motorcycle': {'in': 20, 'out': 15}, ...}
```

---

## 7. Reset Counter

### 7.1 Periodic Reset

```python
from datetime import datetime

last_reset_hour = -1

while True:
    # ... processing ...
    
    # Reset setiap jam
    current_hour = datetime.now().hour
    if current_hour != last_reset_hour:
        # Log before reset
        hourly_data = {
            'hour': last_reset_hour,
            'in_count': line_zone.in_count,
            'out_count': line_zone.out_count
        }
        save_to_database(hourly_data)
        
        # Reset
        line_zone.in_count = 0
        line_zone.out_count = 0
        last_reset_hour = current_hour
```

### 7.2 Manual Reset API

```python
class CountingSession:
    """Manage counting sessions with reset capability"""
    
    def __init__(self, line_zone):
        self.line_zone = line_zone
        self.session_start = datetime.now()
        self.history = []
    
    def reset(self, save_history=True):
        if save_history:
            self.history.append({
                'start': self.session_start,
                'end': datetime.now(),
                'in_count': self.line_zone.in_count,
                'out_count': self.line_zone.out_count
            })
        
        self.line_zone.in_count = 0
        self.line_zone.out_count = 0
        self.session_start = datetime.now()
    
    def get_total_history(self):
        total_in = sum(h['in_count'] for h in self.history)
        total_out = sum(h['out_count'] for h in self.history)
        return {'total_in': total_in, 'total_out': total_out}
```

---

## 8. Data Output

### 8.1 JSON Lines Output

```python
import json
from datetime import datetime

def log_count(line_zone, camera_id: str, output_file: str = "counts.jsonl"):
    """Append count data to JSONL file"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "camera_id": camera_id,
        "in_count": line_zone.in_count,
        "out_count": line_zone.out_count,
        "net_flow": line_zone.in_count - line_zone.out_count
    }
    
    with open(output_file, "a") as f:
        f.write(json.dumps(data) + "\n")

# Usage: log every minute
import time
last_log_time = time.time()

while True:
    # ... processing ...
    
    if time.time() - last_log_time >= 60:  # Every minute
        log_count(line_zone, camera_id="cam_001")
        last_log_time = time.time()
```

### 8.2 Real-time Event Streaming

```python
def on_crossing_event(tracker_id, direction, class_name):
    """Callback for each crossing event"""
    event = {
        'timestamp': datetime.now().isoformat(),
        'tracker_id': int(tracker_id),
        'direction': direction,
        'class': class_name
    }
    # Send to message queue, webhook, etc.
    print(json.dumps(event))

# Usage dengan crossed_in/crossed_out
crossed_in, crossed_out = line_zone.trigger(detections)

for i, tracker_id in enumerate(detections.tracker_id):
    if tracker_id is None:
        continue
    
    class_name = model.names[detections.class_id[i]]
    
    if crossed_in[i]:
        on_crossing_event(tracker_id, 'IN', class_name)
    if crossed_out[i]:
        on_crossing_event(tracker_id, 'OUT', class_name)
```

---

## 9. Debugging

### 9.1 Garis Tidak Menghitung

**Checklist:**

| Check | How to Verify |
|-------|---------------|
| Tracking berjalan? | Print `detections.tracker_id` — should not be all None |
| Posisi garis benar? | Gambar garis di frame dan verifikasi visual |
| Objek melewati garis? | Cek apakah objek berhenti sebelum garis |
| Anchor point crossing? | Gunakan `BOTTOM_CENTER` dan pastikan box bottom crosses |

### 9.2 Double Counting

**Penyebab**: ID switch di dekat garis

**Debug:**
```python
# Print setiap crossing event
crossed_in, crossed_out = line_zone.trigger(detections)
for i, tid in enumerate(detections.tracker_id):
    if crossed_in[i] or crossed_out[i]:
        print(f"Crossing: ID={tid}, IN={crossed_in[i]}, OUT={crossed_out[i]}")
```

**Solusi**: Improve tracking stability (lihat Lab 04)

### 9.3 Tidak Menghitung Semua

**Penyebab**: Anchor point tidak melewati garis

**Solusi**: 
- Gunakan `BOTTOM_CENTER`
- Perlebar jangkauan garis
- Turunkan garis sedikit

---

## Kesimpulan

| Feature | LineZone Handling |
|---------|-------------------|
| **Intersection Math** | Automatic |
| **Direction Detection** | IN/OUT built-in |
| **Double Count Prevention** | Automatic per ID |
| **Visualization** | LineZoneAnnotator |

---

## Langkah Selanjutnya

LineZone untuk counting melewati garis. Untuk counting **di dalam area**, gunakan PolygonZone:

👉 **[PolygonZone untuk Area Counting →](03_polygon_zone.md)**
