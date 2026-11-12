# Matematika Line Crossing

**Durasi Baca**: 15 Menit \| **Level**: Menengah

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan memahami:

- Teori geometri di balik line crossing detection  
- Algoritma cross product untuk intersection test  
- Cara menentukan arah crossing (IN/OUT)  
- Edge cases dan solusinya  
- Implementasi line crossing counter dari scratch

---

## 🎬 Pendahuluan

**Vehicle counting** memerlukan deteksi event "crossing" — kapan objek **melewati garis virtual**. Ini berbeda dengan sekedar "apakah objek di atas/bawah garis":

```
❌ Simple position check:
   "Apakah objek sekarang ada di atas garis?"
   Problem: Tidak tahu kapan MELEWATI garis

✅ Line crossing detection:
   "Apakah objek MELEWATI garis antara frame t-1 dan frame t?"
```

![Line Crossing Detection Principle](assets/line_crossing.png)

---

## 1\. Definisi Masalah

### 1.1 Input

Kita memiliki:

- **Garis counting**: Dari titik P1 (x1, y1) ke P2 (x2, y2)  
- **Movement vector**: Dari posisi objek di frame t-1 (A) ke frame t (B)

### 1.2 Pertanyaan

**Apakah segment AB memotong segment P1P2?**

```
Visual:
                P1
                 •
                 │╲
                 │ ╲
            A────┼──╲────→ Movement vector
                 │   ╲
                 │    ╲
                 •     B
                P2

Crossing terjadi karena segment AB memotong segment P1P2
```

### 1.3 Output

1. **Boolean**: Apakah crossing terjadi?  
2. **Direction**: Arah crossing (IN \= kiri→kanan, OUT \= kanan→kiri)

---

## 2\. Teori: Line Segment Intersection

### 2.1 Representasi Parametrik

Sebuah segment dapat direpresentasikan secara **parametrik**:

**Segment AB:**

$$
P(t) = A + t \cdot (B - A), \quad 0 \leq t \leq 1
$$

&nbsp;

- Saat $t=0$ → $P=A$  
- Saat $t=1$ → $P=B$  
- Saat $t=0.5$ → $P$ \= midpoint

**Segment P1P2:**

$$
P(u) = P_1 + u \cdot (P_2 - P_1), \quad 0 \leq u \leq 1
$$

&nbsp;

### 2.2 Kondisi Intersection

Dua segment berpotongan jika:

1. Ada nilai $t$ dan $u$ dalam range $[0,1]$  
2. Yang menghasilkan Point yang sama

Ini berarti menyelesaikan sistem persamaan:

$$A+t(B-A)={P}_{1}+u({P}_{2}-{P}_{1})$$

2

&nbsp;

### 2.3 Solusi dengan Cross Product

Cara lebih **efisien** menggunakan **cross product 2D**.

**Cross Product 2D:**

$$
\text{cross}(O, A, B) = (A_x - O_x) \cdot (B_y - O_y) - (A_y - O_y) \cdot (B_x - O_x)
$$

&nbsp;

**Interpretasi geometris:**

- Hasil $>0$: B berada di **sebelah kiri** vector $\overrightarrow{OA}$
- Hasil $<0$: B berada di **sebelah kanan** vector $\overrightarrow{OA}$  
- Hasil $=0$: A, O, B **collinear** (segaris)

```py
def cross_product_2d(o: tuple, a: tuple, b: tuple) -> float:
    """
    Compute 2D cross product of vectors OA and OB
    
    Args:
        o: Origin point (x, y)
        a: Point A (x, y)
        b: Point B (x, y)
    
    Returns:
        Positive if B is left of OA
        Negative if B is right of OA
        Zero if collinear
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
```

---

## 3\. Algoritma Intersection Test

### 3.1 Core Algorithm

Dua segment berpotongan jika mereka **saling mengapit**:

- P1 dan P2 berada di sisi berbeda dari segment AB  
- A dan B berada di sisi berbeda dari segment P1P2

```py
def segments_intersect(p1: tuple, p2: tuple, a: tuple, b: tuple) -> bool:
    """
    Check if segment P1P2 intersects with segment AB
    
    Args:
        p1, p2: Counting line endpoints
        a, b: Object trajectory (previous pos, current pos)
    
    Returns:
        True if segments intersect
    """
    # Calculate cross products
    d1 = cross_product_2d(a, b, p1)
    d2 = cross_product_2d(a, b, p2)
    d3 = cross_product_2d(p1, p2, a)
    d4 = cross_product_2d(p1, p2, b)
    
    # Check if P1 and P2 are on opposite sides of AB
    opposite_1 = (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)
    
    # Check if A and B are on opposite sides of P1P2
    opposite_2 = (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    
    # Intersection occurs if both conditions are true
    return opposite_1 and opposite_2
```

### 3.2 Penjelasan Visual

```
                   d1 > 0
                     │
    P1 •─────────────┼─────────────• P2
                     │
                     │  d2 < 0
                     │
    Jika d1 dan d2 punya tanda berbeda,
    berarti P1 dan P2 di sisi berbeda dari AB
```

### 3.3 Edge Cases

```py
def segments_intersect_robust(p1, p2, a, b):
    """
    Robust intersection test handling collinear cases
    """
    d1 = cross_product_2d(a, b, p1)
    d2 = cross_product_2d(a, b, p2)
    d3 = cross_product_2d(p1, p2, a)
    d4 = cross_product_2d(p1, p2, b)
    
    # Standard intersection test
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # Collinear cases (segments overlap)
    if d1 == 0 and on_segment(a, p1, b):
        return True
    if d2 == 0 and on_segment(a, p2, b):
        return True
    if d3 == 0 and on_segment(p1, a, p2):
        return True
    if d4 == 0 and on_segment(p1, b, p2):
        return True
    
    return False

def on_segment(p, q, r):
    """Check if q lies on segment pr when all are collinear"""
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))
```

---

## 4\. Menentukan Arah Crossing

### 4.1 Menggunakan Tanda Cross Product

Arah crossing ditentukan oleh perubahan "sisi" objek:

```py
def get_crossing_direction(
    line_start: tuple, 
    line_end: tuple, 
    point_before: tuple, 
    point_after: tuple
) -> int:
    """
    Determine crossing direction relative to line orientation
    
    Args:
        line_start, line_end: Counting line endpoints
        point_before: Object position at frame t-1
        point_after: Object position at frame t
    
    Returns:
        +1: Crossed from left to right (IN)
        -1: Crossed from right to left (OUT)
         0: No crossing
    """
    d_before = cross_product_2d(line_start, line_end, point_before)
    d_after = cross_product_2d(line_start, line_end, point_after)
    
    # Crossed from left (+) to right (-)
    if d_before > 0 and d_after < 0:
        return 1  # IN
    
    # Crossed from right (-) to left (+)
    elif d_before < 0 and d_after > 0:
        return -1  # OUT
    
    else:
        return 0  # No crossing
```

### 4.2 Pengertian "Left" dan "Right"

Tergantung orientasi garis:

```
Garis horizontal KIRI→KANAN:
┌──────────────────────────────────┐
│         LEFT (d > 0)             │
│                                  │
•──────────────────────────────────• → Line direction
│                                  │
│         RIGHT (d < 0)            │
└──────────────────────────────────┘

Garis vertikal ATAS→BAWAH:
         │
  LEFT   │   RIGHT
 (d > 0) │  (d < 0)
         │
         ↓
```

### 4.3 Konfigurasi untuk Traffic

```py
class CountingLine:
    """
    Counting line with configurable direction
    """
    
    def __init__(
        self, 
        start: tuple, 
        end: tuple,
        in_direction: str = "left_to_right"
    ):
        self.start = start
        self.end = end
        self.in_direction = in_direction
    
    def get_direction(self, point_before, point_after):
        raw_direction = get_crossing_direction(
            self.start, self.end, point_before, point_after
        )
        
        # Adjust based on configuration
        if self.in_direction == "right_to_left":
            return -raw_direction
        
        return raw_direction

# Contoh: Jalan satu arah dari kiri ke kanan
line = CountingLine(
    start=(0, 400),
    end=(1920, 400),
    in_direction="left_to_right"  # Kendaraan masuk dari atas
)
```

---

## 5\. Implementasi Complete Counter

### 5.1 Line Crossing Counter Class

```py
from collections import defaultdict
from typing import Dict, Tuple, Set
import numpy as np

class LineCrossingCounter:
    """
    Count vehicles crossing a virtual line
    """
    
    def __init__(
        self, 
        line_start: Tuple[int, int], 
        line_end: Tuple[int, int],
        use_bottom_center: bool = True
    ):
        self.line_start = line_start
        self.line_end = line_end
        self.use_bottom_center = use_bottom_center
        
        # State tracking
        self.previous_positions: Dict[int, Tuple[int, int]] = {}
        self.crossed_ids: Set[int] = set()
        
        # Counters
        self.in_count = 0
        self.out_count = 0
        
        # Debug info
        self.last_crossing_id = None
        self.last_crossing_direction = None
    
    def _get_anchor_point(self, xyxy: np.ndarray) -> Tuple[int, int]:
        """Extract anchor point from bounding box"""
        if self.use_bottom_center:
            # Bottom center for ground-level counting
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int(xyxy[3])  # Bottom y
        else:
            # Center
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)
        
        return (cx, cy)
    
    def update(self, detections) -> Dict[str, any]:
        """
        Update counter with new detections
        
        Args:
            detections: supervision.Detections with tracker_id
        
        Returns:
            Dict with crossing events
        """
        events = []
        
        for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
            if tracker_id is None:
                continue
            
            current_pos = self._get_anchor_point(xyxy)
            
            # Check crossing if we have previous position
            if tracker_id in self.previous_positions:
                prev_pos = self.previous_positions[tracker_id]
                
                # Get crossing direction
                direction = get_crossing_direction(
                    self.line_start, self.line_end,
                    prev_pos, current_pos
                )
                
                # Count only once per ID
                if direction != 0 and tracker_id not in self.crossed_ids:
                    self.crossed_ids.add(tracker_id)
                    
                    if direction == 1:
                        self.in_count += 1
                        self.last_crossing_direction = "IN"
                    else:
                        self.out_count += 1
                        self.last_crossing_direction = "OUT"
                    
                    self.last_crossing_id = tracker_id
                    
                    events.append({
                        'tracker_id': tracker_id,
                        'direction': 'IN' if direction == 1 else 'OUT',
                        'position': current_pos
                    })
            
            # Update position
            self.previous_positions[tracker_id] = current_pos
        
        # Cleanup old tracks (memory management)
        active_ids = set(tid for tid in detections.tracker_id if tid is not None)
        old_ids = set(self.previous_positions.keys()) - active_ids
        for old_id in old_ids:
            del self.previous_positions[old_id]
        
        return {
            'in_count': self.in_count,
            'out_count': self.out_count,
            'total': self.in_count + self.out_count,
            'net_flow': self.in_count - self.out_count,
            'events': events
        }
    
    def reset(self):
        """Reset counters"""
        self.in_count = 0
        self.out_count = 0
        self.crossed_ids.clear()
```

### 5.2 Complete Usage Example

```py
import cv2
from ultralytics import YOLO
import supervision as sv

# Initialize
model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()

# Counting line (horizontal across screen)
counter = LineCrossingCounter(
    line_start=(0, 400),
    line_end=(1920, 400),
    use_bottom_center=True
)

cap = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detection
    results = model(frame, conf=0.1, classes=[2, 3, 5, 7], verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Tracking
    detections = tracker.update_with_detections(detections)
    
    # Counting
    stats = counter.update(detections)
    
    # Draw line
    cv2.line(frame, counter.line_start, counter.line_end, (0, 255, 255), 3)
    
    # Draw counts
    mid_x = (counter.line_start[0] + counter.line_end[0]) // 2
    mid_y = counter.line_start[1]
    
    cv2.putText(frame, f"IN: {stats['in_count']}", 
                (mid_x - 100, mid_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {stats['out_count']}", 
                (mid_x - 100, mid_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw bounding boxes
    for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
        if tracker_id is None:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"#{tracker_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("Counting", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
print(f"\nFinal Count: IN={stats['in_count']}, OUT={stats['out_count']}")
```

---

## 6\. Edge Cases dan Solusi

### 6.1 Objek Berhenti di Atas Garis (Jitter)

**Problem**: Jitter deteksi menyebabkan posisi berfluktuasi bolak-balik.

```
Frame 10: pos = (500, 399)  ← Above line
Frame 11: pos = (502, 401)  ← Below line → COUNT!
Frame 12: pos = (501, 399)  ← Above line → COUNT AGAIN?
```

**Solusi**: Gunakan `crossed_ids` set untuk track yang sudah dihitung.

```py
if direction != 0 and tracker_id not in self.crossed_ids:
    self.crossed_ids.add(tracker_id)  # Prevent double counting
    # Count...
```

### 6.2 ID Switch di Dekat Garis

**Problem**: Jika ID berubah tepat saat crossing, objek mungkin tidak terhitung.

```
Frame 10: Track #5 at (500, 390)
Frame 11: Track #5 lost, Track #42 created at (510, 410)
          → Crossing not detected because different IDs!
```

**Solusi**: Pastikan tracking stabil dengan tuning ByteTrack parameters.

### 6.3 Kecepatan Sangat Tinggi / FPS Rendah

**Problem**: Objek "teleport" melewati garis.

```
Frame 10: pos = (500, 300)  ← Far above line
Frame 11: pos = (550, 500)  ← Far below line

If we only check if line intersects with movement vector,
this will be caught. But if using simple position checks,
it might be missed.
```

**Solusi**: Selalu gunakan intersection test, bukan position check.

### 6.4 Garis Diagonal

```py
# Garis diagonal berfungsi dengan algoritma yang sama!
counter = LineCrossingCounter(
    line_start=(100, 200),
    line_end=(800, 600)  # Diagonal line
)
```

---

## 7\. Multiple Lines

### 7.1 Multi-Lane Counting

```py
class MultiLineCounter:
    """Count vehicles across multiple lanes"""
    
    def __init__(self, lines: list):
        self.counters = [
            LineCrossingCounter(line['start'], line['end'])
            for line in lines
        ]
        self.line_names = [line.get('name', f'Line_{i}') 
                          for i, line in enumerate(lines)]
    
    def update(self, detections):
        results = {}
        total_in = 0
        total_out = 0
        
        for name, counter in zip(self.line_names, self.counters):
            stats = counter.update(detections)
            results[name] = stats
            total_in += stats['in_count']
            total_out += stats['out_count']
        
        results['total'] = {'in': total_in, 'out': total_out}
        return results

# Usage
lines = [
    {'name': 'Lane_1', 'start': (0, 300), 'end': (640, 300)},
    {'name': 'Lane_2', 'start': (640, 400), 'end': (1280, 400)},
    {'name': 'Lane_3', 'start': (1280, 300), 'end': (1920, 300)},
]

multi_counter = MultiLineCounter(lines)
```

---

## Kesimpulan

| Konsep | Penjelasan |
| :---- | :---- |
| **Cross Product** | Menentukan sisi titik terhadap garis |
| **Intersection Test** | Cek apakah dua segment berpotongan |
| **Direction** | Perubahan tanda cross product \= crossing |
| **Double Count Prevention** | Track crossed IDs in set |

---

## Langkah Selanjutnya

Teori sudah dipahami. Artikel berikutnya menggunakan library Supervision untuk implementasi yang lebih mudah:

👉 [**Supervision LineZone →**](http://02_supervision_linezone.md)