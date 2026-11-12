# PolygonZone untuk Area Counting

**Durasi Baca**: 15 Menit | **Level**: Menengah

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan mampu:
- Menggunakan `sv.PolygonZone` untuk area-based counting
- Membuat zone interaktif dengan click-to-create
- Track entry/exit dari zone
- Menghitung dwell time (berapa lama objek di dalam zone)
- Implementasi use cases: parking, queue, restricted area

---

## 🎬 Pendahuluan

**LineZone** bagus untuk counting lalu lintas yang **melewati** suatu titik. Tapi bagaimana jika Anda ingin tahu:
- *"Berapa kendaraan ada DI DALAM parkiran?"*
- *"Berapa orang antri di kasir?"*
- *"Apakah ada kendaraan di area terlarang?"*

Untuk ini, kita gunakan **PolygonZone** — area berbentuk poligon yang bisa di-query.

![PolygonZone Analytics](assets/polygon_zone.png)

---

## 1. Konsep PolygonZone

### 1.1 Apa Itu PolygonZone?

PolygonZone adalah area **berbentuk poligon** yang bisa kita query:
- Berapa objek di dalam zone saat ini?
- Objek mana saja yang ada di dalamnya?
- Apakah zone kosong atau terisi?

### 1.2 Perbedaan dengan LineZone

| Aspect | LineZone | PolygonZone |
|--------|----------|-------------|
| **Shape** | Garis (2 titik) | Polygon (N titik) |
| **Detection** | Crossing event | Presence in area |
| **Output** | Count crossing | Count inside |
| **Use Case** | Traffic flow | Occupancy, dwell |

---

## 2. Definisi Polygon

### 2.1 Basic Polygon

```python
import numpy as np
import supervision as sv

# Koordinat vertex polygon (urutan searah jarum jam atau sebaliknya)
polygon = np.array([
    [100, 200],   # Top-left
    [500, 200],   # Top-right
    [550, 600],   # Bottom-right
    [50, 600]     # Bottom-left
])

zone = sv.PolygonZone(
    polygon=polygon,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)
```

### 2.2 Ordering Vertices

Vertices harus dalam urutan yang **konsisten**:

```
       (0)●───────────────●(1)
          │               │
          │    INSIDE     │
          │               │
       (3)●───────────────●(2)

Clockwise: 0 → 1 → 2 → 3
Counter-clockwise: 0 → 3 → 2 → 1
```

### 2.3 Complex Shapes

```python
# L-shaped zone
l_shape = np.array([
    [100, 100],
    [300, 100],
    [300, 200],
    [200, 200],
    [200, 400],
    [100, 400]
])

# Triangular zone
triangle = np.array([
    [400, 100],
    [600, 400],
    [200, 400]
])
```

---

## 3. Kode Dasar

### 3.1 Minimal Example

```python
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()

# Definisikan zone
parking_zone = np.array([
    [100, 300], [400, 300], [400, 700], [100, 700]
])

zone = sv.PolygonZone(
    polygon=parking_zone,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.GREEN,
    thickness=2
)

box_annotator = sv.BoxAnnotator()

cap = cv2.VideoCapture("parking.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detection & Tracking
    results = model(frame, conf=0.1, classes=[2, 3, 5, 7], verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    
    # Check objek di dalam zone
    is_in_zone = zone.trigger(detections=detections)
    
    # Hitung objek di dalam
    count_in_zone = np.sum(is_in_zone)
    
    # Annotate
    frame = zone_annotator.annotate(scene=frame)
    frame = box_annotator.annotate(frame, detections)
    
    # Display count
    cv2.putText(frame, f"Vehicles in parking: {count_in_zone}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Parking Zone", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
```

### 3.2 Return Value dari trigger()

```python
is_in_zone = zone.trigger(detections)

# is_in_zone: boolean array
# True untuk objek yang anchor-nya berada DI DALAM zone
# False untuk objek di luar zone

# Contoh penggunaan
for i, (tracker_id, inside) in enumerate(zip(detections.tracker_id, is_in_zone)):
    if inside:
        print(f"Vehicle {tracker_id} is in the zone")
```

---

## 4. Multiple Zones

### 4.1 Parking Slots

```python
class ParkingLotMonitor:
    """Monitor multiple parking slots"""
    
    def __init__(self, slot_polygons: list):
        """
        Args:
            slot_polygons: List of np.array polygons for each slot
        """
        self.zones = [
            sv.PolygonZone(
                polygon=poly,
                triggering_anchors=(sv.Position.BOTTOM_CENTER,)
            )
            for poly in slot_polygons
        ]
        self.num_slots = len(slot_polygons)
    
    def get_occupancy(self, detections) -> dict:
        """Get occupancy status for each slot"""
        occupancy = []
        
        for i, zone in enumerate(self.zones):
            is_in = zone.trigger(detections)
            is_occupied = np.any(is_in)
            occupancy.append(is_occupied)
        
        return {
            'slots': occupancy,
            'occupied': sum(occupancy),
            'available': self.num_slots - sum(occupancy),
            'occupancy_rate': sum(occupancy) / self.num_slots
        }

# Usage
slots = [
    np.array([[100, 100], [200, 100], [200, 200], [100, 200]]),
    np.array([[250, 100], [350, 100], [350, 200], [250, 200]]),
    np.array([[400, 100], [500, 100], [500, 200], [400, 200]]),
    # ... more slots
]

parking = ParkingLotMonitor(slots)

# Dalam loop
status = parking.get_occupancy(detections)
print(f"Available: {status['available']}/{parking.num_slots}")
print(f"Occupancy: {status['occupancy_rate']:.1%}")
```

### 4.2 Grid Layout Generator

```python
def generate_grid_zones(
    start_x: int, start_y: int,
    slot_width: int, slot_height: int,
    rows: int, cols: int,
    gap_x: int = 10, gap_y: int = 10
) -> list:
    """Generate grid of rectangular zones"""
    zones = []
    
    for row in range(rows):
        for col in range(cols):
            x = start_x + col * (slot_width + gap_x)
            y = start_y + row * (slot_height + gap_y)
            
            polygon = np.array([
                [x, y],
                [x + slot_width, y],
                [x + slot_width, y + slot_height],
                [x, y + slot_height]
            ])
            zones.append(polygon)
    
    return zones

# Generate 3x4 grid of parking slots
slot_polygons = generate_grid_zones(
    start_x=100, start_y=200,
    slot_width=80, slot_height=150,
    rows=3, cols=4,
    gap_x=15, gap_y=20
)

parking = ParkingLotMonitor(slot_polygons)
```

---

## 5. Tracking Entry/Exit dari Zone

### 5.1 Zone Entry/Exit Tracker

```python
from typing import Set, Dict
from datetime import datetime

class ZoneEntryExitTracker:
    """Track entry and exit events for a zone"""
    
    def __init__(self, zone: sv.PolygonZone):
        self.zone = zone
        self.ids_currently_in: Set[int] = set()
        self.entry_count = 0
        self.exit_count = 0
    
    def update(self, detections) -> Dict:
        """
        Update zone status and return events
        
        Returns:
            Dict with current count, entries, and exits
        """
        is_in = self.zone.trigger(detections)
        
        current_in_zone: Set[int] = set()
        entries = []
        exits = []
        
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue
            
            if is_in[i]:
                current_in_zone.add(tracker_id)
                
                # Entry: was not in zone, now is
                if tracker_id not in self.ids_currently_in:
                    self.entry_count += 1
                    entries.append(tracker_id)
        
        # Exit: was in zone, now is not
        exited_ids = self.ids_currently_in - current_in_zone
        for tracker_id in exited_ids:
            self.exit_count += 1
            exits.append(tracker_id)
        
        # Update state
        self.ids_currently_in = current_in_zone
        
        return {
            'current_count': len(current_in_zone),
            'entry_count': self.entry_count,
            'exit_count': self.exit_count,
            'entries_this_frame': entries,
            'exits_this_frame': exits
        }

# Usage
zone = sv.PolygonZone(polygon=parking_zone)
tracker = ZoneEntryExitTracker(zone)

# Dalam loop
status = tracker.update(detections)

for tid in status['entries_this_frame']:
    print(f"Vehicle {tid} entered the zone!")

for tid in status['exits_this_frame']:
    print(f"Vehicle {tid} left the zone!")
```

---

## 6. Dwell Time Analysis

### 6.1 Dwell Time Tracker

```python
import time
from typing import Dict, Optional

class DwellTimeTracker:
    """Track how long vehicles stay in a zone"""
    
    def __init__(self, zone: sv.PolygonZone):
        self.zone = zone
        self.entry_times: Dict[int, float] = {}
        self.completed_dwells: list = []
    
    def update(self, detections) -> Dict:
        """
        Update dwell tracking
        
        Returns:
            Dict with active dwells and completed dwells this frame
        """
        is_in = self.zone.trigger(detections)
        current_time = time.time()
        
        current_in_zone: Set[int] = set()
        completed_this_frame = []
        
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue
            
            if is_in[i]:
                current_in_zone.add(tracker_id)
                
                # Record entry time
                if tracker_id not in self.entry_times:
                    self.entry_times[tracker_id] = current_time
            else:
                # Check if was in zone
                if tracker_id in self.entry_times:
                    dwell_duration = current_time - self.entry_times[tracker_id]
                    completed = {
                        'tracker_id': tracker_id,
                        'duration': dwell_duration,
                        'entry_time': self.entry_times[tracker_id],
                        'exit_time': current_time
                    }
                    self.completed_dwells.append(completed)
                    completed_this_frame.append(completed)
                    del self.entry_times[tracker_id]
        
        # Calculate active dwells
        active_dwells = {}
        for tracker_id, entry_time in self.entry_times.items():
            if tracker_id in current_in_zone:
                active_dwells[tracker_id] = current_time - entry_time
        
        return {
            'active_count': len(current_in_zone),
            'active_dwells': active_dwells,
            'completed_this_frame': completed_this_frame,
            'avg_dwell': self._calculate_avg_dwell()
        }
    
    def _calculate_avg_dwell(self) -> Optional[float]:
        if not self.completed_dwells:
            return None
        return sum(d['duration'] for d in self.completed_dwells) / len(self.completed_dwells)
    
    def get_statistics(self) -> Dict:
        """Get dwell time statistics"""
        if not self.completed_dwells:
            return {'count': 0}
        
        durations = [d['duration'] for d in self.completed_dwells]
        
        return {
            'count': len(durations),
            'avg': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
            'total': sum(durations)
        }

# Usage
zone = sv.PolygonZone(polygon=loading_dock)
dwell_tracker = DwellTimeTracker(zone)

# Dalam loop
status = dwell_tracker.update(detections)

for completed in status['completed_this_frame']:
    print(f"Vehicle {completed['tracker_id']} spent {completed['duration']:.1f}s in zone")

# Untuk aktif
for tid, duration in status['active_dwells'].items():
    print(f"Vehicle {tid} has been in zone for {duration:.1f}s")
```

---

## 7. Click-to-Create Zone Tool

### 7.1 Interactive Zone Creator

```python
import cv2
import numpy as np

class ZoneCreator:
    """Interactive tool to create polygon zones by clicking"""
    
    def __init__(self, image: np.ndarray, window_name: str = "Create Zone"):
        self.image = image.copy()
        self.display = image.copy()
        self.points = []
        self.window_name = window_name
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._click_event)
    
    def _click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.points.append([x, y])
            self._redraw()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last point
            if self.points:
                self.points.pop()
                self._redraw()
    
    def _redraw(self):
        self.display = self.image.copy()
        
        # Draw points
        for i, pt in enumerate(self.points):
            cv2.circle(self.display, tuple(pt), 5, (0, 255, 0), -1)
            cv2.putText(self.display, str(i), (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw lines between points
        if len(self.points) > 1:
            pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.display, [pts], False, (0, 255, 0), 2)
        
        # Draw closing line (preview)
        if len(self.points) > 2:
            cv2.line(self.display, 
                    tuple(self.points[-1]), 
                    tuple(self.points[0]),
                    (0, 255, 255), 2, cv2.LINE_AA)
    
    def run(self) -> np.ndarray:
        """
        Run the interactive zone creation
        
        Controls:
            Left Click: Add point
            Right Click: Remove last point
            'c': Clear all points
            'q': Finish and return polygon
            
        Returns:
            np.array of polygon vertices
        """
        print("Controls:")
        print("  Left Click: Add point")
        print("  Right Click: Remove last point")
        print("  'c': Clear all")
        print("  'q': Finish")
        
        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.points = []
                self._redraw()
        
        cv2.destroyWindow(self.window_name)
        
        if len(self.points) < 3:
            print("Warning: Polygon needs at least 3 points")
            return None
        
        return np.array(self.points)

# Usage
cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()
cap.release()

if ret:
    creator = ZoneCreator(frame)
    polygon = creator.run()
    
    if polygon is not None:
        print(f"Polygon coordinates:\n{polygon.tolist()}")
        
        # Save to file
        np.save("zone_polygon.npy", polygon)
        print("Saved to zone_polygon.npy")
```

### 7.2 Load Saved Zone

```python
# Load zone from file
polygon = np.load("zone_polygon.npy")

zone = sv.PolygonZone(
    polygon=polygon,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)
```

---

## 8. Use Cases

### 8.1 Parking Occupancy Display

```python
class ParkingDisplay:
    """Real-time parking availability display"""
    
    def __init__(self, zone: sv.PolygonZone, capacity: int):
        self.zone = zone
        self.capacity = capacity
    
    def draw_status(self, frame: np.ndarray, detections) -> np.ndarray:
        is_in = self.zone.trigger(detections)
        occupied = np.sum(is_in)
        available = self.capacity - occupied
        
        # Status color
        if available == 0:
            color = (0, 0, 255)  # Red - FULL
            status = "FULL"
        elif available < self.capacity * 0.2:
            color = (0, 165, 255)  # Orange - Almost full
            status = "ALMOST FULL"
        else:
            color = (0, 255, 0)  # Green - Available
            status = "AVAILABLE"
        
        # Draw overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        cv2.putText(frame, f"PARKING: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Available: {available}/{self.capacity}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        return frame
```

### 8.2 Queue / Antrian Detection

```python
class QueueMonitor:
    """Monitor queue length at checkout/service point"""
    
    def __init__(self, zone: sv.PolygonZone, alert_threshold: int = 5):
        self.zone = zone
        self.alert_threshold = alert_threshold
        self.history = []
    
    def update(self, detections) -> Dict:
        is_in = self.zone.trigger(detections)
        queue_length = np.sum(is_in)
        
        self.history.append({
            'timestamp': time.time(),
            'length': queue_length
        })
        
        # Keep last 5 minutes
        cutoff = time.time() - 300
        self.history = [h for h in self.history if h['timestamp'] > cutoff]
        
        # Calculate average
        avg_length = sum(h['length'] for h in self.history) / len(self.history)
        
        return {
            'current_length': queue_length,
            'avg_length_5min': avg_length,
            'alert': queue_length >= self.alert_threshold
        }
```

### 8.3 Restricted Area Alert

```python
class RestrictedAreaMonitor:
    """Alert when vehicles enter restricted zone"""
    
    def __init__(self, zone: sv.PolygonZone, alert_callback=None):
        self.zone = zone
        self.alert_callback = alert_callback
        self.alerted_ids = set()
    
    def update(self, detections) -> list:
        is_in = self.zone.trigger(detections)
        alerts = []
        
        for i, (tracker_id, inside) in enumerate(zip(detections.tracker_id, is_in)):
            if tracker_id is None:
                continue
            
            if inside and tracker_id not in self.alerted_ids:
                self.alerted_ids.add(tracker_id)
                
                alert = {
                    'tracker_id': tracker_id,
                    'timestamp': datetime.now(),
                    'position': detections.xyxy[i].tolist()
                }
                alerts.append(alert)
                
                if self.alert_callback:
                    self.alert_callback(alert)
        
        return alerts

# Usage dengan alert callback
def send_alert(alert):
    print(f"⚠️ ALERT: Vehicle {alert['tracker_id']} in restricted area!")
    # Send notification, log event, etc.

restricted = RestrictedAreaMonitor(
    zone=sv.PolygonZone(polygon=restricted_area),
    alert_callback=send_alert
)
```

---

## Kesimpulan

| Feature | PolygonZone Capability |
|---------|------------------------|
| **Shape** | Any polygon shape |
| **Query** | Objects inside area |
| **Entry/Exit** | With custom tracking |
| **Dwell Time** | With timestamp tracking |
| **Use Cases** | Parking, queue, alerts |

---

## Langkah Selanjutnya

Vehicle counting sudah selesai! Lab berikutnya menganalisis **arah belok** kendaraan:

👉 **[Lab 06: Turn Analytics →](../06_turn_analytics/)**
