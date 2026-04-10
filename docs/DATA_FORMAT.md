# Data Format Specification

## Landmark CSV Format

Each CSV file contains landmark time-series data for one or more swallowing sequences. Required columns:

```
subject_id, condition, frame, x1, y1, x2, y2, ..., x14, y14
```

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | string | Unique subject identifier (e.g., "HNC_001") |
| `condition` | string | Clinical label: "healthy", "patient", "pre_rt", "post_rt", etc. |
| `frame` | int | Frame index (0-based) |
| `time` | float | (Optional) Physical time in seconds. If absent, computed as frame/fps. |
| `x1, y1` | float | Coordinates of landmark 1 (tongue dorsum anterior) |
| `x2, y2` | float | Coordinates of landmark 2 (tongue dorsum mid-anterior) |
| ... | ... | ... |
| `x14, y14` | float | Coordinates of landmark 14 (UES inferior margin) |

### Landmark Definitions

| ID | Name | Anatomical Description |
|----|------|----------------------|
| 1 | Tongue anterior | Anterior tongue dorsum at midsagittal plane |
| 2 | Tongue mid-anterior | Mid-anterior tongue dorsum |
| 3 | Tongue mid-posterior | Mid-posterior tongue dorsum |
| 4 | Tongue posterior | Posterior tongue dorsum / tongue base |
| 5 | Hyoid anterior | Anterior-superior corner of hyoid body |
| 6 | Hyoid posterior | Posterior-inferior corner of hyoid body |
| 7 | Larynx superior | Superior margin of laryngeal vestibule |
| 8 | Larynx inferior | Inferior margin of cricoid cartilage |
| 9 | Epiglottis tip | Free tip of epiglottis |
| 10 | Epiglottis base | Base of epiglottis at hyoepiglottic ligament |
| 11 | Pharynx superior | Superior pharyngeal wall at velopharyngeal level |
| 12 | Pharynx inferior | Inferior pharyngeal wall at pyriform sinus level |
| 13 | UES superior | Superior margin of UES (cricopharyngeus) |
| 14 | UES inferior | Inferior margin of UES |

### Coordinate System

- **Origin**: Anterior-inferior corner of C2 vertebral body (or other fixed bony landmark).
- **X-axis**: Anterior-posterior (positive = anterior).
- **Y-axis**: Superior-inferior (positive = superior).
- **Units**: Millimeters.

### Example

```csv
subject_id,condition,frame,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12,x13,y13,x14,y14
HNC_001,pre_rt,0,45.2,62.1,38.5,58.3,31.2,52.1,24.8,44.6,22.1,38.5,18.3,35.2,15.8,42.3,12.1,32.1,28.5,48.2,22.3,42.1,8.5,55.2,5.2,38.1,4.8,32.5,3.2,28.1
HNC_001,pre_rt,1,45.3,62.0,38.6,58.2,31.3,52.0,24.9,44.5,22.2,38.4,18.4,35.1,15.9,42.4,12.2,32.0,28.6,48.1,22.4,42.0,8.6,55.1,5.3,38.0,4.9,32.4,3.3,28.0
...
```

## Clinical Scores File (Optional)

For clinical comparison, provide a CSV with:

```csv
subject_id,FOIS,MBSImP_oral,MBSImP_pharyngeal,PAS
HNC_001,5,8,12,2
HNC_002,7,2,4,1
...
```

| Column | Description | Range |
|--------|------------|-------|
| FOIS | Functional Oral Intake Scale | 1–7 |
| MBSImP_oral | MBSImP Oral Total score | 0–22 |
| MBSImP_pharyngeal | MBSImP Pharyngeal Total score | 0–27 |
| PAS | Penetration-Aspiration Scale | 1–8 |

## NIfTI Format (Alternative)

For volumetric 3D+time MRI data, place NIfTI files (.nii.gz) in `data/real/raw/`:

```
data/real/raw/
├── HNC_001_cine.nii.gz      # 4D: (X, Y, Z, T)
├── HNC_001_seg.nii.gz       # 4D segmentation masks
├── HNC_001_metadata.json    # {"fps": 25, "condition": "pre_rt"}
└── ...
```

Segmentation labels: 1=tongue, 2=airway, 3=pharyngeal wall, 4=larynx, 5=epiglottis, 6=UES, 0=background.
