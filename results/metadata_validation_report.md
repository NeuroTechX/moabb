# Metadata Validation Report

Generated: 2026-01-28 10:34:17

## Summary

| Metric | Count |
|--------|-------|
| Total Datasets | 84 |
| Datasets with Errors | 3 |
| Datasets with Warnings | 14 |
| Datasets with Data Available | 30 |
| Clean (No Issues) | 65 |

### Issue Counts by Severity

| Severity | Count |
|----------|-------|
| Errors | 3 |
| Warnings | 14 |
| Info | 8 |

## Datasets with Errors

### MAMEM1

Data path: `/Users/bruaristimunha/mne_data/MNE-ssvepexo-data`

- **ERROR** `acquisition.sampling_rate`: Sampling rate mismatch: catalog=250.0, actual=256.0
  - Suggestion: Update catalog: sampling_rate=256.0

### MAMEM2

Data path: `/Users/bruaristimunha/mne_data/MNE-ssvepexo-data`

- **ERROR** `acquisition.sampling_rate`: Sampling rate mismatch: catalog=250.0, actual=256.0
  - Suggestion: Update catalog: sampling_rate=256.0

### MAMEM3

Data path: `/Users/bruaristimunha/mne_data/MNE-ssvepexo-data`

- **ERROR** `acquisition.sampling_rate`: Sampling rate mismatch: catalog=128.0, actual=256.0
  - Suggestion: Update catalog: sampling_rate=256.0

## Datasets with Warnings

### Dreyer2023A

- **WARNING** `acquisition.n_channels`: Channel count mismatch: catalog=27, actual=32
  - Suggestion: Update catalog: n_channels=32

### ErpCore2021_ERN

- **WARNING** `acquisition.n_channels`: Channel count mismatch: catalog=30, actual=33
  - Suggestion: Update catalog: n_channels=33

### ErpCore2021_LRP

- **WARNING** `acquisition.n_channels`: Channel count mismatch: catalog=30, actual=33
  - Suggestion: Update catalog: n_channels=33

### ErpCore2021_MMN

- **WARNING** `acquisition.n_channels`: Channel count mismatch: catalog=30, actual=33
  - Suggestion: Update catalog: n_channels=33

### ErpCore2021_N170

- **WARNING** `acquisition.n_channels`: Channel count mismatch: catalog=30, actual=33
  - Suggestion: Update catalog: n_channels=33

### ErpCore2021_N2pc

- **WARNING** `acquisition.n_channels`: Channel count mismatch: catalog=30, actual=33
  - Suggestion: Update catalog: n_channels=33

### ErpCore2021_N400

- **WARNING** `acquisition.n_channels`: Channel count mismatch: catalog=30, actual=33
  - Suggestion: Update catalog: n_channels=33

### ErpCore2021_P3

- **WARNING** `acquisition.n_channels`: Channel count mismatch: catalog=30, actual=33
  - Suggestion: Update catalog: n_channels=33

### PhysionetMI

- **WARNING** `experiment.n_classes`: n_classes (4) != len(events) (5)

### Thielen2015

- **WARNING** `experiment.n_classes`: n_classes (36) != len(events) (2)

### Thielen2021

- **WARNING** `experiment.n_classes`: n_classes (20) != len(events) (2)

## Missing/Optional Field Summary

Fields that could be populated:


## Extracted Values (from available data)

### Dreyer2023A

```
  sampling_rate: 512.0
  n_channels: 32
  sensors: [32 channels]
  channel_types: {'eeg': 32}
  has_montage: False
```

### ErpCore2021_ERN

```
  sampling_rate: 1024.0
  n_channels: 33
  sensors: [33 channels]
  channel_types: {'eeg': 33}
  has_montage: True
```

### ErpCore2021_LRP

```
  sampling_rate: 1024.0
  n_channels: 33
  sensors: [33 channels]
  channel_types: {'eeg': 33}
  has_montage: True
```

### ErpCore2021_MMN

```
  sampling_rate: 1024.0
  n_channels: 33
  sensors: [33 channels]
  channel_types: {'eeg': 33}
  has_montage: True
```

### ErpCore2021_N170

```
  sampling_rate: 1024.0
  n_channels: 33
  sensors: [33 channels]
  channel_types: {'eeg': 33}
  has_montage: True
```

### ErpCore2021_N2pc

```
  sampling_rate: 1024.0
  n_channels: 33
  sensors: [33 channels]
  channel_types: {'eeg': 33}
  has_montage: True
```

### ErpCore2021_N400

```
  sampling_rate: 1024.0
  n_channels: 33
  sensors: [33 channels]
  channel_types: {'eeg': 33}
  has_montage: True
```

### ErpCore2021_P3

```
  sampling_rate: 1024.0
  n_channels: 33
  sensors: [33 channels]
  channel_types: {'eeg': 33}
  has_montage: True
```

### MAMEM1

```
  sampling_rate: 256.0
  n_channels: 9
  sensors: [9 channels]
  channel_types: {'eeg': 8, 'stim': 1}
  line_freq: 50.0
  has_montage: False
```

### MAMEM2

```
  sampling_rate: 256.0
  n_channels: 9
  sensors: [9 channels]
  channel_types: {'eeg': 8, 'stim': 1}
  line_freq: 50.0
  has_montage: False
```

### MAMEM3

```
  sampling_rate: 256.0
  n_channels: 9
  sensors: [9 channels]
  channel_types: {'eeg': 8, 'stim': 1}
  line_freq: 50.0
  has_montage: False
```
