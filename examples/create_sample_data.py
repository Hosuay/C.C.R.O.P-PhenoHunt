#!/usr/bin/env python3
"""
Create sample strain database for PhenoHunter CLI examples.
"""

import pandas as pd
from pathlib import Path

# Sample strain data
sample_strains = {
    'Blue Dream': {
        'type': 'hybrid',
        'thc_pct': 19.5, 'cbd_pct': 0.8, 'cbg_pct': 0.9, 'cbc_pct': 0.2, 'cbda_pct': 0.15,
        'thcv_pct': 0.3, 'cbn_pct': 0.5, 'delta8_thc_pct': 0.1, 'thca_pct': 18.0,
        'myrcene_pct': 1.2, 'limonene_pct': 1.5, 'pinene_pct': 0.8, 'linalool_pct': 0.4,
        'caryophyllene_pct': 0.9, 'humulene_pct': 0.3, 'terpinolene_pct': 0.2,
        'ocimene_pct': 0.1, 'camphene_pct': 0.05, 'bisabolol_pct': 0.15
    },
    'OG Kush': {
        'type': 'hybrid',
        'thc_pct': 22.5, 'cbd_pct': 0.3, 'cbg_pct': 0.6, 'cbc_pct': 0.25, 'cbda_pct': 0.1,
        'thcv_pct': 0.4, 'cbn_pct': 0.8, 'delta8_thc_pct': 0.15, 'thca_pct': 21.0,
        'myrcene_pct': 1.8, 'limonene_pct': 1.2, 'pinene_pct': 0.7, 'linalool_pct': 0.6,
        'caryophyllene_pct': 1.4, 'humulene_pct': 0.5, 'terpinolene_pct': 0.15,
        'ocimene_pct': 0.08, 'camphene_pct': 0.04, 'bisabolol_pct': 0.12
    },
    'Sour Diesel': {
        'type': 'sativa',
        'thc_pct': 20.8, 'cbd_pct': 0.2, 'cbg_pct': 0.7, 'cbc_pct': 0.18, 'cbda_pct': 0.12,
        'thcv_pct': 0.5, 'cbn_pct': 0.3, 'delta8_thc_pct': 0.12, 'thca_pct': 19.5,
        'myrcene_pct': 0.8, 'limonene_pct': 2.1, 'pinene_pct': 1.3, 'linalool_pct': 0.3,
        'caryophyllene_pct': 0.9, 'humulene_pct': 0.4, 'terpinolene_pct': 0.25,
        'ocimene_pct': 0.12, 'camphene_pct': 0.06, 'bisabolol_pct': 0.08
    },
    'Girl Scout Cookies': {
        'type': 'hybrid',
        'thc_pct': 24.2, 'cbd_pct': 0.5, 'cbg_pct': 0.8, 'cbc_pct': 0.3, 'cbda_pct': 0.2,
        'thcv_pct': 0.6, 'cbn_pct': 0.6, 'delta8_thc_pct': 0.18, 'thca_pct': 23.0,
        'myrcene_pct': 1.1, 'limonene_pct': 1.4, 'pinene_pct': 0.6, 'linalool_pct': 0.7,
        'caryophyllene_pct': 1.6, 'humulene_pct': 0.45, 'terpinolene_pct': 0.18,
        'ocimene_pct': 0.09, 'camphene_pct': 0.03, 'bisabolol_pct': 0.14
    },
    'Granddaddy Purple': {
        'type': 'indica',
        'thc_pct': 21.5, 'cbd_pct': 0.9, 'cbg_pct': 0.5, 'cbc_pct': 0.22, 'cbda_pct': 0.18,
        'thcv_pct': 0.2, 'cbn_pct': 1.2, 'delta8_thc_pct': 0.08, 'thca_pct': 20.0,
        'myrcene_pct': 2.4, 'limonene_pct': 0.7, 'pinene_pct': 0.5, 'linalool_pct': 1.0,
        'caryophyllene_pct': 1.5, 'humulene_pct': 0.6, 'terpinolene_pct': 0.12,
        'ocimene_pct': 0.06, 'camphene_pct': 0.02, 'bisabolol_pct': 0.18
    },
    'Wedding Cake': {
        'type': 'indica',
        'thc_pct': 25.3, 'cbd_pct': 0.4, 'cbg_pct': 0.9, 'cbc_pct': 0.28, 'cbda_pct': 0.16,
        'thcv_pct': 0.7, 'cbn_pct': 0.9, 'delta8_thc_pct': 0.2, 'thca_pct': 24.0,
        'myrcene_pct': 1.3, 'limonene_pct': 1.8, 'pinene_pct': 0.7, 'linalool_pct': 0.8,
        'caryophyllene_pct': 1.7, 'humulene_pct': 0.55, 'terpinolene_pct': 0.16,
        'ocimene_pct': 0.1, 'camphene_pct': 0.04, 'bisabolol_pct': 0.16
    },
    'Gelato': {
        'type': 'hybrid',
        'thc_pct': 23.8, 'cbd_pct': 0.6, 'cbg_pct': 0.85, 'cbc_pct': 0.26, 'cbda_pct': 0.19,
        'thcv_pct': 0.5, 'cbn_pct': 0.7, 'delta8_thc_pct': 0.16, 'thca_pct': 22.5,
        'myrcene_pct': 1.4, 'limonene_pct': 1.6, 'pinene_pct': 0.8, 'linalool_pct': 0.65,
        'caryophyllene_pct': 1.5, 'humulene_pct': 0.5, 'terpinolene_pct': 0.14,
        'ocimene_pct': 0.09, 'camphene_pct': 0.04, 'bisabolol_pct': 0.13
    },
    'Northern Lights': {
        'type': 'indica',
        'thc_pct': 18.9, 'cbd_pct': 1.2, 'cbg_pct': 0.6, 'cbc_pct': 0.2, 'cbda_pct': 0.14,
        'thcv_pct': 0.2, 'cbn_pct': 1.0, 'delta8_thc_pct': 0.09, 'thca_pct': 17.5,
        'myrcene_pct': 2.1, 'limonene_pct': 0.8, 'pinene_pct': 0.9, 'linalool_pct': 0.9,
        'caryophyllene_pct': 1.3, 'humulene_pct': 0.55, 'terpinolene_pct': 0.11,
        'ocimene_pct': 0.07, 'camphene_pct': 0.03, 'bisabolol_pct': 0.17
    },
    'Jack Herer': {
        'type': 'sativa',
        'thc_pct': 20.3, 'cbd_pct': 0.4, 'cbg_pct': 0.75, 'cbc_pct': 0.21, 'cbda_pct': 0.13,
        'thcv_pct': 0.6, 'cbn_pct': 0.4, 'delta8_thc_pct': 0.14, 'thca_pct': 19.0,
        'myrcene_pct': 0.9, 'limonene_pct': 1.9, 'pinene_pct': 1.6, 'linalool_pct': 0.4,
        'caryophyllene_pct': 1.0, 'humulene_pct': 0.45, 'terpinolene_pct': 0.22,
        'ocimene_pct': 0.11, 'camphene_pct': 0.07, 'bisabolol_pct': 0.09
    },
    'Durban Poison': {
        'type': 'sativa',
        'thc_pct': 18.5, 'cbd_pct': 0.2, 'cbg_pct': 0.9, 'cbc_pct': 0.17, 'cbda_pct': 0.1,
        'thcv_pct': 0.8, 'cbn_pct': 0.2, 'delta8_thc_pct': 0.11, 'thca_pct': 17.0,
        'myrcene_pct': 0.7, 'limonene_pct': 2.0, 'pinene_pct': 1.5, 'linalool_pct': 0.3,
        'caryophyllene_pct': 0.85, 'humulene_pct': 0.35, 'terpinolene_pct': 0.28,
        'ocimene_pct': 0.13, 'camphene_pct': 0.08, 'bisabolol_pct': 0.07
    }
}

# Convert to DataFrame
rows = []
for name, profile in sample_strains.items():
    row = {'strain_name': name}
    row.update(profile)
    rows.append(row)

df = pd.DataFrame(rows)

# Create examples directory if it doesn't exist
examples_dir = Path(__file__).parent
examples_dir.mkdir(exist_ok=True)

# Save to CSV
output_file = examples_dir / 'sample_strains.csv'
df.to_csv(output_file, index=False)

print(f"✓ Created sample database: {output_file}")
print(f"✓ {len(df)} strains")
print(f"✓ {len(df.columns)} columns")
print(f"\nColumns: {', '.join(df.columns)}")
