# File Structure Migration Notes

## What Changed

The repository has been reorganized from a flat structure to a clean research-oriented layout.

### Before (Messy)
```
contrastive-pair-gen/
├── contrast_pair_data/          # Mixed with root
│   ├── tom_pairs.json
│   ├── irony_pairs_2.json
│   └── ...
├── contrast_pair_generation/    # Unclear naming
│   ├── tom/
│   ├── irony/
│   └── harmlessness/
├── steering_chat_interface.py   # Root-level experiment
├── testing.ipynb                # Root-level notebook
└── steering_vector_basic_usage.md
```

### After (Clean)
```
contrastive-pair-gen/
├── claude.md                    # 🆕 Main research documentation
├── requirements.txt
│
├── data/                        # 🆕 Clean data organization
│   └── contrast_pairs/
│       ├── theory_of_mind.json  (was: tom_pairs.json)
│       ├── self_other.json      (was: self_other_pairs.json)
│       ├── irony.json           (was: irony_pairs_2.json)
│       └── harmfulness.json     (was: harmfulness_pairs.json)
│
├── data_generation/             (was: contrast_pair_generation)
│   ├── theory_of_mind/         (was: tom/)
│   ├── irony/
│   ├── harmfulness/            (was: harmlessness/)
│   └── self_other/             (potential future addition)
│
├── experiments/                 # 🆕 Experiment isolation
│   ├── steering_interface.py   (was: steering_chat_interface.py)
│   └── notebooks/
│       └── testing.ipynb
│
└── docs/                        # 🆕 Documentation folder
    └── steering_vector_basics.md (was: steering_vector_basic_usage.md)
```

## Key Improvements

### 1. Clear Separation of Concerns
- **data/** - Curated, ready-to-use datasets
- **data_generation/** - Scripts to create/regenerate data
- **experiments/** - Active research code
- **docs/** - Documentation and references

### 2. Consistent Naming
- `tom_pairs.json` → `theory_of_mind.json` (descriptive)
- `irony_pairs_2.json` → `irony.json` (clean)
- `harmlessness/` → `harmfulness/` (accurate)

### 3. Research-Oriented Documentation
- **claude.md** - Comprehensive research overview
- Explains motivation, hypotheses, and experimental design
- Documents all contrast pair types and their purpose

### 4. Updated References
- `experiments/steering_interface.py` now uses `Path` for robust file loading
- All paths updated to reference `data/contrast_pairs/`

## Migration Checklist

✅ Created new directory structure
✅ Moved and renamed data files
✅ Reorganized generation scripts
✅ Moved experiment files to experiments/
✅ Moved documentation to docs/
✅ Updated steering_interface.py paths
✅ Created comprehensive claude.md
✅ Removed old directories

## Files Removed

The following temporary files I created were removed per your request:
- ~~train_steering_vectors.py~~ (not needed yet)
- ~~steering_chat_interface_fast.py~~ (not needed yet)
- ~~USAGE_GUIDE.md~~ (replaced by claude.md)

## Next Steps

1. Review [claude.md](claude.md) for complete research overview
2. Run experiments from `experiments/` directory:
   ```bash
   cd experiments
   python steering_interface.py
   ```
3. Data generation scripts remain in `data_generation/` if you need to regenerate
4. All curated data is in `data/contrast_pairs/` (clean, version-controlled)

## Why This Structure is Better

### For Research
- Clear distinction between data generation (one-time) and experiments (iterative)
- Documentation lives with the code
- Easy to understand project goals from claude.md

### For Development
- Predictable file locations
- Clean imports and path handling
- Separation allows independent testing of components

### For Collaboration
- Self-documenting structure
- Clear research motivation in claude.md
- Easy onboarding for new contributors

---

**All changes preserve existing functionality** - only paths and organization changed, not the actual code logic.
