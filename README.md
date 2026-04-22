# Virtual ICU - AI-Driven Monitoring System 🏥

> **Educational demonstration** of an AI-powered ICU patient monitoring system with early warning predictions for critical care scenarios.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/paracelsus12-crypto/virtual-icu-demo/actions/workflows/tests.yml/badge.svg)](https://github.com/paracelsus12-crypto/virtual-icu-demo/actions)

## Features ✨

- 🏥 **6 Critical Clinical Scenarios** — Sepsis, Cardiac Arrest, Respiratory Failure, Hypotension, Hypoxemia, Arrhythmias
- 🤖 **AI-Powered Predictions** — NEWS2, qSOFA, CART clinical scoring systems
- 📊 **Real-Time Vital Signs Visualization** — Interactive Plotly charts with live updates
- 🎓 **Interactive Teaching Interface** — "Invigilator Control Panel" for instructor demonstrations
- 💻 **Local Deployment** — No cloud required, runs on any computer with Docker
- 🔐 **Privacy-First** — 100% synthetic patient data, fully anonymized
- 📚 **Comprehensive Documentation** — User guides, developer docs, clinical validation notes

## Quick Start 🚀

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/paracelsus12-crypto/virtual-icu-demo.git
cd virtual-icu-demo

# Run with Docker Compose
docker-compose up

# Visit http://localhost:8501 in your browser
```

### Option 2: Local Python Environment

```bash
# Clone repository
git clone https://github.com/paracelsus12-crypto/virtual-icu-demo.git
cd virtual-icu-demo

# Create virtual environment
python -m venv venv

# Activate
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app/app.py
```

The app will open at `http://localhost:8501`

## Project Structure 📁

```
virtual-icu-demo/
├── synthetic_data_generator/      # Patient data synthesis
│   ├── config.py                  # Configuration & parameters
│   ├── base_generator.py          # Abstract base class
│   ├── scenario_generators/       # 6 clinical scenarios
│   ├── physiology/                # Physiological correlations
│   ├── validators/                # Data validation
│   └── exporters/                 # CSV, JSON exports
│
├── scoring/                       # Clinical scoring systems
│   ├── news2.py                   # NEWS2 implementation
│   ├── qsofa.py                   # qSOFA scoring
│   ├── cart.py                    # CART risk stratification
│   └── tests/                     # Unit tests
│
├── ml/                            # Machine learning models
│   ├── pipeline.py                # Training/inference pipeline
│   ├── models/                    # Model implementations
│   └── tests/
│
├── streamlit_app/                 # Web interface
│   ├── app.py                     # Main app entry
│   ├── pages/                     # Multi-page dashboard
│   │   ├── dashboard.py
│   │   ├── monitoring.py
│   │   ├── invigilator.py
│   │   └── education.py
│   ├── utils/                     # Helpers
│   └── assets/                    # Images, CSS
│
├── alerts/                        # Alert generation engine
├── tests/                         # Integration tests
├── docs/                          # Documentation
├── data/                          # Datasets & models
│
├── Dockerfile                     # Docker image
├── docker-compose.yml             # Multi-container setup
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
└── README.md                      # This file
```

## Clinical Scenarios 🏥

### 1️⃣ **SEPSIS** (Septic Shock)
- **Timeline:** 4-8 hours to critical state
- **Progression:** Normal → Fever/Tachycardia → Hypotension → Shock
- **Key Parameters:** Heart rate ↑, BP ↓, Lactate ↑↑, pH ↓
- **Learning Focus:** SOFA scoring, bundle therapy (antibiotics, fluids, pressors)

### 2️⃣ **CARDIAC ARREST** (Cardiopulmonary Emergency)
- **Timeline:** Minutes (4-6 until brain damage)
- **Types:** VF, PEA, Asystole
- **Key Interventions:** BLS (100-120 compressions/min), Defibrillation, ACLS
- **Learning Focus:** Rhythm recognition, CPR technique, medication dosing

### 3️⃣ **RESPIRATORY FAILURE** (Ventilatory Insufficiency)
- **Types:** Type 1 (Hypoxemic), Type 2 (Hypercapnic)
- **Progression:** SpO₂ 98% → 75%, RR ↑↑, respiratory muscle fatigue
- **Interventions:** O₂ therapy, CPAP/BiPAP, mechanical ventilation, intubation
- **Learning Focus:** ABG interpretation, ventilator modes, weaning parameters

### 4️⃣ **HYPOTENSION** (Shock States)
- **Definition:** SBP < 90 mmHg with inadequate perfusion
- **Types:** Hypovolemic, Cardiogenic, Distributive (septic), Obstructive
- **Markers:** MAP < 65, Urine output ↓, Lactate ↑, Altered mental status
- **Learning Focus:** Fluid resuscitation, vasopressors, MAP targets

### 5️⃣ **HYPOXEMIA** (Low Oxygen)
- **Definition:** PaO₂ < 60 mmHg or SpO₂ < 88%
- **Causes:** Hypoventilation, V/Q mismatch, Diffusion impairment
- **Compensation:** Tachypnea (RR ↑), Tachycardia (HR ↑), Cyanosis
- **Learning Focus:** O₂ masks, CPAP/BiPAP, prone positioning, suctioning

### 6️⃣ **ARRHYTHMIAS** (Abnormal Heart Rhythms)
- **Critical Types:** VF, VT, Severe Bradycardia, Rapid AF
- **Parameters:** HR, Regularity, QRS width, P waves, ST changes
- **Interventions:** ACLS algorithms, Cardioversion, Pacing, Medications
- **Learning Focus:** ECG interpretation, ACLS protocols, drug dosing

## Clinical Validation 📋

All vital sign ranges, progression timelines, and clinical endpoints are based on:
- ✅ Evidence-based clinical literature
- ✅ Published scoring systems (NEWS2, qSOFA, CART)
- ✅ Expert clinician review & validation
- ✅ Reference implementation validation

## Using Virtual ICU 🎓

### For Educators
```
1. Launch Invigilator Control Panel
2. Select clinical scenario (e.g., Sepsis)
3. Simulate patient deterioration (T+0 → T+8 hours)
4. Students observe vital signs, scores, alerts in real-time
5. Discuss clinical decision-making at each timepoint
```

### For Self-Study Students
```
1. Select "Education Mode"
2. Choose a scenario difficulty (Easy → Expert)
3. View vital signs & predict next steps
4. Get instant feedback & clinical explanations
5. Link to evidence-based guidelines
```

### For Researchers
```
1. Export 100+ synthetic patients (CSV/JSON)
2. Train/validate ML models
3. Benchmark against clinical scoring systems
4. Publish results with proper attribution
```

## Technical Stack 🛠️

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.10+ |
| **Framework** | Streamlit | 1.28+ |
| **Visualization** | Plotly | 5.18+ |
| **Data Processing** | Pandas, NumPy | Latest |
| **Machine Learning** | Scikit-learn, TensorFlow | Latest |
| **Containerization** | Docker | Latest |
| **Testing** | pytest | 7.4+ |
| **CI/CD** | GitHub Actions | - |

## Installation 📦

### Prerequisites
- Python 3.10 or higher
- Docker (optional, for containerized deployment)
- Git

### Development Setup

```bash
# Clone repo
git clone https://github.com/paracelsus12-crypto/virtual-icu-demo.git
cd virtual-icu-demo

# Create venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov

# Run app
streamlit run streamlit_app/app.py
```

## Documentation 📚

- **[INSTALLATION.md](docs/INSTALLATION.md)** — Detailed setup guide
- **[USER_GUIDE.md](docs/USER_GUIDE.md)** — How to use the system
- **[DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** — Contributing to the project
- **[CLINICAL_VALIDATION.md](docs/CLINICAL_VALIDATION.md)** — Validation methodology
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** — Technical API documentation

## Testing 🧪

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=synthetic_data_generator --cov=scoring --cov=ml

# Run specific test file
pytest tests/scoring/test_news2.py -v

# Run with markers
pytest -m "clinical" -v  # Only clinical validation tests
```

## Contributing 🤝

We welcome contributions! Please see [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for:
- Code style guidelines (Black, Flake8)
- Branch strategy (main, develop, feature/*)
- Pull request process
- Testing requirements

## License 📄

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

Permission is granted for:
- ✅ Educational use
- ✅ Research & publication
- ✅ Commercial adaptation (with attribution)
- ✅ Modification & distribution

## Citation 📚

If you use Virtual ICU in research or education, please cite:

```bibtex
@software{virtual_icu_2024,
  title = {Virtual ICU: AI-Driven Monitoring System for Educational Purposes},
  author = {Paracelsus12 Team},
  year = {2024},
  url = {https://github.com/paracelsus12-crypto/virtual-icu-demo},
  note = {Educational demonstration platform}
}
```

## Acknowledgments 🙏

Built with inspiration from:
- Royal College of Physicians (NEWS2 scoring)
- Sepsis-3 consensus definitions (qSOFA)
- Emergency Medicine literature (CART scoring)
- Clinical care best practices

## Troubleshooting 🔧

### Issue: "Streamlit port 8501 already in use"
```bash
# Kill the process
lsof -i :8501 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Or use different port
streamlit run streamlit_app/app.py --server.port 8502
```

### Issue: Docker build fails
```bash
# Clean and rebuild
docker-compose down
docker system prune -a
docker-compose up --build
```

### Issue: Import errors
```bash
# Reinstall requirements
pip install --upgrade --force-reinstall -r requirements.txt
```

## Roadmap 🗺️

- [x] **Phase 1:** Synthetic data generation + Clinical scoring
- [ ] **Phase 2:** ML models for early prediction
- [ ] **Phase 3:** Real-time hospital data integration
- [ ] **Phase 4:** Multi-patient dashboard for teams
- [ ] **Phase 5:** Clinical validation study

## Contact & Support 📧

- **Issues:** Use [GitHub Issues](https://github.com/paracelsus12-crypto/virtual-icu-demo/issues)
- **Discussions:** Use [GitHub Discussions](https://github.com/paracelsus12-crypto/virtual-icu-demo/discussions)
- **Email:** paracelsus12@example.com

---

**Last Updated:** April 2024  
**Status:** 🔴 Phase 1 - Alpha (Pre-release)  
**Next Release:** Phase 2 with ML models
