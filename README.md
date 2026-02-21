# chi-e2macs-research

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Preparing-orange)]()

**chi-e2macs-research** documents the design exploration, simulations, and early engineering thinking behind **CHI (Cross-platform Heterogeneous Intelligence)** and its scheduling model **E²-MACS**.

> ⚠️ This repository contains research results and simulation records only. It is **not a production-ready framework or runtime**.

---

## 🎯 What is CHI?

**CHI (Cross-platform Heterogeneous Intelligence)** refers to a class of multi-model, edge AI tasks spanning CPU, GPU, and NPU resources, designed to satisfy **user perception and interaction requirements**.
CHI tasks are defined by their **multi-modal processing needs** (e.g., ASR + CV + TTS) and their dependence on **dynamic system constraints** like battery, temperature, and real-time deadlines.

---

## 🎯 What is E²-MACS?

**E²-MACS (End-to-End Multi-Architecture Computing Scheduling)** is an **application-level dynamic scheduler** for CHI tasks.

It **decides where to run each AI task** (CPU/GPU/NPU) and **in which order**, aiming to optimize:

* ✅ **Success Rate** – Meet task deadlines under variable workloads
* ⚡ **Energy Efficiency** – Minimize overall power consumption
* ⚖️ **Resource Utilization** – Balance heterogeneous compute units

> E²-MACS **works alongside OS scheduling**, providing guidance on task-to-resource mapping and execution order. It does **not replace the OS scheduler**.

---

## 📊 Key Simulation Results

Based on extensive simulations (**>1.5 million tasks**) over:

* 20 load levels (λ = 2–40)
* 3 task distributions (uniform, detection-heavy, voice-heavy)
* 6 scheduling strategies (including Oracle upper bound)

| Scenario                         | Load (λ) | E²-MACS   | Best Baseline       | Improvement |
| -------------------------------- | -------- | --------- | ------------------- | ----------- |
| **Detection Heavy** (60% detect) | 20       | **99.0%** | 0.25% (Fastest)     | +98.75%     |
| **Voice Heavy** (60% voice)      | 40       | **99.5%** | 98.3% (LoadBalance) | +1.2%       |
| **Uniform Distribution**         | 30       | **99.7%** | 98.7% (LoadBalance) | +1.0%       |

### Key Insights

* **Prevents catastrophic failure**: FastestResponse (always pick NPU) collapses to 0.25% at λ=20 in detection-heavy workloads, while E²-MACS maintains 99.0%
* **Adapts to workload**: NPU load automatically adjusts from 0.33 (voice-heavy) to 0.62 (detection-heavy)
* **Strategic trade-offs**: Accepts slightly higher latency in detection-heavy scenarios to maintain high success rate

| Metric             | Uniform | Heavy Detect | Heavy Voice |
| ------------------ | ------- | ------------ | ----------- |
| P95 Latency @ λ=20 | 94 ms   | 210 ms       | 113 ms      |
| NPU Load @ λ=20    | 0.30    | 0.62         | 0.33        |
| Energy @ λ=20      | 792 J   | 1108 J       | 650 J       |

> Note: These results are **from simulations only**. Real-world performance may vary depending on hardware, OS scheduling, thermal conditions, and workload variability.

---

## 📁 Repository Structure

```
chi-e2macs-research/
├── docs/          # Simulation datasets, plots and analysis         
└── papers/        # Academic publications (coming soon)
```

> `docs/` is **research records only**. The E²-MACS runtime is **proprietary and not included**.

---

## 🔬 Why Trust These Results

1. **Rigorous Simulation**

   * 20 load levels, 3 task distributions, 6 strategies
   * Fixed random seeds for reproducibility

2. **Theoretical Validation**

   * M/G/1 queueing model (<3% error)
   * Pollaczek-Khinchin formula verification
   * Sensitivity analysis on all parameters

3. **Statistical Significance**

   * 10+ runs per configuration
   * Confidence intervals reported
   * P95/P99 metrics for tail behavior

---

## 📄 Citation

```bibtex
@misc{chi2026e2macs,
  title={E²-MACS: End-to-End Multi-Architecture Computing Scheduling for Heterogeneous Edge AI},
  author={Chi, [chiversal]},
  year={2026},
  howpublished={
    \url{https://gitee.com/chiversal/chi-e2macs-research}
  }
}
```

---

## 📈 What's Next

* [ ] Real-world validation on Hailo-8 / Raspberry Pi 5 (Q2 2026)
* [ ] Additional platforms (RK, Qualcomm, Huawei) (**Hardware donation or cooperative research is welcome**)
* [ ] Distributed / federated scheduling (planned for 2026)

---

## 🤝 Connect

* 📧 Email: zc_linuxer@163.com


---

⭐ Star this repo if you find these research results interesting!

---
