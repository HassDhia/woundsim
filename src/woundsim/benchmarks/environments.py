"""Benchmark environment configurations."""

BENCHMARK_ENVS = [
    {
        "env_id": "woundsim/WoundMacrophage-v0",
        "name": "Macrophage Polarization",
        "difficulties": ["easy", "medium", "hard"],
        "default_difficulty": "medium",
    },
    {
        "env_id": "woundsim/WoundIschemic-v0",
        "name": "Ischemic Wound",
        "difficulties": ["mild", "moderate", "severe"],
        "default_difficulty": "moderate",
    },
    {
        "env_id": "woundsim/WoundHBOT-v0",
        "name": "HBOT Angiogenesis",
        "difficulties": ["acute", "chronic", "non-healing"],
        "default_difficulty": "chronic",
    },
    {
        "env_id": "woundsim/WoundDiabetic-v0",
        "name": "Diabetic Wound",
        "difficulties": ["well-controlled", "moderate", "uncontrolled"],
        "default_difficulty": "moderate",
    },
]
