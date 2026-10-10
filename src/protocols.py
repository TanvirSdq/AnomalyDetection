"""
protocols.py — NASA Flight Surgeon Protocol Recommender
KUET_CHAYAPOTH | NASA Space Apps 2026
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RecommendedProtocol:
    code: str
    title: str
    urgency: str  # ROUTINE, ELEVATED, IMMEDIATE
    operational_action: str
    rationale: str


class NASAProtocolEngine:
    """
    Translates explainable anomaly states into non-diagnostic,
    NASA flight-rule grounded operational recommendations.
    """

    PROTOCOLS: Dict[str, RecommendedProtocol] = {
        "ACUTE_PHYSIOLOGICAL_STRAIN": RecommendedProtocol(
            code="FS-CARDIO-02",
            title="Cardiovascular Deconditioning & Strain Check",
            urgency="ELEVATED",
            operational_action="Initiate 500ml oral fluid intake. Rest from current strenuous EVA/workstation task for 15 minutes. Check suit/cabin CO2 levels.",
            rationale="Sustained heart rate elevation coupled with autonomic vagal withdrawal (depressed RMSSD) indicates acute cardiovascular or thermal load."
        ),
        "PARASYMPATHETIC_RECOVERY": RecommendedProtocol(
            code="FS-RECOVERY-01",
            title="Nominal Rest & Recovery Verification",
            urgency="ROUTINE",
            operational_action="No corrective action required. Active alarms suppressed.",
            rationale="Heart rate deceleration paired with elevated RMSSD corresponds to intentional relaxation, meditation, or restorative sleep."
        ),
        "TACHYCARDIA_ACTIVE_LOAD": RecommendedProtocol(
            code="FS-LOAD-01",
            title="Workload / Activity Cross-Check",
            urgency="ROUTINE",
            operational_action="Confirm correlation with scheduled cycle ergometer / ARED resistive exercise. If unscheduled, monitor for 10 minutes.",
            rationale="Isolated elevated heart rate without autonomic collapse typically corresponds to physical exertion."
        ),
        "BRADYCARDIA_EXTREME_REST": RecommendedProtocol(
            code="FS-NEURO-03",
            title="Deep Sleep / Neurovestibular Check",
            urgency="ROUTINE",
            operational_action="Verify crew member awake state and cabin oxygen partial pressure.",
            rationale="Marked bradycardia without recovery dynamics."
        ),
        "ATYPICAL_MULTI_SIGNAL_DEVIATION": RecommendedProtocol(
            code="FS-SENSOR-04",
            title="Wearable Sensor Lead & Coupling Verification",
            urgency="ROUTINE",
            operational_action="Inspect biometric sensor electrode contact, battery status, and skin impedance.",
            rationale="Atypical multi-signal decoupling often stems from partial sensor displacement or intermittent telemetry packet drops."
        )
    }

    @classmethod
    def get_recommendation(cls, state_type: str) -> RecommendedProtocol:
        return cls.PROTOCOLS.get(state_type, cls.PROTOCOLS["ATYPICAL_MULTI_SIGNAL_DEVIATION"])
