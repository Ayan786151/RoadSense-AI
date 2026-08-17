"""
================================================================================
ROAD SENSE AI - DIGITAL E-CHALLAN & MUNICIPAL ENFORCEMENT GENERATOR
MODULE: AUTOMATED TRAFFIC CITATION & LEGAL EVIDENCE ENGINE
================================================================================

This module generates legally formatted Digital E-Challan citations from detected
traffic safety violations (No-Helmet, Red-Light Breaking, Triple-Riding, Speeding):
1. Assigns unique Challan Number (e.g. DL-2026-RS-849201).
2. Maps violation type to Motor Vehicles Act (MVA) penal sections and fine amounts.
3. Generates printable, high-aesthetic HTML E-Challan tickets with QR verification.
4. Exports digital ticket records and bulk enforcement summaries.
================================================================================
"""

import os
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


# Penal codes and fine structure (Motor Vehicles Amendment Act)
PENAL_CODE_DIRECTORY = {
    "NO_HELMET": {
        "title": "Driving Two-Wheeler Without Protective Headgear (Helmet)",
        "section": "Section 129 read with Section 194D, MVA 1988",
        "fine_inr": 1000,
        "points": 3,
        "penalty_description": "₹1,000 fine and possible disqualification of driving license for 3 months."
    },
    "RED_LIGHT_RUNNING": {
        "title": "Violation of Mandatory Traffic Signal (Red Light Jumping)",
        "section": "Section 119 read with Section 177A & 184, MVA 1988",
        "fine_inr": 5000,
        "points": 5,
        "penalty_description": "₹5,000 fine for dangerous driving / jumping automated traffic light signal."
    },
    "TRIPLE_RIDING": {
        "title": "Carrying More Than One Pillion Rider on Two-Wheeler",
        "section": "Section 128 read with Section 194C, MVA 1988",
        "fine_inr": 1000,
        "points": 2,
        "penalty_description": "₹1,000 fine and suspension of license."
    },
    "SPEED_VIOLATION": {
        "title": "Exceeding Prescribed Speed Limit (Over-Speeding)",
        "section": "Section 112 read with Section 183, MVA 1988",
        "fine_inr": 2000,
        "points": 4,
        "penalty_description": "₹2,000 fine for Light Motor Vehicles exceeding zone speed limit."
    }
}


def generate_challan_id(track_id: int, violation_type: str, timestamp: float) -> str:
    """Generates a verifiable unique E-Challan number."""
    raw = f"{track_id}-{violation_type}-{timestamp}-{time.time()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:6].upper()
    return f"RS-2026-{digest}"


def create_echallan_record(
    violation_dict: Dict[str, Any],
    location_name: str = "MG Road Junction, Bengaluru",
    camera_id: str = "CAM-BLR-04",
    officer_id: str = "AI-ENFORCER-V1"
) -> Dict[str, Any]:
    """
    Constructs a structured E-Challan document from a detected violation.
    """
    v_type = violation_dict.get("violation_type", "NO_HELMET")
    t_id = violation_dict.get("track_id", 0)
    ts = violation_dict.get("timestamp_seconds", 0.0)
    conf = violation_dict.get("confidence", "90%")

    penal_info = PENAL_CODE_DIRECTORY.get(v_type, {
        "title": "Traffic Safety Infraction",
        "section": "Section 177, Motor Vehicles Act",
        "fine_inr": 500,
        "points": 1,
        "penalty_description": "General penalty for traffic violation."
    })

    challan_id = generate_challan_id(t_id, v_type, ts)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    record = {
        "challan_id": challan_id,
        "violation_type": v_type,
        "offense_title": penal_info["title"],
        "mva_section": penal_info["section"],
        "fine_amount_inr": penal_info["fine_inr"],
        "penalty_points": penal_info["points"],
        "penalty_details": penal_info["penalty_description"],
        "track_id": t_id,
        "vehicle_type": violation_dict.get("vehicle_type", "2-Wheeler"),
        "timestamp_seconds": ts,
        "confidence": conf,
        "location_name": location_name,
        "camera_id": camera_id,
        "issued_at": now_iso,
        "payment_status": "UNPAID (Notice Pending)",
        "verification_hash": hashlib.md5(f"{challan_id}-{penal_info['fine_inr']}".encode()).hexdigest().upper()
    }
    return record


def render_echallan_html(challan: Dict[str, Any]) -> str:
    """
    Renders a print-ready, professional digital traffic citation ticket.
    """
    html_ticket = f"""
    <div style="background: #ffffff; color: #1e293b; border: 2px solid #0f172a; border-radius: 12px; padding: 24px; max-width: 650px; margin: 15px auto; font-family: 'Segoe UI', Arial, sans-serif; box-shadow: 0 10px 25px rgba(0,0,0,0.15);">
        <!-- Header -->
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px;">
            <div>
                <div style="font-size: 18px; font-weight: 800; color: #0f172a; letter-spacing: 0.5px;">DEPARTMENT OF TRAFFIC POLICE & CIVIC SAFETY</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 2px;">AUTOMATED AI E-CHALLAN ENFORCEMENT CITATION</div>
            </div>
            <div style="background: #fee2e2; color: #ef4444; font-weight: 700; font-size: 12px; padding: 4px 10px; border-radius: 6px; border: 1px solid #fca5a5;">
                {challan['payment_status']}
            </div>
        </div>

        <!-- Challan Details Grid -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 18px 0; font-size: 13px;">
            <div>
                <span style="color: #64748b;">Challan Number:</span><br>
                <strong style="color: #0f172a; font-size: 15px;">{challan['challan_id']}</strong>
            </div>
            <div>
                <span style="color: #64748b;">Date & Time:</span><br>
                <strong>{challan['issued_at']}</strong>
            </div>
            <div>
                <span style="color: #64748b;">Location / Camera ID:</span><br>
                <strong>{challan['location_name']} ({challan['camera_id']})</strong>
            </div>
            <div>
                <span style="color: #64748b;">Detected Target:</span><br>
                <strong>{challan['vehicle_type']} (Track ID: #{challan['track_id']})</strong>
            </div>
        </div>

        <!-- Violation Box -->
        <div style="background: #f8fafc; border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 4px; margin-bottom: 18px;">
            <div style="font-size: 14px; font-weight: 700; color: #b91c1c;">🚨 Offense: {challan['offense_title']}</div>
            <div style="font-size: 12px; color: #334155; margin-top: 4px;"><b>Legal Section:</b> {challan['mva_section']}</div>
            <div style="font-size: 12px; color: #64748b; margin-top: 2px;">{challan['penalty_details']}</div>
        </div>

        <!-- Fine Amount -->
        <div style="display: flex; justify-content: space-between; align-items: center; background: #0f172a; color: #ffffff; padding: 14px 20px; border-radius: 8px;">
            <div>
                <div style="font-size: 11px; opacity: 0.8; text-transform: uppercase;">Total Penalty Fine Amount</div>
                <div style="font-size: 24px; font-weight: 800; color: #38bdf8;">₹{challan['fine_amount_inr']:,} INR</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 11px; opacity: 0.8;">Verification Hash</div>
                <div style="font-size: 11px; font-family: monospace; color: #a5f3fc;">{challan['verification_hash']}</div>
            </div>
        </div>

        <!-- Footer Notice -->
        <div style="margin-top: 16px; font-size: 11px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 10px;">
            Generated automatically by RoadSense AI Vision Enforcement Pipeline. Pay online within 15 days to avoid judicial summons.
        </div>
    </div>
    """
    return html_ticket
