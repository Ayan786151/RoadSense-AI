"""LLM-powered zone intelligence briefing using Groq/Gemini with key rotation."""

import os
import time
import json
from typing import Dict, Any, Optional, List
from openai import OpenAI


def _load_keys_from_env(env_name: str) -> List[str]:
    """Loads comma-separated keys from environment or .env file."""
    val = os.environ.get(env_name, "")
    if not val:
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_file):
            try:
                with open(env_file, "r") as f:
                    for line in f:
                        if line.strip().startswith(env_name):
                            val = line.split("=", 1)[1].strip().strip('"\'')
                            break
            except Exception:
                pass
    if val:
        return [k.strip() for k in val.split(",") if k.strip()]
    return []


# Dynamic key pools with rotation
GROQ_KEYS = _load_keys_from_env("GROQ_API_KEYS")
GEMINI_KEYS = _load_keys_from_env("GEMINI_API_KEYS")

_groq_idx = 0
_gemini_idx = 0
_cooldowns: Dict[str, float] = {}


def _get_next_key(provider: str = "groq") -> Optional[str]:
    """Round-robin key selection with cooldown tracking."""
    global _groq_idx, _gemini_idx
    keys = GROQ_KEYS if provider == "groq" else GEMINI_KEYS
    if not keys:
        return None

    now = time.time()
    start = _groq_idx if provider == "groq" else _gemini_idx

    for i in range(len(keys)):
        idx = (start + i) % len(keys)
        key = keys[idx]
        if _cooldowns.get(key, 0) <= now:
            if provider == "groq":
                _groq_idx = (idx + 1) % len(keys)
            else:
                _gemini_idx = (idx + 1) % len(keys)
            return key

    return keys[start % len(keys)]  # fallback


def _call_llm(prompt: str, provider: str = "groq", max_tokens: int = 300) -> Optional[str]:
    """Call LLM with automatic key rotation and fallback between providers."""
    providers_to_try = [provider, "gemini" if provider == "groq" else "groq"]

    for prov in providers_to_try:
        key = _get_next_key(prov)
        if not key:
            continue

        try:
            if prov == "groq":
                client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
                model = "llama-3.3-70b-versatile"
            else:
                client = OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
                model = "gemini-2.5-flash"

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise traffic intelligence analyst for a smart city system. Give actionable, data-driven insights in 2-3 sentences. Focus on social impact: citizen safety, time saved, pollution reduced. Never mention AI model names, LLMs, architectures, or training details; present insights directly as an 'AI Review'."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            # Cooldown this key for 60 seconds on rate limit
            if "429" in str(e) or "rate" in str(e).lower():
                _cooldowns[key] = time.time() + 60
            continue

    return None


def generate_zone_briefing(zone_data: Dict[str, Any], signal_data: Dict[str, Any] = None, co2_data: Dict[str, Any] = None) -> str:
    """
    Generates a plain-English intelligence briefing for a zone's current state.
    Falls back to template-based generation if LLM is unavailable.
    """
    zone_id = zone_data.get("zone_id", "Unknown")
    location = zone_data.get("location_name", zone_id)
    congestion = zone_data.get("congestion", 0)
    speed = zone_data.get("average_speed", 0)
    risk_prob = zone_data.get("risk_prob", 0)
    weather = zone_data.get("weather", "Normal")
    violations = zone_data.get("red_light_violations", 0)
    week = zone_data.get("week", 0)
    zone_type = zone_data.get("zone_type", "Urban")
    population = zone_data.get("population_density", 0)
    event = zone_data.get("special_event", 0)

    prompt = f"""Analyze this traffic zone and give a 2-3 sentence actionable briefing:

Zone: {location} ({zone_type})
Week: {week}/52 | Population: {population}/km²
Congestion: {congestion:.1f}/100 | Speed: {speed:.1f} km/h
Risk Probability: {risk_prob*100 if risk_prob else 0:.1f}%
Red-Light Violations: {violations} | Weather: {weather}
Special Event: {"Yes" if event else "No"}"""

    if signal_data:
        prompt += f"\nSignal Timing: {signal_data['base_green_seconds']}s → {signal_data['recommended_green_seconds']}s ({signal_data['urgency']})"

    if co2_data:
        prompt += f"\nWeekly CO2: {co2_data['total_co2_kg_per_week']} kg | Potential Savings: {co2_data['potential_savings_kg_per_week']} kg/week"

    prompt += "\n\nFocus on: citizen safety impact, what action to take, and social benefit."

    result = _call_llm(prompt)

    if result:
        return result

    # Template fallback if LLM unavailable
    return _template_briefing(location, congestion, speed, risk_prob, violations, weather, co2_data)


def _template_briefing(location, congestion, speed, risk_prob, violations, weather, co2_data) -> str:
    """Deterministic template fallback when LLM is unavailable."""
    risk_pct = (risk_prob or 0) * 100

    if risk_pct > 60:
        risk_text = f"{location} is in a critical safety state with {risk_pct:.0f}% incident risk."
    elif risk_pct > 35:
        risk_text = f"{location} shows elevated risk at {risk_pct:.0f}%."
    else:
        risk_text = f"{location} is within safe operating parameters ({risk_pct:.0f}% risk)."

    if congestion > 60:
        action = f"Signal optimization recommended — congestion at {congestion:.0f}/100 with speeds averaging {speed:.0f} km/h."
    elif violations > 8:
        action = f"Enforcement focus needed — {violations} red-light violations recorded."
    else:
        action = "Continue routine monitoring."

    savings = ""
    if co2_data and co2_data.get("potential_savings_kg_per_week", 0) > 10:
        savings = f" Optimizing signals could save {co2_data['potential_savings_kg_per_week']:.0f} kg CO2/week, equivalent to {co2_data['trees_equivalent_per_year']} trees/year."

    return f"{risk_text} {action}{savings}"


def generate_city_summary(zones_data: List[Dict[str, Any]]) -> str:
    """Generates a city-wide daily summary across all zones."""
    if not zones_data:
        return "No zone data available."

    critical_zones = [z for z in zones_data if z.get("risk_prob", 0) > 0.6]
    avg_congestion = sum(z.get("congestion", 0) for z in zones_data) / len(zones_data)
    total_violations = sum(z.get("red_light_violations", 0) for z in zones_data)

    prompt = f"""Generate a 3-4 sentence city-wide traffic intelligence summary:

Total Zones: {len(zones_data)}
Critical Risk Zones: {len(critical_zones)} ({', '.join(z.get('location_name', z.get('zone_id', '')) for z in critical_zones[:5])})
City Average Congestion: {avg_congestion:.1f}/100
Total Red-Light Violations: {total_violations}

Focus on: overall city safety, top priorities, and social impact for citizens."""

    result = _call_llm(prompt, max_tokens=400)
    if result:
        return result

    # Template fallback
    if critical_zones:
        names = ", ".join(z.get("location_name", z.get("zone_id", "")) for z in critical_zones[:3])
        return f"{len(critical_zones)} zones are in critical condition: {names}. City-wide congestion averages {avg_congestion:.0f}/100 with {total_violations} total violations. Immediate attention recommended for high-density residential areas."
    return f"City traffic is within normal parameters. Average congestion: {avg_congestion:.0f}/100."
