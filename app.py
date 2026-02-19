import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st


def sigmoid(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class EmployerType:
    name: str
    baseline_utilization_rate: float
    habit_mean: float
    trust_mean: float
    price_sensitivity_mean: float
    convenience_sensitivity_mean: float
    social_sensitivity_mean: float


EMPLOYER_TYPES = [
    EmployerType(
        name="Blue collar",
        baseline_utilization_rate=0.22,
        habit_mean=0.70,
        trust_mean=0.40,
        price_sensitivity_mean=0.55,
        convenience_sensitivity_mean=0.60,
        social_sensitivity_mean=0.50,
    ),
    EmployerType(
        name="White collar",
        baseline_utilization_rate=0.18,
        habit_mean=0.62,
        trust_mean=0.48,
        price_sensitivity_mean=0.50,
        convenience_sensitivity_mean=0.55,
        social_sensitivity_mean=0.40,
    ),
    EmployerType(
        name="Mixed workforce",
        baseline_utilization_rate=0.20,
        habit_mean=0.66,
        trust_mean=0.45,
        price_sensitivity_mean=0.52,
        convenience_sensitivity_mean=0.58,
        social_sensitivity_mean=0.45,
    ),
    EmployerType(
        name="Rural spread out",
        baseline_utilization_rate=0.20,
        habit_mean=0.68,
        trust_mean=0.43,
        price_sensitivity_mean=0.56,
        convenience_sensitivity_mean=0.62,
        social_sensitivity_mean=0.42,
    ),
]


@dataclass
class AdoptionInputs:
    ease: int
    safety: int
    speed: int
    visibility: int
    reward: int
    authority: int
    loss_framing: int
    identity: int


def adoption_index(inputs: AdoptionInputs) -> float:
    # 0 to 100
    # Primary drivers weighted most
    score_100 = (
        0.28 * inputs.ease
        + 0.24 * inputs.safety
        + 0.18 * inputs.speed
        + 0.12 * inputs.visibility
        + 0.08 * inputs.reward
        + 0.05 * inputs.authority
        + 0.03 * inputs.loss_framing
        + 0.02 * inputs.identity
    ) * 5.0
    return clamp(score_100, 0.0, 100.0)


def adoption_probability(inputs: AdoptionInputs) -> float:
    # Convert inputs (0-20 each) into probability
    # Ease and Safety dominate, Speed next, Visibility next, Reward modest
    x = (
        0.09 * inputs.ease
        + 0.08 * inputs.safety
        + 0.07 * inputs.speed
        + 0.05 * inputs.visibility
        + 0.03 * inputs.reward
        + 0.03 * inputs.authority
        + 0.02 * inputs.loss_framing
        + 0.01 * inputs.identity
        - 5.2
    )
    return sigmoid(x)


@dataclass
class AvoidedCostAssumptions:
    er_events_per_1000_per_month: float
    imaging_events_per_1000_per_month: float
    procedure_events_per_1000_per_month: float
    er_legacy_cost: float
    er_dvg_cost: float
    imaging_legacy_cost: float
    imaging_dvg_cost: float
    procedure_legacy_cost: float
    procedure_dvg_cost: float
    reward_share_of_avoided: float


def compute_avoided_cost(
    lives: int,
    months: int,
    dvg_adoption_share_by_month: List[float],
    a: AvoidedCostAssumptions,
) -> Dict[str, float]:
    total_avoided = 0.0
    total_rewards = 0.0
    total_redirected_events = 0.0

    period_months = min(months, len(dvg_adoption_share_by_month))
    for m in range(period_months):
        share = dvg_adoption_share_by_month[m]
        factor = lives / 1000.0

        er_events = a.er_events_per_1000_per_month * factor * share
        imaging_events = a.imaging_events_per_1000_per_month * factor * share
        procedure_events = a.procedure_events_per_1000_per_month * factor * share

        er_avoided = er_events * max(0.0, a.er_legacy_cost - a.er_dvg_cost)
        imaging_avoided = imaging_events * max(0.0, a.imaging_legacy_cost - a.imaging_dvg_cost)
        procedure_avoided = procedure_events * max(0.0, a.procedure_legacy_cost - a.procedure_dvg_cost)

        month_avoided = er_avoided + imaging_avoided + procedure_avoided
        month_rewards = month_avoided * clamp(a.reward_share_of_avoided, 0.0, 1.0)

        total_avoided += month_avoided
        total_rewards += month_rewards
        total_redirected_events += er_events + imaging_events + procedure_events

    avoided_per_life = total_avoided / max(1, lives)
    rewards_per_life = total_rewards / max(1, lives)

    return {
        "total_avoided": total_avoided,
        "total_rewards": total_rewards,
        "avoided_per_life": avoided_per_life,
        "rewards_per_life": rewards_per_life,
        "redirected_events": total_redirected_events,
    }


def simulate_adoption_path(
    inputs: AdoptionInputs,
    employer: EmployerType,
    months: int,
) -> Tuple[List[float], List[float]]:
    # Social proof grows as adoption happens, and also as visibility improves
    # Visibility comes from the avoided cost engine reporting, testimonials, dashboards
    base_p = adoption_probability(inputs)

    dvg_share_by_month: List[float] = []
    social_proof_by_month: List[float] = []

    social_proof = 0.02
    trust_uplift = (inputs.safety + inputs.authority + inputs.identity) / 60.0
    friction_relief = inputs.ease / 20.0
    speed_relief = inputs.speed / 20.0

    for _ in range(months):
        # simple compounding adoption model:
        # adoption share increases based on base_p, social proof, and habit resistance
        # employer habit resistance approximated from habit_mean
        habit_resistance = employer.habit_mean

        visibility_boost = (inputs.visibility / 20.0) * 0.20
        social_boost = social_proof * (0.25 + 0.35 * (inputs.visibility / 20.0))

        month_p = clamp(base_p + visibility_boost + social_boost, 0.01, 0.95)

        # convert probability to share change
        # higher habit means slower conversion
        delta = (month_p - 0.20) * (0.10 + 0.25 * friction_relief + 0.20 * speed_relief + 0.15 * trust_uplift)
        delta = delta * (1.0 - 0.70 * habit_resistance)

        prev = dvg_share_by_month[-1] if dvg_share_by_month else employer.baseline_utilization_rate
        new_share = clamp(prev + delta, 0.0, 0.95)
        dvg_share_by_month.append(new_share)

        # update social proof with diffusion and visibility
        social_proof = clamp(social_proof + 0.30 * new_share * (1.0 - social_proof), 0.0, 1.0)
        social_proof_by_month.append(social_proof)

    return dvg_share_by_month, social_proof_by_month


st.set_page_config(page_title="DVG Adoption and Savings Simulator", layout="wide")
st.title("DVG Health - Adoption and Savings Simulator")

colL, colR = st.columns([1, 1])

with colL:
    st.subheader("Inputs (0 to 20)")
    ease = st.slider("Ease (less hassle and fewer steps)", 0, 20, 12)
    safety = st.slider("Safety (no surprise bills, protected experience)", 0, 20, 12)
    speed = st.slider("Speed (faster access to care)", 0, 20, 12)
    visibility = st.slider("Visibility (people can see savings and results)", 0, 20, 10)
    reward = st.slider("Reward (immediate personal benefit)", 0, 20, 8)
    authority = st.slider("Authority (doctor, HR, and leadership endorsement)", 0, 20, 10)
    loss_framing = st.slider("Loss framing (clear cost of doing nothing)", 0, 20, 8)
    identity = st.slider("Identity (fits culture and values)", 0, 20, 8)

    employer_name = st.selectbox("Employer type", [e.name for e in EMPLOYER_TYPES], index=2)
    employer = next(e for e in EMPLOYER_TYPES if e.name == employer_name)

    lives = st.number_input("Covered lives", min_value=500, max_value=250000, value=10000, step=500)
    months = st.number_input("Months to simulate", min_value=6, max_value=60, value=24, step=6)

inputs = AdoptionInputs(
    ease=ease,
    safety=safety,
    speed=speed,
    visibility=visibility,
    reward=reward,
    authority=authority,
    loss_framing=loss_framing,
    identity=identity,
)

lives_i = int(lives)
months_i = int(months)

idx = adoption_index(inputs)
p = adoption_probability(inputs)

with colR:
    st.subheader("Summary")
    st.write(f"DVG Adoption Index: {idx:.1f} out of 100")
    st.write(f"Base adoption probability (before compounding): {p*100:.1f} percent")

    if idx < 55:
        st.warning("Adoption likely stalls. Ease and safety usually fix this fastest.")
    elif idx < 75:
        st.info("Moderate adoption. Improve ease, safety, or speed to get compounding.")
    else:
        st.success("Strong compounding adoption likely if the member experience stays consistent.")

st.divider()
st.subheader("Adoption path over time")

dvg_share, social_proof = simulate_adoption_path(inputs, employer, months_i)

st.line_chart(
    {"DVG share (monthly)": dvg_share, "Social proof": social_proof},
    height=320,
)

st.divider()
st.subheader("Avoided Cost Accounting Engine")

colA, colB, colC = st.columns([1, 1, 1])

with colA:
    er_events = st.number_input("ER events per 1000 lives per month", min_value=0.0, max_value=50.0, value=6.0, step=0.5)
    imaging_events = st.number_input("Imaging events per 1000 lives per month", min_value=0.0, max_value=80.0, value=14.0, step=1.0)
    procedure_events = st.number_input("Procedure episodes per 1000 lives per month", min_value=0.0, max_value=30.0, value=2.5, step=0.5)

with colB:
    er_legacy = st.number_input("Legacy ER average cost", min_value=0.0, max_value=20000.0, value=2200.0, step=100.0)
    er_dvg = st.number_input("DVG ER diversion pathway cost", min_value=0.0, max_value=20000.0, value=450.0, step=50.0)
    imaging_legacy = st.number_input("Legacy imaging average cost", min_value=0.0, max_value=20000.0, value=1200.0, step=100.0)
    imaging_dvg = st.number_input("DVG imaging pathway cost", min_value=0.0, max_value=20000.0, value=450.0, step=50.0)

with colC:
    procedure_legacy = st.number_input("Legacy procedure episode average cost", min_value=0.0, max_value=200000.0, value=28000.0, step=500.0)
    procedure_dvg = st.number_input("DVG bundled procedure episode cost", min_value=0.0, max_value=200000.0, value=19000.0, step=500.0)
    reward_share = st.slider("Share of avoided cost returned as rewards", 0.0, 0.5, 0.15, 0.01)

assumptions = AvoidedCostAssumptions(
    er_events_per_1000_per_month=er_events,
    imaging_events_per_1000_per_month=imaging_events,
    procedure_events_per_1000_per_month=procedure_events,
    er_legacy_cost=er_legacy,
    er_dvg_cost=er_dvg,
    imaging_legacy_cost=imaging_legacy,
    imaging_dvg_cost=imaging_dvg,
    procedure_legacy_cost=procedure_legacy,
    procedure_dvg_cost=procedure_dvg,
    reward_share_of_avoided=reward_share,
)

ac = compute_avoided_cost(lives=lives_i, months=months_i, dvg_adoption_share_by_month=dvg_share, a=assumptions)

st.write(f"Redirected events (approx): {ac['redirected_events']:.0f}")
st.write(f"Total avoided cost: ${ac['total_avoided']:,.0f}")
st.write(f"Avoided cost per life: ${ac['avoided_per_life']:,.0f}")
st.write(f"Total rewards funded: ${ac['total_rewards']:,.0f}")
st.write(f"Rewards per life: ${ac['rewards_per_life']:,.0f}")

st.divider()
st.subheader("Stress test across employer types")

results = []
for e in EMPLOYER_TYPES:
    share_e, _ = simulate_adoption_path(inputs, e, months_i)
    results.append((e.name, share_e[0], share_e[-1]))

for name, m1, mN in results:
    st.write(f"{name}: month 1 share {m1*100:.1f} percent, month {months_i} share {mN*100:.1f} percent")
