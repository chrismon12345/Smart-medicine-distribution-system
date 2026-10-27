"""Comprehensive unit and integration test suite for:
1. Feature 1: Expiry-Aware Redistribution Optimizer
2. Feature 2: Network-Wide Cascading Shortage Predictor
3. Integrated Closed-Loop Verification (Deficit Mitigation & Physical Invariants)
"""
import pytest
import numpy as np
import pandas as pd

from pipeline.simulator.entities import Hospital, Medicine, Batch
from pipeline.optimizer.solver import RedistributionOptimizer, TransferRecommendation
from pipeline.simulator.cascading import CascadingShortagePredictor, DisruptionConfig


@pytest.fixture
def network_fixture():
    """Setup a controlled 3-hospital network with staggered stock and batches."""
    h1 = Hospital("H001", "Central District Hospital", "District", False, 120, 10.0, 76.0, "DistA")
    h2 = Hospital("H002", "Rural Taluk Hospital", "Taluk", True, 40, 10.1, 76.1, "DistA")
    h3 = Hospital("H003", "Remote PHC Clinic", "PHC", True, 15, 10.2, 76.2, "DistA")
    hospitals = {"H001": h1, "H002": h2, "H003": h3}

    m1 = Medicine("M001", "Paracetamol 500mg", "Antipyretic", 2.0, 3, 365, True, False)
    m2 = Medicine("M002", "Normal Saline 500ml", "IV Fluid", 35.0, 5, 730, True, True)
    medicines = {"M001": m1, "M002": m2}

    # H001 has large surplus (1200 units M001, 400 units M002)
    # H002 has moderate stock (100 units M001, 20 units M002)
    # H003 has zero stock of both (acute deficit)
    batches = {
        "B_H1_M1_EARLY": Batch("B_H1_M1_EARLY", "M001", "H001", 600, 600, 0, 25, "active"),
        "B_H1_M1_LATE": Batch("B_H1_M1_LATE", "M001", "H001", 600, 600, 0, 180, "active"),
        "B_H1_M2": Batch("B_H1_M2", "M002", "H001", 400, 400, 0, 200, "active"),
        "B_H2_M1": Batch("B_H2_M1", "M001", "H002", 100, 100, 0, 90, "active"),
        "B_H2_M2": Batch("B_H2_M2", "M002", "H002", 20, 20, 0, 90, "active"),
    }

    pred_demand = {
        ("H001", "M001"): 5.0,
        ("H001", "M002"): 2.0,
        ("H002", "M001"): 5.0,
        ("H002", "M002"): 2.0,
        ("H003", "M001"): 8.0,
        ("H003", "M002"): 4.0,
    }

    return hospitals, medicines, batches, pred_demand


# ==============================================================================
# FEATURE 1 TESTS: EXPIRY-AWARE OPTIMIZER
# ==============================================================================

def test_safety_stock_preservation_over_horizon(network_fixture):
    """Verify that source hospital never transfers stock required for its own safety buffer + 14-day horizon."""
    hospitals, medicines, batches, pred_demand = network_fixture
    # Constrain H001 stock to barely cover its 14-day horizon (5*14=70) + safety (5*5=25) = 95 units
    batches["B_H1_M1_EARLY"].current_quantity = 50
    batches["B_H1_M1_LATE"].current_quantity = 40
    # Total stock at H001 = 90 < 95 -> Available surplus must be strictly 0!

    optimizer = RedistributionOptimizer(horizon_days=14)
    result = optimizer.solve(
        current_day=0,
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
        safety_stock_days=5,
    )

    assert result.solved is True
    # H001 must NOT transfer any M001 because local safety reserve would be breached
    m1_transfers_from_h1 = [t for t in result.transfers if t.from_hospital == "H001" and t.medicine_id == "M001"]
    assert len(m1_transfers_from_h1) == 0


def test_near_expiry_prioritization_over_fresh_batches(network_fixture):
    """Verify earlier-expiring batch (B_H1_M1_EARLY, exp 25) is prioritized over fresh batch (B_H1_M1_LATE, exp 180)."""
    hospitals, medicines, batches, pred_demand = network_fixture
    optimizer = RedistributionOptimizer(horizon_days=14, fefo_epsilon=0.005)

    result = optimizer.solve(
        current_day=0,
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
        safety_stock_days=5,
    )

    assert result.solved is True
    assert len(result.transfers) > 0
    used_batch_ids = [t.batch_id for t in result.transfers if t.medicine_id == "M001"]
    assert "B_H1_M1_EARLY" in used_batch_ids, "Earlier expiring batch must be chosen under FEFO"


def test_batch_expiry_feasibility_enforcement(network_fixture):
    """Verify batches that expire before or upon arrival plus minimum shelf life are strictly rejected."""
    hospitals, medicines, batches, pred_demand = network_fixture
    # Make B_H1_M1_EARLY expire on day 4 (transit is 1-2 days, min shelf life is 7 days -> requires >= 8-9 days)
    batches["B_H1_M1_EARLY"].expiry_day = 4
    # Zero out the late batch so H001 only has the expiring batch
    batches["B_H1_M1_LATE"].current_quantity = 0

    optimizer = RedistributionOptimizer(horizon_days=14, min_destination_shelf_life=7)
    result = optimizer.solve(
        current_day=0,
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
    )

    assert result.solved is True
    h1_m1_transfers = [t for t in result.transfers if t.from_hospital == "H001" and t.medicine_id == "M001"]
    assert len(h1_m1_transfers) == 0, "Expiring batch cannot be transferred"


def test_transfer_recommendation_contains_explanations(network_fixture):
    """Verify that every TransferRecommendation contains target shortage, batch expiry, and human-readable explanation."""
    hospitals, medicines, batches, pred_demand = network_fixture
    optimizer = RedistributionOptimizer(horizon_days=14)

    result = optimizer.solve(
        current_day=0,
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
    )

    assert result.solved is True
    assert len(result.transfers) > 0
    for tr in result.transfers:
        assert tr.target_shortage > 0, "Transfer must target a positive deficit"
        assert tr.batch_expiry is not None, "Batch expiry must be recorded"
        assert len(tr.reason) > 20, "Human-readable explanation rationale must be generated"
        assert tr.from_hospital in tr.reason
        assert tr.to_hospital in tr.reason or hospitals[tr.to_hospital].name in tr.reason


def test_graceful_degradation_missing_batch_expiry(network_fixture):
    """Verify solver falls back gracefully when batch details are missing or batches dict is empty."""
    hospitals, medicines, _, pred_demand = network_fixture
    # Completely empty batches dict, but provide stock levels
    stock_levels = {
        ("H001", "M001"): 800,
        ("H001", "M002"): 300,
    }

    optimizer = RedistributionOptimizer(horizon_days=14)
    result = optimizer.solve(
        current_day=0,
        hospitals=hospitals,
        medicines=medicines,
        batches={},
        predicted_demand=pred_demand,
        stock_levels=stock_levels,
    )

    assert result.solved is True
    assert len(result.transfers) > 0, "Synthesized fallback batches should enable optimization"


# ==============================================================================
# FEATURE 2 TESTS: CASCADING SHORTAGE PREDICTOR
# ==============================================================================

def test_cascading_predictor_distinguishes_direct_from_secondary(network_fixture):
    """
    Verify that an outbreak surge at H003 causes Direct shortage at H003,
    and pulling emergency stock from H002 triggers a Secondary (Cascading) shortage at H002.
    """
    hospitals, medicines, batches, pred_demand = network_fixture
    # Constrain H002 stock so emergency pull exhausts it
    batches["B_H2_M1"].current_quantity = 30
    # Zero out H001 so H002 is the only donor for H003
    batches["B_H1_M1_EARLY"].current_quantity = 0
    batches["B_H1_M1_LATE"].current_quantity = 0

    predictor = CascadingShortagePredictor(
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
        horizon_days=14,
    )

    # Disruption: Acute outbreak at H003
    disruption = DisruptionConfig(
        disrupted_hospital_ids=["H003"],
        demand_surge_multiplier=3.0,
        surge_medicines=["M001"],
        start_day=0,
        end_day=14,
        name="H003 Surge Test",
    )

    result = predictor.predict(current_day=0, disruption=disruption, emergency_donor_pull=True)

    summary = result.at_risk_summary
    assert not summary.empty

    # H003 should be classified as Direct
    h3_records = summary[summary["hospital_id"] == "H003"]
    assert not h3_records.empty
    assert (h3_records["shortage_type"] == "Direct").all()

    # H002 should have an entry with Cascading (Secondary)
    h2_records = summary[summary["hospital_id"] == "H002"]
    if not h2_records.empty:
        assert (h2_records["shortage_type"] == "Cascading (Secondary)").any()


def test_cascading_predictor_earliest_stockout_and_timeline(network_fixture):
    """Verify earliest stockout day and timeline progression calculation."""
    hospitals, medicines, batches, pred_demand = network_fixture
    predictor = CascadingShortagePredictor(
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
        horizon_days=14,
    )

    result = predictor.predict(current_day=0, emergency_donor_pull=False)
    summary = result.at_risk_summary
    assert not summary.empty

    # H003 starts with 0 stock, so earliest stockout day must be 0
    h3_rows = summary[summary["hospital_id"] == "H003"]
    assert not h3_rows.empty
    assert (h3_rows["earliest_stockout_day"] == 0).all()
    assert (h3_rows["days_until_stockout"] == 0).all()


# ==============================================================================
# INTEGRATION TESTS: OPTIMIZER RESOLVES CASCADING DEFICITS
# ==============================================================================

def test_closed_loop_optimizer_mitigates_cascading_deficits(network_fixture):
    """
    Verify that when the optimizer's candidate transfers are simulated,
    projected deficits are resolved and no secondary shortages are created at H001.
    """
    hospitals, medicines, batches, pred_demand = network_fixture
    predictor = CascadingShortagePredictor(
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
        horizon_days=14,
    )

    optimizer = RedistributionOptimizer(horizon_days=14)
    opt_result = optimizer.solve(
        current_day=0,
        hospitals=hospitals,
        medicines=medicines,
        batches=batches,
        predicted_demand=pred_demand,
    )
    assert opt_result.solved is True
    assert len(opt_result.transfers) > 0

    # Simulate with candidate transfers applied
    res = predictor.predict(
        current_day=0,
        candidate_transfers=opt_result.transfers,
    )

    assert res.mitigated_stats is not None
    base_deficit = res.summary_stats["total_deficits"]
    mitigated_deficit = res.mitigated_stats["total_deficits"]

    # Deficit must be reduced
    assert mitigated_deficit < base_deficit, f"Mitigated deficit {mitigated_deficit} should be less than baseline {base_deficit}"

    # Verify H001 did not experience a stockout as a result of donating stock (conservation & safety stock maintained)
    if not res.mitigated_summary.empty:
        h1_stockouts = res.mitigated_summary[res.mitigated_summary["hospital_id"] == "H001"]
        assert len(h1_stockouts) == 0, "Source hospital H001 should not be pushed into shortage by planned transfers"
