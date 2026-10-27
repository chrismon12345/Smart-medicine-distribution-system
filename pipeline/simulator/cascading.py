"""Network-Wide Cascading Shortage Predictor for Multi-Facility Hospital Networks.
Simulates daily inventory trajectories across a multi-day horizon, injects supply/demand shocks,
distinguishes primary (direct) disruptions from secondary (cascading) knock-on deficits,
identifies earliest stockout dates and propagation timelines, and validates redistribution plans.
"""
import copy
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
import pandas as pd
import numpy as np

from pipeline.simulator.entities import Hospital, Medicine, Batch, TransferOrder, SupplierOrder
from pipeline.optimizer.solver import TransferRecommendation

logger = logging.getLogger(__name__)


@dataclass
class DisruptionConfig:
    """Defines an operational shock or disruption applied to the hospital network."""
    disrupted_hospital_ids: List[str] = field(default_factory=list)
    disrupted_supplier_hospital_ids: List[str] = field(default_factory=list)
    disrupted_transit_routes: List[Tuple[str, str]] = field(default_factory=list)
    demand_surge_multiplier: float = 1.0
    surge_medicines: Optional[List[str]] = None
    start_day: int = 0
    end_day: int = 14
    name: str = "Standard Disruption"


@dataclass
class ShortageRecord:
    """Represents a projected shortage at a specific facility and medicine."""
    hospital_id: str
    hospital_name: str
    medicine_id: str
    medicine_name: str
    earliest_stockout_day: int
    days_until_stockout: int
    total_deficit: int
    shortage_type: str  # "Direct" or "Cascading (Secondary)"
    root_cause: str
    propagation_path: str


@dataclass
class CascadingPredictionResult:
    """Encapsulates full multi-day cascading shortage prediction outputs."""
    at_risk_summary: pd.DataFrame
    timeline: pd.DataFrame
    daily_stock_trajectories: pd.DataFrame
    summary_stats: Dict[str, Any]
    mitigated_summary: Optional[pd.DataFrame] = None
    mitigated_timeline: Optional[pd.DataFrame] = None
    mitigated_stats: Optional[Dict[str, Any]] = None


class CascadingShortagePredictor:
    """
    Simulates multi-day network dynamics over the planning horizon to predict
    how local supply deficits, surges, or logistics disruptions cascade across facilities.
    """

    def __init__(
        self,
        hospitals: Dict[str, Hospital],
        medicines: Dict[str, Medicine],
        batches: Dict[str, Batch],
        predicted_demand: Dict[Tuple[str, str], float],
        pending_transfers: Optional[List[TransferOrder]] = None,
        pending_supplier_orders: Optional[List[SupplierOrder]] = None,
        horizon_days: int = 14,
        safety_stock_days: int = 5,
        rural_transit_multiplier: float = 1.0,
    ):
        self.hospitals = hospitals
        self.medicines = medicines
        self.batches = batches
        self.predicted_demand = predicted_demand
        self.pending_transfers = pending_transfers or []
        self.pending_supplier_orders = pending_supplier_orders or []
        self.horizon_days = horizon_days
        self.safety_stock_days = safety_stock_days
        self.rural_transit_multiplier = rural_transit_multiplier

    def predict(
        self,
        current_day: int = 0,
        disruption: Optional[DisruptionConfig] = None,
        candidate_transfers: Optional[List[TransferRecommendation]] = None,
        emergency_donor_pull: bool = True,
    ) -> CascadingPredictionResult:
        """
        Simulate multi-day network trajectories day-by-day.
        When a hospital enters deficit, if emergency_donor_pull is True, it draws from
        nearest neighbors with stock, potentially causing secondary shortages at those donors.
        """
        disr = disruption or DisruptionConfig()
        
        # 1. Run Baseline Unmitigated Simulation (with optional emergency donor propagation)
        base_summary, base_timeline, base_trajectories, base_stats = self._simulate_network(
            current_day=current_day,
            disruption=disr,
            transfers_to_apply=[],
            allow_emergency_pull=emergency_donor_pull,
        )

        mitigated_summary = None
        mitigated_timeline = None
        mitigated_stats = None

        # 2. If candidate transfers are provided, re-simulate to evaluate mitigation
        if candidate_transfers:
            mit_summary, mit_timeline, _, mit_stats = self._simulate_network(
                current_day=current_day,
                disruption=disr,
                transfers_to_apply=candidate_transfers,
                allow_emergency_pull=False,  # Managed by planned transfers
            )
            mitigated_summary = mit_summary
            mitigated_timeline = mit_timeline
            mitigated_stats = mit_stats

        return CascadingPredictionResult(
            at_risk_summary=base_summary,
            timeline=base_timeline,
            daily_stock_trajectories=base_trajectories,
            summary_stats=base_stats,
            mitigated_summary=mitigated_summary,
            mitigated_timeline=mitigated_timeline,
            mitigated_stats=mitigated_stats,
        )

    def _simulate_network(
        self,
        current_day: int,
        disruption: DisruptionConfig,
        transfers_to_apply: List[TransferRecommendation],
        allow_emergency_pull: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Core multi-day deterministic daily network projection."""
        
        # Deepcopy batches to ensure simulation sandbox isolation
        sim_batches: Dict[str, Batch] = {
            b_id: copy.deepcopy(b) for b_id, b in self.batches.items()
        }

        # In-transit orders tracker
        active_transfers: List[Dict[str, Any]] = []
        for tr in self.pending_transfers:
            active_transfers.append({
                "to_hospital": tr.to_hospital_id,
                "from_hospital": tr.from_hospital_id,
                "medicine_id": tr.medicine_id,
                "batch_id": tr.batch_id,
                "quantity": tr.quantity,
                "arrival_day": tr.arrival_day,
                "expiry_day": tr.original_expiry_day,
            })

        # Scheduled candidate transfers (dispatched on current_day)
        dispatched_by_hospital: Dict[Tuple[str, str], int] = {}
        transfers_drawn_to_assist: Dict[Tuple[str, str], Set[str]] = {}  # (hosp, med) -> set of beneficiary hospitals

        for tr in transfers_to_apply:
            key = (tr.from_hospital, tr.medicine_id)
            dispatched_by_hospital[key] = dispatched_by_hospital.get(key, 0) + tr.quantity
            b_key = (tr.from_hospital, tr.medicine_id)
            if b_key not in transfers_drawn_to_assist:
                transfers_drawn_to_assist[b_key] = set()
            transfers_drawn_to_assist[b_key].add(tr.to_hospital)

            arrival = current_day + tr.transit_days
            b_orig = sim_batches.get(tr.batch_id)
            exp_day = b_orig.expiry_day if b_orig else (current_day + self.horizon_days + 30)

            # Deduct from source batch in sandbox
            if b_orig:
                actual_deduct = min(b_orig.current_quantity, tr.quantity)
                b_orig.current_quantity -= actual_deduct

            active_transfers.append({
                "to_hospital": tr.to_hospital,
                "from_hospital": tr.from_hospital,
                "medicine_id": tr.medicine_id,
                "batch_id": tr.batch_id,
                "quantity": tr.quantity,
                "arrival_day": arrival,
                "expiry_day": exp_day,
            })

        # Supplier deliveries tracker
        active_supplier_orders: List[Dict[str, Any]] = []
        for so in self.pending_supplier_orders:
            # Check if hospital is subject to supplier disruption
            delivery_day = so.delivery_day
            if so.hospital_id in disruption.disrupted_supplier_hospital_ids:
                delivery_day += 10  # Supplier delivery heavily delayed or blocked
            active_supplier_orders.append({
                "hospital_id": so.hospital_id,
                "medicine_id": so.medicine_id,
                "quantity": so.quantity,
                "delivery_day": delivery_day,
                "shelf_life": so.shelf_life_days,
            })

        # Records to populate
        timeline_records: List[Dict[str, Any]] = []
        trajectory_records: List[Dict[str, Any]] = []
        first_stockout_map: Dict[Tuple[str, str], int] = {}
        cumulative_deficit_map: Dict[Tuple[str, str], int] = {}
        shortage_nature_map: Dict[Tuple[str, str], str] = {}
        root_cause_map: Dict[Tuple[str, str], str] = {}
        propagation_path_map: Dict[Tuple[str, str], str] = {}

        # Daily iteration
        for day in range(current_day, current_day + self.horizon_days):
            
            # A. Process transfer arrivals
            remaining_transfers = []
            for t_item in active_transfers:
                if t_item["arrival_day"] <= day:
                    # Check if batch has expired
                    if t_item["expiry_day"] > day:
                        new_bid = f"SANDBOX_TR_{day}_{t_item['to_hospital']}_{t_item['batch_id']}"
                        sim_batches[new_bid] = Batch(
                            batch_id=new_bid,
                            medicine_id=t_item["medicine_id"],
                            hospital_id=t_item["to_hospital"],
                            initial_quantity=t_item["quantity"],
                            current_quantity=t_item["quantity"],
                            manufacture_day=day,
                            expiry_day=t_item["expiry_day"],
                            status="active",
                        )
                else:
                    remaining_transfers.append(t_item)
            active_transfers = remaining_transfers

            # B. Process supplier order arrivals
            remaining_supplier = []
            for s_item in active_supplier_orders:
                if s_item["delivery_day"] <= day:
                    new_bid = f"SANDBOX_SUP_{day}_{s_item['hospital_id']}_{s_item['medicine_id']}"
                    sim_batches[new_bid] = Batch(
                        batch_id=new_bid,
                        medicine_id=s_item["medicine_id"],
                        hospital_id=s_item["hospital_id"],
                        initial_quantity=s_item["quantity"],
                        current_quantity=s_item["quantity"],
                        manufacture_day=day,
                        expiry_day=day + s_item["shelf_life"],
                        status="active",
                    )
                else:
                    remaining_supplier.append(s_item)
            active_supplier_orders = remaining_supplier

            # C. Expire batches
            for b in sim_batches.values():
                if b.is_valid and b.expiry_day <= day:
                    b.current_quantity = 0
                    b.status = "expired"

            # D. Hospital daily consumption and shortage detection
            for h_id, hosp in self.hospitals.items():
                for m_id, med in self.medicines.items():
                    key = (h_id, m_id)

                    # Compute projected daily demand
                    base_rate = max(0.5, self.predicted_demand.get(
                        key, max(1.0, (hosp.bed_count / 50.0) * med.criticality * 2.0)
                    ))
                    is_in_surge = (
                        h_id in disruption.disrupted_hospital_ids
                        and (disruption.start_day <= day <= disruption.end_day)
                        and (disruption.surge_medicines is None or m_id in disruption.surge_medicines)
                    )
                    surge_mult = disruption.demand_surge_multiplier if is_in_surge else 1.0
                    daily_demand = max(1.0, base_rate * surge_mult)

                    # Available valid stock at hospital
                    hosp_med_batches = sorted(
                        [
                            b for b in sim_batches.values()
                            if b.hospital_id == h_id and b.medicine_id == m_id and b.is_valid and b.expiry_day > day
                        ],
                        key=lambda b: b.expiry_day,
                    )
                    current_stock = sum(b.current_quantity for b in hosp_med_batches)

                    trajectory_records.append({
                        "day": day,
                        "hospital_id": h_id,
                        "hospital_name": hosp.name,
                        "medicine_id": m_id,
                        "medicine_name": med.name,
                        "projected_stock": current_stock,
                        "daily_demand": round(daily_demand, 1),
                        "is_rural": hosp.is_rural,
                    })

                    # Consume stock under FEFO
                    rem_demand = daily_demand
                    for b in hosp_med_batches:
                        consume = min(b.current_quantity, int(math.ceil(rem_demand)))
                        b.current_quantity -= consume
                        rem_demand = max(0.0, rem_demand - consume)
                        if b.current_quantity <= 0:
                            b.status = "depleted"

                    day_deficit = int(math.ceil(rem_demand))

                    # If deficit occurred
                    if day_deficit > 0:
                        if key not in first_stockout_map:
                            first_stockout_map[key] = day
                            days_until = max(0, day - current_day)

                            # Determine Shortage Nature (Direct vs Cascading)
                            is_primary = (
                                h_id in disruption.disrupted_hospital_ids
                                or h_id in disruption.disrupted_supplier_hospital_ids
                            )
                            had_sent_transfers = dispatched_by_hospital.get(key, 0) > 0

                            if is_primary:
                                shortage_type = "Direct"
                                cause = "Primary Disruption (Demand Surge / Supplier Halt)"
                                path = f"Epicenter: {hosp.name}"
                            elif had_sent_transfers:
                                shortage_type = "Cascading (Secondary)"
                                recipient_names = [
                                    self.hospitals[r].name for r in transfers_drawn_to_assist.get(key, [])
                                    if r in self.hospitals
                                ]
                                recipient_str = ", ".join(recipient_names) or "neighboring facilities"
                                cause = f"Stock drained assisting {recipient_str}"
                                path = f"Transferred to {recipient_str} -> Stock depleted -> Shortage on Day {day}"
                            else:
                                shortage_type = "Cascading (Secondary)"
                                cause = "Depletion without external replenishment buffer"
                                path = f"Local buffer depleted under sustained patient demand"

                            shortage_nature_map[key] = shortage_type
                            root_cause_map[key] = cause
                            propagation_path_map[key] = path

                        cumulative_deficit_map[key] = cumulative_deficit_map.get(key, 0) + day_deficit

                        timeline_records.append({
                            "day": day,
                            "hospital_id": h_id,
                            "hospital_name": hosp.name,
                            "medicine_id": m_id,
                            "medicine_name": med.name,
                            "deficit": day_deficit,
                            "shortage_type": shortage_nature_map.get(key, "Direct"),
                            "root_cause": root_cause_map.get(key, ""),
                        })

                        # Emergency pull logic: unmitigated deficit pulls stock from nearest neighbor on subsequent days
                        if allow_emergency_pull and day + 1 < current_day + self.horizon_days:
                            self._execute_emergency_neighbor_pull(
                                sim_batches=sim_batches,
                                deficit_hospital_id=h_id,
                                medicine_id=m_id,
                                deficit_qty=day_deficit,
                                dispatch_day=day,
                                active_transfers=active_transfers,
                                dispatched_by_hospital=dispatched_by_hospital,
                                transfers_drawn_to_assist=transfers_drawn_to_assist,
                            )

        # Build Summary DataFrame
        summary_rows: List[Dict[str, Any]] = []
        for (h_id, m_id), first_day in first_stockout_map.items():
            tot_def = cumulative_deficit_map.get((h_id, m_id), 0)
            hosp = self.hospitals[h_id]
            med = self.medicines[m_id]
            summary_rows.append({
                "hospital_id": h_id,
                "hospital_name": hosp.name,
                "district": hosp.district,
                "is_rural": hosp.is_rural,
                "medicine_id": m_id,
                "medicine_name": med.name,
                "criticality": med.criticality,
                "earliest_stockout_day": first_day,
                "days_until_stockout": max(0, first_day - current_day),
                "total_deficit_units": tot_def,
                "shortage_type": shortage_nature_map.get((h_id, m_id), "Direct"),
                "root_cause": root_cause_map.get((h_id, m_id), "Unknown"),
                "propagation_path": propagation_path_map.get((h_id, m_id), ""),
            })

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(
                ["earliest_stockout_day", "criticality", "total_deficit_units"],
                ascending=[True, False, False],
            ).reset_index(drop=True)

        timeline_df = pd.DataFrame(timeline_records)
        trajectories_df = pd.DataFrame(trajectory_records)

        # Compute summary statistics
        total_def = int(summary_df["total_deficit_units"].sum()) if not summary_df.empty else 0
        direct_def = int(summary_df[summary_df["shortage_type"] == "Direct"]["total_deficit_units"].sum()) if not summary_df.empty else 0
        cascading_def = int(summary_df[summary_df["shortage_type"].str.contains("Cascading")]["total_deficit_units"].sum()) if not summary_df.empty else 0
        earliest_day = int(summary_df["earliest_stockout_day"].min()) if not summary_df.empty else None
        affected_facs = int(summary_df["hospital_id"].nunique()) if not summary_df.empty else 0

        stats = {
            "total_deficits": total_def,
            "direct_deficit_units": direct_def,
            "cascading_deficit_units": cascading_def,
            "earliest_stockout_day": earliest_day,
            "affected_facilities_count": affected_facs,
            "disruption_epicenter_count": len(disruption.disrupted_hospital_ids),
        }

        return summary_df, timeline_df, trajectories_df, stats

    def _execute_emergency_neighbor_pull(
        self,
        sim_batches: Dict[str, Batch],
        deficit_hospital_id: str,
        medicine_id: str,
        deficit_qty: int,
        dispatch_day: int,
        active_transfers: List[Dict[str, Any]],
        dispatched_by_hospital: Dict[Tuple[str, str], int],
        transfers_drawn_to_assist: Dict[Tuple[str, str], Set[str]],
    ) -> None:
        """Finds nearest neighbor with available stock to fulfill emergency replenishment."""
        def_hosp = self.hospitals[deficit_hospital_id]
        sorted_neighbors = sorted(
            [h for h_id, h in self.hospitals.items() if h_id != deficit_hospital_id],
            key=lambda h: def_hosp.distance_to(h),
        )

        pull_remaining = deficit_qty
        for neighbor in sorted_neighbors:
            if pull_remaining <= 0:
                break
            n_id = neighbor.hospital_id
            n_batches = [
                b for b in sim_batches.values()
                if b.hospital_id == n_id and b.medicine_id == medicine_id and b.is_valid and b.expiry_day > dispatch_day + 2
            ]
            avail_stock = sum(b.current_quantity for b in n_batches)
            if avail_stock <= 0:
                continue

            pull_amount = min(avail_stock, pull_remaining)
            # Deduct from neighbor batches
            rem_to_deduct = pull_amount
            for b in sorted(n_batches, key=lambda x: x.expiry_day):
                ded = min(b.current_quantity, rem_to_deduct)
                b.current_quantity -= ded
                rem_to_deduct -= ded
                if rem_to_deduct <= 0:
                    break

            dist = def_hosp.distance_to(neighbor)
            transit_days = max(1, int(math.ceil(dist / 120.0)))
            arrival = dispatch_day + transit_days

            active_transfers.append({
                "to_hospital": deficit_hospital_id,
                "from_hospital": n_id,
                "medicine_id": medicine_id,
                "batch_id": f"EMERGENCY_{n_id}_{dispatch_day}",
                "quantity": pull_amount,
                "arrival_day": arrival,
                "expiry_day": dispatch_day + 60,
            })

            key = (n_id, medicine_id)
            dispatched_by_hospital[key] = dispatched_by_hospital.get(key, 0) + pull_amount
            if key not in transfers_drawn_to_assist:
                transfers_drawn_to_assist[key] = set()
            transfers_drawn_to_assist[key].add(deficit_hospital_id)

            pull_remaining -= pull_amount
