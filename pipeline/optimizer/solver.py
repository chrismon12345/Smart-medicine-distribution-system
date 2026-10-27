"""OR-Tools Integer Linear Programming (ILP) Redistribution Optimizer."""
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from ortools.linear_solver import pywraplp

from pipeline.simulator.entities import Hospital, Medicine, Batch

logger = logging.getLogger(__name__)


@dataclass
class TransferRecommendation:
    from_hospital: str
    to_hospital: str
    medicine_id: str
    batch_id: str
    quantity: int
    transit_days: int
    distance_km: float
    estimated_transport_cost: float
    target_shortage: int = 0
    batch_expiry: Optional[int] = None
    reason: str = ""


@dataclass
class OptimizationResult:
    status: str  # "OPTIMAL", "FEASIBLE", "INFEASIBLE"
    objective_value: float
    transfers: List[TransferRecommendation]
    unmet_shortages: Dict[Tuple[str, str], int]
    solved: bool
    message: str = ""


class RedistributionOptimizer:
    """
    Mathematical optimization engine for medicine redistribution.
    Formulated as a Mixed-Integer Linear Program (MILP) solved using CBC / GLOP.
    Enforces strict receiving caps, batch limits, expiry feasibility, and FEFO tie-breaking.
    Generates transparent transfer explanations with batch expiry and shortage contexts.
    """

    def __init__(
        self,
        fixed_transport_cost: float = 100.0,
        cost_per_km: float = 2.5,
        horizon_days: int = 14,
        fefo_epsilon: float = 0.001,
        min_destination_shelf_life: int = 7,
        wastage_penalty_weight: float = 5.0,
    ):
        self.fixed_transport_cost = fixed_transport_cost
        self.cost_per_km = cost_per_km
        self.horizon_days = horizon_days
        self.fefo_epsilon = fefo_epsilon
        self.min_shelf_life = min_destination_shelf_life
        self.wastage_penalty_weight = wastage_penalty_weight

    def solve(
        self,
        current_day: int,
        hospitals: Dict[str, Hospital],
        medicines: Dict[str, Medicine],
        batches: Dict[str, Batch],
        predicted_demand: Dict[Tuple[str, str], float],
        safety_stock_days: int = 5,
        in_transit_incoming: Optional[Dict[Tuple[str, str], int]] = None,
        rural_transit_multiplier: float = 1.0,
        stock_levels: Optional[Dict[Tuple[str, str], int]] = None,
    ) -> OptimizationResult:
        """
        Formulate and solve the redistribution ILP for the current day across the hospital network.
        Accounts for pipeline inventory (in-transit transfers) and rural transit conditions.
        """
        if in_transit_incoming is None:
            in_transit_incoming = {}

        solver = pywraplp.Solver.CreateSolver("CBC")
        if not solver:
            # Fallback to SCIP or GLOP if CBC is not compiled
            solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("GLOP")
            if not solver:
                raise RuntimeError("Failed to create OR-Tools mathematical solver.")

        # Graceful fallback: synthesize batches for aggregate inventory if batch details are missing
        effective_batches: Dict[str, Batch] = dict(batches)
        if stock_levels:
            for (h_id, m_id), s_qty in stock_levels.items():
                if s_qty > 0 and not any(
                    b.hospital_id == h_id and b.medicine_id == m_id and b.is_valid
                    for b in effective_batches.values()
                ):
                    synth_id = f"BAT_SYNTH_{h_id}_{m_id}"
                    effective_batches[synth_id] = Batch(
                        batch_id=synth_id,
                        medicine_id=m_id,
                        hospital_id=h_id,
                        initial_quantity=s_qty,
                        current_quantity=s_qty,
                        manufacture_day=current_day,
                        expiry_day=current_day + self.horizon_days + self.min_shelf_life + 60,
                        status="active",
                    )
                    logger.warning(
                        f"Missing batch-level expiry for hospital {h_id}, medicine {m_id}. "
                        "Synthesized aggregate batch with nominal shelf-life."
                    )

        # Fallback for batches with missing/None expiry_day
        for b_id, b in effective_batches.items():
            if getattr(b, "expiry_day", None) is None:
                b.expiry_day = current_day + self.horizon_days + self.min_shelf_life + 60
                logger.warning(
                    f"Batch {b_id} missing expiry_day; defaulting to nominal shelf-life."
                )

        # 1. Precalculate availability and need per (hospital, medicine)
        avail: Dict[Tuple[str, str], int] = {}
        need: Dict[Tuple[str, str], int] = {}

        for h_id, hosp in hospitals.items():
            for m_id, med in medicines.items():
                key = (h_id, m_id)
                # Current active valid stock
                active_batches = [
                    b for b in effective_batches.values()
                    if b.hospital_id == h_id and b.medicine_id == m_id and b.is_valid and b.expiry_day > current_day
                ]
                current_stock = sum(b.current_quantity for b in active_batches)
                incoming = in_transit_incoming.get(key, 0)
                inventory_pos = current_stock + incoming

                est_daily = max(0.5, predicted_demand.get(key, max(1.0, (hosp.bed_count / 50.0) * med.criticality * 2.0)))
                # Reorder threshold: only trigger deficit if inventory position drops below safety stock buffer
                reorder_threshold = int(est_daily * safety_stock_days)
                target_stock = int(est_daily * (safety_stock_days + min(self.horizon_days, 7)))

                # Need accounts for in-transit arrivals (prevents bullwhip duplicate orders)
                if inventory_pos < reorder_threshold:
                    need[key] = max(0, target_stock - inventory_pos)
                else:
                    need[key] = 0

                # Availability = surplus stock strictly above local consumption + safety buffer over horizon
                min_reserve = int(math.ceil(est_daily * (safety_stock_days + self.horizon_days)))
                avail[key] = max(0, current_stock - min_reserve)

        # 1b. Precalculate expected local consumption under FEFO to identify at-risk expiring batches
        at_risk_units: Dict[str, int] = {}
        for h_id in hospitals:
            for m_id in medicines:
                hosp_batches = sorted(
                    [
                        b for b in effective_batches.values()
                        if b.hospital_id == h_id and b.medicine_id == m_id and b.is_valid and b.current_quantity > 0 and b.expiry_day > current_day
                    ],
                    key=lambda b: b.expiry_day,
                )
                est_daily = predicted_demand.get((h_id, m_id), 1.0)
                local_consumption_horizon = int(est_daily * self.horizon_days)
                rem_demand = local_consumption_horizon
                for b in hosp_batches:
                    cons = min(b.current_quantity, rem_demand)
                    rem_demand = max(0, rem_demand - cons)
                    # If batch expires within horizon window and has units that will not be consumed locally
                    if b.expiry_day <= current_day + self.horizon_days + safety_stock_days:
                        unconsumed = max(0, b.current_quantity - cons)
                        if unconsumed > 0:
                            at_risk_units[b.batch_id] = unconsumed

        # 2. Decision Variables
        # x[i, j, m, b]: integer transfer units from source i to dest j of medicine m and batch b
        x_vars: Dict[Tuple[str, str, str, str], pywraplp.Variable] = {}
        # u[j, m]: unmet shortage at hospital j
        u_vars: Dict[Tuple[str, str], pywraplp.Variable] = {}
        # w[i, m, b]: units left to expire unused
        w_vars: Dict[str, pywraplp.Variable] = {}

        objective = solver.Objective()
        objective.SetMinimization()

        for j_id, dst_hosp in hospitals.items():
            for m_id, med in medicines.items():
                req = need.get((j_id, m_id), 0)
                u_var = solver.IntVar(0, req, f"u_{j_id}_{m_id}")
                u_vars[(j_id, m_id)] = u_var
                # Shortage penalty: heavily penalized by price, criticality, rural status, and urgency
                rural_weight = 1.5 if dst_hosp.is_rural else 1.0
                curr_stock = sum(
                    b.current_quantity for b in effective_batches.values()
                    if b.hospital_id == j_id and b.medicine_id == m_id and b.is_valid and b.expiry_day > current_day
                )
                est_d = max(0.5, predicted_demand.get((j_id, m_id), 1.0))
                days_cover = curr_stock / est_d
                # Urgency multiplier for acute stockouts
                urgency_weight = 3.0 if curr_stock <= 0 else (2.0 if days_cover < 2.0 else (1.5 if days_cover < safety_stock_days else 1.0))
                shortage_cost = med.price * med.criticality * rural_weight * 100.0 * urgency_weight
                objective.SetCoefficient(u_var, shortage_cost)

        for b_id, at_risk_qty in at_risk_units.items():
            b = effective_batches[b_id]
            med = medicines[b.medicine_id]
            w_var = solver.IntVar(0, at_risk_qty, f"w_{b.hospital_id}_{b.medicine_id}_{b_id}")
            w_vars[b_id] = w_var
            # Wastage penalty term in objective: gamma * V[m] * w[i,m,b]
            objective.SetCoefficient(w_var, self.wastage_penalty_weight * med.price)

        for i_id, src_hosp in hospitals.items():
            for j_id, dst_hosp in hospitals.items():
                if i_id == j_id:
                    continue

                dist_km = src_hosp.distance_to(dst_hosp)
                base_transit = max(1, int(math.ceil(dist_km / 120.0)))
                if src_hosp.is_rural or dst_hosp.is_rural:
                    base_transit = int(math.ceil(base_transit * rural_transit_multiplier))
                transit_days = max(1, base_transit)
                # Timing feasibility: cannot ship if transit exceeds horizon
                if transit_days > self.horizon_days:
                    continue

                trans_cost_per_unit = (self.fixed_transport_cost / 50.0) + (self.cost_per_km * dist_km / 100.0)

                for m_id, med in medicines.items():
                    if need.get((j_id, m_id), 0) <= 0 or avail.get((i_id, m_id), 0) <= 0:
                        continue

                    # Active batches at source hospital
                    candidate_batches = [
                        b for b in effective_batches.values()
                        if b.hospital_id == i_id and b.medicine_id == m_id and b.is_valid and b.current_quantity > 0
                    ]

                    for b in candidate_batches:
                        # Expiry feasibility constraint:
                        # Batch must arrive with at least min_shelf_life remaining before expiring!
                        remaining_days_at_arrival = b.expiry_day - (current_day + transit_days)
                        if remaining_days_at_arrival < self.min_shelf_life:
                            continue

                        var_name = f"x_{i_id}_{j_id}_{m_id}_{b.batch_id}"
                        upper_bound = min(b.current_quantity, need[(j_id, m_id)], avail[(i_id, m_id)])
                        x_var = solver.IntVar(0, upper_bound, var_name)
                        x_vars[(i_id, j_id, m_id, b.batch_id)] = x_var

                        # Objective coefficient: Transport cost + FEFO tie-break epsilon
                        # b.expiry_day - current_day: smaller expiry day has lower cost, breaking ties!
                        fefo_cost = self.fefo_epsilon * float(b.expiry_day - current_day)
                        objective.SetCoefficient(x_var, trans_cost_per_unit + fefo_cost)

        # 3. Constraints
        # C1: Source Availability Constraint
        # Sum of all transfers out of source (i, m) across all batches and destinations <= avail[i, m]
        for i_id in hospitals:
            for m_id in medicines:
                src_out = [
                    x_var for (si, dj, sm, sb), x_var in x_vars.items()
                    if si == i_id and sm == m_id
                ]
                if src_out:
                    solver.Add(solver.Sum(src_out) <= avail[(i_id, m_id)])

        # C2: Batch Capacity Constraint
        # Sum of transfers out of batch b to all destinations <= batch current_quantity
        for b_id, b in effective_batches.items():
            if b.is_valid and b.current_quantity > 0:
                batch_out = [
                    x_var for (si, dj, sm, sb), x_var in x_vars.items()
                    if sb == b_id
                ]
                if batch_out:
                    solver.Add(solver.Sum(batch_out) <= b.current_quantity)

        # C3: Shortage Balance Constraint
        # Inflow + Unmet Need >= Need
        for j_id in hospitals:
            for m_id in medicines:
                req = need.get((j_id, m_id), 0)
                if req > 0:
                    dest_in = [
                        x_var for (si, dj, sm, sb), x_var in x_vars.items()
                        if dj == j_id and sm == m_id
                    ]
                    u_var = u_vars[(j_id, m_id)]
                    solver.Add(solver.Sum(dest_in) + u_var >= req)

        # C4: Receiving Cap (MANDATORY ARCHITECTURE CONSTRAINT)
        # Total inflow cannot exceed actual destination need! Prevents dumping excess stock!
        for j_id in hospitals:
            for m_id in medicines:
                dest_in = [
                    x_var for (si, dj, sm, sb), x_var in x_vars.items()
                    if dj == j_id and sm == m_id
                ]
                if dest_in:
                    solver.Add(solver.Sum(dest_in) <= need[(j_id, m_id)])

        # C5: Wastage Penalty Constraint
        # w[i, m, b] >= at_risk - Sum_j x[i, j, m, b]
        for b_id, w_var in w_vars.items():
            batch_out = [
                x_var for (si, dj, sm, sb), x_var in x_vars.items()
                if sb == b_id
            ]
            at_risk = at_risk_units[b_id]
            if batch_out:
                solver.Add(w_var + solver.Sum(batch_out) >= at_risk)
            else:
                solver.Add(w_var >= at_risk)


        # 4. Solve Problem
        solver_status = solver.Solve()
        status_str = "OPTIMAL" if solver_status == pywraplp.Solver.OPTIMAL else (
            "FEASIBLE" if solver_status == pywraplp.Solver.FEASIBLE else "INFEASIBLE"
        )

        if solver_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            logger.warning(f"Optimization solver returned status: {status_str}")
            return OptimizationResult(
                status=status_str,
                objective_value=0.0,
                transfers=[],
                unmet_shortages=need,
                solved=False,
            )

        # 5. Extract Recommended Transfers
        recommendations: List[TransferRecommendation] = []
        for (si, dj, sm, sb), x_var in x_vars.items():
            qty = int(round(x_var.solution_value()))
            if qty > 0:
                dist = hospitals[si].distance_to(hospitals[dj])
                base_transit = max(1, int(math.ceil(dist / 120.0)))
                if hospitals[si].is_rural or hospitals[dj].is_rural:
                    base_transit = int(math.ceil(base_transit * rural_transit_multiplier))
                trans_days = max(1, base_transit)
                cost = self.fixed_transport_cost + (self.cost_per_km * dist)
                b = effective_batches.get(sb)
                b_exp = b.expiry_day if b else None
                tgt_need = need.get((dj, sm), 0)

                # Formulate transparent operational explanation
                shelf_at_arrival = (b_exp - (current_day + trans_days)) if b_exp is not None else 999
                med_obj = medicines.get(sm)
                med_name = f"{med_obj.name} ({sm})" if med_obj else sm
                src_name = f"{hospitals[si].name} ({si})" if si in hospitals else si
                dst_name = f"{hospitals[dj].name} ({dj})" if dj in hospitals else dj

                exp_urgency = "near-expiry batch prioritized to prevent write-off" if (b_exp is not None and b_exp <= current_day + self.horizon_days + safety_stock_days) else "standard shelf-life batch"

                reason = (
                    f"Redistributed {qty} units of {med_name} from {src_name} to {dst_name}: "
                    f"fulfills urgent shortage (target deficit {tgt_need} units); "
                    f"transit takes {trans_days}d leaving {shelf_at_arrival}d shelf life ({exp_urgency}); "
                    f"estimated transport cost ₹{cost:.1f}."
                )

                recommendations.append(TransferRecommendation(
                    from_hospital=si,
                    to_hospital=dj,
                    medicine_id=sm,
                    batch_id=sb,
                    quantity=qty,
                    transit_days=trans_days,
                    distance_km=round(dist, 2),
                    estimated_transport_cost=round(cost, 2),
                    target_shortage=tgt_need,
                    batch_expiry=b_exp,
                    reason=reason,
                ))

        # 6. Post-Solve Plan Validation (Mandatory Audit)
        self.validate_plan(
            transfers=recommendations,
            avail_map=avail,
            need_map=need,
            batches_map=effective_batches,
            hospitals_map=hospitals,
            current_day=current_day,
        )

        unmet_shortages_res = {
            (j_id, m_id): int(round(u_var.solution_value()))
            for (j_id, m_id), u_var in u_vars.items()
        }

        return OptimizationResult(
            status=status_str,
            objective_value=float(solver.Objective().Value()),
            transfers=recommendations,
            unmet_shortages=unmet_shortages_res,
            solved=True,
        )

    def validate_plan(
        self,
        transfers: List[TransferRecommendation],
        avail_map: Dict[Tuple[str, str], int],
        need_map: Dict[Tuple[str, str], int],
        batches_map: Dict[str, Batch],
        hospitals_map: Dict[str, Hospital],
        current_day: int,
    ) -> None:
        """
        Validate every generated transfer against physical constraints.
        Raises ValueError if any constraint is violated.
        """
        source_totals: Dict[Tuple[str, str], int] = {}
        dest_totals: Dict[Tuple[str, str], int] = {}
        batch_totals: Dict[str, int] = {}

        for tr in transfers:
            # 1. Non-negativity
            if tr.quantity <= 0:
                raise ValueError(f"Plan validation failed: non-positive transfer quantity {tr.quantity}")

            # 2. Source != Dest
            if tr.from_hospital == tr.to_hospital:
                raise ValueError(f"Plan validation failed: self-transfer at {tr.from_hospital}")

            # 3. Expiry and timing feasibility
            batch = batches_map.get(tr.batch_id)
            if not batch:
                raise ValueError(f"Plan validation failed: batch {tr.batch_id} not found")

            arrival_day = current_day + tr.transit_days
            if batch.expiry_day <= arrival_day:
                raise ValueError(f"Plan validation failed: batch {tr.batch_id} expires before/on arrival")

            # Accumulate totals
            s_key = (tr.from_hospital, tr.medicine_id)
            d_key = (tr.to_hospital, tr.medicine_id)
            source_totals[s_key] = source_totals.get(s_key, 0) + tr.quantity
            dest_totals[d_key] = dest_totals.get(d_key, 0) + tr.quantity
            batch_totals[tr.batch_id] = batch_totals.get(tr.batch_id, 0) + tr.quantity

        # 4. Check source limits
        for (si, sm), total_out in source_totals.items():
            allowed_avail = avail_map.get((si, sm), 0)
            if total_out > allowed_avail:
                raise ValueError(f"Plan violation: Source {si} exceeded available surplus for {sm} ({total_out} > {allowed_avail})")

        # 5. Check receiving cap
        for (dj, dm), total_in in dest_totals.items():
            allowed_need = need_map.get((dj, dm), 0)
            if total_in > allowed_need:
                raise ValueError(f"Plan violation: Destination {dj} exceeded receiving cap for {dm} ({total_in} > {allowed_need})")

        # 6. Check batch stock
        for b_id, total_out in batch_totals.items():
            batch = batches_map[b_id]
            if total_out > batch.current_quantity:
                raise ValueError(f"Plan violation: Batch {b_id} exceeded stock ({total_out} > {batch.current_quantity})")
