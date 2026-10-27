"""Rolling-horizon optimizer orchestrator."""
from typing import Dict, List, Any, Optional
from pipeline.optimizer.solver import RedistributionOptimizer, OptimizationResult
from pipeline.simulator.engine import Simulator


class RollingHorizonOptimizer:
    """
    Executes rolling-horizon optimization.
    At simulated day t:
    - Solves multi-day redistribution problem
    - Applies ONLY the immediate day-1 transfers
    - Discards later horizon decisions for dynamic re-solving at day t+1
    """

    def __init__(
        self,
        optimizer: Optional[RedistributionOptimizer] = None,
        safety_stock_days: int = 5,
    ):
        self.optimizer = optimizer or RedistributionOptimizer()
        self.safety_stock_days = safety_stock_days
        self.last_result: Optional[OptimizationResult] = None

    def __call__(
        self,
        sim: Simulator,
        predicted_demand: Optional[Dict[Any, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate simulator state, execute OR-Tools solver, and return day's transfer actions.
        """
        # Build demand forecast expectation per (hospital, medicine) if not provided
        if not predicted_demand:
            predicted_demand = {}
            for h_id, hosp in sim.hospitals.items():
                for m_id, med in sim.medicines.items():
                    # Base expected daily burn rate
                    rate = max(1.0, (hosp.bed_count / 50.0) * med.criticality * 2.0)
                    predicted_demand[(h_id, m_id)] = rate

        # Extract in-transit pipeline stock arriving at destinations from inter-hospital transfers
        in_transit: Dict[Any, int] = {}
        for tr in getattr(sim, "pending_transfers", []):
            k = (tr.to_hospital_id, tr.medicine_id)
            in_transit[k] = in_transit.get(k, 0) + tr.quantity

        result = self.optimizer.solve(
            current_day=sim.current_day,
            hospitals=sim.hospitals,
            medicines=sim.medicines,
            batches=sim.batches,
            predicted_demand=predicted_demand,
            safety_stock_days=self.safety_stock_days,
            in_transit_incoming=in_transit,
            rural_transit_multiplier=getattr(sim, "rural_transit_multiplier", 1.0),
        )
        self.last_result = result

        actions: List[Dict[str, Any]] = []
        if result.solved and result.transfers:
            for tr in result.transfers:
                actions.append({
                    "from_hospital": tr.from_hospital,
                    "to_hospital": tr.to_hospital,
                    "medicine_id": tr.medicine_id,
                    "batch_id": tr.batch_id,
                    "quantity": tr.quantity,
                    "transit_days": tr.transit_days,
                    "distance_km": tr.distance_km,
                    "estimated_transport_cost": tr.estimated_transport_cost,
                    "target_shortage": getattr(tr, "target_shortage", 0),
                    "batch_expiry": getattr(tr, "batch_expiry", None),
                    "reason": getattr(tr, "reason", ""),
                })

        return actions
