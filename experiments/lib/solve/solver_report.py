# viz/data_center_report.py
#
# Reporting utilities for the data-center block-job model.
# Assumes the optimization model has:
#   - Sets:  S (sites), T (hours), J (jobs)
#   - Params: Ncpu[s], Pdc[s,t], JobLoad[s,t]
#   - Vars: x[j,s]  (assignment of job j to site s)
#   - Objective: Cost
#
# The "jobs" argument is a dict[int, BlockJob] from
# workload.block_jobs_loader.load_hp_block_jobs().

from typing import Any, Dict
import math
import pyomo.environ as pyo


def _safe_value(expr, default=None):
    """Safely evaluate a Pyomo expression/Var/Param to a float."""
    try:
        return pyo.value(expr)
    except Exception:
        return default


def _extract_solver_info(result):
    """
    Robustly extract solver name, status, and termination condition
    from a Pyomo SolverResults object.
    """
    solver_name = "<undefined>"
    status = "unknown"
    term_cond = "unknown"

    try:
        s = result.solver

        # Case 1: dict-like or ComponentMap-like
        try:
            # Pyomo often stores info in a dict-like object
            if hasattr(s, "get"):
                n = s.get("name", None)
                st = s.get("status", None)
                tc = s.get("termination_condition", None)
                if n:
                    solver_name = str(n)
                if st:
                    status = str(st)
                if tc:
                    term_cond = str(tc)
        except Exception:
            pass

        # Case 2: attribute-based
        if solver_name == "<undefined>":
            if hasattr(s, "name") and s.name:
                solver_name = str(s.name)
            elif hasattr(s, "_name") and s._name:
                solver_name = str(s._name)

        if status == "unknown":
            if hasattr(s, "status") and s.status:
                status = str(s.status)

        if term_cond == "unknown":
            if hasattr(s, "termination_condition") and s.termination_condition:
                term_cond = str(s.termination_condition)

    except Exception:
        # Fall back to defaults
        pass

    return solver_name, status, term_cond


def _is_feasible(result):
    """Return True if solver result is feasible/optimal."""
    try:
        tc = result.solver.termination_condition
        return tc in {
            pyo.TerminationCondition.optimal,
            getattr(pyo.TerminationCondition, "locallyOptimal", None),
            pyo.TerminationCondition.feasible,
        }
    except Exception:
        return False
