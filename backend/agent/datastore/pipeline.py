"""
Pipeline and opportunity operations for CRM Data Store.

Provides opportunity queries, pipeline summaries, forecasts, and deals at risk.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.agent.datastore.base import DataStoreMixinProtocol

    _MixinBase = DataStoreMixinProtocol
else:
    _MixinBase = object


class PipelineMixin(_MixinBase):
    """Mixin providing pipeline and opportunity operations."""

    def get_open_opportunities(self, company_id: str, limit: int = 20) -> list[dict]:
        """Get open opportunities for a company (filters out closed stages)."""
        self._ensure_table("opportunities")

        result = self.conn.execute(
            f"""
            SELECT * FROM opportunities
            WHERE company_id = ?
            AND LOWER(stage) NOT LIKE '%closed%'
            ORDER BY value DESC
            LIMIT {limit}
        """,
            [company_id],
        ).fetchall()

        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def get_pipeline_summary(self, company_id: str) -> dict:
        """
        Get pipeline summary for a company.

        Returns dict with stages, total_count, and total_value.
        """
        self._ensure_table("opportunities")

        result = self.conn.execute(
            """
            SELECT
                stage,
                COUNT(*) as count,
                SUM(COALESCE(value, 0)) as total_value
            FROM opportunities
            WHERE company_id = ?
            AND LOWER(stage) NOT LIKE '%closed%'
            GROUP BY stage
        """,
            [company_id],
        ).fetchall()

        stages = {
            stage: {"count": count, "total_value": float(value or 0)}
            for stage, count, value in result
        }
        return {
            "stages": stages,
            "total_count": sum(s["count"] for s in stages.values()),
            "total_value": sum(s["total_value"] for s in stages.values()),
        }

    def get_all_pipeline_summary(self) -> dict:
        """Get pipeline summary across ALL companies."""
        self._ensure_table("opportunities")

        overall = self.conn.execute("""
            SELECT
                COUNT(*) as total_count,
                SUM(COALESCE(value, 0)) as total_value
            FROM opportunities
            WHERE LOWER(stage) NOT LIKE '%closed%'
        """).fetchone()

        by_stage = self.conn.execute("""
            SELECT
                stage,
                COUNT(*) as count,
                SUM(COALESCE(value, 0)) as total_value
            FROM opportunities
            WHERE LOWER(stage) NOT LIKE '%closed%'
            GROUP BY stage
            ORDER BY total_value DESC
        """).fetchall()

        by_company = self.conn.execute("""
            SELECT
                company_id,
                COUNT(*) as count,
                SUM(COALESCE(value, 0)) as total_value
            FROM opportunities
            WHERE LOWER(stage) NOT LIKE '%closed%'
            GROUP BY company_id
            ORDER BY total_value DESC
            LIMIT 10
        """).fetchall()

        return {
            "total_count": overall[0] if overall else 0,
            "total_value": float(overall[1] or 0) if overall else 0,
            "by_stage": {
                stage: {"count": count, "value": float(value or 0)}
                for stage, count, value in by_stage
            },
            "top_companies": [
                {"company_id": cid, "count": count, "value": float(value or 0)}
                for cid, count, value in by_company
            ],
        }

    def get_upcoming_renewals(
        self, days: int = 90, limit: int = 20, owner: str | None = None
    ) -> list[dict]:
        """Get companies with upcoming renewals within the date window."""
        self._ensure_table("companies")

        today = datetime.now().strftime("%Y-%m-%d")
        cutoff = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        owner_filter = f"AND account_owner = '{owner}'" if owner else ""

        result = self.conn.execute(f"""
            SELECT * FROM companies
            WHERE renewal_date >= '{today}'
            AND renewal_date <= '{cutoff}'
            AND status = 'Active'
            {owner_filter}
            ORDER BY renewal_date ASC
            LIMIT {limit}
        """).fetchall()

        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def get_pipeline_by_owner(self, owner: str | None = None) -> dict:
        """Get pipeline grouped by owner."""
        self._ensure_table("opportunities")

        conditions = ["LOWER(stage) NOT LIKE '%closed%'"]
        params = []

        if owner:
            conditions.append("owner = ?")
            params.append(owner)

        where_clause = " AND ".join(conditions)

        result = self.conn.execute(
            f"""
            SELECT
                owner,
                COUNT(*) as deal_count,
                SUM(COALESCE(value, 0)) as total_value,
                AVG(COALESCE(days_in_stage, 0)) as avg_days_in_stage
            FROM opportunities
            WHERE {where_clause}
            GROUP BY owner
            ORDER BY total_value DESC
        """,
            params,
        ).fetchall()

        total_value = sum(float(row[2] or 0) for row in result)
        total_count = sum(row[1] for row in result)

        breakdown = [
            {
                "owner": owner_id,
                "deal_count": count,
                "total_value": float(value or 0),
                "avg_days_in_stage": round(float(avg_days or 0), 1),
                "percentage": round(float(value or 0) / total_value * 100, 1)
                if total_value > 0
                else 0,
            }
            for owner_id, count, value, avg_days in result
        ]

        return {
            "total_count": total_count,
            "total_value": total_value,
            "owner_filter": owner,
            "breakdown": breakdown,
        }

    def get_deals_at_risk(
        self, owner: str | None = None, days_threshold: int = 45, limit: int = 20
    ) -> list[dict]:
        """Get deals at risk (stale in stage for too long)."""
        self._ensure_table("opportunities")

        conditions = [
            "LOWER(stage) NOT LIKE '%closed%'",
            f"COALESCE(days_in_stage, 0) >= {days_threshold}",
        ]
        params = []

        if owner:
            conditions.append("owner = ?")
            params.append(owner)

        where_clause = " AND ".join(conditions)

        return self._fetch_all_dicts(
            f"""
            SELECT * FROM opportunities
            WHERE {where_clause}
            ORDER BY days_in_stage DESC, value DESC
            LIMIT {limit}
        """,
            params,
        )

    def get_forecast(self, owner: str | None = None) -> dict:
        """Get weighted pipeline forecast using stage probabilities."""
        self._ensure_table("opportunities")

        stage_probs = {
            "new": 0.10,
            "discovery": 0.10,
            "qualified": 0.25,
            "proposal": 0.50,
            "negotiation": 0.75,
        }

        conditions = ["LOWER(stage) NOT LIKE '%closed%'"]
        params = []

        if owner:
            conditions.append("owner = ?")
            params.append(owner)

        where_clause = " AND ".join(conditions)

        opps = self._fetch_all_dicts(
            f"""
            SELECT owner, stage, value, name, expected_close_date
            FROM opportunities
            WHERE {where_clause}
        """,
            params,
        )

        by_stage = {}
        by_owner = {}
        total_pipeline = 0
        total_weighted = 0

        for opp in opps:
            stage = (opp.get("stage") or "").lower()
            value = float(opp.get("value") or 0)
            owner_id = opp.get("owner", "unknown")

            prob = stage_probs.get(stage, 0.20)
            weighted = value * prob

            total_pipeline += value
            total_weighted += weighted

            if stage not in by_stage:
                by_stage[stage] = {"count": 0, "pipeline": 0, "weighted": 0, "probability": prob}
            by_stage[stage]["count"] += 1
            by_stage[stage]["pipeline"] += value
            by_stage[stage]["weighted"] += weighted

            if owner_id not in by_owner:
                by_owner[owner_id] = {"count": 0, "pipeline": 0, "weighted": 0}
            by_owner[owner_id]["count"] += 1
            by_owner[owner_id]["pipeline"] += value
            by_owner[owner_id]["weighted"] += weighted

        return {
            "total_pipeline": total_pipeline,
            "total_weighted": round(total_weighted, 2),
            "owner_filter": owner,
            "by_stage": by_stage,
            "by_owner": by_owner,
        }

    def get_forecast_accuracy(self, owner: str | None = None) -> dict:
        """Get forecast accuracy metrics based on historical closed deals."""
        self._ensure_table("opportunities")

        conditions = ["LOWER(stage) LIKE '%closed%'"]
        params = []

        if owner:
            conditions.append("owner = ?")
            params.append(owner)

        where_clause = " AND ".join(conditions)

        closed = self._fetch_all_dicts(
            f"""
            SELECT owner, stage, value, name, type
            FROM opportunities
            WHERE {where_clause}
        """,
            params,
        )

        by_owner = {}
        total_won = 0
        total_lost = 0
        total_won_value = 0
        total_lost_value = 0

        for opp in closed:
            stage = (opp.get("stage") or "").lower()
            value = float(opp.get("value") or 0)
            owner_id = opp.get("owner", "unknown")

            is_won = "won" in stage

            if owner_id not in by_owner:
                by_owner[owner_id] = {"won": 0, "lost": 0, "won_value": 0, "lost_value": 0}

            if is_won:
                total_won += 1
                total_won_value += value
                by_owner[owner_id]["won"] += 1
                by_owner[owner_id]["won_value"] += value
            else:
                total_lost += 1
                total_lost_value += value
                by_owner[owner_id]["lost"] += 1
                by_owner[owner_id]["lost_value"] += value

        total_closed = total_won + total_lost
        overall_win_rate = (total_won / total_closed * 100) if total_closed > 0 else 0

        for owner_id, stats in by_owner.items():
            owner_total = stats["won"] + stats["lost"]
            stats["win_rate"] = round(stats["won"] / owner_total * 100, 1) if owner_total > 0 else 0
            stats["total_closed"] = owner_total

        return {
            "overall_win_rate": round(overall_win_rate, 1),
            "total_won": total_won,
            "total_lost": total_lost,
            "total_won_value": total_won_value,
            "total_lost_value": total_lost_value,
            "total_closed": total_closed,
            "owner_filter": owner,
            "by_owner": by_owner,
        }
