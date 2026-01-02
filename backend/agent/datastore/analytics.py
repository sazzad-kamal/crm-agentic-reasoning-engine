"""
Analytics operations for CRM Data Store.

Provides breakdowns, aggregations, groups, and attachments functionality.
"""


class AnalyticsMixin:
    """Mixin providing analytics, groups, and attachments operations."""

    # =========================================================================
    # Breakdown Methods
    # =========================================================================

    def get_contact_breakdown(self, company_id: str | None = None, group_by: str = "role") -> dict:
        """Get breakdown of contacts by role or job_title."""
        self._ensure_table("contacts")

        group_field = "role" if group_by == "role" else "job_title"

        if company_id:
            result = self.conn.execute(
                f"""
                SELECT
                    COALESCE({group_field}, 'Unknown') as category,
                    COUNT(*) as count
                FROM contacts
                WHERE company_id = ?
                GROUP BY {group_field}
                ORDER BY count DESC
            """,
                [company_id],
            ).fetchall()

            total = self.conn.execute(
                "SELECT COUNT(*) FROM contacts WHERE company_id = ?", [company_id]
            ).fetchone()[0]
        else:
            result = self.conn.execute(f"""
                SELECT
                    COALESCE({group_field}, 'Unknown') as category,
                    COUNT(*) as count
                FROM contacts
                GROUP BY {group_field}
                ORDER BY count DESC
            """).fetchall()

            total = self.conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]

        breakdown = [
            {
                "category": cat,
                "count": count,
                "percentage": round(count / total * 100, 1) if total > 0 else 0,
            }
            for cat, count in result
        ]

        return {
            "group_by": group_by,
            "company_id": company_id,
            "total": total,
            "breakdown": breakdown,
        }

    def get_activity_breakdown(
        self, company_id: str | None = None, days: int = 30, group_by: str = "type"
    ) -> dict:
        """Get breakdown of activities by type or status."""
        self._ensure_table("activities")

        cutoff = self._get_date_cutoff(days)
        group_field = "type" if group_by == "type" else "status"

        conditions = [f"(due_datetime >= '{cutoff}' OR created_at >= '{cutoff}')"]
        params = []

        if company_id:
            conditions.append("company_id = ?")
            params.append(company_id)

        where_clause = " AND ".join(conditions)

        result = self.conn.execute(
            f"""
            SELECT
                COALESCE({group_field}, 'Unknown') as category,
                COUNT(*) as count
            FROM activities
            WHERE {where_clause}
            GROUP BY {group_field}
            ORDER BY count DESC
        """,
            params,
        ).fetchall()

        total_result = self.conn.execute(
            f"SELECT COUNT(*) FROM activities WHERE {where_clause}", params
        ).fetchone()
        total = total_result[0] if total_result else 0

        breakdown = [
            {
                "category": cat,
                "count": count,
                "percentage": round(count / total * 100, 1) if total > 0 else 0,
            }
            for cat, count in result
        ]

        return {
            "group_by": group_by,
            "company_id": company_id,
            "days": days,
            "total": total,
            "breakdown": breakdown,
        }

    def get_activity_count_by_filter(
        self,
        activity_type: str | None = None,
        days: int = 30,
        company_id: str | None = None,
    ) -> dict:
        """Get count of activities matching filters."""
        self._ensure_table("activities")

        cutoff = self._get_date_cutoff(days)
        conditions = [f"(due_datetime >= '{cutoff}' OR created_at >= '{cutoff}')"]
        params = []

        if activity_type:
            conditions.append("LOWER(type) LIKE ?")
            params.append(f"%{activity_type.lower()}%")

        if company_id:
            conditions.append("company_id = ?")
            params.append(company_id)

        where_clause = " AND ".join(conditions)

        result = self.conn.execute(
            f"SELECT COUNT(*) FROM activities WHERE {where_clause}", params
        ).fetchone()

        return {
            "count": result[0] if result else 0,
            "activity_type": activity_type,
            "days": days,
            "company_id": company_id,
        }

    def get_accounts_needing_attention(
        self, owner: str | None = None, limit: int = 20
    ) -> list[dict]:
        """Get accounts that need immediate attention (Trial, Churned, or at-risk)."""
        self._ensure_table("companies")

        conditions = []
        params = []

        conditions.append("""
            (LOWER(status) = 'trial'
            OR LOWER(status) = 'churned'
            OR LOWER(COALESCE(health_flags, '')) LIKE '%risk%')
        """)

        if owner:
            conditions.append("account_owner = ?")
            params.append(owner)

        where_clause = " AND ".join(conditions)

        return self._fetch_all_dicts(
            f"""
            SELECT * FROM companies
            WHERE {where_clause}
            ORDER BY
                CASE LOWER(status)
                    WHEN 'trial' THEN 1
                    WHEN 'churned' THEN 2
                    ELSE 3
                END,
                name
            LIMIT {limit}
        """,
            params,
        )

    # =========================================================================
    # Group Methods
    # =========================================================================

    def get_group(self, group_id: str) -> dict | None:
        """Get group by ID."""
        if not self._ensure_table("groups"):
            return None
        return self._fetch_one_dict("SELECT * FROM groups WHERE group_id = ?", [group_id])

    def get_all_groups(self) -> list[dict]:
        """Get all groups."""
        if not self._ensure_table("groups"):
            return []
        return self._fetch_all_dicts("SELECT * FROM groups ORDER BY name")

    def get_group_members(self, group_id: str, limit: int = 50) -> list[dict]:
        """Get companies in a group."""
        if not self._ensure_table("group_members"):
            return []
        self._ensure_table("companies")

        return self._fetch_all_dicts(
            f"""
            SELECT c.* FROM companies c
            INNER JOIN group_members gm ON c.company_id = gm.company_id
            WHERE gm.group_id = ?
            LIMIT {limit}
        """,
            [group_id],
        )

    def get_accounts_by_group(self) -> dict:
        """Get count of accounts in each group."""
        if not self._ensure_table("groups") or not self._ensure_table("group_members"):
            return {"total_groups": 0, "breakdown": []}

        self._ensure_table("companies")

        result = self.conn.execute("""
            SELECT
                g.group_id,
                g.name,
                COUNT(gm.company_id) as count
            FROM groups g
            LEFT JOIN group_members gm ON g.group_id = gm.group_id
            GROUP BY g.group_id, g.name
            ORDER BY count DESC
        """).fetchall()

        total_groups = len(result)
        total_memberships = sum(count for _, _, count in result)

        breakdown = [
            {
                "group_id": gid,
                "group_name": name,
                "count": count,
                "percentage": round(count / total_memberships * 100, 1)
                if total_memberships > 0
                else 0,
            }
            for gid, name, count in result
        ]

        return {
            "total_groups": total_groups,
            "total_memberships": total_memberships,
            "breakdown": breakdown,
        }

    def get_pipeline_by_group(self, group_id: str | None = None) -> dict:
        """Get pipeline value breakdown by group or for a specific group."""
        if not self._ensure_table("groups") or not self._ensure_table("group_members"):
            return {"breakdown": []}

        self._ensure_table("opportunities")

        if group_id:
            result = self.conn.execute(
                """
                SELECT
                    g.group_id,
                    g.name as group_name,
                    COUNT(DISTINCT gm.company_id) as company_count,
                    COUNT(o.opportunity_id) as deal_count,
                    SUM(COALESCE(o.value, 0)) as total_value
                FROM groups g
                LEFT JOIN group_members gm ON g.group_id = gm.group_id
                LEFT JOIN opportunities o ON gm.company_id = o.company_id
                    AND LOWER(o.stage) NOT LIKE '%closed%'
                WHERE g.group_id = ?
                GROUP BY g.group_id, g.name
            """,
                [group_id],
            ).fetchone()

            if result:
                return {
                    "group_id": result[0],
                    "group_name": result[1],
                    "company_count": result[2],
                    "deal_count": result[3],
                    "total_value": float(result[4] or 0),
                }
            return {"error": f"Group '{group_id}' not found"}

        result = self.conn.execute("""
            SELECT
                g.group_id,
                g.name as group_name,
                COUNT(DISTINCT gm.company_id) as company_count,
                COUNT(o.opportunity_id) as deal_count,
                SUM(COALESCE(o.value, 0)) as total_value
            FROM groups g
            LEFT JOIN group_members gm ON g.group_id = gm.group_id
            LEFT JOIN opportunities o ON gm.company_id = o.company_id
                AND LOWER(o.stage) NOT LIKE '%closed%'
            GROUP BY g.group_id, g.name
            ORDER BY total_value DESC
        """).fetchall()

        total_value = sum(float(row[4] or 0) for row in result)

        breakdown = [
            {
                "group_id": gid,
                "group_name": name,
                "company_count": company_count,
                "deal_count": deal_count,
                "total_value": float(value or 0),
                "percentage": round(float(value or 0) / total_value * 100, 1)
                if total_value > 0
                else 0,
            }
            for gid, name, company_count, deal_count, value in result
        ]

        return {
            "total_value": total_value,
            "breakdown": breakdown,
        }

    # =========================================================================
    # Attachment Methods
    # =========================================================================

    def search_attachments(
        self, query: str = "", company_id: str = "", file_type: str = "", limit: int = 20
    ) -> list[dict]:
        """Search attachments by title, company, or file type."""
        if not self._ensure_table("attachments"):
            return []

        conditions = []
        params = []

        if query:
            conditions.append("(LOWER(title) LIKE ? OR LOWER(summary) LIKE ?)")
            q = f"%{query.lower()}%"
            params.extend([q, q])

        if company_id:
            conditions.append("company_id = ?")
            params.append(company_id)

        if file_type:
            conditions.append("LOWER(file_type) LIKE ?")
            params.append(f"%{file_type.lower()}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        return self._fetch_all_dicts(
            f"SELECT * FROM attachments WHERE {where_clause} ORDER BY created_at DESC LIMIT {limit}",
            params,
        )
