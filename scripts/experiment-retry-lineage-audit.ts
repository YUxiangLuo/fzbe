import type { Connection, RowDataPacket } from "mysql2/promise";

interface InvalidLineageRow extends RowDataPacket {
  child_experiment_id: number;
  child_student_id: number;
  parent_experiment_id: number;
  parent_student_id: number;
  parent_status: string;
  parent_report_status: string | null;
}

interface AmbiguousLegacyLineageRow extends RowDataPacket {
  parent_experiment_id: number;
  student_id: number;
  first_unlinked_experiment_id: number;
  unlinked_later_experiment_count: number;
}

/**
 * Fail closed when persisted retry relationships are invalid or legacy rows
 * cannot be linked without guessing. Kept separate from migration 009 so a
 * later ledger migration can enforce the same audit for databases where the
 * historical 009 row predates this contract.
 */
export async function auditExperimentRetryLineage(connection: Connection) {
  const [invalidLineage] = await connection.query<InvalidLineageRow[]>(
    `SELECT child.experiment_id AS child_experiment_id,
            child.student_id AS child_student_id,
            parent.experiment_id AS parent_experiment_id,
            parent.student_id AS parent_student_id,
            parent.status AS parent_status,
            parent_report.status AS parent_report_status
       FROM experiment_status child
       JOIN experiment_status parent
         ON parent.experiment_id = child.retry_of_experiment_id
       LEFT JOIN experiment_reports parent_report
         ON parent_report.experiment_id = parent.experiment_id
        AND parent_report.student_id = parent.student_id
      WHERE child.retry_of_experiment_id IS NOT NULL
        AND (
          child.student_id <> parent.student_id
          OR parent.experiment_id >= child.experiment_id
          OR EXISTS (
            SELECT 1
              FROM experiment_status intervening
             WHERE intervening.student_id = child.student_id
               AND intervening.experiment_id > parent.experiment_id
               AND intervening.experiment_id < child.experiment_id
          )
          OR parent.status <> 'Completed'
          OR parent_report.status IS NULL
          OR parent_report.status <> 'rejected'
        )
      ORDER BY child.experiment_id ASC
      LIMIT 20`,
  );
  if (invalidLineage.length > 0) {
    throw new Error(
      "Existing retry lineage violates the rejected-parent contract; manual repair required: "
      + JSON.stringify(invalidLineage),
    );
  }

  const [ambiguousLegacyLineage] = await connection.query<AmbiguousLegacyLineageRow[]>(
    `SELECT parent.experiment_id AS parent_experiment_id,
            parent.student_id AS student_id,
            MIN(later.experiment_id) AS first_unlinked_experiment_id,
            COUNT(*) AS unlinked_later_experiment_count
       FROM experiment_status parent
       JOIN experiment_reports parent_report
         ON parent_report.experiment_id = parent.experiment_id
        AND parent_report.student_id = parent.student_id
        AND parent_report.status = 'rejected'
       JOIN experiment_status later
         ON later.student_id = parent.student_id
        AND later.experiment_id > parent.experiment_id
      WHERE later.retry_of_experiment_id IS NULL
      GROUP BY parent.experiment_id, parent.student_id
      ORDER BY parent.experiment_id ASC
      LIMIT 20`,
  );
  if (ambiguousLegacyLineage.length > 0) {
    // The legacy schema has no rejected_at timestamp. A later attempt could
    // have been created while the parent report was merely awaiting review,
    // so adjacency/start_time is not proof of a retry. Never guess a parent.
    throw new Error(
      "Ambiguous legacy retry lineage detected; verify and backfill explicitly before retrying: "
      + JSON.stringify(ambiguousLegacyLineage),
    );
  }
}
