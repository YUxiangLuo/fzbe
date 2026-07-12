import crypto from "node:crypto";
import mysql, { type PoolConnection, type ResultSetHeader, type RowDataPacket } from "mysql2/promise";

const requireEnv = (name: string) => {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing required env var: ${name}`);
  return value;
};

const database = requireEnv("DB_DATABASE");
if (!/(?:^|_)(?:test|ci)(?:_|$)/i.test(database)) {
  throw new Error("Concurrency verification must run against a dedicated test/CI database");
}

const pool = mysql.createPool({
  host: requireEnv("DB_HOST"),
  user: requireEnv("DB_USER"),
  password: process.env.DB_PASSWORD ?? "",
  database,
  connectionLimit: 8,
});

const assert: (condition: unknown, message: string) => asserts condition = (condition, message) => {
  if (!condition) throw new Error(message);
};

const claim = async (
  connection: PoolConnection,
  sessionId: string,
  expectedVersion: number,
  token: string,
) => {
  const [result] = await connection.query<ResultSetHeader>(
    `UPDATE model_guided_training_sessions
        SET status = 'running',
            current_step_id = 'prepare_data',
            run_token = ?,
            run_version = run_version + 1,
            last_heartbeat_at = NOW(3),
            lease_expires_at = TIMESTAMPADD(SECOND, 90, NOW(3))
      WHERE session_id = ?
        AND run_version = ?
        AND (
          status IN ('ready', 'failed')
          OR (status = 'running' AND lease_expires_at <= NOW(3))
        )`,
    [token, sessionId, expectedVersion],
  );
  return result.affectedRows;
};

const connectionA = await pool.getConnection();
const connectionB = await pool.getConnection();
const experimentId = 910_000_001;
const studentId = 910_000_001;
const sessionId = crypto.randomUUID();
const cleanupRaceSessionId = crypto.randomUUID();

try {
  await pool.query("DELETE FROM users WHERE user_id = ?", [studentId]);
  await pool.query("INSERT INTO users (user_id) VALUES (?)", [studentId]);
  await pool.query(
    `INSERT INTO experiment_status
       (experiment_id, student_id, state_version, training_input_version)
     VALUES (?, ?, 0, 0)`,
    [experimentId, studentId],
  );
  await pool.query(
    `INSERT INTO model_guided_training_sessions
       (session_id, experiment_id, student_id, model_type, request_hash,
        request_payload, status, current_step_id, completed_step_ids,
        step_outputs, step_plan, artifact_dir, training_input_version, expires_at)
     VALUES (?, ?, ?, 'ma', REPEAT('a', 64), JSON_OBJECT(), 'ready',
             'prepare_data', JSON_ARRAY(), JSON_OBJECT(), JSON_ARRAY(), ?, 0,
             DATE_ADD(NOW(), INTERVAL 1 DAY))`,
    [sessionId, experimentId, studentId, `/tmp/fz-guided-concurrency-${sessionId}`],
  );
  await pool.query(
    `INSERT INTO model_guided_training_sessions
       (session_id, experiment_id, student_id, model_type, request_hash,
        request_payload, status, current_step_id, completed_step_ids,
        step_outputs, step_plan, artifact_dir, training_input_version, expires_at)
     VALUES (?, ?, ?, 'ma', REPEAT('c', 64), JSON_OBJECT(), 'ready',
             'prepare_data', JSON_ARRAY(), JSON_OBJECT(), JSON_ARRAY(), ?, 0,
             DATE_SUB(NOW(), INTERVAL 1 SECOND))`,
    [
      cleanupRaceSessionId,
      experimentId,
      studentId,
      `/tmp/fz-guided-cleanup-race-${cleanupRaceSessionId}`,
    ],
  );

  const tokenA = crypto.randomUUID();
  const tokenB = crypto.randomUUID();
  const firstClaims = await Promise.all([
    claim(connectionA, sessionId, 0, tokenA),
    claim(connectionB, sessionId, 0, tokenB),
  ]);
  assert(firstClaims[0] + firstClaims[1] === 1, "exactly one concurrent claimant must win");

  const [ownerRows] = await pool.query<RowDataPacket[]>(
    "SELECT run_token, run_version FROM model_guided_training_sessions WHERE session_id = ?",
    [sessionId],
  );
  const firstOwnerToken = String(ownerRows[0]?.run_token);
  const staleToken = firstOwnerToken;
  const replacementToken = crypto.randomUUID();

  await pool.query(
    "UPDATE model_guided_training_sessions SET lease_expires_at = DATE_SUB(NOW(3), INTERVAL 1 SECOND) WHERE session_id = ?",
    [sessionId],
  );
  const [expiredOwnerCommit] = await connectionA.query<ResultSetHeader>(
    `UPDATE model_guided_training_sessions
        SET status = 'ready', completed_step_ids = JSON_ARRAY('prepare_data')
      WHERE session_id = ?
        AND status = 'running'
        AND run_token = ?
        AND run_version = 1
        AND lease_expires_at > NOW(3)`,
    [sessionId, staleToken],
  );
  assert(
    expiredOwnerCommit.affectedRows === 0,
    "an owner must be fenced as soon as its lease expires",
  );
  assert(
    await claim(connectionB, sessionId, 1, replacementToken) === 1,
    "an expired lease must be reclaimable",
  );

  const [staleCommit] = await connectionA.query<ResultSetHeader>(
    `UPDATE model_guided_training_sessions
        SET status = 'ready', completed_step_ids = JSON_ARRAY('prepare_data')
      WHERE session_id = ?
        AND status = 'running'
        AND run_token = ?
        AND run_version = 1
        AND lease_expires_at > NOW(3)`,
    [sessionId, staleToken],
  );
  assert(staleCommit.affectedRows === 0, "a stale worker must be fenced from committing");

  const [currentCommit] = await connectionB.query<ResultSetHeader>(
    `UPDATE model_guided_training_sessions
        SET status = 'ready', completed_step_ids = JSON_ARRAY('prepare_data'),
            run_token = NULL, lease_expires_at = NULL
      WHERE session_id = ?
        AND status = 'running'
        AND run_token = ?
        AND run_version = 2
        AND lease_expires_at > NOW(3)`,
    [sessionId, replacementToken],
  );
  assert(currentCommit.affectedRows === 1, "the current lease owner must be able to commit");

  const stateCas = async (connection: PoolConnection) => {
    const [result] = await connection.query<ResultSetHeader>(
      `UPDATE experiment_status
          SET state_version = state_version + 1,
              training_input_version = training_input_version + 1
        WHERE experiment_id = ? AND state_version = 0`,
      [experimentId],
    );
    return result.affectedRows;
  };
  const stateWrites = await Promise.all([stateCas(connectionA), stateCas(connectionB)]);
  assert(stateWrites[0] + stateWrites[1] === 1, "state-version CAS must admit one writer");

  assert(
    await claim(connectionA, cleanupRaceSessionId, 0, crypto.randomUUID()) === 1,
    "a worker must be able to claim the initially expired cleanup candidate",
  );
  const [cleanupAfterClaim] = await connectionB.query<ResultSetHeader>(
    `DELETE FROM model_guided_training_sessions
      WHERE session_id = ?
        AND (
          status <> 'running'
          OR lease_expires_at IS NULL
          OR lease_expires_at <= NOW(3)
        )
        AND expires_at IS NOT NULL
        AND expires_at <= NOW(3)
        AND NOT EXISTS (
          SELECT 1
            FROM experiment_model_artifacts active_artifact
           WHERE active_artifact.source_session_id = model_guided_training_sessions.session_id
        )`,
    [cleanupRaceSessionId],
  );
  assert(
    cleanupAfterClaim.affectedRows === 0,
    "cleanup must recheck eligibility and preserve a newly claimed session",
  );

  const productionTokenA = crypto.randomUUID();
  const productionTokenB = crypto.randomUUID();
  await pool.query(
    `INSERT INTO experiment_model_artifacts
       (experiment_id, student_id, model_type, revision_token,
        training_input_version, source_session_id, backtest_artifact_path,
        request_payload, result_payload, production_artifact_path,
        production_revision_token)
     VALUES (?, ?, 'ma', ?, 1, NULL, '/tmp/backtest.pkl',
             JSON_OBJECT(), JSON_OBJECT(), NULL, NULL)`,
    [experimentId, studentId, sessionId],
  );
  const claimProduction = async (token: string) => {
    const [result] = await pool.query<ResultSetHeader>(
      `UPDATE experiment_model_artifacts
          SET production_artifact_path = NULL,
              production_revision_token = ?
        WHERE experiment_id = ?
          AND model_type = 'ma'
          AND revision_token = ?
          AND training_input_version = 1`,
      [token, experimentId, sessionId],
    );
    return result.affectedRows;
  };
  assert(await claimProduction(productionTokenA) === 1, "first production attempt must claim");
  assert(await claimProduction(productionTokenB) === 1, "replacement production attempt must claim");
  const publishProduction = async (token: string, path: string) => {
    const [result] = await pool.query<ResultSetHeader>(
      `UPDATE experiment_model_artifacts
          SET production_artifact_path = ?,
              production_revision_token = revision_token
        WHERE experiment_id = ?
          AND model_type = 'ma'
          AND revision_token = ?
          AND production_revision_token = ?`,
      [path, experimentId, sessionId, token],
    );
    return result.affectedRows;
  };
  const staleProductionPublish = await publishProduction(productionTokenA, "/tmp/stale-production.pkl");
  const replacementProductionPublish = await publishProduction(productionTokenB, "/tmp/current-production.pkl");
  assert(staleProductionPublish === 0, "obsolete production worker must be fenced from publishing");
  assert(replacementProductionPublish === 1, "replacement production worker must be able to publish");

  const lockName = `fangzhen-model:exp:${experimentId}`;
  const [[lockA]] = await connectionA.query<RowDataPacket[]>("SELECT GET_LOCK(?, 0) AS acquired", [lockName]);
  const [[lockBWhileBusy]] = await connectionB.query<RowDataPacket[]>("SELECT GET_LOCK(?, 0) AS acquired", [lockName]);
  assert(Number(lockA?.acquired) === 1, "first experiment lock acquisition must succeed");
  assert(Number(lockBWhileBusy?.acquired) === 0, "second experiment lock acquisition must fail while busy");
  await connectionA.query("SELECT RELEASE_LOCK(?)", [lockName]);
  const [[lockBAfterRelease]] = await connectionB.query<RowDataPacket[]>("SELECT GET_LOCK(?, 0) AS acquired", [lockName]);
  assert(Number(lockBAfterRelease?.acquired) === 1, "experiment lock must be reusable after release");
  await connectionB.query("SELECT RELEASE_LOCK(?)", [lockName]);

  const [[finalSession]] = await pool.query<RowDataPacket[]>(
    "SELECT status, run_version, completed_step_ids FROM model_guided_training_sessions WHERE session_id = ?",
    [sessionId],
  );
  const [[finalExperiment]] = await pool.query<RowDataPacket[]>(
    "SELECT state_version, training_input_version FROM experiment_status WHERE experiment_id = ?",
    [experimentId],
  );
  console.log(JSON.stringify({
    concurrent_claim_results: firstClaims,
    expired_owner_commit_affected_rows: expiredOwnerCommit.affectedRows,
    stale_commit_affected_rows: staleCommit.affectedRows,
    replacement_commit_affected_rows: currentCommit.affectedRows,
    state_cas_results: stateWrites,
    cleanup_after_claim_affected_rows: cleanupAfterClaim.affectedRows,
    stale_production_publish_affected_rows: staleProductionPublish,
    replacement_production_publish_affected_rows: replacementProductionPublish,
    final_session: finalSession,
    final_experiment: finalExperiment,
  }, null, 2));
} finally {
  await connectionA.query("SELECT RELEASE_ALL_LOCKS()").catch(() => {});
  await connectionB.query("SELECT RELEASE_ALL_LOCKS()").catch(() => {});
  connectionA.release();
  connectionB.release();
  await pool.query("DELETE FROM users WHERE user_id = ?", [studentId]).catch(() => {});
  await pool.end();
}
