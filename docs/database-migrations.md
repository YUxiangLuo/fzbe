# Database migrations and forecast evidence

## Deployment migration path

Run all database migrations through the ordered, idempotent entry point:

```bash
bun run db:migrate
```

The runner executes the academic-term, question-explanation, quiz-snapshot,
guided-training, model-integrity, production-preparation, canonical phone,
production-forecast-evidence, experiment-retry-lineage, and report-status-index
migrations, followed by the retry-delete-rule and canonical-phone convergence
audits and the production-inventory precision widening, in dependency order. It acquires a
database-scoped MySQL/MariaDB advisory lock before inspecting the
`_fzbe_schema_migrations` ledger, so concurrent deployments serialize instead
of running DDL at the same time. Only migrations without a successful ledger
row run; each row is inserted immediately after its script exits successfully.

On the first deployment that introduces the ledger, the new table starts empty
and the runner executes every migration. It deliberately does not infer a
baseline from a few columns or blindly mark an existing database as migrated:
the underlying scripts are idempotent, so this first pass both verifies an
already-current schema and finishes a partially migrated one. If a process
stops after DDL succeeds but before its ledger insert, the same idempotent script
runs again on the next deployment and is recorded only after a clean exit.

Migration connections use the same parsed database settings as the application,
including `DB_PORT`, `DB_CONNECT_TIMEOUT_MS`, `DB_TIMEZONE`, `DB_DATE_STRINGS`,
`DB_SSL_MODE`, `DB_SSL_CA_PATH`, and `DB_SSL_REJECT_UNAUTHORIZED`. The migration
helper intentionally loads only database configuration, so deployment does not
initialize the HTTP application or its connection pool. `DB_TIMEZONE` defaults
to `Z` and accepts only `Z` or a fixed offset such as `+08:00`. Both migration
connections and every pooled runtime connection execute `SET SESSION time_zone`
before application SQL. Startup then verifies `@@session.time_zone` and the
`NOW()`/`UTC_TIMESTAMP()` offset. This is required because mysql2's `timezone`
option controls JavaScript Date conversion but does not by itself change the
server session; relying on a MariaDB `SYSTEM` timezone would double-shift values
written by `NOW()` or `CURRENT_TIMESTAMP`.

`update.sh` serializes host-level deploys with `flock`, requires a clean worktree,
and verifies that the active `fangzhen-be.service` uses `KillMode=control-group`,
this checkout as its `WorkingDirectory`, and this checkout's `.env` as its
`EnvironmentFile`. It then
fetches and fast-forward-checks the target commit, creates a detached staging
worktree, copies the validated `.env`, installs dependencies (and the Python
environment when its inputs changed), and runs `actions.sh` there. Database
migrations therefore finish while the currently deployed service is still
running.

The service must have exactly one systemd `EnvironmentFile` and it must resolve
to the project `.env`. Deployment-sensitive direct `Environment=` assignments
(`DB_*`, `HOST`, `PORT`, `JWT*`, `ALLOW_*`, `NODE_ENV`, and `BUN_ENV`) are rejected;
`.env` itself must select production mode and `127.0.0.1:3001`. Staging migrations,
quiesced and post-activation reconciliation, and health-URL resolution run in
isolated transient-service environments that explicitly load the verified
`.env` and do not inherit the operator's shell; shell exports therefore cannot
redirect a migration, change a rollout switch, or move the health check to a
different listener. Deployment Bun commands run as
transient systemd services so the snapshot is parsed with the same
`EnvironmentFile` rules as the application service (in particular, Bun's
dotenv `$VARIABLE` expansion is not applied). The script compares that snapshot
byte-for-byte with the project `.env` before stopping, before final migration,
and across service startup. A concurrent configuration edit therefore aborts
activation; if it occurs after the service has stopped, automatic restart is
refused until an operator reconciles the environment deliberately.

Database changes are not part of the automatic rollback. If activation or its
health check fails, `update.sh` restores the previous application commit and
dependencies, but it does not remove migration-ledger rows or revert schema and
data changes that already succeeded. Treat the database as forward-only and
repair a failed deployment by fixing the problem and rerunning the migration.

Migration `013-production-inventory-precision` widens
`experiment_status.production_initial_inventory` from `FLOAT` to `DOUBLE`.
Production-plan MPS rows are stored as JSON numbers at JavaScript double
precision; keeping the scalar input as single precision could make a plan pass
when saved and fail the same formula replay when submitted. The migration also
repairs an existing scalar only when the first MPS row differs by no more than a
known MySQL/MariaDB representation of the same single-precision value, guarded
by `state_version`; merely close or materially different snapshots remain
untouched for the normal integrity validator to reject. The migration is
automatic and introduces no deployment setting. Because staging runs before
the legacy service is stopped, a compare-and-swap miss re-reads a still-safe
legacy snapshot and retries it at most three times. A state-versioned business
write that materially changes the scalar/MPS pair owns that new state and is
left untouched; repeated version-only races fail the migration before its
ledger row is written, so a later deployment can retry after write traffic
subsides without relying on an unreachable post-stop migration step.

Retry-lineage auditing is deliberately not trusted as a one-shot staging check.
The legacy release can create a later unlinked attempt or reject an earlier
report while staging runs. After stopping that writer, `update.sh` therefore
reruns the lineage audit from the staged release before advancing the live
checkout. A failure in this quiesced window cannot activate the new release: the
EXIT recovery path restores and health-checks the previous application commit.

Rejected-grade normalization is also deliberately not a one-shot ledger
migration. The old release can still accept legacy rejection writes while
staging runs, and it may run for longer after an activation rollback. Once the
new service is healthy, `update.sh` marks activation complete and then reruns
both the lineage audit as defense in depth and the idempotent rejected-grade
normalizer. A failure in this post-activation phase leaves the healthy new
writer active instead of restoring the legacy writer; the command exits nonzero
so the operator can repair the reported data and rerun the deployment.
Subsequent deploys retry both steps.
The normalizer walks the `(status, report_id)` index with a monotonic cursor and
commits at most 500 candidate reports at a time. This bounds row-lock duration
and makes partial progress safely repeatable instead of holding one transaction
across the complete report history.

Every deployment migration must therefore follow an expand/contract sequence
and remain backward compatible with both the currently running release and the
release being staged. Add nullable/defaulted columns, tables, or indexes first;
deploy code that can tolerate both schema versions; and remove, rename, or
tighten old structures only in a later release after every old application
instance has been retired. Do not combine a destructive or otherwise
backward-incompatible contract step with the release that first introduces its
replacement.

Staging also imports the target release's environment and derived concurrency/
timeout configuration under the verified snapshot, so invalid production secrets,
removed settings, listener values, or budgets fail before downtime. Only after
staging succeeds does the script stop the service, rerun the lineage
and canonical-phone audits with the legacy writer quiesced, fast-forward the
live checkout, refresh its dependencies, and start the service. Immediately
before stopping it rechecks that both `HEAD` and the worktree are unchanged, so
manual changes made during a long migration cannot be overwritten. Readiness is
confirmed through the minimal public `/api/v1/health` endpoint rather than by trusting the
initial `systemd` process state. Detailed `runtime-info` remains restricted to administrators in normal runtimes. A failure before that bounded stop window
leaves the old service untouched. A quiesced-audit or other failure after
stopping restores the previous commit, dependencies, and changed Python
environment before restarting and health-checking the old release; rollback
refuses to overwrite any unexpected checkout change. Both deployment scripts
use `set -euo pipefail`; resolve the reported schema, data, or host-preflight
problem before rerunning deployment.

The two new invariants can be verified after migration with:

```sql
SHOW INDEX FROM users WHERE Key_name = 'users_phone_number_unique';
SHOW COLUMNS FROM users LIKE 'phone_number_canonical';
SHOW INDEX FROM users WHERE Key_name = 'users_phone_number_canonical_unique';
SHOW COLUMNS FROM experiment_status LIKE 'production_forecast_evidence';
SHOW COLUMNS FROM experiment_status LIKE 'retry_of_experiment_id';
SHOW INDEX FROM experiment_status WHERE Key_name = 'experiment_status_retry_parent_unique';
SHOW INDEX FROM experiment_reports WHERE Key_name = 'idx_experiment_reports_status_report';
SELECT DELETE_RULE
  FROM information_schema.REFERENTIAL_CONSTRAINTS
 WHERE CONSTRAINT_SCHEMA = DATABASE()
   AND TABLE_NAME = 'experiment_status'
   AND CONSTRAINT_NAME = 'experiment_status_retry_parent_fk';
```

The rejected-report normalization clears only official report grades
(`experiment_reports.grade` and `experiment_grade.report_quality`). It deliberately
preserves flow, knowledge-test, and model-quality scores as historical evidence;
those scores are hidden from official grade views until a replacement attempt is
graded.

Migration `012-user-phone-canonical-unique` is intentionally separate from 007.
Databases may already have recorded 007 while the old release can still write
untrimmed phone values. Migration 012 normalizes existing values and adds a
stored `phone_number_canonical = NULLIF(TRIM(phone_number), '')` generated column
with a unique index. The initial duplicate audit reports only counts, not phone
numbers. If a legacy write races after that audit, the atomic index build either
includes it safely or fails the deployment; after the index exists, old writers
cannot insert a whitespace variant of an existing phone. An old writer can
still store a brand-new unique phone with surrounding whitespace, so after
stopping that writer `update.sh` unconditionally reruns 012 from the staged
release before activation. This bounded quiesced pass normalizes raw values even
when the ledger already contains 007 and 012. A failure restores and restarts
the previous release; successful DDL remains forward-compatible. Direct reruns
remain idempotent definition and data audits.

The destructive SQL regression for this migration is intentionally outside the
ordinary unit suite. Run it only against an empty disposable local database
accepted by the shared reset guard, with explicit opt-in, for example:

```bash
E2E_ALLOW_DB_RESET=1 DB_DATABASE=fz_test_phone_canonical \
  bun run test:system:user-phone-canonical
```

The retry self-reference uses `ON DELETE SET NULL`, not a recursive cascade.
Account deletion already removes every attempt through the `student_id` foreign
key. Severing the self-reference avoids InnoDB's finite cascade-depth limit for
students with a long rejection/retry chain, while normal experiment APIs still
forbid deleting completed parent attempts.

Migration `011-experiment-retry-delete-rule` is intentionally separate from
the original lineage migration. Some databases already recorded migration 009
while its self-reference still used `ON DELETE CASCADE`; the ledger must never
silently rerun or rewrite an applied migration. Migration 011 therefore runs on
both fresh databases and those already at 009/010, upgrades the known CASCADE
definition to SET NULL, and re-reads `information_schema` to audit the final
constraint. It also invokes the same fail-closed lineage audit as 009, because
an already-ledgered 009 may predate that audit. The direct
`db:migrate:experiment-retry-lineage` deployment audit remains defense in depth,
but bare `db:migrate` is sufficient for both the delete-rule and lineage-data
contracts.

The delete-rule migration can also be audited independently with:

```bash
bun run db:migrate:experiment-retry-delete-rule
```

## Legacy retry-lineage audit

The legacy schema recorded neither `retry_of_experiment_id` nor a rejection
timestamp. It therefore cannot prove that a later attempt was created after a
specific rejection: a student could create an attempt while a report was merely
awaiting review, and a teacher with an already-open review could reject the older
report afterward. The migration must not guess from adjacent ids or `start_time`.

During staging, again after the legacy writer is stopped, and once more after
the new writer is healthy, the idempotent 009 script validates two complementary
invariants. Every explicit retry parent must be the same student's immediately
preceding attempt by `experiment_id`, with no same-student row between parent
and child, and that parent must be completed and
rejected. In addition, once a rejected attempt exists, every later attempt must
remain in the explicit retry chain: a later row whose `retry_of_experiment_id` is
NULL fails the audit even when the rejected parent already has a valid explicit
child. This prevents an existing child from masking a still-ambiguous legacy
attempt and rejects stale-ancestor or branched manual backfills. Migration 011
calls that same shared audit so databases whose recorded 009 predates this check
are also protected by the ordinary ledger runner. Its error contains only
student and experiment ids. Resolve each candidate against operational/audit
records, then backfill only a confirmed, adjacent pair in a transaction, for
example:

```sql
START TRANSACTION;

SELECT parent.experiment_id AS parent_experiment_id,
       parent.student_id,
       parent.status AS parent_status,
       parent_report.status AS report_status,
       child.experiment_id AS child_experiment_id,
       child.retry_of_experiment_id
  FROM experiment_status parent
  JOIN experiment_reports parent_report
    ON parent_report.experiment_id = parent.experiment_id
   AND parent_report.student_id = parent.student_id
  JOIN experiment_status child
    ON child.student_id = parent.student_id
 WHERE parent.experiment_id = :verified_parent_experiment_id
   AND child.experiment_id = :verified_child_experiment_id
 FOR UPDATE;

UPDATE experiment_status
   SET retry_of_experiment_id = :verified_parent_experiment_id
 WHERE experiment_id = :verified_child_experiment_id
   AND student_id = :verified_student_id
   AND retry_of_experiment_id IS NULL;

COMMIT;
```

Rerun `bun run db:migrate:experiment-retry-lineage` after the verified update.
Do not assign a relationship when the external evidence is inconclusive.

## Forecast-evidence signing

Production forecast evidence is signed with a domain-separated HMAC key derived
from `JWT_SECRET`. The fixed derivation domain keeps evidence signatures
separate from authentication-token signatures while avoiding another deployment
secret. The single backend process needs no additional evidence-key environment
variables to configure or rotate.

Rotating `JWT_SECRET` invalidates evidence attached to production plans that
have not yet been submitted. Affected users must run production prediction
again before saving or submitting the plan. Completed experiment history is
immutable and is not revalidated against the current signing key, so it remains
available after a JWT rotation.

For the least disruption, rotate `JWT_SECRET` outside active classroom sessions
or notify users with pending plans that they will need to predict again.

`ALLOW_LEGACY_PRODUCTION_PLAN_WRITES` is a separate frontend-compatibility switch.
It defaults to `false` and should remain off for fresh and normal deployments.
Set it explicitly to `true` only while already-open old frontend pages may call
the newly restarted backend, then restore `false` as soon as every frontend uses
the v2 production-plan contract. When disabled, an active v1 write returns
`PRODUCTION_FORECAST_LEGACY_REJECTED` so the frontend can force a fresh
prediction; immutable completed v1 history remains readable.
